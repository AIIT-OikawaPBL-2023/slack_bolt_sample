import os
import re
from urllib.parse import unquote

import faiss
import numpy as np
import pandas as pd
import spacy
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


class VectorSearch:
    def __init__(
        self,
        model_name="bert-large-uncased",
        device="auto",
        chunk_size=80,
        chunk_overlap=15,
    ):
        """
        ベクトルデータベースの設定値を渡して初期化するコンストラクタ。
        「良いベクトル」を料理するためにはかなり何回も実験回して調整が必要だという見通しなうえに、
        Notion全データが対象となるとベクトル化を毎回OpenAI APIで試すと料金がかかりすぎる心配があるので
        ベクトル化の手元実験用にBERTを実装しました。
        デプロイ時にBERT使っちゃうと重すぎるので、その時にはOpenAI APIに置き換える想定です。

        Args:
            model_name (str): 使用するモデル名
            device (str): GPU使うのかCPU使うのかを指定
            chunk_size (int): テキストを分割する際の文字数
            chunk_overlap (int): テキストを分割する際のオーバーラップする文字数
                チャンク切った時に文脈が失われる箇所が生じないようにするためのもの
                例えばchunk_size=80, chunk_overlap=15の場合は、
                1-80文字目、65-145文字目、130-210文字目...というように分割する
        """
        if device == "auto":
            # autoに設定しているときはGPUが使える場合はGPUを使う
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # auto以外の場合は指定されたデバイスを使用
            self.device = device
        print(f"Device: {self.device}")  # GPU使ってるかCPU使ってるかを表示
        # 実験作業用にOpenAI APIではなくローカルで動くBERTを使っています
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        # BERTのモデルを読み込むとき警告出ますが無視してOKなのでメッセージ出しとく
        print("上記警告は無視してOK")
        # embeddingするモデルごとにベクトル次元数が異なるので、モデル名に応じて次元数を設定
        # ちなみにOpenAI APIのtext-embedding-ada-002を使う場合は1536次元になります
        # 基本傾向としては次元数が大きい方が精度は上がりますが計算量も増えます
        if model_name == "bert-large-uncased":
            self.dim_vector = 1024
        elif model_name == "bert-base-uncased":
            self.dim_vector = 768
        # テキスト前処理用の自然言語モデルの読み込み
        self.nlp = spacy.load("ja_core_news_lg")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def init_dataset(self, dir_path):
        """
        Notionから取得したデータセットを読み取ってベクトルデータベース化するメソッド

        Args:
            dir_path (str): データセットの読み取り先ディレクトリパス

        Returns:
            self.df_corpus (pd.DataFrame): 処理後のデータセット
        """
        rows_list = []
        # エクスポートしたNotionファイルを保存した`./OikawaPBL2023`フォルダ配下の子フォルダごとに処理を行うためのループ文
        for j, (dirpath, dirnames, filenames) in enumerate(tqdm(os.walk(dir_path), total=self.file_counting(dir_path)[1])):
            for filename in filenames:  # 一つの子フォルダの中の各ファイルごとに処理を実行
                file_path = os.path.join(dirpath, filename)  # ディレクトリ名とファイル名を結合してファイルパスを作成
                chunks = self.load_file(file_path)  # 各ファイルからテキストを抽出してchunk化し、各chunkを格納したリストを返す
                if chunks is not None:  # テキストが抽出できた場合のみ処理を実行
                    for chunk in chunks:  # chunkされた全データを格納したリストから各chunkを取り出して処理を実行
                        chunk = self.remove_stopwords(chunk)  # ストップワード（要らん文字列）を除去
                        chunk_vectorized = self.vectorize_text(chunk)  # ここでchunkをベクトル化
                        # あとでDataFrameの各行に変換される辞書型データを作成
                        row = {
                            "filename": file_path,  # ファイル保存先パス
                            "chunk_vectorized": chunk_vectorized,  # chunkをベクトル化したもの
                            "chunk_source": chunk,
                        }  # ベクトル化する前のchunk
                        rows_list.append(row)  # 作成した辞書型データをリストに追加。あとでこれをDataFrameに変換する

        # ここでrows_listを変換してベクトルデータベースを作成（データベースといってもDataFrameですが…）
        self.df_corpus = pd.DataFrame(rows_list)
        # 重複の削除（Notion上にテーブルビューがたくさんあるので重複めっちゃ多いです）
        self.df_corpus.drop_duplicates(subset="chunk_source", inplace=True)
        # 検索に用いるベクトルのndarrayをDataFrameから作成
        self.chunk_vectorized = np.array(self.df_corpus["chunk_vectorized"].tolist()).astype("float32")
        # ベクトルの大きさによる傾向差が生じるのを抑えるためL2正規化を行う
        # L2正則化を行ってもなおベクトルの大きさによる傾向差が生じるのが悩み。ここは模索中。
        # 直接変数が更新されるため返り値は出ない
        faiss.normalize_L2(self.chunk_vectorized)
        # DataFrameのインデックスを正規化後の値に更新
        self.df_corpus["chunk_vectorized"] = list(self.chunk_vectorized)
        # 探索用のインデックスを作成
        self.init_faiss_indexer()
        return self.df_corpus

    def answer(self, query, n_answer=15):
        # query文字列をベクトル化
        query_vectorized = self.vectorize_text(query)
        # 探索を実行。上位n件のテキストのindexを取得
        idx = self.search_index(query_vectorized, n_answer)
        # 類似度上位のテキストを返す（idxがネストしているので[0]を渡して使用）
        results = self.df_corpus.chunk_source.iloc[idx[0]]
        return list(results)

    def load_file(self, file_path):
        _, ext = os.path.splitext(file_path)
        # 拡張子に応じて読み込み方を変える
        # マークダウンファイルの場合の読み取り
        if ext == ".md":
            with open(file_path, "r") as f:
                content = f.read()
            chunks = self.split_text(content)  # 読み取ったテキストをchunk化
            return chunks
        # csvファイルの読み取り。基本、NotionDBはcsvでエクスポートされます
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            # この辺はまだ雑な処理。シンプルに各行を空白で結合してプレーンテキスト化しまっている
            # 本当はこの辺もそのまま表データとしてChatGPTに渡せた方が良い
            content = " ".join(df.values.flatten().astype(str))
            chunks = self.split_text(content)  # 読み取ったテキストをchunk化
            return chunks
        else:
            return None

    def split_text_recursive_character(self, content):
        """
        LangChainのRecursiveCharacterTextSplitterを用いて
        分割する場合のメソッド。今回は使わないことにしたので、
        このメソッドはこのクラス内では利用されていません。参考用。
        でもうまく使えればこっちの方が最終的には性能出せるかも。

        Args:
            content (str): テキストデータ

        Returns:
            list: テキストデータを分割したリスト
        """

        # このメソッドを実際に使う場合にはtext_splitterはここじゃなくて
        # コンストラクタ(__init__)でインスタンス変数として一度だけ作成する想定
        # ここに配置しているとメソッドが呼び出されるたびにインスタンス化されるので計算効率がわるい
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["。", "\n"],  # セパレータを指定
            keep_separator=False,  # チャンキングしたときにセパレータを残すかどうか
            chunk_size=self.chunk_size,  # チャンクの文字数
            chunk_overlap=self.chunk_overlap,  # チャンクオーバーラップの文字数
        )
        chunks = text_splitter.split_text(content)
        return chunks

    def split_text(self, content):
        """
        色々な手法を試しましたが結局一律の文字数で区切った方が
        クエリの文字数に依存しない検索が可能になるという点で
        優位性がみられたため、いったん文字数ベースのスプリッターを採用しました。
        文字数ベース以外には句読点や特定の記号など意味的な区切り文字を
        使ってスプリットする方法もあり、LangchainのRecursiveCharacterTextSplitterなど
        既に色々試行済みではありますが、逆に性能さがったりするので今回は不採用。
        幅広い手法があるので今後も検討の余地があります。

        Args:
            content (str): テキストデータ

        Returns:
            list: テキストデータを分割したリスト
        """
        chunks = []  # いっこいっこのchunkを格納するリスト型変数
        start = 0
        # whileループを使って、startがcontentの長さを超えない間、ループを継続
        # 各イテレーションでは、startからstart + chunk_sizeまでの部分文字列（つまりチャンク）を取得
        while start < len(content):
            end = start + self.chunk_size
            chunk = content[start:end]
            chunks.append(chunk)
            # 次のイテレーションに移る前に、startをchunk_overlapだけ後退させる。
            # これにより、次のチャンクは現在のチャンクの最後のchunk_overlap文字から始まる。
            start = end - self.chunk_overlap
        return chunks

    def vectorize_text(self, text):
        text_tokenized = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        # 入力データを辞書型にしてdevice(gpu/cpu)に転送
        text_tokenized = {key: value.to(self.device) for key, value in text_tokenized.items()}
        # ベクトル化
        outputs = self.model(**text_tokenized)
        # ベクトルの平均を取る（この時点ではTensor型になっているので次の行でnumpy配列に変換）
        text_vectorized = outputs[0][0].mean(dim=0)
        # ベクトルをCPUに戻してnumpy配列に戻して返り値とする
        return text_vectorized.detach().cpu().numpy()

    def file_counting(self, directory):
        total_files = 0
        total_dirs = 1

        for dirpath, dirnames, filenames in os.walk(directory):
            total_files += len(filenames)
            total_dirs += len(dirnames)

        return total_files, total_dirs

    def init_faiss_indexer(self):
        # このメソッドは以前プルリク立て済みの内容です。
        """
        クラスタリングによってデータ空間をボロノイ領域に分割することにより
        高速な近傍探索を可能にするためのindexerを初期化するメソッド
        ボロノイ領域の紹介はこちら --> https://ja.wikipedia.org/?curid=91418
        このアルゴリズムでは質問クエリに対して、その周辺領域のみで探索できるので
        総当たりせずに済むので探索にかかる時間を大幅に短縮できる
        """

        # ボロノイ領域の数
        # クラスタリング空間を定義する量子化器（Quantizer）つくる（コサイン類似度）
        self.quantizer = faiss.IndexFlatIP(self.dim_vector)
        # 下記引数の20はボロノイ領域の数（ボロノイの数は増やすと精度上がって処理時間増えるトレードオフ）
        self.indexer = faiss.IndexIVFFlat(self.quantizer, self.dim_vector, 20)
        # ベクトルデータベースからボロノイ領域を生成
        self.indexer.train(self.chunk_vectorized)

        # インデックス作成器にデータを追加
        self.indexer.add(self.chunk_vectorized)

    def search_index(self, query_vectorized, n_answer):
        """
        近傍探索を実行するメソッド

        Args:
            query_vectorized (np.ndarray): 質問クエリをembeddingしたやつ

        Returns:
            results (pandas.DataFrame): 上位n件のテキスト情報を格納したデータフレーム
        """
        query_vectorized = np.array([query_vectorized]).astype("float32")
        faiss.normalize_L2(query_vectorized)  # ベクトルをL2正規化
        # 近傍探索の実行
        _, idx = self.indexer.search(query_vectorized, n_answer)
        return idx

    def remove_stopwords(self, text):
        """
        入力データにかなり多量の記号が含まれるので、現状はかなり破壊的に
        記号を除去する処理をしている。でも本当はそうではなく、マークダウン形式
        における連想配列構造やCSVデータのテーブル構造は保ちたい。
        それらを保ったデータをDataFrameに格納する方法は現在探し中。

        Args:
            text (str): テキストデータ

        Returns:
            text (str): ストップワードを除去したテキストデータ
        """

        # 記号を除去
        text = self.remove_sym(text)
        # 各種形式のハッシュ値を除去
        text = re.sub(r"[a-zA-Z0-9]{10,}", "", text)
        # URLエンコードをデコード
        text = unquote(text)
        # 大文字小文字の英数字と記号の10文字以上連続する文字列を除去
        text = re.sub(r"[a-zA-Z0-9\.\?]{10,}", "", text)
        # 'nan'を除去
        text = re.sub(r"nan", "", text)
        # 連続する空白を除去
        text = re.sub(r"\s+", " ", text)
        # 全角スペースを半角に変換（後続の処理で連続する空白が一括で削除される）
        text = re.sub(r"　", " ", text)
        # 全角英数字を半角に変換
        text = re.sub(r"[０-９Ａ-Ｚａ-ｚ]", lambda x: chr(ord(x.group(0)) - 0xFEE0), text)
        # 半角カタカナを全角に変換
        text = re.sub(r"[ｦ-ﾟ]", lambda x: chr(ord(x.group(0)) + 0xFEE0), text)
        # '�'を除去
        text = re.sub(r"�", "", text)
        return text

    def remove_sym(self, raw_text):
        processed_text = ""
        text = self.nlp(str(raw_text))

        for t in text:
            if t.pos_ != "SYM":  # 記号を除去
                processed_text += str(t)
        return processed_text


if __name__ == "__main__":
    # ベクトル検索をするためのインスタンスを作成。上記で定義しているクラス。
    vector_search = VectorSearch(
        model_name="bert-large-uncased", device="auto", chunk_size=300, chunk_overlap=80  # BERTのモデル。`bert-base-uncased`という速度重視のモデルもある。  # cpu使うんかgpu使うんかのやつ  # 長い文章を何文字で区切るか
    )  # 区切る時に変なところで切れてしまった文章が保持していた文脈を保つために文字列の重複を何文字にするか

    # データを読み込ませてベクトルデータベース化するメソッドを実行。これで検索できるようになる
    df_corpus = vector_search.init_dataset(dir_path="./search_all_document/data")

    while True:
        print("さあ、なんでも聞いてね。（`Ctrl + C`で停止します。）")
        # ユーザー質問文入力。いろいろ変えてみてください！（まだなかなかいい結果でません(;ω;)）
        query = str(input())
        print(vector_search.answer(query, n_answer=15), '\n')  # 類似度上位何位まで表示するかを引数で指定できるよーにしています。
