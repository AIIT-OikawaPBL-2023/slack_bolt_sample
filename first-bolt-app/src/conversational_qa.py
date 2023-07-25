import argparse
import os

import rich
from dotenv import load_dotenv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chains.conversational_retrieval.prompts import \
    CONDENSE_QUESTION_PROMPT
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()


class ConversationalRetrievalQAChain:
    def __init__(self, chain_type='stuff', max_chunk_size=1000):
        """
        コンストラクタ。インスタンスの設定値を受け取り初期化する

        Args:
            chain_type (str, optional):
                'stuff' or 'map_reduce'
                'stuff'は通常回答。'map_reduce'は要約文生成後に回答
            max_chunk_size (int, optional): 元のドキュメントをチャンキングする際の最大文字数
        """
        self.chain_type = chain_type  # -> str 'stuff' or 'map_reduce'
        self.max_chunk_size = max_chunk_size  # -> int チャンキングする際の最大文字数
        self.init_text_splitter()  # -> CharacterTextSplitter ドキュメントをチャンキングするための分割器
        self.init_embeddings()  # -> OpenAIEmbeddings ベクトル変換器
        self.chat_history = []  # -> List[Tuple[str, str]] 対話の履歴を保持する変数

    def load_data(self, src_path):
        """
        与えられたパスからデータを読み取ってベクトルデータストアを作成し、
        これを使って質問応答チェーンを初期化する

        Args:
            src_path (str): 読み取るデータのパス
        """
        loader = TextLoader(src_path)
        documents = loader.load()
        self.texts = self.text_splitter.split_documents(documents)  # -> List[Document] テキスト分割を実行
        self.init_vectorstore()  # -> Chroma ベクトルデータストアを作成
        self.init_conversational_retrieval_qa_chain()  # -> ConversationalRetrievalChain 質問応答チェーンを初期化

    def question(self, question):
        """
        質問を受け取り、回答を返すメソッド

        Args:
            question (str): ユーザーが入力した質問文

        Returns:
            response (dict): 質問に対する回答。回答の根拠となる文の情報も含まれる。
        """

        # __call__はもちろん省略しても実行されるけど明示するためここではこう書いた
        response = self.qa.__call__(inputs={"question": question,
                                            'chat_history': self.chat_history},
                                    return_only_outputs=False,
                                    include_run_info=True,
                                    callbacks=None)
        self.chat_history.append((question, response['answer']))  # 対話履歴に追加
        return response

    def init_text_splitter(self):
        self.text_splitter = CharacterTextSplitter(chunk_size=self.max_chunk_size,
                                                   chunk_overlap=0,
                                                   length_function=len,
                                                   keep_separator=False,
                                                   add_start_index=False)

    def init_embeddings(self):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            embedding_ctx_length=8191,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),  # 環境変数`OPENAI_API_KEY`に格納されているなら省略可
            chunk_size=self.max_chunk_size,
            max_retries=6,
            show_progress_bar=True,
        )

    def init_vectorstore(self):
        """ベクトルデータストアを作成する
        """
        self.vectorstore = Chroma.from_documents(documents=self.texts, embedding=self.embeddings)

    def init_conversational_retrieval_qa_chain(self):
        """今回の実装における本体部分を構成する質問応答チェーンを初期化する"""
        self.qa = ConversationalRetrievalChain(
            combine_docs_chain=self.combine_docs_chain,  # -> BaseCombineDocumentsChain ドキュメントを結合するチェーン
            question_generator=self.question_generator,  # -> BaseLanguageModel 質問文を生成する言語モデル
            retriever=self.vectorstore.as_retriever(),  # -> BaseRetriever ベクトルデータストアを検索するリトリーバー
            return_source_documents=True,  # -> bool ドキュメントの内容を返すかどうか
            return_generated_question=False,  # -> bool 生成された質問文を返すかどうか
            rephrase_question=True,  # -> bool 質問文を再構成するかどうか
            max_tokens_limit=None,  # -> int 最大トークン数
            output_key='answer',  # -> str 出力のキー
        )

    @property
    def main_llm(self):
        if self.chain_type == 'stuff':
            # stuffの場合、callback_managerを使って対話をリアルタイム表示する
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        else:
            # map_reduceの場合、callback_managerを使うと反復表示され違和感を生じるためNoneにしている
            callback_manager = None

        # 質問応答チェーンの本体部分で使う言語モデルを初期化する
        main_llm = ChatOpenAI(temperature=0,
                              openai_api_key=os.environ.get("OPENAI_API_KEY"),
                              model_name="gpt-3.5-turbo-0613",
                              streaming=True,
                              callback_manager=callback_manager)  # -> BaseLanguageModel
        return main_llm

    @property
    def condense_llm(self):
        # 質問文再生成で使う言語モデルを初期化する
        # callback機能が不要であるほか、main_llmにgpt4を使う場合でもこちらは
        # より軽量のモデルを使うことが想定されるので、別のインスタンスを作成している
        condense_llm = ChatOpenAI(temperature=0,
                                  model="gpt-3.5-turbo",
                                  openai_api_key=os.environ.get("OPENAI_API_KEY"))
        return condense_llm

    @property
    def combine_docs_chain(self):  # -> BaseCombineDocumentsChain
        """
        ドキュメントを結合するチェーンを初期化する

        Returns:
            combine_docs_chain (BaseCombineDocumentsChain): ドキュメントを結合するチェーン
        """
        map_reduce_params = {}
        if self.chain_type == 'map_reduce':
            # map_reduceの場合、質問文生成用の言語モデルを設定する
            map_reduce_params['question_prompt'] = self.question_prompt_template
            map_reduce_params['combine_prompt'] = self.combine_prompt_template

        # ドキュメントを結合するチェーンを生成する関数を呼び出す
        # 回答文生成の根拠となる文を抽出する処理が含まれる
        combine_docs_chain = load_qa_with_sources_chain(llm=self.main_llm, # -> BaseLanguageModel 質問応答チェーンの本体部分で使う言語モデル
                                                        chain_type=self.chain_type, # -> str 'stuff' or 'map_reduce'
                                                        verbose=None, # -> bool ログを出力するかどうか
                                                        **map_reduce_params) # -> dict 質問文生成用の言語モデルをchain_typeに応じて設定する
        return combine_docs_chain

    @property
    def question_generator(self):
        # 質問文を再生成する言語モデルを初期化する
        # 質問文が過度に長かったり、文法がおかしかったりする場合に調整する役割
        question_generator = LLMChain(llm=self.condense_llm,
                                      prompt=CONDENSE_QUESTION_PROMPT)  # -> BaseLanguageModel 質問文を再生成する言語モデル
        return question_generator

    @property
    def question_prompt_template(self):
        if self.chain_type == 'stuff':
            return None

        self.question_prompt = """
            Use the following portion of a long document to see if any of the text is relevant to answer the question.
            Return any relevant text translated into Japanese.
            {context}
            Question: {question}
            Relevant text, if any, in Japanese:"""

        return PromptTemplate(template=self.question_prompt,
                              input_variables=["context", "question"])

    @property
    def combine_prompt_template(self):
        if self.chain_type == 'stuff':
            return None

        combine_prompt = """
            Given the following extracted parts of a long document and a question, create a final answer Japanese.
            If you don't know the answer, just say that you don't know. Don't try to make up an answer.

            QUESTION: {question}
            =========
            {summaries}
            =========
            Answer in Japanese:"""

        return PromptTemplate(template=combine_prompt,
                              input_variables=["summaries", "question"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', '-s', type=str,
                        default='./first-bolt-app/data/[sample]scram_rules.txt',
                        help='path to source text file')
    parser.add_argument('--chain_type', '-c', type=str,
                        default='stuff',
                        choices=['stuff', 'map_reduce'])
    parser.add_argument('--max_chunk_size', '-m', type=int,
                        default=1000)
    args = parser.parse_args()

    qa_instance = ConversationalRetrievalQAChain(args.chain_type, args.max_chunk_size)
    qa_instance.load_data(args.src_path)

    while True:
        print("質問を入力してください。（終了する場合：`Ctrl + C`）")
        question = str(input())
        response = qa_instance.question(question)
        rich.print(response)
