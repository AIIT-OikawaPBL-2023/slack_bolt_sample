import os
import thesis_interactor as thesis

from dotenv import load_dotenv
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import AIMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate  # System メッセージテンプレート; assistant メッセージテンプレート; user メッセージテンプレート
from langchain.schema import AIMessage, HumanMessage, SystemMessage  # それぞれ GPT-3.5-turbo API の assistant, user, system role に対応
from notion_client import Client
from rich import print
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


# 環境変数を設定
load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# ボットトークンとソケットモードハンドラーを使ってアプリを初期化します
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

notion = Client(auth=os.environ["NOTION_API_KEY"])

# モデル作成
# わかりやすい解説
# https://note.com/npaka/n/n403fc29a02c7
llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name="gpt-3.5-turbo")


# todo: ここにテンプレートをいれる
# 日本語で ChatGPT っぽく丁寧に説明させる
system_message_prompt = SystemMessagePromptTemplate.from_template(
    """
You are an assistant who thinks step by step and includes a thought path in your response. Your answers are in Japanese."""
)

# ユーザーからの入力
human_template = """
{chat_history}
{text}"""


# User role のテンプレートに
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# ひとつのChatTemplateに
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

# (参考) https://note.com/npaka/n/n155e66a263a2
chat_prompt.input_variables = ["text", "chat_history"]
memory = ConversationBufferMemory(memory_key="chat_history")

# カスタムプロンプトを入れてchain 化
chain = LLMChain(llm=llm, prompt=chat_prompt, memory=memory)


@app.event("app_mention")
def command_handler(body, say, client):
    # メンションの内容を取得
    mention_text = body["event"]["text"]
    # LLMを動作させてチャンネルで発言
    say(chain.run(text=mention_text))


@app.shortcut("post-thesis-modal")
def post_thesis_modal_open(ack, body, client):
    """
    論文検索モーダルを開く
    """
    ack()
    client.views_open(
        trigger_id=body["trigger_id"],
        view=thesis.ThesisInteractor.build_arxiv_modal(),
    )


@app.action("post-thesis-count")
def handle_some_action(ack, body, logger):
    """
    出力件数の選択時の処理: 何もしない
    """
    ack()
    return


@app.action("post-thesis-type")
def handle_some_action2(ack, body, logger):
    """
    動作タイプ選択時: 何もしない
    """
    ack()
    return


@app.view("post-thesis-modal-input")
def post_thesis_modal_events(ack, body, say):
    """
    論文検索モーダルのイベントを処理
    """
    c_id = os.environ.get("SLACK_CHANNEL_ID", "")
    if not c_id:
        print("SLACK_CHANNEL_ID is not set.")
        return

    ack()
    result = thesis.ThesisInteractor.search_from_arxiv(body)
    say(result, channel=c_id)


# アプリを起動します
if __name__ == "__main__":
    mention_text = (
        "-困ったときは学習が目的というところに立ち返って、学習に最適な進め方を行う。-たとえば、プロダクトの締め切りに追われたとしても、リソース効率が落ちるからという理由でモブプログラミングを軽視しない。プロダクトを作ることが目的にならないように注意を払う。-モブプロにおいては、実装を実際に行う時間と、いったん立ち止まって情報の整理や質疑応答を行う時間に分けて運用すると学習効果が高まります。"
    )
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
