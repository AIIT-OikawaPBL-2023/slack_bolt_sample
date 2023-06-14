import os

from notion_client import Client
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from dotenv import load_dotenv

# GPT-3.5-turbo
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    # System メッセージテンプレート
    SystemMessagePromptTemplate,
    # assistant メッセージテンプレート
    AIMessagePromptTemplate,
    # user メッセージテンプレート
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    # それぞれ GPT-3.5-turbo API の assistant, user, system role に対応
    AIMessage,
    HumanMessage,
    SystemMessage,
)

# 環境変数を設定
load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# ボットトークンとソケットモードハンドラーを使ってアプリを初期化します
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

# モデル作成
# わかりやすい解説
# https://note.com/npaka/n/n403fc29a02c7
llm = ChatOpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"), model_name="gpt-3.5-turbo")

# 日本語で ChatGPT っぽく丁寧に説明させる
system_message_prompt = SystemMessagePromptTemplate.from_template("You are an assistant who thinks step by step and includes a thought path in your response. Your answers are in Japanese.")
# ユーザーからの入力
human_template = (
    "{text}" + "-困ったときは学習が目的というところに立ち返って、学習に最適な進め方を行う。-たとえば、プロダクトの締め切りに追われたとしても、リソース効率が落ちるからという理由でモブプログラミングを軽視しない。プロダクトを作ることが目的にならないように注意を払う。-モブプロにおいては、実装を実際に行う時間と、いったん立ち止まって情報の整理や質疑応答を行う時間に分けて運用すると学習効果が高まります。"
)

# User role のテンプレートに
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# ひとつのChatTemplateに
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
chat_prompt.input_variables = ["text"]

# カスタムプロンプトを入れてchain 化
chain = LLMChain(llm=llm, prompt=chat_prompt)


# @app.message("hello")
# def message_hello(message, say):
#     # イベントがトリガーされたチャンネルへ say() でメッセージを送信します
#     say(f"Hey there <@{message['user']}>!")
ym_select = {"2023年5月": "2023-06-10", "2023年6月": "2023-06-07", "2023年7月": "2023-06-05"}

options = []
for k, v in ym_select.items():
    options.append({"text": {"type": "plain_text", "text": k}, "value": v})

select_block = [
    {
        "type": "section",
        "block_id": "section678",
        "text": {"type": "mrkdwn", "text": "Pick an item from the dropdown list"},
        "accessory": {
            "action_id": "text1234",
            "type": "static_select",
            "placeholder": {"type": "plain_text", "text": "Select an item"},
            "options": options,
        },
    }
]


@app.event("app_mention")
def command_handler(body, say):
    # メンションの内容を取得
    mention_text = body["event"]["text"]
    if "マンスリーレビューをまとめてください" in mention_text:
        say(blocks=select_block)
    # LLMを動作させてチャンネルで発言
    say(chain.run(text=mention_text))

def fetch_records_for_week(semester=None, week=None):
    pages = notion_client.databases.query(
        database_id="01be2b6ddec849d199e6c4f555accc98",
        # filter={
        #     "and": [
        #         {"property": "Semester", "select": {"equals": semester}},
        #         {"property": "Week", "select": {"equals": week}},
        #     ]
        # },
    )
    return pages

# アプリを起動します
if __name__ == "__main__":
    mention_text = (
        "-困ったときは学習が目的というところに立ち返って、学習に最適な進め方を行う。-たとえば、プロダクトの締め切りに追われたとしても、リソース効率が落ちるからという理由でモブプログラミングを軽視しない。プロダクトを作ることが目的にならないように注意を払う。-モブプロにおいては、実装を実際に行う時間と、いったん立ち止まって情報の整理や質疑応答を行う時間に分けて運用すると学習効果が高まります。"
    )
    # print(chain.run(text=mention_text))
    print(fetch_records_for_week("2023-06-10", "2023-06-10"))

    # SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
