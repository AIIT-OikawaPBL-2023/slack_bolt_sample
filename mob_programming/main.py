import os

from dotenv import load_dotenv
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (  # System メッセージテンプレート; assistant メッセージテンプレート; user メッセージテンプレート
    AIMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate,
    SystemMessagePromptTemplate)
from langchain.schema import (  # それぞれ GPT-3.5-turbo API の assistant, user, system role に対応
    AIMessage, HumanMessage, SystemMessage)
from notion_client import Client
from rich import print
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from flask import Flask, request
from slack_bolt.adapter.flask import SlackRequestHandler

flask_app = Flask(__name__)
handler = SlackRequestHandler(app)

@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    payload = request.get_json()
    if 'challenge' in payload:
        # これはチャレンジリクエストです。アプリを確認するためにチャレンジ値で応答してください。
        # ref: https://qiita.com/masa_masa_ra/items/618779e698921cb53cec
        return payload['challenge']
    else:
        # これは通常のイベントです。SlackRequestHandlerで処理してください。
        return handler.handle(request)
    
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
system_message_prompt = SystemMessagePromptTemplate.from_template("""
You are an assistant who thinks step by step and includes a thought path in your response. Your answers are in Japanese.""")

# ユーザーからの入力
human_template = """
５月の週報の内容を以下のルールに従って要約してね。
要約というのは、文章全体の中から重要なトピックを見つけ出し、そのトピックごとに要点をまとめ、エッセンスとなる情報だけに絞り込むことだよ。
長い文章は要約とは言えないから注意してね。
    ＜ルール＞
    ・「活動内容と成果の実績」、「課題と解決策」、「できごと・気づき」の３つの大項目を立て、この大項目ごとに要約を記載すること（これ以外の項目は立てないこと）
    ・上記３つの各項目につき、３箇条ずつ箇条書きで要約を記載すること（４箇条以上は記載しないこと）
    ・一文につき各37文字以内とすること
    ・回答全体の総文字数は330文字以内とすること
    ・回答は12行以内とすること
    ・「今週の活動と成果の実績」という名前の項目は立てないこと

    回答作成に当たっては次の記載例を参考にすること。
    文字数や構成は下記記載例から大きく逸脱しないこと。
    ＜記載例＞
        【活動内容と成果の実績】
        ・モブプログラミングを実施し、APIの使い方を学んだ
        ・スクラム運営方針を決定し、スプリントの設計を行った
        ・マンスリーレビューの指摘事項について振り返りを行った
        【課題と解決策】
        ・スケジュール遅延が繰り返し生じている
        ・まだ悲鳴を上げることに抵抗感が残っている
        ・評価指標が未作成である
        【できごと・気づき】
        ・モブプロにより他のメンバーとの知見共有がスムーズになった
        ・輪読会での学びがスプリントの設計に活用できた
        ・スクラムの精神を学び失敗しても大丈夫な雰囲気の醸成ができた

    ＜以下の行から先が週報の内容。下記を要約すること。＞

{text}"""

human_template_en = """
Summarize the contents of the Weekly Reports in May in Japanese according to the following rules.
"Summarize" here means finding the important topics in the whole text, extracting the main points of each topic, and reducing the text to the essential information.
Note that long sentences are not a summary.
    <Rules>
    - Create three main categories: "活動内容と成果の実績", "課題と解決策" and "できごと・気づき" (no other categories are allowed).
    - Provide 3 bullets summary (four bullets or more are not allowed) for each of the above three categories.
    - Each sentence should be no longer than 37 characters.
    - Total length of response should not exceed 330 characters.
    - Responses should be no longer than 12 lines.
    - Do not include a section named "今週の活動と成果の実績".
Please refer to the following example when drafting your response.
    The number of characters and structure should not deviate greatly from the following example.
    <Example>
    Activities and Achievements
        Mob programming was conducted and participants learned how to use the API.
        Decided on a Scrum management policy and designed sprints.
        Reviewed the points pointed out in the monthly review.
    Issues and solutions
            Schedule delays are occurring repeatedly.
            There is still a sense of resistance to screaming.
            The evaluation index has not yet been created.
    What happened and what we noticed
            MobPro has facilitated the sharing of knowledge with other members.
            The learning from the reading session was utilized in the sprint design.
            The spirit of Scrum was learned, and an atmosphere in which it was okay to make mistakes was fostered.

        < The following lines are the contents of the weekly report. The following should be summarized. >
{text}"""


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
    filter_obj = {
        "and": [
            {"property": "活動報告", "rich_text": {"contains": "1Q-05"}},
            {"property": "タグ", "multi_select": {"does_not_contain": "来週の活動と成果の予定"}},
            {"property": "タグ", "multi_select": {"contains": "今週の活動と成果の実績"}},
        ]
    }

    # filter_obj = {
    #     "or": [
    #         {"property": "活動報告", "rich_text": {"contains": "1Q-05"}},
    #         # {"property": "活動報告", "rich_text": {"contains": "1Q-06"}},
    #         # {"property": "活動報告", "rich_text": {"contains": "1Q-07"}},
    #         {"property": "活動報告", "rich_text": {"contains": "1Q-08"}},
    #         {"property": "活動報告", "rich_text": {"contains": "1Q-09"}},
    #     ],
    #     "and": [
    #         {"property": "タグ", "multi_select": {"does_not_contain": "来週の活動と成果の予定"}},
    #     ],
    # }
    pages = notion.databases.query(**{"database_id": "01be2b6ddec849d199e6c4f555accc98", "filter": filter_obj})["results"]
    results = []
    # 各レコードに対して
    for page in pages:
        # すべてのプロパティを取得
        properties = page["properties"]

        # 対象期間絞り込みのために活動報告カラムの値を取得
        target_period = properties["活動報告"]["rich_text"][0]["plain_text"]
        page_semester, page_week = target_period.split("-")

        # レコードの学期と週が指定された学期と週と一致するか確認
        # if semester == page_semester and week == page_week:
        # 一致する場合、各プロパティの値を取得
        row = {}
        for name, prop in properties.items():
            # プロパティのタイプによって異なる形式の値が存在するため、それを判定
            if prop["type"] == "title":
                # タイトルプロパティの場合、テキストを取得
                value = prop["title"][0]["plain_text"]  # userid
            elif prop["type"] == "rich_text":
                # リッチテキストプロパティの場合、テキストを取得
                value = prop["rich_text"][0]["plain_text"]  # 内容, 活動報告
            elif prop["type"] == "multi_select":  # タグ
                # マルチセレクトプロパティの場合、選択されたオプションの名前をすべて取得
                value = ", ".join(option["name"] for option in prop["multi_select"])
            else:
                value = "Unsupported property type: " + prop["type"]
            row[name] = value
        results.append(row)

    return results

# MOB flask_app.runで書き換え
# アプリを起動します
if __name__ == "__main__":
    mention_text = (
        "-困ったときは学習が目的というところに立ち返って、学習に最適な進め方を行う。-たとえば、プロダクトの締め切りに追われたとしても、リソース効率が落ちるからという理由でモブプログラミングを軽視しない。プロダクトを作ることが目的にならないように注意を払う。-モブプロにおいては、実装を実際に行う時間と、いったん立ち止まって情報の整理や質疑応答を行う時間に分けて運用すると学習効果が高まります。"
    )
    print(chain.run(text=fetch_records_for_week("2023-06-10", "2023-06-10")))
    print(fetch_records_for_week("2023-06-10", "2023-06-10"))

    # SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
