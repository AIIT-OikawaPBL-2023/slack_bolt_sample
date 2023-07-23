import random
from thesis import Arxiv


class ThesisInteractor:
    def __init__(self):
        # モーダルで未入力の場合の初期値
        self.word = ""
        self.type = "summary"  # 要約
        self.count = "3"  # 件数

    def build_arxiv_modal(self):
        """
        論文検索用のモーダルの中身
        """
        return {
            "type": "modal",
            "callback_id": "post-thesis-modal-input",
            "title": {"type": "plain_text", "text": "論文まとめ", "emoji": True},
            "submit": {"type": "plain_text", "text": "Start", "emoji": True},
            "close": {"type": "plain_text", "text": "Cancel", "emoji": True},
            "blocks": [
                {"type": "input", "element": {"type": "plain_text_input", "action_id": "post-thesis-search-word"}, "label": {"type": "plain_text", "text": "検索キーワード (英語)", "emoji": True}},
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "動作タイプ"},
                    "accessory": {
                        "type": "radio_buttons",
                        "options": [
                            {"text": {"type": "plain_text", "text": "要約文", "emoji": True}, "value": "summary"},
                            {"text": {"type": "plain_text", "text": "要点のみ", "emoji": True}, "value": "keypoint"},
                        ],
                        "action_id": "post-thesis-type",
                    },
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": "出力する件数"},
                    "accessory": {
                        "type": "static_select",
                        "placeholder": {"type": "plain_text", "text": "-", "emoji": True},
                        "options": [
                            {"text": {"type": "plain_text", "text": "1", "emoji": True}, "value": "1"},
                            {"text": {"type": "plain_text", "text": "3", "emoji": True}, "value": "3"},
                            {"text": {"type": "plain_text", "text": "5", "emoji": True}, "value": "5"},
                        ],
                        "action_id": "post-thesis-count",
                    },
                },
            ],
        }

    def search_from_arxiv(self, body):
        """
        論文検索を呼び出し、件数分の論文情報リストを返す
        """
        modal_value = body["view"]["state"]["values"]

        w = "post-thesis-search-word"
        t = "post-thesis-type"
        c = "post-thesis-count"

        for d in modal_value.values():
            try:
                if d.get(w):
                    self.word = d[w]["value"]
                elif d.get(t):
                    self.type = d[t]["selected_option"]["value"]
                elif d.get(c):
                    self.count = d[c]["selected_option"]["value"]
            except TypeError:
                pass

        res = Arxiv.get_list(self.word)

        if not res:
            print("検索結果が0件です")
            exit()
        elif len(res) < int(self.count):
            print("検索結果が少ないため、全件出力します")
            return res

        # 件数分ランダムで取得
        return random.sample(res, k=int(self.count))

    def gpt_message(self, summary):
        """
        要約または要点のみのGPT用メッセージを返す
        """
        if self.type == "summary":
            # 要約の場合
            return self.summary_message(summary)
        else:
            # 要点のみの場合
            return self.keypoint_message(summary)

    def summary_message(self, summary):
        """
        要約文を生成するGPT用メッセージ
        """
        return f"""
        次の文章の要約文を300字程度で生成して日本語で出力してください。
        ----------------------------------
        {summary}
        """

    def keypoint_message(self, summary):
        """
        要点のみを生成するGPT用メッセージ
        """
        return f"""
        次の文章の要点を3行でまとめ日本語で出力してください。
        要点に番号をつけて改行してください。
        1: 要点1
        2: 要点2
        3: 要点3
        ----------------------------------
        {summary}
        """

    def output_message(self, summary, data):
        """
        Slackに投稿するメッセージを生成
        """
        message = f"""
タイトル: *{data.title}*
発行日: {data.published.strftime("%Y/%m/%d")}
カテゴリ: {self.cateogry_long(data.primary_category)}
URL: {data.entry_id}
{self.type_ja().replace("のみ", "")}:\n
{summary}
"""
        return message

    def type_ja(self):
        if self.type == "summary":
            return "要約文"
        else:
            return "要点のみ"

    def cateogry_long(self, category):
        if "cs" in category:
            return "Computer Science"
        elif "math" in category:
            return "Mathematics"
        elif "stat" in category:
            return "Statistics"
        else:
            return "Other"
