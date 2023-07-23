class ThesisInteractor:
    @classmethod
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
                {"type": "input", "element": {"type": "plain_text_input", "action_id": "post-thesis-search-word"}, "label": {"type": "plain_text", "text": "検索キーワード(英単語1つ)", "emoji": True}},
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

    def search_from_arxiv(body):
        """
        論文検索を実行
        """
        modal_value = body["view"]["state"]["values"]

        w = "post-thesis-search-word"
        t = "post-thesis-type"
        c = "post-thesis-count"

        condition = {}
        for d in modal_value.values():
            if d.get(w):
                condition["word"] = d[w]["value"]
            elif d.get(t):
                condition["type"] = d[t]["selected_option"]["value"]
            elif d.get(c):
                condition["count"] = d[c]["selected_option"]["value"]

        # TODO: 論文検索処理を呼び出す
        return "入力内容: " + str(condition)
