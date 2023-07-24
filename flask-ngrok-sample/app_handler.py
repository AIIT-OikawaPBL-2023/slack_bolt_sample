#!/usr/bin/env python3

import os
from rich import print


class SlackAppController(object):
    def __init__(self, slack_app, handler) -> None:
        self.handler = handler

        @slack_app.event("message")
        def messageHandler(body, say):
            print("message handler")
            print(body)

        @slack_app.event("reaction_added")
        def reactionAddedHandler(body, say):
            print("reaction added handler")
            print(body)

    def startSocketMode(self):
        self.handler.start()

    def runFlask(self, flask_app):  # ngrok のポートと合わせてください
        flask_app.run(host="0.0.0.0", port=11111)


# 下記はデバックでapp_handler単体をソケットモードで動かす
if __name__ == "__main__":
    from dotenv import load_dotenv
    from configparser import ConfigParser
    from slack_bolt import App
    from slack_bolt.adapter.socket_mode import SocketModeHandler

    # 設定情報をロード
    config = ConfigParser()
    config.read("config.ini", encoding="utf-8")
    # 秘密情報をロード
    load_dotenv()

    # ロードした情報を格納
    access_token = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")

    slack_app = App(token=access_token)
    handler = SocketModeHandler(slack_app, app_token)
    slack_controller = SlackAppController(slack_app, handler)

    slack_controller.startSocketMode()
