#!/usr/bin/env python3

import os
import sys

from rich import print
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import slack_event_st
import slack_block_st

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../api-data-struct")


class SlackAppController(object):
    def __init__(self, slack_app, handler) -> None:
        self.handler = handler

        @slack_app.event(slack_event_st.EventType_Const.message)
        def messageHandler(body, say):
            print("message handler")
            body_st = slack_event_st.purseEventMessageCB(body)
            print(body_st.event)

            builder = slack_block_st.HelloBuilder(say_text="元気にしてましたか? >")
            blocks_dict = builder.hello_build
            say(**blocks_dict)

        # Reactionすると日付選択が出てくる
        @slack_app.event(slack_event_st.EventType_Const.reaction_added)
        def reactionAddedHandler(body, say):
            print("reaction added handler")
            body_st = slack_event_st.purseEventReactionAddedCB(body)
            print(body_st.event)

            builder = slack_block_st.DatePickerBuilder()
            blocks_dict = builder.input_block_build
            say(**blocks_dict)

        @slack_app.event(slack_event_st.EventType_Const.reaction_removed)
        def reactionRemovedHandler(body, say):
            print("reaction removed handler")
            body_st = slack_event_st.purseEventReactionRemovedCB(body)
            print(body_st.event)

            builder = slack_block_st.DatePickerBuilder()
            blocks_dict = builder.section_block_build
            say(**blocks_dict)

        @slack_app.event(slack_event_st.EventType_Const.app_mention)
        def AppMentionHandler(body, say):
            print("app mention handler")
            print(body)
            body_st = slack_event_st.purseEventAppMentionCB(body)
            print(body_st.event)

        # 日付選択すると下記が呼ばれる
        @slack_app.action("datepicker_01")
        def dataPicker01Handler(ack, body, logger):
            print("data picker 01 handler")
            ack()
            print(body["type"], body["actions"][0]["selected_date"])  # ここは力尽きたので、パースしていない

    def startSocketMode(self):
        self.handler.start()


if __name__ == "__main__":
    # 秘密情報をロード
    load_dotenv()

    # ロードした情報を格納
    access_token = os.environ.get("SLACK_BOT_TOKEN")
    app_token = os.environ.get("SLACK_APP_TOKEN")

    slack_app = App(token=access_token)
    handler = SocketModeHandler(slack_app, app_token)
    slack_controller = SlackAppController(slack_app, handler)

    slack_controller.startSocketMode()
