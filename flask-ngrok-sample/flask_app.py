#!/usr/bin/env python3

import sys
import os
from flask import Flask, request
import ngrok

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../slack-app-manifest")


def initFlaskApp(flask_app):
    @flask_app.route("/slack/events", methods=["POST"])
    def slack_events():
        """
        Slackからのイベントを受け取る
        チャレンジリクエストと通常のイベントを区別して処理する。
        チャレンジリクエストの処理は、下記ページを参考にした。
        https://qiita.com/masa_masa_ra/items/618779e698921cb53cec

        Returns:
            dict: Slackからのレスポンス
        """
        payload = request.get_json()
        if "challenge" in payload:
            # これはチャレンジリクエストです。アプリを確認するためにチャレンジ値で応答してください。
            return payload["challenge"]
        else:
            # これは通常のイベントです。SlackRequestHandlerで処理してください。
            return handler.handle(request)
        return handler.handle(request)


def getNgrokUrl(ngrok_api_key) -> str:
    """
    ngrokのURLを取得する
    https://python-api.docs.ngrok.com/client.html#ngrok.Client.tunnels

    Args:
        ngrok_api_key (str): ngrokのAPIキー

    Returns:
        str: ngrokのURL
    """
    ngrok_client = ngrok.Client(ngrok_api_key)
    for tunnel in ngrok_client.tunnels.list():
        print("ngrok URL -> ", tunnel.public_url)

    return tunnel.public_url


if __name__ == "__main__":
    from dotenv import load_dotenv
    from configparser import ConfigParser
    from slack_bolt import App
    from slack_bolt.adapter.flask import SlackRequestHandler

    # 自作クラス
    from app_handler import SlackAppController
    from create_app import Manifest

    # 設定情報をロード
    config = ConfigParser()
    config.read("./flask-ngrok-sample/config.ini", encoding="utf-8")
    # 秘密情報をロード
    load_dotenv()

    # ロードした情報を格納
    access_token = os.environ.get("SLACK_BOT_TOKEN")
    signing_secret = os.environ.get("SLACK_SIGNING_SECRET")
    ngrok_api_key = os.environ.get("NGROK_API_KEY")
    slack_app_id = config.get("Slack", "app_id")
    slacK_conf_refresh_token = os.environ.get("SLACK_CONF_REFRESH_TOKEN")
    slack_conf_token = os.environ.get("SLACK_CONF_TOKEN")

    # ngrokのURLを取得
    ngrok_url = getNgrokUrl(ngrok_api_key)
    slack_request_url = ngrok_url + "/slack/events"

    # Slack Botの設定
    manifest = Manifest(slacK_conf_refresh_token, slack_conf_token)
    manifest.getFileFromApp(slack_app_id)
    manifest.switchSocketModeAndEventSubsc(slack_app_id, socket_mode_enabled=False, request_url=slack_request_url)
    manifest.changeAppConfig(slack_app_id)

    # インスタンス生成
    slack_app = App(signing_secret=signing_secret, token=access_token)
    handler = SlackRequestHandler(slack_app)
    slack_controller = SlackAppController(slack_app, handler)

    # Flaskを初期化
    flask_app = Flask(__name__)
    initFlaskApp(flask_app)

    # Flaskをslack_controllerで実行
    slack_controller.runFlask(flask_app)
