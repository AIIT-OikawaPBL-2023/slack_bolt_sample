#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from configparser import ConfigParser

from rich import print
import requests
import json


class SlackToken(object):
    """
    
    """
    def __init__(self, refresh_token, config_token=None) -> None:
        self.refresh = refresh_token
        self.config = config_token
        self.base_url = "https://slack.com/api/"

    def refreshConfigToken(self) -> str:
        """
        Refresh Tokenを使ってConfig Tokenを取得する
        ※期限切れ前の古いConfig Tokenも引き続き使用可能である。
        https://api.slack.com/methods/tooling.tokens.rotate

        """
        rotate_api = self.base_url + "tooling.tokens.rotate"
        post_data = {"refresh_token": self.refresh}
        print(self.refresh)
        response = requests.post(rotate_api, post_data)
        if response.json()["ok"] is True:
            # 環境変数のCONF_TOKENを更新する
            self.config = response.json()["token"]
            self.refresh = response.json()["refresh_token"]
            self.issue_at = response.json()["iat"]
            self.expires_in = response.json()["exp"]
            env_config = ConfigParser()
            env_config.optionxform = str
            env_config.read("./.env", encoding="utf-8")
            env_config.set("Env", "SLACK_CONF_TOKEN", self.config)
            env_config.set("Env", "SLACK_CONF_REFRESH_TOKEN", self.refresh)

            # 実際の.envファイルに反映させる
            with open("./.env", "w") as f:
                env_config.write(f)

            # 環境変数を更新する
            print("Config Token & Refresg Token is updated in .evn file.\nDev ContainerをRebuildしてください。")

        else:
            print(response.json())
            self.config = None

        return self.config

    def checkConfigToken(self) -> bool:
        """
        Config Tokenが有効かどうかをチェックする。
        https://api.slack.com/methods/auth.test

        Returns:
            bool: True:有効, False:無効
        """
        auth_api = self.base_url + "auth.test"
        post_data = {"token": self.config}
        response = requests.post(auth_api, post_data)

        ret = response.json()["ok"]
        if ret is True:
            # トークンは有効である。
            print("Config Token is valid.")
            return True
        else:
            # トークンは下記の理由で無効である。
            print("checkConfigToken result: ", response.json()["error"])
            print("Your ConfigToken: ", self.config)
            return False


class Manifest(object):
    def __init__(self, refresh_token, config_token=None):
        """
        初期化関数

        Args:
            token (str): Authentication token bearing required scopes.
                         Tokens should be passed as an HTTP Authorization header or alternatively, as a POST parameter.
        """
        self.token = SlackToken(refresh_token, config_token)
        if self.token.checkConfigToken() is False:
            self.config_token = self.token.refreshConfigToken()
        else:
            self.config_token = config_token

        self.issue_at = None
        self.expires_in = None
        self.base_url = "https://slack.com/api/"
        self.manifest_config = None

    def getFileFromApp(self, app_id):
        """
        既存のSlack AppからManifestファイルを取得する
        https://api.slack.com/methods/apps.manifest.export

        Args:
            app_id (str): The ID of the app whose configuration you want to export as a manifest.
        """
        export_api = self.base_url + "apps.manifest.export"
        post_data = {"token": self.config_token, "app_id": app_id}
        response = requests.post(export_api, post_data)
        if self.__checkResponse(response, "getFileFromApp") is True:
            self.manifest_config = response.json()["manifest"]

    def saveFile(self, json_path):
        """
        Manifest Fileを保存する

        Args:
            json_path (str): File名.jsonをパス付きで指定する
        """
        with open(json_path, mode="w", encoding="utf-8") as json_file:
            json.dump(self.manifest_config, json_file, indent=4, ensure_ascii=False)

    def loadFile(self, json_path):
        """
        Manifest Fileをロードする

        Args:
            json_path (str): File名.jsonをパス付きで指定する
        """
        with open(json_path, mode="r", encoding="utf-8") as json_file:
            self.manifest_config = json.load(json_file)

    def createApp(self):
        """
        新しいAppを作成する。
        本メソッド実行前にgetFileFromAppまたはloadFileを実行している必要がある
        https://api.slack.com/methods/apps.manifest.create
        """
        create_api = self.base_url + "apps.manifest.create"
        post_data = {"token": self.config_token, "manifest": json.dumps(self.manifest_config)}
        response = requests.post(create_api, post_data)
        self.__checkResponse(response, "createApp")

    def deleteApp(self, app_id):
        """
        既存のAppを削除する
        https://api.slack.com/methods/apps.manifest.delete

        Args:
            app_id (str): The ID of the app whose configuration you want to export as a manifest.
        """
        delete_api = self.base_url + "apps.manifest.delete"
        post_data = {"token": self.config_token, "app_id": app_id}
        response = requests.post(delete_api, post_data)
        self.__checkResponse(response, "deleteApp")

    def changeAppConfig(self, app_id):
        """
        Applicationの設定を変更する
        本メソッド実行前にgetFileFromAppまたはloadFileを実行している必要がある
        https://api.slack.com/methods/apps.manifest.update

        Args:
            app_id (str): The ID of the app whose configuration you want to export as a manifest.
        """
        update_api = self.base_url + "apps.manifest.update"
        post_data = {"token": self.config_token, "app_id": app_id, "manifest": json.dumps(self.manifest_config)}
        response = requests.post(update_api, post_data)
        self.__checkResponse(response, "changeAppConfig")

    def switchSocketModeAndEventSubsc(self, app_id: str, socket_mode_enabled: bool, request_url: str = "https://XXX-XXX-XXX-XXX/slack/events"):
        """
        Socket ModeとEvent Subscriptionを切り替える

        Args:
            app_id (str): The ID of the app whose configuration you want to export as a manifest.
            socket_mode_enabled (bool): Socket Modeを有効無効を切り替える。無効時はEvent Subscriptionが有効になる。
            request_url (str): Event SubscriptionのRequest URL
        """
        self.manifest_config["settings"]["event_subscriptions"]["request_url"] = request_url
        self.manifest_config["settings"]["socket_mode_enabled"] = socket_mode_enabled
        if socket_mode_enabled is True:
            self.manifest_config["settings"]["interactivity"] = {"is_enabled": True}
        else:
            self.manifest_config["settings"]["interactivity"] = {"is_enabled": False}

        self.changeAppConfig(app_id)

    def __checkResponse(self, response, method_name=""):
        """
        This Method check the response for requests.

        Args:
            response (object): response object returned requests function.
        """
        if response.status_code == 200:
            post_result = response.json()
            if post_result["ok"] is False:
                print(method_name, "エラー内容：", post_result["error"], "Slackへの反映に失敗しました。")
                if post_result["error"] == "invalid_auth":
                    print("ヒント: Configuration Tokensの期限が切れていないか確認してください。-> https://api.slack.com/apps")
                    print(".envを更新した場合、Dev ContainerのRebuildが必要な場合があります。")
                else:
                    print(post_result["errors"])
                ret = False
            else:
                # print(method_name, post_result)
                ret = True
        else:
            print(method_name, response.status_code, response.reason)
            ret = False

        return ret


if __name__ == "__main__":
    # 環境変数を設定
    load_dotenv()
    # 設定値をロード
    config = ConfigParser()
    config.read("./slack-app-manifest/config.ini", encoding="utf-8")

    slack_conf_refresh_token = os.environ.get("SLACK_CONF_REFRESH_TOKEN")
    slack_conf_token = os.environ.get("SLACK_CONF_TOKEN")
    app_id = config.get("Slack", "app_id")
    file_path = config.get("Manifest", "file_path")

    manifest = Manifest(slack_conf_refresh_token, slack_conf_token)

    # ManifestファイルをSlack Applicationからもってきて、PCに保存する
    manifest.getFileFromApp(app_id)
    manifest.saveFile(file_path)

    # PCに保存したManifestファイルをSlack Applicationに反映する
    # manifest.loadFile(file_path)
    # manifest.switchSocketModeAndEventSubsc(app_id, socket_mode_enabled=False, request_url="https://AAA-AAA-XXX-XXX/slack/events")
    # manifest.changeAppConfig(app_id)

    # 新しいSlack Applicationを作成する
    # manifest.createApp()

    # 既存のSlack Applicationを削除する
    # delete_app_id = XXXXX
    # manifest.deleteApp(delete_app_id)
