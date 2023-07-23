#!/usr/bin/env python3

import os
import sys

from dotenv import load_dotenv
from rich import print

from notion_client import Client

# from time import sleep

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../api-data-struct")

if __name__ == "__main__":
    # 実装した構造化スクリプト
    import notion_st

    table_row = 0
    database_id = "01be2b6ddec849d199e6c4f555accc98"

    # 秘密情報をロード
    load_dotenv()
    api_key = os.environ.get("NOTION_API_KEY")
    notion = Client(auth=api_key)

    # notionからのResponse情報を得る
    response_dict = notion.databases.query(**{"database_id": database_id})

    # Responseをパースする
    db_query_res = notion_st.purseDbQueryRes(response_dict)

    # Responseの中身のpropertiesをパースする
    # purseDbQueryResですべてパースできないのは、Notionに置いてある表の情報を教えてあげる必要があるため。
    # 列ごとに、ヘッダー名とtypeが異なる。その情報をパーサーに入力する必要がある
    user_id = notion_st.pursePageTitle(db_query_res.results[table_row].properties["userid"])  # typeがTitleのプロパティをパース
    quarter_num = notion_st.pursePageText(db_query_res.results[table_row].properties["活動報告"])  # typeがTextのプロパティをパース
    tag = notion_st.pursePageMultiSelect(db_query_res.results[table_row].properties["タグ"])  # typeがMultiSelectのプロパティをパース
    contents = notion_st.pursePageText(db_query_res.results[table_row].properties["内容"])  # typeがTextのプロパティをパース

    # typeに関しては自動識別可能なパーサーも別途準備した。
    # ただし、返り値がUnionになるので型ヒントとして不要な要素もサジェストされる
    tag_union = notion_st.pursePagePropaty(db_query_res.results[table_row].properties["タグ"])

    """
        表記例
    """
    print(db_query_res.results[table_row].last_edited_by)
    print(user_id.title[0].text.content)
    print(tag.multi_select[0])
    print(quarter_num.rich_text[0].text)
    print(tag_union.multi_select[0].name)
    print('\n"-------ここから動作確認-------"')
    """
        下記の変数についてVS CODEの型ヒントが利くことを動作確認してください。
    """

    db_query_res
    user_id
    quarter_num
    contents
    tag
    tag_union
