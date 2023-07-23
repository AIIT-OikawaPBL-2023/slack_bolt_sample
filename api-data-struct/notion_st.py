#!/usr/bin/env python3

from dotenv import load_dotenv
from rich import print
import dataclasses

from notion_client import Client


# ここから基本要素
# https://developers.notion.com/reference/rich-text#text
@dataclasses.dataclass(frozen=True)
class Text_ST:
    content: str
    link: str = None


# https://developers.notion.com/reference/rich-text#the-annotation-object (color)
# 定数であることを_Constで表現している。
# もっと良い方法があれば教えてください。
@dataclasses.dataclass(frozen=True)
class Color_Const:
    blue: str = "blue"
    brown: str = "brown"
    gray: str = "gray"
    green: str = "green"
    orange: str = "orange"
    pink: str = "pink"
    purple: str = "purple"
    red: str = "red"
    yellow: str = "yellow"
    default: str = "default"
    blue_background: str = "blue_background"
    brown_background: str = "brown_background"
    gray_background: str = "gray_background"
    green_background: str = "green_background"
    orange_background: str = "orange_background"
    pink_background: str = "pink_background"
    purple_background: str = "purple_background"
    red_background: str = "red_background"
    yellow_background: str = "yellow_background"


# https://developers.notion.com/reference/rich-text#the-annotation-object
@dataclasses.dataclass(frozen=True)
class PageAnnotations_ST:
    bold: bool = False
    italic: bool = False
    strikethrough: bool = False
    underline: bool = False
    code: bool = False
    color: Color_Const = Color_Const.default


# https://developers.notion.com/reference/rich-text
@dataclasses.dataclass(frozen=True)
class RichText_ST:
    text: Text_ST
    annotations: PageAnnotations_ST = PageAnnotations_ST()
    plain_text: str = "undefined"
    href: str = None
    type: str = "text"


# https://developers.notion.com/reference/page-property-values#select
@dataclasses.dataclass(frozen=True)
class Select_ST:
    id: str  # 既存のIDを指定する
    color: Color_Const  # IDに紐づいている色を指定する
    name: str = "undefined"


@dataclasses.dataclass(frozen=True)
# 新しくセレクタを作りたい場合にこちらを選択する
class NewSelect_ST:
    name: str
    color: Color_Const = Color_Const.default


# https://developers.notion.com/reference/user#all-users (type)
# 定数であることを_Constで表現している。
# もっと良い方法があれば教えてください。
@dataclasses.dataclass(frozen=True)
class UsersType_Const:
    person: str = "person"
    bot: str = "bot"


# https://developers.notion.com/reference/user#all-users
@dataclasses.dataclass(frozen=True)
class AllUsers_ST:
    id: str
    object: str = "user"
    type: UsersType_Const = UsersType_Const.person
    name: str = None
    avatar_url: str = None


# https://developers.notion.com/reference/emoji-object
@dataclasses.dataclass(frozen=True)
class Emoji_ST:
    emoji: str = "😻"
    type: str = "emoji"


# ここまで基本要素
# ここから下はPageの要素
# Pageの要素は上の基本要素の組み合わせで成り立っている。
# https://developers.notion.com/reference/page-property-values#rich-text
@dataclasses.dataclass(frozen=True)
class PageText_ST:
    rich_text: list[RichText_ST]
    type: str = "rich_text"
    id: str = "undefined"


# https://developers.notion.com/reference/page-property-values#title
@dataclasses.dataclass(frozen=True)
class PageTitle_ST:
    title: list[RichText_ST]
    id: str = "title"
    type: str = "title"


# https://developers.notion.com/reference/page-property-values#url
@dataclasses.dataclass(frozen=True)
class PageUrl_ST:
    url: str
    type: str = "url"
    id: str = "undefined"


# https://developers.notion.com/reference/page-property-values#multi-select
@dataclasses.dataclass(frozen=True)
class PageMultiSelect_ST:
    multi_select: list[Select_ST | NewSelect_ST]
    type: str = "multi_select"
    id: str = "undefined"


# ここから下はParentの要素
# https://developers.notion.com/reference/parent-object#database-parent
@dataclasses.dataclass(frozen=True)
class DbParent_ST:
    database_id: str
    type: str = "database_id"


# https://developers.notion.com/reference/parent-object#page-parent
@dataclasses.dataclass(frozen=True)
class PageParent_ST:
    page_id: str
    type: str = "page_id"


# ここからはENDPOINTSの要素
# https://developers.notion.com/reference/post-page
@dataclasses.dataclass(frozen=True)
class CreatePageInDb_ST:
    parent: DbParent_ST
    properties: dict


# https://developers.notion.com/reference/post-database-query
# RESPONSE["results"]の要素
@dataclasses.dataclass(frozen=True)
class DbQueryResResult_ST:
    object: str
    id: str
    created_time: str
    last_edited_time: str
    created_by: AllUsers_ST
    last_edited_by: AllUsers_ST
    cover: str
    icon: Emoji_ST
    parent: DbParent_ST
    archived: bool
    properties: dict
    url: str
    public_url: str


# https://developers.notion.com/reference/post-database-query
# RESPONSEの要素
@dataclasses.dataclass(frozen=True)
class DbQueryRes_ST:
    object: str
    results: list[DbQueryResResult_ST]
    next_cursor: str
    has_more: bool
    type: str
    page: dict


# ここからパーサーの関数を定義
# 基本要素
def __purseRichText(purse_target: dict) -> RichText_ST:
    """
    BlockのRich textのNotionデータをパースして出力する
    参照: https://developers.notion.com/reference/rich-text

    Args:
        purse_target (dict): Notionのデータ。辞書形式(Json)

    Returns:
        RichText_ST: パース結果
    """
    # 辞書からクラスに戻す時はアンパックを使う: https://qiita.com/ttyszk/items/01934dc42cbd4f6665d2
    rich_text = RichText_ST(**purse_target)

    # アンパックした要素をデータ構造に当てはめる
    ret = RichText_ST(
        plain_text=rich_text.plain_text,
        text=Text_ST(**rich_text.text),
        annotations=PageAnnotations_ST(**rich_text.annotations),
        href=rich_text.href,
        type=rich_text.type,
    )
    return ret


# Propaties関連のパーサー
def pursePagePropaty(purse_target: dict) -> PageText_ST | PageTitle_ST | PageUrl_ST | PageMultiSelect_ST:
    """
    PropatyのNotionデータをパースして出力する。
    参照: https://developers.notion.com/reference/page-property-values
    返り値がUnion型なので、型ヒントは不要な選択肢が含まれる。

    Args:
        purse_target (dict): Notionのデータ。辞書形式(JSON)

    Returns:
        PageText_ST: パース結果
    """
    if "title" == purse_target["type"]:
        ret = pursePageTitle(purse_target)
    elif "rich_text" == purse_target["type"]:
        ret = pursePageText(purse_target)
    elif "url" == purse_target["type"]:
        ret = pursePageUrl(purse_target)
    elif "multi_select" == purse_target["type"]:
        ret = pursePageMultiSelect(purse_target)
    else:
        print("Propertyのtypeが実装されていません。")
        print("type:", purse_target["type"])
        ret = None

    return ret


# 以降Propaties関連のパーサーでPropatriesのtypeまで指定
# 型ヒントで提示される選択肢が必要十分になる。
def pursePageMultiSelect(purse_target: dict) -> PageMultiSelect_ST:
    """
    Multi Select型のNotionデータをパースして出力する
    参照: https://developers.notion.com/reference/page-property-values#multi-select

    Args:
        purse_target (dict): Notionのデータ。辞書形式(JSON)

    Returns:
        PageTitle_ST: パース結果
    """
    # 辞書からクラスに戻す時はアンパックを使う
    page_multi_select = PageMultiSelect_ST(**purse_target)

    # API仕様を見るとField:optionsはOptionsが配列になっていることが分かる
    page_options = []
    for option_elem in page_multi_select.multi_select:
        page_options.append(Select_ST(**option_elem))

    # アンパックした要素をデータ構造に当てはめる
    ret = PageMultiSelect_ST(
        id=page_multi_select.id,
        type=page_multi_select.type,
        multi_select=page_options,
    )
    return ret


def pursePageText(purse_target: dict) -> PageText_ST:
    """
    Rich text型のNotionデータをパースして出力する
    参照: https://developers.notion.com/reference/page-property-values#rich-text

    Args:
        purse_target (dict): Notionのデータ。辞書形式(JSON)

    Returns:
        PageText_ST: パース結果
    """
    # 辞書からクラスに戻す時はアンパックを使う
    page_text = PageText_ST(**purse_target)

    # API仕様を見るとField:rich_textはRich Textが配列になっていることが分かる
    rich_texts = []
    for rich_text_elem in page_text.rich_text:
        rich_texts.append(__purseRichText(rich_text_elem))

    # アンパックした要素をデータ構造に当てはめる
    ret = PageText_ST(
        rich_text=rich_texts,
        type=page_text.type,
        id=page_text.id,
    )
    return ret


def pursePageTitle(purse_target: dict) -> PageTitle_ST:
    """
    Title型のNotionデータをパースして出力する
    参照: https://developers.notion.com/reference/page-property-values#title

    Args:
        purse_target (dict): Notionのデータ。辞書形式(JSON)

    Returns:
        PageTitle_ST: パース結果
    """
    # 辞書からクラスに戻す時はアンパックを使う
    page_title = PageTitle_ST(**purse_target)

    # API仕様を見るとField:titleはRich Textが配列になっていることが分かる
    title_texts = []
    for title_elem in page_title.title:
        title_texts.append(__purseRichText(title_elem))

    # アンパックした要素をデータ構造に当てはめる
    ret = PageTitle_ST(
        title=title_texts,
        type=page_title.type,
        id=page_title.id,
    )
    return ret


def pursePageUrl(purse_target: dict) -> PageUrl_ST:
    """
    Url型のNotionデータをパースして出力する
    参照: https://developers.notion.com/reference/page-property-values#url

    Args:
        purse_target (dict): Notionのデータ。辞書形式(JSON)

    Returns:
        PageUrl_ST: パース結果
    """
    # 辞書からクラスに戻す時はアンパックを使う
    page_url = PageUrl_ST(**purse_target)

    # アンパックした要素をデータ構造に当てはめる
    ret = PageTitle_ST(
        url=page_url.url,
    )
    return ret


# Endpoints関連
def purseDbQueryRes(purse_target: dict) -> DbQueryRes_ST:
    """
    database queryのRESPONSEのNotionデータをパースして出力する
    参照: https://developers.notion.com/reference/post-database-query

    Args:
        purse_target (dict): Notionのデータ。辞書形式(JSON)

    Returns:
        DbQueryRes_ST: パース結果
    """
    # 辞書からクラスに戻す時はアンパックを使う
    res_result = DbQueryRes_ST(**purse_target)

    # API仕様を見るとField:resultsはQuery結果が配列になっていることが分かる
    results_list = []
    for result in res_result.results:
        results_list.append(__purseDbQueryResResult(result))

    # アンパックした要素をデータ構造に当てはめる
    ret = DbQueryRes_ST(
        object=res_result.object,
        results=results_list,
        next_cursor=res_result.next_cursor,
        has_more=res_result.has_more,
        type=res_result.type,
        page=res_result.page,
    )
    return ret


def __purseDbQueryResResult(purse_target: dict) -> DbQueryResResult_ST:
    """
    database queryのRESPONSE["results"]のNotionデータをパースして出力する
    参照: https://developers.notion.com/reference/post-database-query

    Args:
        purse_target (dict): Notionのデータ。辞書形式(JSON)

    Returns:
        DbQueryResResult_ST: パース結果
    """
    # 辞書からクラスに戻す時はアンパックを使う
    res_result = DbQueryResResult_ST(**purse_target)

    # Field:iconはAPI結果でNoneの場合がある
    if res_result.icon is not None:
        icon = Emoji_ST(**res_result.icon)
    else:
        icon = None

    # アンパックした要素をデータ構造に当てはめる
    ret = DbQueryResResult_ST(
        object=res_result.object,
        id=res_result.id,
        created_time=res_result.created_time,
        last_edited_time=res_result.last_edited_time,
        created_by=AllUsers_ST(**res_result.created_by),
        last_edited_by=AllUsers_ST(**res_result.last_edited_by),
        cover=res_result.cover,
        icon=icon,
        parent=DbParent_ST(**res_result.parent),
        archived=res_result.archived,
        properties=res_result.properties,  # propertiesの中身はNotionの表の状態によって変わるため、辞書型(Json)のままとする
        url=res_result.url,
        public_url=res_result.public_url,
    )
    return ret


if __name__ == "__main__":
    import os

    database_id = "01be2b6ddec849d199e6c4f555accc98"

    # 秘密情報をロード
    load_dotenv()
    api_key = os.environ.get("NOTION_API_KEY")
    notion = Client(auth=api_key)

    create_page = CreatePageInDb_ST(
        parent=DbParent_ST(database_id),
        properties={  # ここはNotionの表によって変わるので、補助が出せない…
            "userid": PageTitle_ST(
                [
                    RichText_ST(
                        Text_ST("abcdefg"),
                    )
                ]
            ),
            "活動報告": PageText_ST(
                [
                    RichText_ST(
                        Text_ST("5Q-11"),
                        PageAnnotations_ST(italic=True, color=Color_Const.pink),
                    )
                ]
            ),
            "タグ": PageMultiSelect_ST(
                [
                    Select_ST(
                        id="e2804533-6825-4c61-9350-e3e4e6cf858e",
                        color=Color_Const.brown,
                    )
                ]
            ),
            "内容": PageText_ST(
                [
                    RichText_ST(
                        Text_ST("投稿テスト"),
                    )
                ]
            ),
        },
    )
    post_data = dataclasses.asdict(create_page)
    # notion.pages.create(**post_data)
