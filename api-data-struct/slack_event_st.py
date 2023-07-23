#!/usr/bin/env python3

import dataclasses


# 要素
# https://api.slack.com/apis/connections/events-api#event-type-structure
@dataclasses.dataclass(frozen=True)
class Item_ST:
    type: str
    channel: str
    ts: str


# https://api.slack.com/apis/connections/events-api#authorizations
@dataclasses.dataclass(frozen=True)
class Authorization_ST:
    enterprise_id: str
    team_id: str
    user_id: str
    is_bot: bool


# Eventタイプごとのデータ構造を定義
# https://api.slack.com/events
# 定数であることを_Constで表現している。
@dataclasses.dataclass(frozen=True)
class EventType_Const:
    message: str = "message"
    app_mention: str = "app_mention"
    reaction_added: str = "reaction_added"
    reaction_removed: str = "reaction_removed"
    user_status_changed: str = "user_status_changed"


# https://api.slack.com/events/message#subtypes
@dataclasses.dataclass(frozen=True)
class EventMsgSubtypes_Const:
    bot_message: str = "bot_message"
    me_message: str = "me_message"
    message_deleted: str = "message_deleted"
    message_changed: str = "message_changed"
    message_replied: str = "message_replied"


# https://api.slack.com/events/message/message_changed
# editedの中身
@dataclasses.dataclass(frozen=True)
class Edited_ST:
    user: str
    ts: str


# ここから下がEventに関するデータ構造
# https://api.slack.com/events/message
@dataclasses.dataclass(frozen=True)
class EventMessage_ST:
    type: str = EventType_Const.message
    subtype: str = None
    text: str = None
    blocks: list[dict] = None
    team: str = None
    channel: str = None
    channel_type: str = None
    user: str = None
    client_msg_id: str = None
    ts: str = None
    event_ts: str = None

    # スレッドメッセージの場合は下記に値が入っている
    # https://api.slack.com/messaging/retrieving#finding_threads
    thread_ts: str = None

    # スレッドにぶら下がっているメッセージの場合は下記に値が入っている
    # https://api.slack.com/messaging/retrieving#pulling_threads
    parent_user_id: str = None

    # スレッドの親メッセージの場合は下記に値が入っている
    # https://api.slack.com/messaging/retrieving#pulling_threads
    reply_count: int = 0
    reply_users_count: int = 0
    reply_users: list[str] = None  # Max 5 by slack api document
    latest_reply: str = None
    is_locked: bool = None
    subscribed: bool = None
    last_read: str = None

    # メッセージを変更した場合は下記に値が入っている
    # https://api.slack.com/events/message/message_changed
    edited: Edited_ST = None
    source_team: str = None
    user_team: str = None


# https://api.slack.com/events/message/message_changed
@dataclasses.dataclass(frozen=True)
class EventMsgChange_ST:
    message: EventMessage_ST = None
    previous_message: EventMessage_ST = None
    type: str = EventType_Const.message
    subtype: str = EventMsgSubtypes_Const.message_changed
    channel: str = None
    hidden: bool = None
    ts: str = None
    event_ts: str = None
    channel_type: str = None


# https://api.slack.com/events/message/message_deleted
@dataclasses.dataclass(frozen=True)
class EventMsgDelete_ST:
    previous_message: EventMessage_ST = None
    channel: str = None
    hidden: bool = None
    deleted_ts: str = None
    event_ts: str = None
    ts: str = None
    channel_type: str = None
    type: str = EventType_Const.message
    subtype: str = EventMsgSubtypes_Const.message_deleted


# https://api.slack.com/events/reaction_added
@dataclasses.dataclass(frozen=True)
class EventReactionAdded_ST:
    user: str
    reaction: str
    item: Item_ST
    item_user: str
    event_ts: str
    type: str = EventType_Const.reaction_added


# https://api.slack.com/events/reaction_removed
@dataclasses.dataclass(frozen=True)
class EventReactionRemoved_ST:
    user: str
    reaction: str
    item: Item_ST
    item_user: str
    event_ts: str
    type: str = EventType_Const.reaction_removed


# https://api.slack.com/events/app_mention
@dataclasses.dataclass(frozen=True)
class EventAppMention_ST:
    user: str
    text: str
    team: str
    channel: str
    ts: str
    event_ts: str
    type: str = EventType_Const.app_mention
    client_msg_id: str = None
    blocks: dict = None


# Event Callbackで得たデータ構造
# https://api.slack.com/apis/connections/events-api#callback-field
@dataclasses.dataclass(frozen=False)
class __EventCB_ST:
    token: str
    team_id: str
    context_team_id: str
    context_enterprise_id: str
    api_app_id: str
    event: object
    type: str
    event_id: str
    event_time: int
    authorizations: list[Authorization_ST]
    is_ext_shared_channel: bool
    event_context: str


# https://api.slack.com/apis/connections/events-api#callback-field
@dataclasses.dataclass(frozen=True)
class EventMessageCB_ST:
    token: str
    team_id: str
    api_app_id: str
    event: EventMessage_ST | EventMsgDelete_ST
    type: str
    event_id: str
    event_time: int
    authorizations: list[Authorization_ST]
    is_ext_shared_channel: bool
    event_context: str
    context_team_id: str = None
    context_enterprise_id: str = None


# https://api.slack.com/apis/connections/events-api#callback-field
@dataclasses.dataclass(frozen=True)
class EventReactionAddedCB_ST:
    token: str
    team_id: str
    context_team_id: str
    context_enterprise_id: str
    api_app_id: str
    event: EventReactionAdded_ST
    type: str
    event_id: str
    event_time: int
    authorizations: list[Authorization_ST]
    is_ext_shared_channel: bool
    event_context: str


# https://api.slack.com/apis/connections/events-api#callback-field
@dataclasses.dataclass(frozen=True)
class EventReactionRemovedCB_ST:
    token: str
    team_id: str
    context_team_id: str
    context_enterprise_id: str
    api_app_id: str
    event: EventReactionRemoved_ST
    type: str
    event_id: str
    event_time: int
    authorizations: list[Authorization_ST]
    is_ext_shared_channel: bool
    event_context: str


# https://api.slack.com/apis/connections/events-api#callback-field
@dataclasses.dataclass(frozen=True)
class EventAppMentionCB_ST:
    token: str
    team_id: str
    context_team_id: str
    context_enterprise_id: str
    api_app_id: str
    event: EventAppMention_ST
    type: str
    event_id: str
    event_time: int
    authorizations: list[Authorization_ST]
    is_ext_shared_channel: bool
    event_context: str


# ここからパーサーの関数を定義
def __purseEventCB(purse_target: dict) -> __EventCB_ST:
    """
    EventのCallbackで得たデータのパース共通処理
    # https://api.slack.com/apis/connections/events-api#callback-field

    Args:
        purse_target (dict): パース対象

    Returns:
        __EventCB_ST: パース結果
    """
    channel_msg_cb = EventMessageCB_ST(**purse_target)
    authorizations = []
    for authorization in authorizations:
        authorizations.append(Authorization_ST(**authorization))

    ret = __EventCB_ST(
        token=channel_msg_cb.token,
        team_id=channel_msg_cb.team_id,
        context_team_id=channel_msg_cb.context_team_id,
        context_enterprise_id=channel_msg_cb.context_enterprise_id,
        api_app_id=channel_msg_cb.api_app_id,
        event=None,  # event.typeによって変わる
        event_id=channel_msg_cb.event_id,
        event_time=channel_msg_cb.event_time,
        authorizations=authorizations,
        is_ext_shared_channel=channel_msg_cb.is_ext_shared_channel,
        event_context=channel_msg_cb.event_context,
        type=channel_msg_cb.type,
    )

    return ret


def purseEventMessageCB(purse_target: dict) -> EventMessageCB_ST:
    """
    EventのCallbackで得たデータにおいて
    event.type = "message"だった場合のパース処理
    # https://api.slack.com/apis/connections/events-api#callback-field

    Args:
        purse_target (dict): パース対象

    Returns:
        EventMessageCB_ST: パース結果
    """
    # Eventをパースする
    purse_result = __purseEventCB(purse_target)
    purse_result.event = __purseEventMessage(purse_target["event"])
    # 出力の型に変換する
    purse_result_dict = dataclasses.asdict(purse_result)
    ret = EventMessageCB_ST(**purse_result_dict)
    return ret


def purseEventReactionAddedCB(purse_target: dict) -> EventReactionAddedCB_ST:
    """
    EventのCallbackで得たデータにおいて
    event.type = "reaction_added"だった場合のパース処理
    # https://api.slack.com/apis/connections/events-api#callback-field

    Args:
        purse_target (dict): パース対象

    Returns:
        EventReactionAddedCB_ST: パース結果
    """
    purse_result = __purseEventCB(purse_target)
    purse_result.event = EventReactionAdded_ST(**purse_target["event"])
    # 出力の型に変換する
    purse_result_dict = dataclasses.asdict(purse_result)
    ret = EventReactionAddedCB_ST(**purse_result_dict)
    return ret


def purseEventReactionRemovedCB(purse_target: dict) -> EventReactionRemovedCB_ST:
    """
    EventのCallbackで得たデータにおいて
    event.type = "reaction_removed"だった場合のパース処理
    # https://api.slack.com/apis/connections/events-api#callback-field

    Args:
        purse_target (dict): パース対象

    Returns:
        EventReactionRemovedCB_ST: パース結果
    """
    purse_result = __purseEventCB(purse_target)
    purse_result.event = EventReactionRemoved_ST(**purse_target["event"])
    # 出力の型に変換する
    purse_result_dict = dataclasses.asdict(purse_result)
    ret = EventReactionRemovedCB_ST(**purse_result_dict)
    return ret


def purseEventAppMentionCB(purse_target: dict) -> EventAppMentionCB_ST:
    """
    EventのCallbackで得たデータにおいて
    event.type = "app_mention"だった場合のパース処理
    # https://api.slack.com/apis/connections/events-api#callback-field

    Args:
        purse_target (dict): パース対象

    Returns:
        EventAppMentionCB_ST: パース結果
    """
    purse_result = __purseEventCB(purse_target)
    purse_result.event = EventAppMention_ST(**purse_target["event"])
    # 出力の型に変換する
    purse_result_dict = dataclasses.asdict(purse_result)
    ret = EventAppMentionCB_ST(**purse_result_dict)
    return ret


def __purseEventMessage(purse_target: dict) -> EventMessage_ST | EventMsgChange_ST | EventMsgDelete_ST:
    """
    event.type = "message"のパース処理
    # https://api.slack.com/events/message
    # https://api.slack.com/events/message/message_changed
    # https://api.slack.com/events/message/message_deleted

    Args:
        purse_target (dict): パース対象

    Returns:
        EventMessage_ST | EventMsgChange_ST | EventMsgDelete_ST: パース結果
    """
    # MessageのSubtypeによってパースするクラスを変える
    if "subtype" not in purse_target.keys():
        ret = EventMessage_ST(**purse_target)
        return ret

    # Messageが削除されたとき
    if purse_target["subtype"] == EventMsgSubtypes_Const.message_deleted:
        channel_msg_delete = EventMsgDelete_ST(**purse_target)
        ret = EventMsgDelete_ST(
            previous_message=EventMessage_ST(**channel_msg_delete.previous_message),
            channel=channel_msg_delete.channel,
            hidden=channel_msg_delete.hidden,
            deleted_ts=channel_msg_delete.deleted_ts,
            event_ts=channel_msg_delete.event_ts,
            ts=channel_msg_delete.ts,
            channel_type=channel_msg_delete.channel_type,
            subtype=channel_msg_delete.subtype,
            type=channel_msg_delete.type,
        )
    # Messageが編集されたとき
    elif purse_target["subtype"] == EventMsgSubtypes_Const.message_changed:
        channel_msg_change = EventMsgChange_ST(**purse_target)
        ret = EventMsgChange_ST(
            message=EventMessage_ST(**channel_msg_change.message),
            previous_message=EventMessage_ST(**channel_msg_change.previous_message),
            type=channel_msg_change.type,
            subtype=channel_msg_change.subtype,
            channel=channel_msg_change.channel,
            hidden=channel_msg_change.hidden,
            ts=channel_msg_change.ts,
            event_ts=channel_msg_change.event_ts,
            channel_type=channel_msg_change.channel_type,
        )
    # その他Messageに関するアクションがあったとき
    else:
        ret = EventMessage_ST(**purse_target)

    return ret


if __name__ == "__main__":
    pass
