#!/usr/bin/env python3

from rich import print
import dataclasses

# Blockの一番細かい要素
# Composition objects
# https://api.slack.com/reference/block-kit/composition-objects


# Text Type Field
# https://api.slack.com/reference/block-kit/composition-objects#text
@dataclasses.dataclass(frozen=True)
class Compo_TextType_Const:
    plain_text: str = "plain_text"
    mrkdwn: str = "mrkdwn"


# Text Object
# https://api.slack.com/reference/block-kit/composition-objects#text
@dataclasses.dataclass(frozen=True)
class Compo_Text_ST:
    type: Compo_TextType_Const
    text: str
    emoji: bool = None
    verbatim: bool = None


# Dialog Style Field
# https://api.slack.com/reference/block-kit/block-elements#button__fields (style)
class Compo_DialogStyle_Const:
    default: str = "default"
    primary: str = "primary"
    danger: str = "danger"


# Confirmation Dialog Object
# https://api.slack.com/reference/block-kit/composition-objects
@dataclasses.dataclass(frozen=True)
class Compo_Dialog_ST:
    title: Compo_Text_ST
    text: Compo_Text_ST
    confirm: Compo_Text_ST
    deny: Compo_Text_ST
    style: Compo_DialogStyle_Const = Compo_DialogStyle_Const.primary


# Option Object
# https://api.slack.com/reference/block-kit/composition-objects#option
# The url attribute is only available in overflow menus.
# https://api.slack.com/reference/block-kit/block-elements#overflow
@dataclasses.dataclass(frozen=True)
class Compo_Option_ST:
    text: Compo_Text_ST
    value: str
    description: Compo_Text_ST = None
    url: str = None


# Option Group Object
# Provides a way to group options in a select menu or multi-select menu.
# https://api.slack.com/reference/block-kit/composition-objects#option_group
@dataclasses.dataclass(frozen=True)
class Compo_OptionGroup_ST:
    label: Compo_Text_ST
    options: list[Compo_Option_ST]


# Dispatch Action Configuration Trigger field
# https://api.slack.com/reference/block-kit/composition-objects#dispatch_action_config (trigger_actions_on)
class Compo_DispatchTrigger_Const:
    on_enter_pressed: str = "on_enter_pressed"
    on_character_entered: str = "on_character_entered"


# Dispatch Action Configuration Object
# https://api.slack.com/reference/block-kit/composition-objects#dispatch_action_config
@dataclasses.dataclass(frozen=True)
class Compo_DispatchActionConfig_ST:
    trigger_actions_on: list[Compo_DispatchTrigger_Const]


# Input Parameter Object
# https://api.slack.com/reference/block-kit/composition-objects#input_parameter
@dataclasses.dataclass(frozen=True)
class Compo_InputParameter_ST:
    name: str
    value: str


# Trigger Object
# https://api.slack.com/reference/block-kit/composition-objects#trigger
@dataclasses.dataclass(frozen=True)
class Compo_Trigger_ST:
    url: str
    customizable_input_parameters: list[Compo_InputParameter_ST] = None


# Workflow Object
# https://api.slack.com/reference/block-kit/composition-objects#workflow
@dataclasses.dataclass(frozen=True)
class Compo_Workflow_ST:
    trigger: Compo_Trigger_ST


# Blockの一番細かい要素はここまで
# ここからはBlock要素 Object
# https://api.slack.com/reference/block-kit/block-elements


# Button Element Object
# https://api.slack.com/reference/block-kit/block-elements#button
@dataclasses.dataclass(frozen=True)
class Elem_Button_ST:
    text: Compo_Text_ST  # type: plain_textのみ可能
    action_id: str
    type: str = "button"
    url: str = None
    value: str = None
    style: Compo_DialogStyle_Const = Compo_DialogStyle_Const.primary
    confirm: Compo_Dialog_ST = None
    accessibility_label: str = None


# Checkbox Element Object
# https://api.slack.com/reference/block-kit/block-elements#checkboxes
@dataclasses.dataclass(frozen=True)
class Elem_Checkbox_ST:
    action_id: str
    options: list[Compo_Option_ST]
    type: str = "checkboxes"
    initial_options: list[Compo_Option_ST] = None
    confirm: Compo_Dialog_ST = None
    focus_on_load: bool = None


# Date picker element object
# https://api.slack.com/reference/block-kit/block-elements#datepicker
@dataclasses.dataclass(frozen=True)
class Elem_DatePicker_ST:
    action_id: str
    type: str = "datepicker"
    placeholder: Compo_Text_ST = None
    initial_date: str = None
    confirm: Compo_Dialog_ST = None
    focus_on_load: bool = None
    placeholder: Compo_Text_ST = None


# Image Element Object
# https://api.slack.com/reference/block-kit/block-elements#image
@dataclasses.dataclass(frozen=True)
class Elem_Image_ST:
    type: str = "image"
    image_url: str = "http://placekitten.com/700/500"
    alt_text: str = "Multiple cute kittens"


# Select menu of static option element object
# https://api.slack.com/reference/block-kit/block-elements#static_select
@dataclasses.dataclass(frozen=True)
class Elem_SelectStatic_ST:
    action_id: str
    type: str = "static_select"
    options: list[Compo_Option_ST] = None
    option_groups: list[Compo_OptionGroup_ST] = None
    initial_option: Compo_Option_ST = None
    confirm: Compo_Dialog_ST = None
    focus_on_load: bool = None
    placeholder: Compo_Text_ST = None


# ここからは Block Object
# https://api.slack.com/reference/block-kit/blocks


# Actions Block Object
# https://api.slack.com/reference/block-kit/blocks#actions
@dataclasses.dataclass(frozen=True)
class ActionsBlock_ST:
    elements: list[Elem_Button_ST | Elem_DatePicker_ST | Elem_SelectStatic_ST]
    type: str = "actions"
    block_id: str = None


# Section Block Object
# https://api.slack.com/reference/block-kit/blocks#section
@dataclasses.dataclass(frozen=True)
class SectionBlock_ST:
    type: str = "section"
    text: Compo_Text_ST = None
    block_id: str = None
    fields: list[Compo_Text_ST] = None
    accessory: Elem_Button_ST | Elem_Checkbox_ST | Elem_DatePicker_ST | Elem_Image_ST | Elem_SelectStatic_ST = None


# Input Block Object
# https://api.slack.com/reference/block-kit/blocks#input
@dataclasses.dataclass(frozen=True)
class InputBlock_ST:
    label: Compo_Text_ST
    type: str = "input"
    element: Elem_Checkbox_ST | Elem_DatePicker_ST | Elem_SelectStatic_ST = None
    dispatch_action: bool = None
    block_id: str = None
    hint: Compo_Text_ST = None
    optional: bool = None


# https://api.slack.com/messaging/composing/layouts#stack_of_blocks
# Top-levelの要約テキストを提供する必要がある。これは通知などに使われている。
# https://github.com/slackapi/python-slack-sdk/issues/965#issuecomment-906383420
@dataclasses.dataclass(frozen=True)
class Blocks_ST:
    blocks: list[ActionsBlock_ST | SectionBlock_ST | InputBlock_ST]
    text: str = "Notification!"


def __deleteNoneKey(slack_dict: dict) -> dict:
    """
    slack_dictの中にNoneのkeyがあれば削除する

    Args:
        slack_dict (dict): 削除対象のdict

    Returns:
        dict: 処理結果
    """
    for key, dict_value in list(slack_dict.items()):
        if dict_value is None:
            del slack_dict[key]
        else:
            if isinstance(dict_value, dict):
                __deleteNoneKey(dict_value)
            elif isinstance(dict_value, list):
                for list_elem in dict_value:
                    if isinstance(list_elem, dict):
                        __deleteNoneKey(list_elem)

    return slack_dict


def slackStructAsDict(blocks: Blocks_ST) -> dict:
    """
    slack blockの構造体をdictに変換する
    Noneの要素をSlack APIに投げるとErrorになってしまうので、削除する
    そのために__deleteNoneKey()を再帰的に呼び出している。

    Args:
        blocks (Blocks_ST): slack blockの構造体

    Returns:
        dict: slack blockの構造体をdictに変換したもの
    """
    blocks_dict = dataclasses.asdict(blocks)
    blocks_dict = __deleteNoneKey(blocks_dict)

    return blocks_dict


# 以下builderのサンプル実装
# SectionBlock_ST | ActionsBlock_ST | InputBlock_STのBlockを組み合わせて作る。
class HelloBuilder(object):
    def __init__(self, say_text="Hello!!!", notify_text="Hello Notification!!!") -> None:
        """
        Botに喋らせるためのBlockを作成する。

        Args:
            say_text (str, optional): 喋らせることを記す. Defaults to "Hello!!!".
            notify_text (str, optional): Slack通知のテキストを記す. Defaults to "Hello Notification!!!".
        """
        # Block要素でBlockを定義
        hello_in_section_block = SectionBlock_ST(text=Compo_Text_ST(type=Compo_TextType_Const.plain_text, text="Hello!!!"))
        text_in_section_block = SectionBlock_ST(text=Compo_Text_ST(type=Compo_TextType_Const.plain_text, text=say_text), accessory=Elem_Image_ST())

        # Blockリストを定義
        self._hello_build = Blocks_ST([hello_in_section_block, text_in_section_block], text=notify_text)

    @property
    def hello_build(self) -> dict:
        return slackStructAsDict(self._hello_build)


class DatePickerBuilder(object):
    def __init__(self) -> None:
        """
        Dateを選択するブロックを作成する。
        """
        # Elementを定義
        date_picker_elem = Elem_DatePicker_ST(
            action_id="datepicker_01",  # action_idの値でHandlingしている。このIDの場合、@slack_app.action("datepicker_01")で捕まえられる。
            placeholder=Compo_Text_ST(type=Compo_TextType_Const.plain_text, text="日付選択"),
        )

        # Block要素でBlockを定義
        text_in_section_block = SectionBlock_ST(text=Compo_Text_ST(type=Compo_TextType_Const.plain_text, text="日付を選択してください"))

        data_in_input_block = InputBlock_ST(
            label=Compo_Text_ST(type=Compo_TextType_Const.plain_text, text="日付"),
            element=date_picker_elem,
            hint=Compo_Text_ST(type=Compo_TextType_Const.plain_text, text="ヒントは出せない"),
        )

        data_in_section_block = SectionBlock_ST(
            text=Compo_Text_ST(type=Compo_TextType_Const.plain_text, text="日付選択Section"),
            accessory=date_picker_elem,
        )

        # Blockリストを定義
        self._input_block_build = Blocks_ST([text_in_section_block, data_in_input_block])
        self._section_block_build = Blocks_ST([text_in_section_block, data_in_section_block])

    @property
    def input_block_build(self) -> dict:
        return slackStructAsDict(self._input_block_build)

    @property
    def section_block_build(self) -> dict:
        return slackStructAsDict(self._section_block_build)


if __name__ == "__main__":
    builder = DatePickerBuilder()
    input_blocks_dict = builder.input_block_build
    section_block_dict = builder.section_block_build

    # 出力結果を https://app.slack.com/block-kit-builder/ に貼り付ける
    # print('{ "blocks": ', input_blocks_dict["blocks"], "}")
    print('{ "blocks": ', section_block_dict["blocks"], "}")
