import os
import re
import shutil

from dotenv import load_dotenv
from langchain import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate

load_dotenv()

# pptxとpblで使っているpythonの喰い合わせが悪いのでいっこモジュールを置き換える
src = "./first-bolt-app/data/for_replace/__init__.py"
dst = "/home/vscode/.local/lib/python3.11/site-packages/pptx/compat/__init__.py"
shutil.copy(src, dst)


class MyChatGPT:
    def __init__(self):
        pass

    def create_chain(self, llm):
        system_template = """
        You are an assistant who thinks step by step and includes a thought path in your response.
        Your answers are in Japanese.
        """
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        human_template = "{user_prompt}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chat_prompt.input_variables = ["user_prompt"]

        chain = LLMChain(llm=llm, prompt=chat_prompt)

        return chain

    def get_llm(self):
        return ChatOpenAI(
            temperature=0,
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo-16k-0613",
            streaming=True,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        )

    def exec_python_code(self, response):
        """
        ChatGPTが作ってくれたスライド作成コードを実行する

        Args:
            response (str): ChatGPTの返答
        """

        # Pythonコードの抽出（開始マーカーは```python、終了マーカーは```）
        python_code = re.search("```python\n(.*?)\n```", response, re.DOTALL).group(1)
        # ChatGPTが作ってくれたスライド作成コードを実行
        exec(python_code)

    def main(self):
        with open("./first-bolt-app/data/sample_prompt.txt", "r") as file:
            user_prompt = file.read()

        self.chain = self.create_chain(self.get_llm())
        response = self.chain.run(user_prompt=user_prompt)
        self.exec_python_code(response)


if __name__ == "__main__":
    bot = MyChatGPT()
    bot.main()
