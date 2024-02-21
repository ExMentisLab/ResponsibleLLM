import json
import os.path
import string
import sys
from os import environ
from argparse import ArgumentParser
from logging import getLogger, basicConfig
from sys import argv
from prompt_toolkit import prompt

from llama_cpp import Llama
from icecream.icecream import ic

logging = getLogger()
basicConfig(filename="responsiblellm.log", filemode='a', level=environ.get("LOGLEVEL", "INFO"))


def load_config(config_path: string) -> dict:
    """ Load prompts configuration from json"""
    try:
        with open(config_path) as f:
            config = json.load(f)
        return config
    except (IOError, ValueError):
        logging.error(f"{config_path} is not a valid json")
        return {}


class ResponsibleLLM:
    def __init__(self,
                 model_path: string,
                 config_path: string,
                 conversation_topic: string = "Intelligenza Artificiale",
                 validate_input: bool = False,
                 validate_output: bool = False):
        """ Initialize the bot with the validation options"""

        self.model_path = model_path

        config = load_config(config_path=config_path)

        self.validate_input = validate_input
        self.validate_output = validate_output

        self.llm = Llama(model_path=self.model_path, n_threads=8, verbose=False, chat_format="chatml")

        self.assistant_name = config['assistant_name']
        self.user_name = config['user_name']
        self.conversation_topic = conversation_topic
        self.appropriate_validation_prompt = config['appropriate_validation_prompt']

        self.topic_validation_prompt = config['topic_validation_prompt']

        self.inappropriate_query = config['inappropriate_query']
        self.validation_response = config['validation_response']

        # Load history from file if exists:
        self.history = []
        if os.path.isfile(f"{self.user_name}_history.json"):
            with open(f"{self.user_name}_history.json") as f:
                self.history = json.load(f)

    def dump_history(self) -> None:
        """ Write user history into a json file"""
        with open(f"{self.user_name}_history.json", "w") as f:
            json.dump(self.history, f, indent=4)

    def message_validation(self, llm: Llama, topic: string, query: string, validation_response: string) -> bool:
        """
            Validate if the query is appropriate for a conversation
            and if it is coherent with the given topic.
        """
        appropriate_validation_query = self.appropriate_validation_prompt.format(assistant_name=self.assistant_name,
                                                                                 query=query)
        ic(appropriate_validation_query)
        result = llm.create_chat_completion(messages=[{"role": "user", "content": appropriate_validation_query}]
                                            , temperature=0.0, max_tokens=2)
        ic(result)
        # ToDo: This might be buggy if the model answers in a weird way.
        appropriate = result['choices'][0]['message']['content'].strip().lower().startswith(validation_response)

        # validate if the query is relevant to the topic
        topic_validation_query = self.topic_validation_prompt.format(topic=topic, query=query)
        ic(topic_validation_query)
        result = llm.create_chat_completion(messages=[{"role": "user", "content": topic_validation_query}]
                                            , temperature=0.0, max_tokens=2)
        ic(result)
        valid = result['choices'][0]['message']['content'].strip().lower().startswith(validation_response)
        if appropriate or valid:
            ic("Returning True for validation")
            return True
        else:
            return False

    def generate_response(self, message: string) -> string:
        """ Generate a response from an input message """
        if self.validate_input:
            validation_result = self.message_validation(llm=self.llm,
                                                        topic=self.conversation_topic,
                                                        query=message,
                                                        validation_response=self.validation_response)
            ic(validation_result)
            if not validation_result:
                message = self.inappropriate_query

        self.history.append({"role": "user", "content": message})
        output = self.llm.create_chat_completion(messages=self.history, temperature=0.2, max_tokens=-1)
        response = output['choices'][0]['message']['content']
        if self.validate_output:
            pass
        self.history.append({"role": "assistant", "content": response})

        return f"{self.assistant_name}: {response}"

    def interactive(self):
        """Run the LLM in interactive mode for the user chat."""
        message = ""

        while message != "quit":
            try:
                message = prompt(f"{self.user_name}: ")
                response = self.generate_response(message)
                print(response)
            except KeyboardInterrupt:
                break

        print("Ok, Byeeee")
        self.dump_history()
        sys.exit(0)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path",
                        help="path of the model to load",
                        default="./mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
                        type=str)
    parser.add_argument("--config_path",
                        help="path to the config file",
                        default="./prompt_config.json",
                        type=str)
    parser.add_argument("--validate_input",
                        help="Whether or not the input from the user is validated by the LLM",
                        default=False,
                        type=bool)

    args = parser.parse_args(argv[1:])

    logging.info(f"CmdLine parameters: {args}")
    bot = ResponsibleLLM(model_path=args.model_path, config_path=args.config_path)
    bot.interactive()
