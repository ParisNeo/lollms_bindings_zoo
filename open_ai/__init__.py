######
# Project       : GPT4ALL-UI
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying binding : Abdeladim's pygptj binding
# Supported by Nomic-AI
# license       : Apache 2.0
# Description   : 
# This is an interface class for GPT4All-ui bindings.

# This binding is a wrapper to open ai's api

######
from pathlib import Path
from typing import Callable
from lollms.binding import LLMBinding, LOLLMSConfig
from lollms.personality import MSG_TYPE
from api.config import load_config
import openai
import yaml
import re

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "OpenAIGPT"
binding_folder_name = ""

class OpenAIGPT(LLMBinding):
    # Define what is the extension of the model files supported by your binding
    # Only applicable for local models for remote models like gpt4 and others, you can keep it empty 
    # and reimplement your own list_models method
    file_extension='*.bin' 
    def __init__(self, config:LOLLMSConfig) -> None:
        """Builds a OpenAIGPT binding

        Args:
            config (LOLLMSConfig): The configuration file
        """
        super().__init__(config, False)
        
        # The local config can be used to store personal information that shouldn't be shared like chatgpt Key 
        # or other personal information
        # This file is never commited to the repository as it is ignored by .gitignore
        self.config = config
        self._local_config_file_path = Path(__file__).parent/"local_config.yaml"
        self.local_config = load_config(self._local_config_file_path)
        openai.api_key = self.local_config["openai_key"]

        # Do your initialization stuff
            
    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        return None

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        return None
    
    def generate(self, 
                 prompt:str,                  
                 n_predict: int = 128,
                 callback: Callable[[str], None] = bool,
                 verbose: bool = False,
                 **gpt_params ):
        """Generates text out of a prompt

        Args:
            prompt (str): The prompt to use for generation
            n_predict (int, optional): Number of tokens to prodict. Defaults to 128.
            callback (Callable[[str], None], optional): A callback function that is called everytime a new text element is generated. Defaults to None.
            verbose (bool, optional): If true, the code will spit many informations about the generation process. Defaults to False.
        """
        try:
            default_params = {
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.96,
                'repeat_penalty': 1.3
            }
            gpt_params = {**default_params, **gpt_params}
            count = 0
            output = ""
            for resp in openai.Completion.create(model=self.config["model"],  # Choose the engine according to your OpenAI plan
                                prompt=prompt,
                                max_tokens=n_predict,  # Adjust the desired length of the generated response
                                n=1,  # Specify the number of responses you want
                                stop=None,  # Define a stop sequence if needed
                                temperature=gpt_params["temperature"],  # Adjust the temperature for more or less randomness in the output
                                stream=True):
                if count >= n_predict:
                    break
                try:
                    word = resp.choices[0].text
                except:
                    word = ""
                if callback is not None:
                    if not callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                        break
                output += word
                count += 1

        except Exception as ex:
            print(ex)
        return ""            

    
    @staticmethod
    def list_models(config:dict):
        """Lists the models for this binding
        """
        return ["ChatGpt by Open AI"]
                
    @staticmethod
    def get_available_models():
        # Create the file path relative to the child class's directory
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        return yaml_data