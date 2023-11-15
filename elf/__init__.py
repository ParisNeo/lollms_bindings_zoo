######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying 
# engine author : Google 
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.

# This binding is a wrapper to open ai's api

######
from pathlib import Path
from typing import Callable
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig
from lollms.helpers import ASCIIColors
from lollms.types import MSG_TYPE
import subprocess
import yaml
import re
import json
import requests
from typing import List, Union

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "Elf"
binding_folder_name = ""

class Elf(LLMBinding):
    
    def __init__(self, 
                config: LOLLMSConfig, 
                lollms_paths: LollmsPaths = None, 
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY,
                notification_callback:Callable=None) -> None:
        """
        Initialize the Binding.

        Args:
            config (LOLLMSConfig): The configuration object for LOLLMS.
            lollms_paths (LollmsPaths, optional): The paths object for LOLLMS. Defaults to LollmsPaths().
            installation_option (InstallOption, optional): The installation option for LOLLMS. Defaults to InstallOption.INSTALL_IF_NECESSARY.
        """
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        # Initialization code goes here

        binding_config = TypedConfig(
            ConfigTemplate([
                {"name":"address","type":"str","value":"http://127.0.0.1:5000"},
            ]),
            BaseConfig(config={
                "google_api_key": "",     # use avx2
            })
        )
        super().__init__(
                            Path(__file__).parent, 
                            lollms_paths, 
                            config, 
                            binding_config, 
                            installation_option,
                            supported_file_extensions=[''],
                            notification_callback=notification_callback
                        )
        
    def build_model(self):
        return self

    def install(self):
        super().install()
        ASCIIColors.success("Installed successfully")
        ASCIIColors.error("----------------------")
        ASCIIColors.error("Attention please")
        ASCIIColors.error("----------------------")
        ASCIIColors.error("The google bard binding uses the Google Bard API which is a paid service. Please create an account on the google cloud website then generate a key and provide it in the configuration file.")
    
    def tokenize(self, text: Union[str, List[str]]) -> List[str]:
        """Tokenizes a text string

        Args:
            text (str): The text to tokenize

        Returns:
            A list of tokens
        """
        if isinstance(text, str):
            return text.split()
        else:
            return text

    def detokenize(self, tokens: List[str]) -> str:
        """Detokenizes a list of tokens

        Args:
            tokens (List[str]): The tokens to detokenize

        Returns:
            A string
        """
        return " ".join(tokens)
    
    def generate(self, 
                 prompt: str,                  
                 n_predict: int = 128,
                 callback: Callable[[str], None] = bool,
                 verbose: bool = False,
                 **gpt_params) -> str:
        """Generates text out of a prompt

        Args:
            prompt (str): The prompt to use for generation
            n_predict (int, optional): Number of tokens to predict. Defaults to 128.
            callback (Callable[[str], None], optional): A callback function that is called everytime a new text element is generated. Defaults to None.
            verbose (bool, optional): If true, the code will spit many informations about the generation process. Defaults to False.
        """

        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer NO_KEY_NEEDED',
        }
        default_params = {
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.96,
            'repeat_penalty': 1.3
        }
        gpt_params = {**default_params, **gpt_params}

        data = {
            'model':'',
            'messages': [{
                'role': "",
                'content': prompt
            }],
            "stream":True,
            "temperature": float(gpt_params["temperature"]),
            "max_tokens": n_predict
        }

        url = f'{self.binding_config.address}/v1/chat/completions'

        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

        text = ""
        for line in response.iter_lines(): 
            decoded = line.decode("utf-8")
            if len(decoded)>5:
                txt = decoded[5:].strip()
                if txt[0]=="{":
                    json_data = json.loads(txt)
                    chunk = json_data["choices"][0]["delta"]["content"]
                else:
                    chunk = decoded
                ## Process the JSON data here
                text +=chunk
                if callback:
                    callback(text, MSG_TYPE.MSG_TYPE_FULL)

        return text

    
    @staticmethod
    def list_models(config:dict):
        """Lists the models for this binding
        """
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        

        return [f["name"] for f in yaml_data]
                
    @staticmethod
    def get_available_models():
        # Create the file path relative to the child class's directory
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        return yaml_data