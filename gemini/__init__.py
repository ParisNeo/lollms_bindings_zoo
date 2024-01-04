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
from lollms.utilities import detect_antiprompt, remove_text_from_string
from lollms.com import LoLLMsCom
import subprocess
import yaml
import re
import json
import requests
from typing import List, Union
from datetime import datetime
__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "Gemini"
binding_folder_name = ""

class Gemini(LLMBinding):
    
    def __init__(self, 
                config: LOLLMSConfig, 
                lollms_paths: LollmsPaths = None, 
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY,
                lollmsCom=None) -> None:
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
                {"name":"google_api_key","type":"str","value":""},
                {"name":"google_api","type":"str","value":"v1beta2","options":["v1beta","v1beta2","v1beta3"],"Help":"API"},
                {"name":"ctx_size","type":"int","value":2048, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"seed","type":"int","value":-1,"help":"Random numbers generation seed allows you to fix the generation making it dterministic. This is useful for repeatability. To make the generation random, please set seed to -1."},

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
                            lollmsCom=lollmsCom
                        )
        self.config.ctx_size=self.binding_config.config.ctx_size

    def settings_updated(self):
        if self.binding_config.config["google_api"]:
            self.error("No API key is set!\nPlease set up your API key in the binding configuration")

        self.config.ctx_size=self.binding_config.config.ctx_size        

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
        PALM_KEY = self.binding_config.google_api_key

        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': PALM_KEY
        }
        default_params = {
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.96,
            'repeat_penalty': 1.3
        }
        gpt_params = {**default_params, **gpt_params}

        data = {
            'prompt': {
                'text': prompt
            },
            "temperature": float(gpt_params["temperature"]),
            "candidateCount": 1
        }

        url = f'https://generativelanguage.googleapis.com/{self.binding_config.google_api}/models/{self.config.model_name}:generateText'

        response = requests.post(url, headers=headers, data=json.dumps(data))
        result = response.json()
        if "error" in result:
            ASCIIColors.error(result["error"]["message"])
            self.error(result["error"]["message"])
            return ''
        else:
            if callback:
                output = result["candidates"][0]["output"]
                antiprompt = detect_antiprompt(output)
                if antiprompt:
                    ASCIIColors.warning(f"\nDetected hallucination with antiprompt: {antiprompt}")
                    output = remove_text_from_string(output, antiprompt)                
                callback(output, MSG_TYPE.MSG_TYPE_FULL)

        return result["candidates"][0]["output"]
    def generate_with_images(self, 
                prompt:str,
                images:list=[],
                n_predict: int = 128,
                callback: Callable[[str, int, dict], bool] = None,
                verbose: bool = False,
                **gpt_params ):
        """Generates text out of a prompt

        Args:
            prompt (str): The prompt to use for generation
            n_predict (int, optional): Number of tokens to prodict. Defaults to 128.
            callback (Callable[[str], None], optional): A callback function that is called everytime a new text element is generated. Defaults to None.
            verbose (bool, optional): If true, the code will spit many informations about the generation process. Defaults to False.
        """
        PALM_KEY = self.binding_config.google_api_key

        headers = {
            'Content-Type': 'application/json',
            'x-goog-api-key': PALM_KEY
        }
        default_params = {
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.96,
            'repeat_penalty': 1.3
        }
        gpt_params = {**default_params, **gpt_params}

        data = {
            'prompt': {
                'text': prompt
            },
            "temperature": float(gpt_params["temperature"]),
            "candidateCount": 1
        }

        url = f'https://generativelanguage.googleapis.com/{self.binding_config.google_api}/models/{self.config.model_name}:generateText'

        response = requests.post(url, headers=headers, data=json.dumps(data))
        result = response.json()
        if "error" in result:
            ASCIIColors.error(result["error"]["message"])
            self.info(result["error"]["message"])
            return ''
        else:
            if callback:
                output = result["candidates"][0]["output"]
                antiprompt = detect_antiprompt(output)
                if antiprompt:
                    ASCIIColors.warning(f"\nDetected hallucination with antiprompt: {antiprompt}")
                    output = remove_text_from_string(output, antiprompt)                
                callback(output, MSG_TYPE.MSG_TYPE_FULL)

        return result["candidates"][0]["output"]
    
    def list_models(self):
        """Lists the models for this binding
        """
        url = f'https://generativelanguage.googleapis.com/{self.binding_config.google_api}/models?key={self.binding_config.google_api_key}'

        response = requests.get(url)
        response_json = response.json()
        return [f["name"] for f in response_json["models"]]                
    
    def get_available_models(self, app:LoLLMsCom=None):
        # Create the file path relative to the child class's directory
        url = f'https://generativelanguage.googleapis.com/{self.binding_config.google_api}/models?key={self.binding_config.google_api_key}'

        response = requests.get(url)
        models = []
        response_json = response.json()
        for model in response_json["models"]:
            md = {
                "category": "generic",
                "datasets": "unknown",
                "icon": '/bindings/gemini/logo.png',
                "last_commit_time": datetime.now().timestamp(),
                "license": "commercial",
                "model_creator": "google",
                "model_creator_link": "https://ai.google.dev/",
                "name": model["name"],
                "quantizer": None,
                "rank": 1.0,
                "type": "api",
                "variants":[
                    {
                        "name":model["name"],
                        "size":999999999999
                    }
                ]
            }
            models.append(md)

        return models
