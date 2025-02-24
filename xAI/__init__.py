######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying 
# engine author : xAI
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.
# This binding is a wrapper for xAI's Grok API
######

from pathlib import Path
from typing import Callable, Any
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors
from lollms.types import MSG_OPERATION_TYPE
from lollms.utilities import detect_antiprompt, remove_text_from_string, trace_exception, PackageManager
from lollms.com import LoLLMsCom
import subprocess
import sys
import json
import requests
from typing import List, Union
from datetime import datetime
from PIL import Image
import base64
import io

if not PackageManager.check_package_installed("tiktoken"):
    PackageManager.install_package('tiktoken')
import tiktoken

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2024, "
__license__ = "Apache 2.0"

binding_name = "Grok"
binding_folder_name = ""

class Grok(LLMBinding):
    def __init__(self, 
                config: LOLLMSConfig, 
                lollms_paths: LollmsPaths = None, 
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY,
                lollmsCom=None) -> None:
        """
        Initialize the Binding.
        """
        if lollms_paths is None:
            lollms_paths = LollmsPaths()

        binding_config = TypedConfig(
            ConfigTemplate([
                {"name":"api_key","type":"str","value":"", "help":"Your xAI API key"},
                {"name":"base_url","type":"str","value":"https://api.x.ai/v1", "help":"API base URL"},
                {"name":"ctx_size","type":"int","value":8192, "min":512, "help":"The current context size"},
                {"name":"max_tokens","type":"int","value":4096, "min":1, "help":"Maximum number of tokens to generate"},
                {"name":"temperature","type":"float","value":0.7, "min":0.0, "max":2.0, "help":"Temperature for sampling"},
                {"name":"top_p","type":"float","value":0.95, "min":0.0, "max":1.0, "help":"Top-p sampling parameter"},
            ]),
            BaseConfig(config={})
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
        
        self.binding_type = BindingType.TEXT_IMAGE  # or BindingType.TEXT if image support is confirmed
        
    def settings_updated(self):
        """Called when settings are updated"""
        if not self.binding_config.api_key:
            self.error("No API key is set! Please set up your xAI API key in the binding configuration")
        else:
            self.build_model()

    def build_model(self, model_name=None):
        """Builds the model"""
        super().build_model(model_name)
        self.headers = {
            "Authorization": f"Bearer {self.binding_config.api_key}",
            "Content-Type": "application/json"
        }
        return self

    def install(self):
        """Installs required packages"""
        super().install()
        PackageManager.install_package("requests")
        ASCIIColors.success("Installed successfully")
        ASCIIColors.warning("----------------------")
        ASCIIColors.warning("Attention please")
        ASCIIColors.warning("----------------------")
        ASCIIColors.warning("The Grok binding uses xAI's API which is a paid service. Please create an account on xAI and provide your API key in the configuration.")

    def tokenize(self, prompt:str):
        """Tokenizes the given prompt"""
        return tiktoken.model.encoding_for_model("gpt-4").encode(prompt)

    def detokenize(self, tokens_list:list):
        """Detokenizes the given tokens"""
        return tiktoken.model.encoding_for_model("gpt-4").decode(tokens_list)

    def generate(self, 
                prompt: str,                  
                n_predict: int = 128,
                callback: Callable[[str], None] = None,
                verbose: bool = False,
                **gpt_params) -> str:
        """Generates text from a prompt"""
        try:
            output = ""
            
            # Prepare the chat message
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare the request payload
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": min(n_predict, self.binding_config.max_tokens),
                "temperature": self.binding_config.temperature,
                "top_p": self.binding_config.top_p,
                "stream": True
            }

            # Make streaming request
            response = requests.post(
                f"{self.binding_config.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True
            )

            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

            for line in response.iter_lines():
                if line:
                    # Remove "data: " prefix and parse JSON
                    try:
                        chunk = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if chunk and 'choices' in chunk and len(chunk['choices']) > 0:
                            content = chunk['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                if callback is not None:
                                    if not callback(content, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                                        break
                                output += content
                    except json.JSONDecodeError:
                        continue

        except Exception as ex:
            trace_exception(ex)
            self.error(ex)
            
        return output
        

    def generate_with_images(self, 
                prompt:str,
                images:list=[],
                n_predict: int = 128,
                callback: Callable[[str, int, dict], bool] = None,
                verbose: bool = False,
                **gpt_params ):
        """Generates text from a prompt with images

        Args:
            prompt (str): The prompt to use for generation
            images (list): List of image file paths to process
            n_predict (int, optional): Number of tokens to predict. Defaults to 128.
            callback (Callable[[str], None], optional): A callback function that is called everytime a new text element is generated. Defaults to None.
            verbose (bool, optional): If true, the code will spit many informations about the generation process. Defaults to False.
        """
        try:
            # Process images
            processed_images = []
            for image_path in images:
                with Image.open(image_path) as img:
                    # Convert image to base64
                    buffered = io.BytesIO()
                    img.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
                    processed_images.append({
                        "type": "image_url",
                        "image_url": {
                            "url":"data:image/png;base64,"+img_str,
                            "detail": "high"
                        }
                    })

            # Prepare messages with both text and images
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *processed_images
                    ]
                }
            ]

            # Prepare request payload
            payload = {
                "model": self.config.model_name,
                "messages": messages,
                "max_tokens": min(n_predict, self.binding_config.max_tokens),
                "temperature": self.binding_config.temperature,
                "top_p": self.binding_config.top_p,
                "stream": True
            }
            
            print(payload)

            # Make streaming request
            response = requests.post(
                f"{self.binding_config.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                stream=True
            )

            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")

            output = ""
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if chunk and 'choices' in chunk and len(chunk['choices']) > 0:
                            content = chunk['choices'][0].get('delta', {}).get('content', '')
                            if content:
                                if callback is not None:
                                    if not callback(content, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                                        break
                                output += content
                    except json.JSONDecodeError:
                        continue

        except Exception as ex:
            trace_exception(ex)
            self.error(ex)
            
        return output


    def list_models(self):
        """Lists available models"""
        try:
            response = requests.get(
                f"{self.binding_config.base_url}/models",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                return [model["id"] for model in data["data"]]
            else:
                self.error(f"Failed to get models list: {response.text}")
                return []
        except Exception as ex:
            trace_exception(ex)
            return []

    def get_available_models(self, app:LoLLMsCom=None):
        """Gets information about available models"""
        models = []
        
        try:
            response = requests.get(
                f"{self.binding_config.base_url}/models",
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                
                for model in data["data"]:
                    md = {
                        "category": "generic",
                        "datasets": "unknown",
                        "icon": '/bindings/xAI/logo.png',
                        "last_commit_time": datetime.now().timestamp(),
                        "license": "commercial",
                        "model_creator": "xAI",
                        "model_creator_link": "https://x.ai/",
                        "name": model["id"],
                        "quantizer": None,
                        "rank": 1.0,
                        "type": "api",
                        "variants":[
                            {
                                "name": model["id"],
                                "size": 999999999999
                            }
                        ]
                    }
                    models.append(md)
                    
        except Exception as ex:
            trace_exception(ex)
            
        return models
