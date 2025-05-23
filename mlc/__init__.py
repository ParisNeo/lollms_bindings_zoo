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
from typing import Callable, Any
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors
from lollms.types import MSG_OPERATION_TYPE
from lollms.utilities import detect_antiprompt, remove_text_from_string, trace_exception, PackageManager
from lollms.com import LoLLMsCom
import subprocess
import yaml
import sys
import json
import requests
from typing import List, Union
from datetime import datetime
from PIL import Image
import base64
import io
import platform
import tempfile
import os


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "MLC"
binding_folder_name = ""

def encode_image(image_path, max_image_width=-1):
    image = Image.open(image_path)
    width, height = image.size

    if max_image_width != -1 and width > max_image_width:
        ratio = max_image_width / width
        new_width = max_image_width
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height))

    # Check and convert image format if needed
    # if image.format not in ['PNG', 'JPEG', 'GIF', 'WEBP']:
    #     image = image.convert('JPEG')

    # Save the image to a BytesIO object
    byte_arr = io.BytesIO()
    image.save(byte_arr, format='PNG')
    byte_arr = byte_arr.getvalue()

    return base64.b64encode(byte_arr).decode('utf-8')
class MLC(LLMBinding):
    
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
                {"name":"ctx_size","type":"int","value":4090, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"max_n_predict","type":"int","value":4090, "min":512, "help":"The maximum amount of tokens to generate"},
                {"name":"seed","type":"int","value":-1,"help":"Random numbers generation seed allows you to fix the generation making it dterministic. This is useful for repeatability. To make the generation random, please set seed to -1."},
            ]),
            BaseConfig(config={
            })
        )
        super().__init__(
                            Path(__file__).parent, 
                            lollms_paths, 
                            config, 
                            binding_config, 
                            installation_option,
                            SAFE_STORE_SUPPORTED_FILE_EXTENSIONS=[''],
                            lollmsCom=lollmsCom
                        )
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict

    def settings_updated(self):
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict

    def build_model(self, model_name="HF://mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC"):
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict
        if model_name is None:
            ASCIIColors.error("Please select a model first")
            return
        super().build_model(model_name)
        from mlc_llm import MLCEngine
        # Create engine
        self.engine = MLCEngine(model_name)
        
        return self

    def install(self):
        super().install()
        url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.4/LLVM-16.0.4-win64.exe"

        subprocess.run([sys.executable, "-m", "pip", "install", "--pre", "-U", "-f", "https://mlc.ai/wheels", "mlc-llm-nightly","mlc-ai-nightly"])
        if platform.system() != 'Windows':
            print("This script is designed to run on Windows systems only.")
            return

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmp_file_path = os.path.join(tmpdirname, 'installer.exe')
            try:
                # Download the file
                print("Downloading the file...")
                response = requests.get(url)
                response.raise_for_status()  # Raise an error for bad responses
                with open(tmp_file_path, 'wb') as tmp_file:
                    tmp_file.write(response.content)
                print(f"File downloaded to temporary directory and saved as {tmp_file_path}")

                # Execute the downloaded file
                print("Executing the file...")
                subprocess.run(tmp_file_path, shell=True)
                print("Execution completed.")

                # Perform the pip install
                self.install()

            except requests.RequestException as e:
                print(f"Error downloading the file: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")

            # The temporary directory and files within are automatically cleaned up        
        ASCIIColors.success("Installed successfully")
    
    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        tokens_list = self.engine.tokenizer.encode(prompt)

        return tokens_list

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        text = self.engine.tokenizer.decode(tokens_list)

        return text

    
    
    def embed(self, text):
        """
        Computes text embedding
        Args:
            text (str): The text to be embedded.
        Returns:
            List[float]
        """
        return 

    def generate(self, 
                 prompt: str,                  
                 n_predict: int = 128,
                 callback: Callable[[str], None] = None,
                 verbose: bool = False,
                 **gpt_params) -> str:
        """Generates text out of a prompt

        Args:
            prompt (str): The prompt to use for generation
            n_predict (int, optional): Number of tokens to predict. Defaults to 128.
            callback (Callable[[str], None], optional): A callback function that is called everytime a new text element is generated. Defaults to None.
            verbose (bool, optional): If true, the code will spit many informations about the generation process. Defaults to False.
        """
        try:
            count = 0
            output = ""
            default_params = {
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.96,
                'repeat_penalty': 1.3
            }
            gpt_params = {**default_params, **gpt_params}

            # Run chat completion in OpenAI API.
            for response in self.engine.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=n_predict,
                temperature=float(gpt_params["temperature"]),
                stream=True,
            ):
                for choice in response.choices:
                    if callback is not None:
                        if not callback(choice.delta.content, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                            break                    
                    output += choice.delta.content
                    count += 1
        except Exception as ex:
            trace_exception(ex)
            self.error(ex)
        return output
    
    def list_models(self):
        """Lists the models for this binding
        """
        # url = f'https://generativelanguage.googleapis.com/{self.binding_config.google_api}/models?key={self.binding_config.google_api_key}'

        # response = requests.get(url)
        # response_json = response.json()
        response_json=[
            {
            "name": "mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
            },
            {
            "name": "mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC",
            },
            {
            "name": "mlc-ai/phi-2-q4f16_1-MLC",
            },
            {
            "name": "mlc-ai/Qwen1.5-1.8B-Chat-q4f16_1-MLC",
            },
        ]        
        return [f["name"] for f in response_json]                
    
    def get_available_models(self, app:LoLLMsCom=None):
        # Create the file path relative to the child class's directory
        # url = f'https://generativelanguage.googleapis.com/{self.binding_config.google_api}/models?key={self.binding_config.google_api_key}'

        # response = requests.get(url)
        models = []
        #response_json = response.json()
        response_json=[
            {
            "name": "mlc-ai/Llama-3-8B-Instruct-q4f16_1-MLC",
            "model_creator": "meta"
            },
            {
            "name": "mlc-ai/Mistral-7B-Instruct-v0.2-q4f16_1-MLC",
            "model_creator": "mistral ai"
            },
            {
            "name": "mlc-ai/phi-2-q4f16_1-MLC",
            "model_creator": "microsoft"
            },
            {
            "name": "mlc-ai/Qwen1.5-1.8B-Chat-q4f16_1-MLC",
            "model_creator": "Qwen"
            },
        ]
        for model in response_json:
            md = {
                "category": "generic",
                "datasets": "unknown",
                "icon": '/bindings/MLC/logo.png',
                "last_commit_time": datetime.now().timestamp(),
                "license": "commercial",
                "model_creator": model["model_creator"],
                "model_creator_link": "",
                "name": model["name"],
                "provider": None,
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
