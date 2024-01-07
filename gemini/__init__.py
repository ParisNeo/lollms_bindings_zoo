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
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors
from lollms.types import MSG_TYPE
from lollms.utilities import detect_antiprompt, remove_text_from_string, trace_exception, PackageManager
from lollms.com import LoLLMsCom
import subprocess
import yaml
import re
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
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "Gemini"
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
                {"name":"google_api","type":"str","value":"v1beta","options":["v1beta","v1beta2","v1beta3"],"Help":"API"},
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
        if not self.binding_config.config["google_api_key"]:
            self.error("No API key is set!\nPlease set up your API key in the binding configuration")
        else:
            self.build_model()

        self.config.ctx_size=self.binding_config.config.ctx_size        

    def build_model(self):
        import google.generativeai as genai
        genai.configure(api_key=self.binding_config.google_api_key)
        if self.config.model_name!="gemini-pro-vision":
            self.model = genai.GenerativeModel(self.config.model_name)
        else:
            self.config.model_name = "gemini-pro"
            self.model = genai.GenerativeModel("gemini-pro")
        self.vision_model = genai.GenerativeModel("gemini-pro-vision")
        self.genai = genai
        self.binding_type=BindingType.TEXT_IMAGE
        
        return self

    def install(self):
        super().install()
        subprocess.run(["pip", "install", "--upgrade", "google-generativeai"])
        ASCIIColors.success("Installed successfully")
        ASCIIColors.error("----------------------")
        ASCIIColors.error("Attention please")
        ASCIIColors.error("----------------------")
        ASCIIColors.error("The google bard binding uses the Google Bard API which is a paid service. Please create an account on the google cloud website then generate a key and provide it in the configuration file.")
    
    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        import tiktoken
        tokens_list = tiktoken.model.encoding_for_model("gpt-3.5-turbo").encode(prompt)

        return tokens_list

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        import tiktoken
        text = tiktoken.model.encoding_for_model("gpt-3.5-turbo").decode(tokens_list)

        return text

    
    
    def embed(self, text):
        """
        Computes text embedding
        Args:
            text (str): The text to be embedded.
        Returns:
            List[float]
        """
        return self.genai.embed_content(
            model="models/embedding-001",
            content="What is the meaning of life?",
            task_type="retrieval_document",
            title="Embedding of single string")['embedding']

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
        try:
            output = ""
            default_params = {
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.96,
                'repeat_penalty': 1.3
            }
            gpt_params = {**default_params, **gpt_params}
            response = self.model.generate_content(
                                                        prompt,
                                                        generation_config=self.genai.types.GenerationConfig(
                                                        # Only one candidate for now.
                                                        candidate_count=1,
                                                        stop_sequences=[],
                                                        max_output_tokens=n_predict,
                                                        temperature=float(gpt_params['temperature'])),
                                                        stream=True)
            count = 0
            for chunk in response:
                if count >= n_predict:
                    break
                try:
                    word = chunk.text
                except Exception as ex:
                    word = ""
                if callback is not None:
                    if not callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                        break
                if word:
                    output += word
                    count += 1
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
        """Generates text out of a prompt

        Args:
            prompt (str): The prompt to use for generation
            n_predict (int, optional): Number of tokens to prodict. Defaults to 128.
            callback (Callable[[str], None], optional): A callback function that is called everytime a new text element is generated. Defaults to None.
            verbose (bool, optional): If true, the code will spit many informations about the generation process. Defaults to False.
        """
        images_list = []
        for image in images:
            images_list.append(Image.open(image))

        default_params = {
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.96,
            'repeat_penalty': 1.3
        }
        gpt_params = {**default_params, **gpt_params}
        response = self.vision_model.generate_content(
                                                    [prompt]+images_list,
                                                    generation_config=self.genai.types.GenerationConfig(
                                                    # Only one candidate for now.
                                                    candidate_count=1,
                                                    stop_sequences=[],
                                                    max_output_tokens=n_predict,
                                                    temperature=float(gpt_params['temperature'])),
                                                    stream=True)

        count = 0
        output = ""
        for chunk in response:
            if count >= n_predict:
                break
            try:
                word = chunk.text
            except Exception as ex:
                word = ""
            if callback is not None:
                if not callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                    break
            if word:
                output += word
                count += 1

        return output
    
    def list_models(self):
        """Lists the models for this binding
        """
        # url = f'https://generativelanguage.googleapis.com/{self.binding_config.google_api}/models?key={self.binding_config.google_api_key}'

        # response = requests.get(url)
        # response_json = response.json()
        response_json=[
            {
            "name": "gemini-pro",
            "version": "001",
            "displayName": "Gemini Pro",
            "description": "The best model for scaling across a wide range of tasks",
            "inputTokenLimit": 30720,
            "outputTokenLimit": 2048,
            "supportedGenerationMethods": [
                "generateContent",
                "countTokens"
            ],
            "temperature": 0.9,
            "topP": 1,
            "topK": 1
            },
            {
            "name": "gemini-pro-vision",
            "version": "001",
            "displayName": "Gemini Pro Vision",
            "description": "The best image understanding model to handle a broad range of applications",
            "inputTokenLimit": 12288,
            "outputTokenLimit": 4096,
            "supportedGenerationMethods": [
                "generateContent",
                "countTokens"
            ],
            "temperature": 0.4,
            "topP": 1,
            "topK": 32
            },
            {
            "name": "gemini-ultra",
            "version": "001",
            "displayName": "Gemini Ultra",
            "description": "The most capable model for highly complex tasks",
            "inputTokenLimit": 30720,
            "outputTokenLimit": 2048,
            "supportedGenerationMethods": [
                "generateContent",
                "countTokens"
            ],
            "temperature": 0.9,
            "topP": 1,
            "topK": 32
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
            "name": "gemini-pro",
            "version": "001",
            "displayName": "Gemini Pro",
            "description": "The best model for scaling across a wide range of tasks",
            "inputTokenLimit": 30720,
            "outputTokenLimit": 2048,
            "supportedGenerationMethods": [
                "generateContent",
                "countTokens"
            ],
            "temperature": 0.9,
            "topP": 1,
            "topK": 1
            },
            {
            "name": "gemini-pro-vision",
            "version": "001",
            "displayName": "Gemini Pro Vision",
            "description": "The best image understanding model to handle a broad range of applications",
            "inputTokenLimit": 12288,
            "outputTokenLimit": 4096,
            "supportedGenerationMethods": [
                "generateContent",
                "countTokens"
            ],
            "temperature": 0.4,
            "topP": 1,
            "topK": 32
            },
            {
            "name": "gemini-ultra",
            "version": "001",
            "displayName": "Gemini Ultra",
            "description": "The most capable model for highly complex tasks",
            "inputTokenLimit": 30720,
            "outputTokenLimit": 2048,
            "supportedGenerationMethods": [
                "generateContent",
                "countTokens"
            ],
            "temperature": 0.9,
            "topP": 1,
            "topK": 32
            },
        ]
        for model in response_json:
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
