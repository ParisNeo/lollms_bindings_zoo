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
from lollms.utilities import detect_antiprompt, remove_text_from_string, trace_exception
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
        self.model = genai.GenerativeModel(self.config.model_name)
        self.genai = genai
        if "vision" in self.config.model_name:
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
        response = self.model.generate_content(
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
            "name": "chat-bison-001",
            "version": "001",
            "displayName": "Chat Bison",
            "description": "Chat-optimized generative language model.",
            "inputTokenLimit": 4096,
            "outputTokenLimit": 1024,
            "supportedGenerationMethods": [
                "generateMessage",
                "countMessageTokens"
            ],
            "temperature": 0.25,
            "topP": 0.95,
            "topK": 40
            },
            {
            "name": "text-bison-001",
            "version": "001",
            "displayName": "Text Bison",
            "description": "Model targeted for text generation.",
            "inputTokenLimit": 8196,
            "outputTokenLimit": 1024,
            "supportedGenerationMethods": [
                "generateText",
                "countTextTokens",
                "createTunedTextModel"
            ],
            "temperature": 0.7,
            "topP": 0.95,
            "topK": 40
            },
            {
            "name": "embedding-gecko-001",
            "version": "001",
            "displayName": "Embedding Gecko",
            "description": "Obtain a distributed representation of a text.",
            "inputTokenLimit": 1024,
            "outputTokenLimit": 1,
            "supportedGenerationMethods": [
                "embedText",
                "countTextTokens"
            ]
            },
            {
            "name": "embedding-gecko-002",
            "version": "002",
            "displayName": "Embedding Gecko 002",
            "description": "Obtain a distributed representation of a text.",
            "inputTokenLimit": 2048,
            "outputTokenLimit": 1,
            "supportedGenerationMethods": [
                "embedText",
                "countTextTokens"
            ]
            },
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
            {
            "name": "embedding-001",
            "version": "001",
            "displayName": "Embedding 001",
            "description": "Obtain a distributed representation of a text.",
            "inputTokenLimit": 2048,
            "outputTokenLimit": 1,
            "supportedGenerationMethods": [
                "embedContent",
                "countTextTokens"
            ]
            },
            {
            "name": "aqa",
            "version": "001",
            "displayName": "Model that performs Attributed Question Answering.",
            "description": "Model trained to return answers to questions that are grounded in provided sources, along with estimating answerable probability.",
            "inputTokenLimit": 7168,
            "outputTokenLimit": 1024,
            "supportedGenerationMethods": [
                "generateAnswer"
            ],
            "temperature": 0.2,
            "topP": 1,
            "topK": 40
            }
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
            "name": "chat-bison-001",
            "version": "001",
            "displayName": "Chat Bison",
            "description": "Chat-optimized generative language model.",
            "inputTokenLimit": 4096,
            "outputTokenLimit": 1024,
            "supportedGenerationMethods": [
                "generateMessage",
                "countMessageTokens"
            ],
            "temperature": 0.25,
            "topP": 0.95,
            "topK": 40
            },
            {
            "name": "text-bison-001",
            "version": "001",
            "displayName": "Text Bison",
            "description": "Model targeted for text generation.",
            "inputTokenLimit": 8196,
            "outputTokenLimit": 1024,
            "supportedGenerationMethods": [
                "generateText",
                "countTextTokens",
                "createTunedTextModel"
            ],
            "temperature": 0.7,
            "topP": 0.95,
            "topK": 40
            },
            {
            "name": "embedding-gecko-001",
            "version": "001",
            "displayName": "Embedding Gecko",
            "description": "Obtain a distributed representation of a text.",
            "inputTokenLimit": 1024,
            "outputTokenLimit": 1,
            "supportedGenerationMethods": [
                "embedText",
                "countTextTokens"
            ]
            },
            {
            "name": "embedding-gecko-002",
            "version": "002",
            "displayName": "Embedding Gecko 002",
            "description": "Obtain a distributed representation of a text.",
            "inputTokenLimit": 2048,
            "outputTokenLimit": 1,
            "supportedGenerationMethods": [
                "embedText",
                "countTextTokens"
            ]
            },
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
            {
            "name": "embedding-001",
            "version": "001",
            "displayName": "Embedding 001",
            "description": "Obtain a distributed representation of a text.",
            "inputTokenLimit": 2048,
            "outputTokenLimit": 1,
            "supportedGenerationMethods": [
                "embedContent",
                "countTextTokens"
            ]
            },
            {
            "name": "aqa",
            "version": "001",
            "displayName": "Model that performs Attributed Question Answering.",
            "description": "Model trained to return answers to questions that are grounded in provided sources, along with estimating answerable probability.",
            "inputTokenLimit": 7168,
            "outputTokenLimit": 1024,
            "supportedGenerationMethods": [
                "generateAnswer"
            ],
            "temperature": 0.2,
            "topP": 1,
            "topK": 40
            }
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
