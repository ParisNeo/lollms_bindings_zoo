######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying 
# engine author : Novita AI
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.

# This binding is a wrapper to Novita AI's api

######
from pathlib import Path
from typing import Callable, Any
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors, trace_exception
from lollms.types import MSG_OPERATION_TYPE
from lollms.utilities import PackageManager, encode_image
from lollms.databases.models_database import ModelsDB

from lollms.com import LoLLMsCom
import subprocess
import yaml
import sys
from PIL import Image
import io
import os
import pipmaster as pm
if not pm.is_installed("openai"):
    pm.install("openai")
if not pm.is_installed("tiktoken"):
    pm.install("tiktoken")

import openai
import tiktoken

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "NovitaAI"
binding_folder_name = ""


  
class NovitaAI(LLMBinding):
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
                {"name":"generation_mode","type":"str", "value":"chat","options":["chat","instruct"],"help":"Select the generation mode. Make sure the model you are using supports the generation mode you have. Instruct models allow more control. Chat models are more likely to stick to the communication but can not be forced to start talking in a certain way easily."},
                {"name":"turn_on_cost_estimation","type":"bool", "value":False,"help":"Turns on measuring the cost of queries"},
                {"name":"total_input_tokens","type":"float", "value":0,"help":"The total number of input tokens in $"},
                {"name":"total_output_tokens","type":"float", "value":0,"help":"The total number of output tokens in $"},
                {"name":"total_input_cost","type":"float", "value":0,"help":"The total cost caused by input tokens in $"},
                {"name":"total_output_cost","type":"float", "value":0,"help":"The total cost caused by output tokens in $"},
                {"name":"total_cost","type":"float", "value":0,"help":"The total cost in $"},
                {"name":"novita_ai_key","type":"str","value":"","help":"A valid Novita AI key to generate text using Novita AI api"},
                {"name":"ctx_size","type":"int","value":4090, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"max_n_predict","type":"int","value":4090, "min":512, "help":"The maximum amount of tokens to generate"},
                {"name":"seed","type":"int","value":-1,"help":"Random numbers generation seed allows you to fix the generation making it dterministic. This is useful for repeatability. To make the generation random, please set seed to -1."},
                {"name":"max_image_width","type":"int","value":-1,"help":"resize the images if they have a width bigger than this (reduces cost). -1 for no change"},

            ]),
            BaseConfig(config={
                "novita_ai_key": "",     # use avx2
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
        self.config.max_n_predict=self.binding_config.max_n_predict
        
    def settings_updated(self):
        # The local key overrides the environment variable key
        if self.binding_config.config["novita_ai_key"] =="":
            # If there is no key, try find one in the environment
            api_key = os.getenv('NOVITA_AI_API_KEY')
            if not api_key:
                self.error("No API key is set!\nPlease set up your API key in the binding configuration")
        else:
            # If there is a key in the configuration, then use it
            api_key = self.binding_config.config["novita_ai_key"]

        self.novita_ai = openai.OpenAI(base_url="https://api.novita.ai/v3/openai",api_key=api_key)

        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict

    def build_model(self, model_name=None):
        super().build_model(model_name)
        # The local key overrides the environment variable key
        if self.binding_config.config["novita_ai_key"] =="":
            # If there is no key, try find one in the environment
            api_key = os.getenv('NOVITA_AI_API_KEY')
            if not api_key:
                self.error("No API key is set!\nPlease set up your API key in the binding configuration")
        else:
            # If there is a key in the configuration, then use it
            api_key = self.binding_config.config["novita_ai_key"]

        self.novita_ai = openai.OpenAI(base_url="https://api.novita.ai/v3/openai",api_key=api_key)

        if self.config.model_name is not None:
            if "vision" in self.config.model_name or "4o" in self.config.model_name:
                self.binding_type = BindingType.TEXT_IMAGE

        # Do your initialization stuff
        return self

    def install(self):
        super().install()
        # install requirements
        self.ShowBlockingMessage("Installing Novita AI api ...")
        try:
            self.HideBlockingMessage()
            ASCIIColors.success("Installed successfully")
            ASCIIColors.error("----------------------")
            ASCIIColors.error("Attention please")
            ASCIIColors.error("----------------------")
            ASCIIColors.error("The chatgpt/gpt4 binding uses the novita_ai API which is a paid service. Please create an account on the openAi website (https://platform.novita_ai.com/) then generate a key and provide it in the configuration of the binding.")
        except:
            self.warning("The chatgpt/gpt4 binding uses the novita_ai API which is a paid service.\nPlease create an account on the openAi website (https://platform.novita_ai.com/) then generate a key and provide it in the configuration of the binding.",20)
            self.HideBlockingMessage()

    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        try:
            tokens_list = tiktoken.model.encoding_for_model(self.config["model_name"]).encode(prompt)
        except:
            tokens_list = tiktoken.model.encoding_for_model("gpt-4-turbo-preview").encode(prompt)

        return tokens_list

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        try:
            text = tiktoken.model.encoding_for_model(self.config["model_name"]).decode(tokens_list)
        except:
            text = tiktoken.model.encoding_for_model("gpt-4-turbo-preview").decode(tokens_list)

        return text

    def embed(self, text):
        """
        Computes text embedding
        Args:
            text (str): The text to be embedded.
        Returns:
            List[float]
        """
        
        pass

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
        if self.binding_config.config.turn_on_cost_estimation:
            self.binding_config.config["total_input_tokens"] +=  len(self.tokenize(prompt))          
            self.binding_config.config["total_input_cost"] =  (self.binding_config.config["total_input_tokens"]/1000) * self.input_costs_by_model.get(self.config["model_name"], 0)
        if not ("vision" in self.config.model_name or "o" in self.config.model_name):
            self.error("You can not call a generate with vision on this model")
            return
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


            messages = [
                        {
                            "role": "user", 
                            "content": [
                                {
                                    "type":"text",
                                    "text":prompt
                                }
                            ]+[
                                {
                                    "type": "image_url",
                                    "image_url": {
                                    "url": f"data:image/jpeg;base64,{encode_image(image_path, self.binding_config.max_image_width)}"
                                    }                                    
                                }
                                for image_path in images
                            ]
                        }
                    ]
            chat_completion = self.novita_ai.chat.completions.create(
                            model=self.config["model_name"],  # Choose the engine according to your OpenAI plan
                            messages=messages,
                            max_tokens=n_predict,  # Adjust the desired length of the generated response
                            n=1,  # Specify the number of responses you want
                            temperature=gpt_params["temperature"],  # Adjust the temperature for more or less randomness in the output
                            stream=True
                            )
            
            for resp in chat_completion:
                if count >= n_predict:
                    break
                try:
                    word = resp.choices[0].delta.content
                except Exception as ex:
                    word = ""
                if callback is not None:
                    if not callback(word, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                        break
                if word:
                    output += word
                    count += 1

            if self.binding_config.config.turn_on_cost_estimation:
                self.binding_config.config["total_output_tokens"] +=  len(self.tokenize(output))          
                self.binding_config.config["total_output_cost"] =  ((self.binding_config.config["total_output_tokens"])/1000) * self.output_costs_by_model.get(self.config["model_name"],0)    
                self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]
        except Exception as ex:
            self.error(f'Error {ex}')
            trace_exception(ex)
        if self.binding_config.config.turn_on_cost_estimation:
            self.info(f'Total consumption since last reset: {self.binding_config.config["total_output_cost"]}$')
            self.binding_config.save()
        return output    


    def generate(self, 
                 prompt:str,                  
                 n_predict: int = 128,
                 callback: Callable[[str], None] = None,
                 verbose: bool = False,
                 **gpt_params ):

        """Generates text out of a prompt

        Args:
            prompt (str): The prompt to use for generation
            n_predict (int, optional): Number of tokens to prodict. Defaults to 128.
            callback (Callable[[str], None], optional): A callback function that is called everytime a new text element is generated. Defaults to None.
            verbose (bool, optional): If true, the code will spit many informations about the generation process. Defaults to False.
        """
        if self.binding_config.config.turn_on_cost_estimation:
            self.binding_config.config["total_input_tokens"] +=  len(self.tokenize(prompt))          
            self.binding_config.config["total_input_cost"] =  (self.binding_config.config["total_input_tokens"]/1000000) * (self.input_costs_by_model[self.config["model_name"]] if self.config["model_name"] in self.input_costs_by_model else 0)
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
            if "vision" in self.config.model_name:
                messages = [
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type":"text",
                                        "text":prompt
                                    }
                                ]
                            }
                        ]
            else:
                messages = [{"role": "user", "content": prompt}]
                

            if self.binding_config.generation_mode=="chat":
                if "o1" in self.model_name or "o3" in self.model_name:
                    chat_completion = self.novita_ai.chat.completions.create(
                                model=self.config["model_name"],  # Choose the engine according to your OpenAI plan
                                messages=messages,
                                n=1,  # Specify the number of responses you want
                                )
                    output = chat_completion.choices[0].message.content
                    callback(output, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK)
                else:
                    chat_completion = self.novita_ai.chat.completions.create(
                                    model=self.config["model_name"],  # Choose the engine according to your OpenAI plan
                                    messages=messages,
                                    max_tokens=n_predict-7 if n_predict>512 else n_predict,  # Adjust the desired length of the generated response
                                    n=1,  # Specify the number of responses you want
                                    temperature=float(gpt_params["temperature"]),  # Adjust the temperature for more or less randomness in the output
                                    stream=True)
                
                    for resp in chat_completion:
                        if count >= n_predict:
                            break
                        try:
                            word = resp.choices[0].delta.content
                        except Exception as ex:
                            word = ""
                        if callback is not None:
                            if not callback(word, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                                break
                        if word:
                            output += word
                            count += 1
            else:
                completion = self.novita_ai.completions.create(
                                model=self.config["model_name"],  # Choose the engine according to your OpenAI plan
                                prompt=prompt,
                                max_tokens=n_predict-7 if n_predict>512 else n_predict,  # Adjust the desired length of the generated response
                                n=1,  # Specify the number of responses you want
                                temperature=float(gpt_params["temperature"]),  # Adjust the temperature for more or less randomness in the output
                                stream=True)
                
                for resp in completion:
                    if count >= n_predict:
                        break
                    try:
                        word = resp.choices[0].text
                    except Exception as ex:
                        word = ""
                    if callback is not None:
                        if not callback(word, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                            break
                    if word:
                        output += word
                        count += 1





        except Exception as ex:
            self.error(f'Error {ex}$')
            trace_exception(ex)
        if self.binding_config.config.turn_on_cost_estimation:
            self.binding_config.config["total_output_tokens"] +=  len(self.tokenize(output))          
            self.binding_config.config["total_output_cost"] =  (self.binding_config.config["total_output_tokens"]/1000000) * self.output_costs_by_model[self.config["model_name"]] if self.config["model_name"] in self.output_costs_by_model else 0    
            self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]
            self.info(f'Total consumption since last reset: {self.binding_config.config["total_output_cost"]}$')
            self.binding_config.save()
        return output

    def list_models(self):
        """Lists the models for this binding
        """
        import requests

        # API endpoint
        url = "https://api.novita.ai/v3/openai/models"

        # Fetch data from the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON data
            data = response.json()

            # Extract model names
            model_names = [model['title'] for model in data['data']]

            # Print the list of model names
            return model_names
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            return []
                
                
    def get_available_models(self, app:LoLLMsCom=None):
        import requests

        # API endpoint
        url = "https://api.novita.ai/v3/openai/models"

        # Fetch data from the API
        response = requests.get(url)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON data
            data = response.json()

            # Initialize an empty list to store the formatted model dictionaries
            model_list = []

            # Iterate through each model in the API response
            for model in data['data']:
                # Create a dictionary for each model in the desired format
                model_dict = {
                    "category": "generic",
                    "datasets": "unknown",
                    "input_token_price_per_m": model.get("input_token_price_per_m", 0),
                    "output_token_price_per_m": model.get("output_token_price_per_m", 0),
                    "icon": "/bindings/novita_ai/logo.png",  # Default icon
                    "last_commit_time": model.get("created", 0),  # Use 'created' as last_commit_time
                    "license": "commercial",  # Default license
                    "model_creator": model.get("owned_by", "unknown"),  # Use 'owned_by' as model_creator
                    "model_creator_link": "https://novita.ai",  # Default link
                    "name": model.get("display_name", "unknown"),  # Use 'display_name' as name
                    "quantizer": None,  # Default quantizer
                    "rank": 1.0,  # Default rank
                    "type": "api",  # Default type
                    "variants": [
                        {
                            "name": model.get("title", "unknown"),  # Use 'display_name' as variant name
                            "size": 999999999999  # Default size
                        }
                    ]
                }
                # Append the model dictionary to the list
                model_list.append(model_dict)
            self.model_list = model_list
            self.input_costs_by_model = {model["name"]: model["input_token_price_per_m"] for model in model_list}
            self.output_costs_by_model = {model["name"]: model["output_token_price_per_m"] for model in model_list}

            # Print the list of model dictionaries
            return model_list
        else:
            print(f"Failed to fetch data. Status code: {response.status_code}")
            self.model_list = []
            return []
    

if __name__=="__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    root_path = Path(__file__).parent
    lollms_paths = LollmsPaths.find_paths(tool_prefix="",force_local=True, custom_default_cfg_path="configs/config.yaml")
    config = LOLLMSConfig.autoload(lollms_paths)
    lollms_app = LollmsApplication("",config, lollms_paths, False, False,False, False)

    oai = NovitaAI(config, lollms_paths,lollmsCom=lollms_app)
    oai.install()
    oai.binding_config.novita_ai_key = input("Novita AI Key:")
    oai.binding_config.save()
    config.binding_name= "open_ai"
    config.model_name="gpt-3.5-turbo"
    config.save_config()
