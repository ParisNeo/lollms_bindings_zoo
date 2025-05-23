######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying 
# engine author : Open AI
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
from lollms.helpers import ASCIIColors, trace_exception
from lollms.types import MSG_OPERATION_TYPE
from lollms.utilities import PackageManager, encode_image, get_media_type
from lollms.com import LoLLMsCom
import subprocess
import yaml
import sys
import os
import json
if not PackageManager.check_package_installed("PIL"):
    PackageManager.install_package("Pillow")
from PIL import Image
import requests

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "GroqLLM"
binding_folder_name = ""


  
class GroqLLM(LLMBinding):
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
                {"name":"turn_on_cost_estimation","type":"bool", "value":True,"help":"Turns on measuring the cost of queries"},
                {"name":"groq_key","type":"str","value":"","help":"A valid open AI key to generate text using groq api"},
                {"name":"mode","type":"str","value":"Chat", "options":["Chat","Instruct"],"help":"The format to be used. Completion anebles the use of some personas, but Chat is generally more stable is you don't want to use any forcing of the AI"},
                {"name":"total_cost","type":"float", "value":0,"help":"The total cost in $"},
                {"name":"ctx_size","type":"int","value":4090, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"max_n_predict","type":"int","value":4090, "min":512, "help":"The maximum amount of tokens to generate"},
                {"name":"seed","type":"int","value":-1,"help":"Random numbers generation seed allows you to fix the generation making it dterministic. This is useful for repeatability. To make the generation random, please set seed to -1."},
                {"name":"max_image_width","type":"int","value":-1,"help":"resize the images if they have a width bigger than this (reduces cost). -1 for no change"},
                {"name":"total_input_tokens","type":"float", "value":0,"help":"The total number of input tokens in $"},
                {"name":"total_output_tokens","type":"float", "value":0,"help":"The total number of output tokens in $"},
                {"name":"total_input_cost","type":"float", "value":0,"help":"The total cost caused by input tokens in $"},
                {"name":"total_output_cost","type":"float", "value":0,"help":"The total cost caused by output tokens in $"},

            ]),
            BaseConfig(config={
                "groq_key": "",     # use avx2
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
        models = self.get_available_models()

        self.input_costs_by_model={ }       
        self.output_costs_by_model={ }
        for model in models:
            self.input_costs_by_model[model["name"]]=model["variants"][0]["input_cost"]
            self.output_costs_by_model[model["name"]]=model["variants"][0]["output_cost"]

        
    def settings_updated(self):
        import groq
        if self.binding_config.config["groq_key"] =="":
            try:
                self.client = groq.Groq(
                    api_key=os.environ.get("GROQ_API_KEY"),
                )        
            except:
                self.error("No API key is set!\nPlease set up your API key in the binding configuration")
        else:
            self.client = groq.Groq(
                api_key=self.binding_config.config["groq_key"],
            )        

        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict

    def build_model(self, model_name=None):
        super().build_model(model_name)
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict
        import groq
        if self.binding_config.config["groq_key"] =="":
            try:
                self.client = groq.Groq(
                    api_key=os.environ.get("GROQ_API_KEY"),
                ) 
                self.binding_type = BindingType.TEXT_IMAGE
            except:
                self.error("No API key is set!\nPlease set up your API key in the binding configuration")
        else:
            self.client = groq.Groq(
                api_key=self.binding_config.config["groq_key"],
            )        
            self.binding_type = BindingType.TEXT_IMAGE
        # Do your initialization stuff
        return self

    def install(self):
        super().install()
        requirements_file = self.binding_dir / "requirements.txt"
        # install requirements
        self.ShowBlockingMessage("Installing groq api ...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])
            self.HideBlockingMessage()
            ASCIIColors.success("Installed successfully")
            ASCIIColors.error("----------------------")
            ASCIIColors.error("Attention please")
            ASCIIColors.error("----------------------")
            ASCIIColors.error("The groq binding uses the groq API which is a paid service. Please create an account on the groq website (https://groq.com/) then generate a key and provide it in the configuration of the binding.")
        except:
            self.warning("The groq binding uses the groq API which is a paid service.\nPlease create an account on the groq website (https://groq.com/) then generate a key and provide it in the configuration of the binding.",20)
            self.HideBlockingMessage()

    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        import tiktoken
        try:
            tokens_list = tiktoken.encoding_for_model(self.config["model_name"]).encode(prompt)
        except:
            tokens_list = tiktoken.encoding_for_model("gpt-4").encode(prompt)
            
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
        try:
            text = tiktoken.model.encoding_for_model(self.config["model_name"]).decode(tokens_list)
        except:
            text = tiktoken.model.encoding_for_model("gpt-4").decode(tokens_list)

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
        if self.binding_config.turn_on_cost_estimation:
            self.binding_config.config["total_input_tokens"] +=  len(self.tokenize(prompt))          
            self.binding_config.config["total_input_cost"] =  self.binding_config.config["total_input_tokens"] * self.input_costs_by_model[self.config["model_name"]] /1000
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
            with self.client.chat.completions.create(
                max_tokens=n_predict,
                messages=[{"role": "user", "content": prompt}],
                model=self.config.model_name, stream=True
            ) as stream:
                for word in stream:
                    if callback is not None:
                        if not callback(word.choices[0].delta.content, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                            break
                    if word.choices[0].delta.content:
                        output += word.choices[0].delta.content
                        count += 1


        except Exception as ex:
            self.error(f'Error {ex}$')
            trace_exception(ex)
        if self.binding_config.turn_on_cost_estimation:
            self.binding_config.config["total_output_tokens"] +=  len(self.tokenize(output))          
            self.binding_config.config["total_output_cost"] =  self.binding_config.config["total_output_tokens"] * self.output_costs_by_model[self.config["model_name"]]/1000    
            self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]
            self.info(f'Consumed {self.binding_config.config["total_output_cost"]}$')
            self.binding_config.save()
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
        
        self.binding_config.config["total_input_tokens"] +=  len(self.tokenize(prompt))          
        self.binding_config.config["total_input_cost"] =  self.binding_config.config["total_input_tokens"] * self.input_costs_by_model.get(self.config["model_name"],0.1) /1000
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
                            "content": 
                            [
                                {
                                    "type": "image",
                                    "source": {
                                        "type":"base64",
                                        "media_type": get_media_type(image_path),
                                        "data": encode_image(image_path, self.binding_config.max_image_width)
                                    }                                    
                                }
                                for image_path in images
                            ]+
                            [
                                {
                                    "type":"text",
                                    "text":prompt
                                }
                            ]
                        }
                    ]
            chat_completion = self.client.messages.create(
                            model=self.config["model_name"],  # Choose the engine according to your OpenAI plan
                            messages=messages,
                            max_tokens=n_predict,  # Adjust the desired length of the generated response
                            temperature=gpt_params["temperature"],  # Adjust the temperature for more or less randomness in the output
                            stream=True
                            )
            for resp in chat_completion:
                if count >= n_predict:
                    break
                try:
                    word = resp.delta.text
                except Exception as ex:
                    word = ""
                if callback is not None:
                    if not callback(word, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                        break
                if word:
                    output += word
                    count += 1


            self.binding_config.config["total_output_tokens"] +=  len(self.tokenize(output))          
            self.binding_config.config["total_output_cost"] =  self.binding_config.config["total_output_tokens"] * self.output_costs_by_model.get(self.config["model_name"],0.1)/1000    
            self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]
        except Exception as ex:
            self.error(f'Error {ex}')
            trace_exception(ex)
        self.info(f'Consumed {self.binding_config.config["total_output_cost"]}$')
        self.binding_config.save()
        return output    

    def list_models(self):
        """Lists the models for this binding
        """
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"
        

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        

        return [f["name"] for f in yaml_data]
                
                

    def list_models(self):
        """Lists the models for this binding
        """
        try:
            full_data = []
            if self.binding_config.config["groq_key"] =="":
                try:
                    api_key=os.environ.get("GROQ_API_KEY"),
                except:
                    self.error("No API key is set!\nPlease set up your API key in the binding configuration")
                    return []
            else:
                api_key=self.binding_config.config["groq_key"]
            url = "https://api.groq.com/openai/v1/models"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            response = requests.get(url, headers=headers)
            full_data = json.loads(response.content.decode("utf-8"))['data']

            return [f["id"] for f in full_data if not "whisper" in f["id"]]
        except Exception as ex:
            trace_exception(ex)
            return []            
                
    def get_available_models(self, app:LoLLMsCom=None):
        try:
            full_data = []
            if self.binding_config.config["groq_key"] =="":
                try:
                    api_key=os.environ.get("GROQ_API_KEY"),
                except:
                    self.error("No API key is set!\nPlease set up your API key in the binding configuration")
                    return []
            else:
                api_key=self.binding_config.config["groq_key"]
            url = "https://api.groq.com/openai/v1/models"

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            response = requests.get(url, headers=headers)
            full_data = json.loads(response.content.decode("utf-8"))['data']
            return [{
                    "category": "generic",
                    "datasets": "unknown",
                    "icon": "/bindings/groq_llm/llama.png" if "llama" in f["id"] else "/bindings/mistral_ai/logo.png" if "mistral" in f["id"] or "mixtral" in f["id"] else "/bindings/gemini/logo.png" if "gemma" in f["id"] else "unknown",
                    "last_commit_time": None,
                    "license": "llama",
                    "model_creator": "meta" if "llama" in f["id"] else "mistralai" if "mistral" in f["id"] or "mixtral" in f["id"] else "google" if "gemma" in f["id"] else "unknown",
                    "model_creator": "meta.com" if "llama" in f["id"] else "mistal.ai" if "mistral" in f["id"] or "mixtral" in f["id"] else "google.com" if "gemma" in f["id"] else "unknown",
                    "name": f["id"],
                    "provider": None,
                    "rank": 1.0,
                    "type": "api",
                    "variants": [
                        {
                        "name":f["id"],
                        "size":None,
                        "ctx_size":8192,
                        "input_cost": 0.0,
                        "output_cost": 0.0            

                        }
                    ],
                } for f in full_data if not "whisper" in f["id"]]
        except Exception as ex:
            binding_path = Path(__file__).parent
            file_path = binding_path/"models.yaml"

            with open(file_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
            
            return yaml_data

if __name__=="__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    root_path = Path(__file__).parent
    lollms_paths = LollmsPaths.find_paths(tool_prefix="",force_local=True, custom_default_cfg_path="configs/config.yaml")
    config = LOLLMSConfig.autoload(lollms_paths)
    lollms_app = LollmsApplication("",config, lollms_paths, False, False,False, False)

    oai = GroqLLM(config, lollms_paths,lollmsCom=lollms_app)
    oai.install()
    oai.binding_config.save()
    config.binding_name= "groq"
    config.model_name="claude-3-opus-20240229"
    config.save_config()
