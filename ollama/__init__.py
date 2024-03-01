######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying 
# engine author : Ollama 
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
from lollms.com import LoLLMsCom
import subprocess
import yaml
import sys
import json
import requests
from datetime import datetime
from typing import List, Union
from lollms.utilities import PackageManager, encode_image, trace_exception

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "Ollama"
binding_folder_name = ""
elf_completion_formats={
    "instruct":"/api",
}

def get_binding_cfg(lollms_paths:LollmsPaths, binding_name):
    cfg_file_path = lollms_paths.personal_configuration_path/"bindings"/f"{binding_name}"/"config.yaml"
    return LOLLMSConfig(cfg_file_path,lollms_paths)

def get_model_info(url, authorization_key):
    url = f'{url}/tags'
    headers = {
                'accept': 'application/json',
                'Authorization': f'Bearer {authorization_key}'
            }
    
    response = requests.get(url, headers=headers)
    data = response.json()
    model_info = []

    for model in data['models']:
        model_name = model['name']
        owned_by = ""
        created_datetime = model["modified_at"]
        model_info.append({'model_name': model_name, 'owned_by': owned_by, 'created_datetime': created_datetime})

    return model_info
class Ollama(LLMBinding):
    
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
                {"name":"address","type":"str","value":"http://127.0.0.1:11434","help":"The server address"},
                {"name":"max_image_width","type":"int","value":1024, "help":"The maximum width of the image in pixels. If the mimage is bigger it gets shrunk before sent to ollama model"},
                {"name":"completion_format","type":"str","value":"instruct","options":["instruct"], "help":"The format supported by the server"},
                {"name":"ctx_size","type":"int","value":4090, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"server_key","type":"str","value":"", "help":"The API key to connect to the server."},
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
                            supported_file_extensions=[''],
                            lollmsCom=lollmsCom
                        )
        self.config.ctx_size=self.binding_config.config.ctx_size
        host = self.binding_config.address.replace("http://","").split(":")[0]
        port = self.binding_config.address.replace("http://","").split(":")[1]
        if  host== self.config.host and port == self.config.port:
            self.binding_config.address = "http://"+host+":"+port+"0"
            self.binding_config.save()
            self.InfoMessage(f"I detected that you are using lollms remotes server with the same address and port number of the current server which will cause an infinite loop.\nTo prevent this I have changed the port number and now the server address is {self.binding_config.address}")

    def settings_updated(self):
        self.config.ctx_size = self.binding_config.config.ctx_size        
        
    def build_model(self):
        if self.config.model_name is None:
            return None
        
        if "llava" in self.config.model_name or "vision" in self.config.model_name:
            self.binding_type = BindingType.TEXT_IMAGE
        return self

    def install(self):
        super().install()
        ASCIIColors.success("Installed successfully")
        ASCIIColors.error("----------------------")
        ASCIIColors.error("Attention please")
        ASCIIColors.error("----------------------")
        ASCIIColors.error("You need to install an ollama server somewhere and run it locally or remotely.")
    
    def install_model(self, model_type:str, model_path:str, variant_name:str, client_id:int=None):
        url = f'{self.binding_config.address}/api/pull'
        headers = {
                    'accept': 'application/json',
                    'Authorization': f'Bearer {self.binding_config.server_key}'
                }
        
        payload = json.dumps({
            'name':variant_name,
            'stream':True
        })

        response = requests.post(url, headers=headers, data=payload, stream=True)
        for line in response.iter_lines():
            line = json.loads(line.decode("utf-8")) 
            if line["status"]=="pulling manifest":
                self.lollmsCom.info("Pulling")
            elif line["status"]=="downloading digestname" or line["status"].startswith("pulling") and "completed" in line.keys():
                self.lollmsCom.notify_model_install(model_path,variant_name,"", model_path,datetime.now().strftime("%Y-%m-%d %H:%M:%S"), line["total"], line["completed"], 100*line["completed"]/line["total"] if line["total"]>0 else 0,0,client_id=client_id)
        self.InfoMessage("Installed")


    def tokenize(self, text: Union[str, List[str]]) -> List[str]:
        """Tokenizes a text string

        Args:
            text (str): The text to tokenize

        Returns:
            A list of tokens
        """
        if isinstance(text, str):
            return text.split(" ")
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
        text = ""
        try:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.binding_config.server_key}',
            }
            default_params = {
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.96,
                'repeat_penalty': 1.3
            }
            gpt_params = {**default_params, **gpt_params}

            data = {
                'model':self.config.model_name,
                'prompt': prompt,
                "stream":True,
                "temperature": float(gpt_params["temperature"]),
                "max_tokens": n_predict
            }

            
            url = f'{self.binding_config.address}{elf_completion_formats[self.binding_config.completion_format]}/generate'

            response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
            for line in response.iter_lines(): 
                decoded = line.decode("utf-8")
                json_data = json.loads(decoded)
                chunk = json_data["response"]
                ## Process the JSON data here
                text +=chunk
                if callback:
                    if not callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                        break
        except Exception as ex:
            trace_exception(ex)
            self.error("Couldn't generate text")
        return text

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
        text = ""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.binding_config.server_key}',
        }
        default_params = {
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.96,
            'repeat_penalty': 1.3
        }
        gpt_params = {**default_params, **gpt_params}
        images_list = []
        for image in images:
            images_list.append(f"{encode_image(image, self.binding_config.max_image_width)}")

        data = {
            'model':self.config.model_name,
            'prompt': prompt,
            'images': images_list,
            "stream":True,
            "temperature": float(gpt_params["temperature"]),
            "max_tokens": n_predict
        }

        try:
            url = f'{self.binding_config.address}{elf_completion_formats[self.binding_config.completion_format]}/generate'

            response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)

            for line in response.iter_lines(): 
                decoded = line.decode("utf-8")
                json_data = json.loads(decoded)
                chunk = json_data["response"]
                ## Process the JSON data here
                text +=chunk
                if callback:
                    if not callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                        break
        except Exception as ex:
            trace_exception(ex)
            self.error(ex)
        return text        
    

    def list_models(self):
        """Lists the models for this binding
        """
        model_names = get_model_info(f'{self.binding_config.address}/api', self.binding_config.server_key)
        entries=[]
        for model in model_names:
            entries.append(model["model_name"])
        return entries
                
    def get_available_models(self, app:LoLLMsCom=None):

        #/pull
        # Create the file path relative to the child class's directory
        #model_names = get_model_info(f'{self.binding_config.address}/api', self.binding_config.server_key)
        model_names=[
            {"model_name":"llama2:latest", "owned_by": "meta"},
            {"model_name":"codellama:latest", "owned_by": "meta"},
            {"model_name":"codellama:70b", "owned_by": "meta"},
            {"model_name":"medllama2", "owned_by": "meta"},
            {"model_name":"falcon:7b", "owned_by": "TII"},
            {"model_name":"falcon:40b", "owned_by": "TII"},
            {"model_name":"falcon:180b", "owned_by": "TII"},
            {"model_name":"nous-hermes", "owned_by": "Nous-research"},
            {"model_name":"openhermes", "owned_by": "Nous-research"},
            {"model_name":"wizardcoder", "owned_by": "WizardLm"},
            {"model_name":"deepseek-coder", "owned_by": "DeepSeek"},
            {"model_name":"neural-chat", "owned_by": "Intel"},
            {"model_name":"bakllava", "owned_by": "mistral based"},
           
            
            
            {"model_name":"llava:latest", "owned_by": "liuhaotian"},
            {"model_name":"mistral:latest", "owned_by": "mistral.ai"},
            {"model_name":"mixtral:latest", "owned_by": "mistral.ai"},
            {"model_name":"neural-chat:latest", "owned_by": "intel"},
            {"model_name":"dolphin-mixtral:latest", "owned_by": "microsoft"},
            {"model_name":"mistral-openorca:latest", "owned_by": "microsoft"},
        ]
        entries=[]
        for model in model_names:
            entry={
                "category": "generic",
                "datasets": "unknown",
                "icon": '/bindings/Ollama/logo.png',
                "last_commit_time": "2023-09-17 17:21:17+00:00",
                "license": "unknown",
                "model_creator": model["owned_by"],
                "model_creator_link": "https://lollms.com/",
                "name": model["model_name"],
                "quantizer": None,
                "rank": "1.0",
                "type": "api",
                "variants":[
                    {
                        "name":model["model_name"],
                        "size":0
                    }
                ]
            }
            entries.append(entry)
        """
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        """
        
        return entries
