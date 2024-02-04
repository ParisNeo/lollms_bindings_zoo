######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying 
# engine author : LollmsRN 
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
import re
import json
import requests
from datetime import datetime
from typing import List, Union
from lollms.utilities import PackageManager, encode_image, trace_exception

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2024, "
__license__ = "Apache 2.0"

binding_name = "LollmsRN"
binding_folder_name = ""
elf_completion_formats={
    "instruct":"/api",
}

def get_binding_cfg(lollms_paths:LollmsPaths, binding_name):
    cfg_file_path = lollms_paths.personal_configuration_path/"bindings"/f"{binding_name}"/"config.yaml"
    return LOLLMSConfig(cfg_file_path,lollms_paths)

def get_model_info(url, authorization_key):
    url = f'{url}/list_models'
    headers = {
                'accept': 'application/json',
                'Authorization': f'Bearer {authorization_key}'
            }
    
    response = requests.get(url, headers=headers)
    return response.json()

class LollmsRN(LLMBinding):
    
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
                {"name":"address","type":"str","value":"http://127.0.0.1:9601","help":"The server address"},
                {"name":"max_image_width","type":"int","value":1024, "help":"The maximum width of the image in pixels. If the mimage is bigger it gets shrunk before sent to lollms remote nodes model"},
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
        ASCIIColors.error("You need to install a lollms remote nodes server somewhere and run it locally or remotely.")
        self.InfoMessage("You need to install a lollms remote nodes server somewhere and run it locally or remotely.")
    
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
                "prompt": prompt,
                "model_name": self.config.model_name,
                "personality": -1,
                "n_predict": n_predict,
                "stream": True,
                "temperature": float(gpt_params["temperature"]),
                "top_k": 50,
                "top_p": 0.95,
                "repeat_penalty": 0.8,
                "repeat_last_n": 40,
                "seed": 0,
                "n_threads": 8

            }
            
            url = f'{self.binding_config.address}{elf_completion_formats[self.binding_config.completion_format]}/lollms_generate'

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
        url = f'{self.binding_config.address}/list_models'
        headers = {
                    'accept': 'application/json',
                    'Authorization': f'Bearer {self.binding_config.server_key}'
                }
        
        response = requests.get(url, headers=headers)
        return response.json()
                
    def get_available_models(self, app:LoLLMsCom=None):
       
        try:
            url = f'{self.binding_config.address}/get_available_models'
            headers = {
                        'accept': 'application/json',
                        'Authorization': f'Bearer {self.binding_config.server_key}'
                    }
            
            response = requests.get(url, headers=headers)
            return response.json()
        except Exception as ex:
            trace_exception()
            self.InfoMessage("Couldn't recover the list of models from the lollms server!\nMake sure the server is running and that you are connected.")
        return {"status":False}

if __name__=="__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    root_path = Path(__file__).parent
    lollms_paths = LollmsPaths.find_paths(tool_prefix="",force_local=True, custom_default_cfg_path="configs/config.yaml")
    config = LOLLMSConfig.autoload(lollms_paths)
    lollms_app = LollmsApplication("",config, lollms_paths, False, False,False, False)

    lrn = LollmsRN(config, lollms_paths,lollmsCom=lollms_app)
    lrn.install()
    lrn.binding_config.openai_key = input("Lollms Remote Nodes Key (If your server don't use keys, please leave it blank):")
    lrn.binding_config.save()
    config.binding_name= "remote_lollms"
    config.model_name=""
    config.save_config()