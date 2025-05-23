######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying 
# engine author : vLLM 
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.

# This binding is a wrapper to open ai's api

######
from pathlib import Path
from typing import Callable, Any
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig
from lollms.helpers import ASCIIColors, trace_exception
from lollms.types import MSG_OPERATION_TYPE
from lollms.com import LoLLMsCom
import subprocess
import yaml
import re
import json
import requests
from datetime import datetime
from typing import List, Union
import sys

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "Vllm"
binding_folder_name = ""
elf_completion_formats={
    "vllm instruct":"/v1/completions",
    "vllm chat":"/v1/chat/completions",
}

def get_binding_cfg(lollms_paths:LollmsPaths, binding_name):
    cfg_file_path = lollms_paths.personal_configuration_path/"bindings"/f"{binding_name}"/"config.yaml"
    return LOLLMSConfig(cfg_file_path,lollms_paths)

def get_model_info(url, completion_format, verify_ssl_certificate=True):
    try:
        url = f'{url}/v1/models'
        headers = {'accept': 'application/json'}
        response = requests.get(url, headers=headers, verify=verify_ssl_certificate)
        data = response.json()
        model_info = [{'model_name': "vllm_remote_model", 'owned_by': "remote server", 'created_datetime': "unknown"}]

        for model in data['data']:
            model_name = model['id']
            owned_by = model['owned_by']
            created_timestamp = model['created']
            created_datetime = datetime.utcfromtimestamp(created_timestamp).strftime('%Y-%m-%d %H:%M:%S')
            model_info.append({'model_name': model_name, 'owned_by': owned_by, 'created_datetime': created_datetime})
    except:
        return [{'model_name': "vllm_remote_model", 'owned_by': "remote server", 'created_datetime': "unknown"}]

        
    return model_info
class Vllm(LLMBinding):
    
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
                {"name":"address","type":"str","value":"http://127.0.0.1:5000","help":"The server address"},
                {"name":"verify_ssl_certificate","type":"bool","value":True,"help":"Deactivate if you don't want the client to verify the SSL certificate"},
                {"name":"completion_format","type":"str","value":"openai instruct","options":list(elf_completion_formats.keys()), "help":"The format supported by the server"},
                {"name":"ctx_size","type":"int","value":4090, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"max_n_predict","type":"int","value":4090, "min":512, "help":"The maximum amount of tokens to generate"},
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
                            SAFE_STORE_SUPPORTED_FILE_EXTENSIONS=[''],
                            lollmsCom=lollmsCom
                        )
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict
        if self.config.model_name is None:
            self.config.model_name = "vllm_remote_model"

    def settings_updated(self):
        if len(self.binding_config.address.strip())>0 and self.binding_config.address.strip().endswith("/"):
            self.binding_config.address = self.binding_config.address.strip()[:-1]
            self.binding_config.save()
        else:
            self.binding_config.address = self.binding_config.address.strip()
            self.binding_config.save()
                        
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict
        
    def build_model(self, model_name=None):
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict
        ASCIIColors.yellow(f"vllm selected model {self.config.model_name}")
        ASCIIColors.yellow(f"vllm custom model {self.binding_config.model_name}")
        super().build_model(model_name)
        return self

    def install(self):
        super().install()
        requirements_file = self.binding_dir / "requirements.txt"
        self.ShowBlockingMessage("Installing vLLM ...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])
            ASCIIColors.success("Installed successfully")
        except Exception as ex:
            ASCIIColors.error(ex)
        finally:
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
        if self.binding_config.server_key:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.binding_config.server_key}',
            }
        else:
            headers = {
                'Content-Type': 'application/json',
            }
        default_params = {
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.96,
            'repeat_penalty': 1.3
        }
        gpt_params = {**default_params, **gpt_params}
        if self.binding_config.completion_format=="vllm instruct":
            data = {
                'model':self.config.model_name if self.config.model_name!="vllm_remote_model" else self.binding_config.model_name,
                'prompt': prompt,
                "stream":True,
                "temperature": float(gpt_params["temperature"]),
                "max_tokens": n_predict
            }
        elif self.binding_config.completion_format=="vllm chat":
            data = {
                'model':self.config.model_name if self.config.model_name!="vllm_remote_model" else self.binding_config.model_name,
                'messages': [{
                    'role': "user",
                    'content': prompt
                }],
                "stream":True,
                "temperature": float(gpt_params["temperature"]),
                "max_tokens": n_predict
            }

        if self.binding_config.address.strip().endswith("/"):
            self.binding_config.address = self.binding_config.address.strip()[:-1]
        else:
            self.binding_config.address = self.binding_config.address.strip()
        url = f'{self.binding_config.address}{elf_completion_formats[self.binding_config.completion_format]}'

        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), stream=True, verify=self.binding_config.verify_ssl_certificate)

            if response.status_code==400:
                content = response.content.decode("utf8")
                content = json.loads(content)
                self.error(content["message"])
                return
            elif response.status_code==404:
                ASCIIColors.error(response.content.decode("utf-8", errors='ignore'))
            text = ""
            for line in response.iter_lines():
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    try:
                        json_data = json.loads(decoded[5:].strip())
                        if "chat" in self.binding_config.completion_format:
                            try:
                                chunk = json_data["choices"][0]["delta"]["content"]
                            except:
                                chunk = ""
                        else:
                            chunk = json_data["choices"][0]["text"]
                        ## Process the JSON data here
                        text +=chunk
                        if callback:
                            if not callback(chunk, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                                break
                    except:
                        break
                else:
                    if decoded.startswith("{"):
                        for line_ in response.iter_lines():
                            decoded += line_.decode("utf-8")
                        try:
                            json_data = json.loads(decoded)
                            if json_data["object"]=="error":
                                self.error(json_data["message"])
                                break
                        except:
                            self.error("Couldn't generate text, verify your key or model name")
                    else:
                        text +=decoded
                        if callback:
                            if not callback(decoded, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                                break
            return text
        except Exception as ex:
            trace_exception(ex)
            self.error("Couldn't connect to server.\nPlease verify your connection or that the server is up.")
    
    def list_models(self):
        """Lists the models for this binding
        """
        model_names = get_model_info(f'{self.binding_config.address}', self.binding_config.completion_format, self.binding_config.verify_ssl_certificate)
        entries=[]
        for model in model_names:
            entries.append(model["model_name"])
        return entries
                
    def get_available_models(self, app:LoLLMsCom=None):
        # Create the file path relative to the child class's directory
        model_names = get_model_info(f'{self.binding_config.address}', self.binding_config.completion_format, self.binding_config.verify_ssl_certificate)
        entries=[]
        for model in model_names:
            entry={
                "category": "generic",
                "datasets": "unknown",
                "icon": '/bindings/vLLM/logo.png',
                "last_commit_time": "2023-09-17 17:21:17+00:00",
                "license": "unknown",
                "model_creator": model["owned_by"],
                "model_creator_link": "https://lollms.com/",
                "name": model["model_name"],
                "provider": None,
                "rank": "1.0",
                "type": "api",
                "variants":[
                    {
                        "name":model,
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
if __name__=="__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    root_path = Path(__file__).parent
    lollms_paths = LollmsPaths.find_paths(tool_prefix="",force_local=True, custom_default_cfg_path="configs/config.yaml")
    config = LOLLMSConfig.autoload(lollms_paths)
    lollms_app = LollmsApplication("",config, lollms_paths, False, False,False, False)

    oai = Vllm(config, lollms_paths,lollmsCom=lollms_app)
    oai.install()
    oai.binding_config.save()
    config.binding_name= "vLLM"
    config.model_name=""
    config.save_config()
