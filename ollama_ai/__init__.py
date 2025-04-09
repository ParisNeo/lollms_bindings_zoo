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
from typing import Callable, Any
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors
from lollms.types import MSG_OPERATION_TYPE
from lollms.com import LoLLMsCom
import subprocess
import yaml
import sys
import json
import requests
from datetime import datetime
from typing import List, Union
from lollms.utilities import PackageManager, encode_image, trace_exception, show_yes_no_dialog
import pipmaster as pm
if not pm.is_installed("ollama"):
    pm.install("ollama")
import ollama

if not pm.is_installed("tiktoken"):
    pm.install("tiktoken")
import tiktoken

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

def get_model_info(url, authorization_key, verify_ssl_certificate=True):

    url = f'{url}/tags'
    headers = {
                'accept': 'application/json',
                'Authorization': f'Bearer {authorization_key}'
            }
    
    response = requests.get(url, headers=headers, verify= verify_ssl_certificate)
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
                {"name":"verify_ssl_certificate","type":"bool","value":True,"help":"Deactivate if you don't want the client to verify the SSL certificate"},
                {"name":"max_image_width","type":"int","value":1024, "help":"The maximum width of the image in pixels. If the mimage is bigger it gets shrunk before sent to ollama model"},
                {"name":"completion_format","type":"str","value":"instruct","options":["instruct"], "help":"The format supported by the server"},
                {"name":"ctx_size","type":"int","value":4090, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"max_n_predict","type":"int","value":4090, "min":512, "help":"The maximum amount of tokens to generate"},
                {"name":"server_key","type":"str","value":"", "help":"The API key to connect to the server."},
                {"name":"timeout","type":"int","value":-1, "help":"the timeout value in ms (-1 for no timeout)."},
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
        self.config.max_n_predict=self.binding_config.max_n_predict
        host = self.binding_config.address.replace("http://","").split(":")[0]
        port = self.binding_config.address.replace("http://","").split(":")[1]
                
        if  host== self.config.host and port == self.config.port:
            self.binding_config.address = "http://"+host+":"+port+"0"
            self.binding_config.save()
            self.InfoMessage(f"I detected that you are using lollms remotes server with the same address and port number of the current server which will cause an infinite loop.\nTo prevent this I have changed the port number and now the server address is {self.binding_config.address}")

    def settings_updated(self):
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict
        if self.binding_config.address.strip().endswith("/") :
            self.binding_config.address = self.binding_config.address.strip()[:-1]
            self.binding_config.save()
        else:
            self.binding_config.address = self.binding_config.address.strip()
            self.binding_config.save() 
        self.build_model()
                   
    def build_model(self, model_name=None):
        super().build_model(model_name)
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict
        if self.binding_config.address.strip().endswith("/") :
            self.binding_config.address = self.binding_config.address.strip()[:-1]
            self.binding_config.save()
        else:
            self.binding_config.address = self.binding_config.address.strip()
            self.binding_config.save()                    
        if self.config.model_name is None:
            return None
        
        if "pixtral" in self.config.model_name or  "llava" in self.config.model_name or "vision" in self.config.model_name:
            self.binding_type = BindingType.TEXT_IMAGE
            
        headers = {
            'Content-Type': 'application/json',
        }
        if self.binding_config.server_key:
            headers['Authorization'] = f'Bearer {self.binding_config.server_key}'
        self.client = ollama.Client(self.binding_config.address, headers=headers)
            
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

        response = requests.post(url, headers=headers, data=payload, stream=True, verify= self.binding_config.verify_ssl_certificate)
        if response.status_code==200:
            for line in response.iter_lines():
                line = json.loads(line.decode("utf-8"))
                if "status" in line:
                    if line["status"]=="pulling manifest":
                        self.lollmsCom.info("Pulling")
                    elif line["status"]=="downloading digestname" or line["status"].startswith("pulling") and "completed" in line.keys():
                        self.lollmsCom.notify_model_install(model_path,variant_name,"", model_path,datetime.now().strftime("%Y-%m-%d %H:%M:%S"), line["total"], line["completed"], 100*line["completed"]/line["total"] if line["total"]>0 else 0,0,client_id=client_id)
                else:
                    self.InfoMessage(line["error"])
                    return
                    
            self.InfoMessage("Installed")
        else:
            try:
                self.InfoMessage(json.loads(response.text)["error"])
            except:
                self.InfoMessage(f"Couldn't generate because of error: {response.status_code}")
            


    def tokenize(self, text: Union[str, List[str]]) -> List[str]:
        """Tokenizes a text string

        Args:
            text (str): The text to tokenize

        Returns:
            A list of tokens
        """
        
        return tiktoken.model.encoding_for_model("gpt-4-turbo-preview").encode(text)

    def detokenize(self, tokens: List[str]) -> str:
        """Detokenizes a list of tokens

        Args:
            tokens (List[str]): The tokens to detokenize

        Returns:
            A string
        """
        return tiktoken.model.encoding_for_model("gpt-4-turbo-preview").decode(tokens)



    def embed(self, text, model="mxbai-embed-large"):
        """
        Computes text embedding
        Args:
            text (str): The text to be embedded.
            model (str, optional): The model to use for embedding. Defaults to "mxbai-embed-large".
        Returns:
            List[float]: The text embedding as a list of float values.
        """
        return self.client.embed(model, text)


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
        text = ""
        try:
            default_params = {
                'temperature': 0.1,
                'top_k': 50,
                'top_p': 0.96,
                'repeat_penalty': 1.3,
                'repeat_last_n': 1.3,
                "num_ctx":self.binding_config.ctx_size,
                "num_predict": n_predict
            }
            gpt_params = {**default_params, **gpt_params}
            for chunk in self.client.chat(model=self.config.model_name, messages=[
                {'role': 'user', 'content': prompt}
            ], stream=True, options = gpt_params):
                text +=chunk['message']['content']
                if callback:
                    if not callback(chunk['message']['content'], MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
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
        try:
            headers = {
                'Content-Type': 'application/json',
                #'Authorization': f'Bearer {self.binding_config.server_key}',
            }

            default_params = {
                'temperature': 0.1,
                'top_k': 50,
                'top_p': 0.96,
                'repeat_penalty': 1.3,
                'repeat_last_n': 1.3,
                "num_predict": n_predict
            }
            gpt_params = {**default_params, **gpt_params}
            for chunk in self.client.chat(model=self.config.model_name, messages=[
                {'role': 'user', 'content': prompt, 'images':images}
            ], stream=True, options = gpt_params):
                text +=chunk['message']['content']
                if callback:
                    if not callback(chunk['message']['content'], MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                        break
        except Exception as ex:
            trace_exception(ex)
            self.error("Couldn't generate text")
        return text    
    

    def list_models(self):
        """Lists the models for this binding
        """
        model_names = get_model_info(f'{self.binding_config.address}/api', self.binding_config.server_key, self.binding_config.verify_ssl_certificate)
        entries=[]
        for model in model_names:
            entries.append(model["model_name"])
        return entries
                
    def get_available_models(self, app:LoLLMsCom=None):

        #/pull
        # Create the file path relative to the child class's directory
        #model_names = get_model_info(f'{self.binding_config.address}/api', self.binding_config.server_key, self.binding_config.verify_ssl_certificate)
        # Load the JSON file
        try:
            with open('zoos/models_zoo/ollama_models.json', 'r', encoding='utf-8') as f:
                model_names = json.load(f)
            print(f"[DEBUG] Successfully loaded {len(model_names)} models")
            print("[DEBUG] First few models:", model_names[:3])  # Print first 3 models as a sample
        except FileNotFoundError:
            print("[ERROR] ollama_models.json file not found")
            model_names = []
        except json.JSONDecodeError:
            print("[ERROR] Error decoding JSON file")
            model_names = []
 
        entries=[]
        for model in model_names:
            entry={
                "category": "generic",
                "datasets": "unknown",
                "icon": '/bindings/ollama_ai/logo.png',
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
    

if __name__=="__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    import platform
    import os
    import shutil
    import urllib.request
    import subprocess
    from pathlib import Path
    root_path = Path(__file__).parent
    lollms_paths = LollmsPaths.find_paths(tool_prefix="",force_local=True, custom_default_cfg_path="configs/config.yaml")
    config = LOLLMSConfig.autoload(lollms_paths)
    lollms_app = LollmsApplication("",config, lollms_paths, False, False,False, False)

    oai = Ollama(config, lollms_paths,lollmsCom=lollms_app)
    oai.install()

    if show_yes_no_dialog("Info","Now it is time to download and install ollama on your system.\nOllama is a separate tool that servs a variaty of llms and lollms can use it as one of its bindings.\nIf you already have it installed, you can press No.\nYou can install it manually from their webite ollama.com.\nPress yes If you want to install it automatically now.\n"):
        system = platform.system()
        download_folder = Path.home() / "Downloads"
        ASCIIColors.yellow("Downloading installer, please wait ...")
        if system == "Windows":
            url = "https://ollama.com/download/OllamaSetup.exe"
            filename = "OllamaSetup.exe"
            urllib.request.urlretrieve(url, download_folder / filename)
            install_process = subprocess.Popen([str(download_folder / filename)])
            install_process.wait()

        elif system == "Linux":
            url = "https://ollama.com/install.sh"
            filename = "install.sh"
            urllib.request.urlretrieve(url, download_folder / filename)
            os.chmod(download_folder / filename, 0o755)
            install_process = subprocess.Popen([str(download_folder / filename)])
            install_process.wait()

        elif system == "Darwin":
            url = "https://ollama.com/download/Ollama-darwin.zip"
            filename = "Ollama-darwin.zip"
            urllib.request.urlretrieve(url, download_folder / filename)
            shutil.unpack_archive(download_folder / filename, extract_dir=download_folder)
            install_process = subprocess.Popen([str(download_folder / "Ollama-darwin" / "install.sh")])
            install_process.wait()

        else:
            print("Unsupported operating system.")


    oai.binding_config.save()
    config.binding_name= "ollama"

    config.model_name="mistral:latest"
    config.save_config()

