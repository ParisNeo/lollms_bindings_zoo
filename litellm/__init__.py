######
# Project       : LiteLLM
# File          : binding.py
# Author        : g1ibby
# Underlying 
# engine author : LiteLLM
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.

# This binding is a wrapper to LiteLLM's api

######
from pathlib import Path
import requests
from typing import Callable, Any
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors, trace_exception
from lollms.types import MSG_OPERATION_TYPE
import subprocess
import base64
import sys
import json 

__author__ = "g1ibby"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2024"
__license__ = "Apache 2.0"

binding_name = "LiteLLM"
binding_folder_name = ""

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_icon_path(model_name):
    model_name = model_name.lower()
    if 'gpt-3' in model_name:
        return '/bindings/open_ai/logo.png'
    elif 'gpt-4' in model_name:
        return '/bindings/open_ai/logo2.png'
    elif 'openai' in model_name:
        return '/bindings/open_ai/logo.png'
    elif 'mistral' in model_name or 'mixtral' in model_name:
        return '/bindings/mistral_ai/logo.png'
    else:
        return '/bindings/litellm/logo.png'

def get_model_info(url, authorization_key, verify_ssl_certificate=True):
    url = f'{url}/model/info'
    headers = {
        'accept': 'application/json',
        'Authorization': f'Bearer {authorization_key}'
    }

    response = requests.get(url, headers=headers, verify=verify_ssl_certificate)
    data = response.json()
    model_info_list = []

    for model in data['data']:
        model_name = model['model_name']
        model_info = model.get('model_info', {})

        # Extracting the required fields, setting default to 0 if not found
        input_cost_per_token = model_info.get('input_cost_per_token', 0)
        output_cost_per_token = model_info.get('output_cost_per_token', 0)
        max_tokens = model_info.get('max_tokens', 0)
        max_input_tokens = model_info.get('max_input_tokens', 0)
        max_output_tokens = model_info.get('max_output_tokens', 0)

        model_details = {
            'model_name': model_name,
            'input_cost_per_token': input_cost_per_token,
            'output_cost_per_token': output_cost_per_token,
            'max_tokens': max_tokens,
            'max_input_tokens': max_input_tokens,
            'max_output_tokens': max_output_tokens
        }

        model_info_list.append(model_details)

    return model_info_list

class LiteLLM(LLMBinding):
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
        self.input_costs_by_model={}
        self.output_costs_by_model={}
        if lollms_paths is None:
            lollms_paths = LollmsPaths()

        binding_config = TypedConfig(
            ConfigTemplate([
                {"name":"address","type":"str","value":"http://localhost:8000","help":"The server address"},
                {"name":"verify_ssl_certificate","type":"bool","value":True,"help":"Deactivate if you don't want the client to verify the SSL certificate"},
                {"name":"server_key","type":"str","value":"anything","help":"The server key"},
                {"name":"total_input_tokens","type":"float", "value":0,"help":"The total number of input tokens in $"},
                {"name":"total_output_tokens","type":"float", "value":0,"help":"The total number of output tokens in $"},
                {"name":"total_input_cost","type":"float", "value":0,"help":"The total cost caused by input tokens in $"},
                {"name":"total_output_cost","type":"float", "value":0,"help":"The total cost caused by output tokens in $"},
                {"name":"total_cost","type":"float", "value":0,"help":"The total cost in $"},
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
            supported_file_extensions=[''],
            lollmsCom=lollmsCom
        )
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict

        address = self.binding_config.config['address']
        server_key = self.binding_config.config['server_key']

        # Fetch model info using get_model_info
        try:
            model_info = get_model_info(address, server_key, verify_ssl_certificate=self.binding_config.verify_ssl_certificate)
        except Exception as ex:
            model_info = []
            self.InfoMessage("Couldn't connect to the server. Please make sure that the server is running with the specified parameters or change the parameters in the settings.")
        # Initialize and populate cost dictionaries
        self.input_costs_by_model = {}
        self.output_costs_by_model = {}
        for model in model_info:
            model_name = model['model_name']
            self.input_costs_by_model[model_name] = model.get('input_cost_per_token', 0)
            self.output_costs_by_model[model_name] = model.get('output_cost_per_token', 0)

    def settings_updated(self):
        if len(self.binding_config.address.strip())>0 and self.binding_config.address.strip().endswith("/"):
            self.binding_config.address = self.binding_config.address.strip()[:-1]
            self.binding_config.save()
            
        self.config.ctx_size = self.binding_config.config.ctx_size

    def build_model(self, model_name=None):
        super().build_model(model_name)
        from openai import OpenAI
        if self.binding_config.address == "":
            self.error("No API url is set!\nPlease set up your API url in the binding configuration")
            raise Exception("No API url is set!\nPlease set up your API url in the binding configuration")
        self.openai = OpenAI(
            base_url=self.binding_config.address,
            api_key=self.binding_config.server_key,
        )

        if self.config.model_name is None:
            return None
        if "llava" in self.config.model_name or "vision" in self.config.model_name:
            self.binding_type = BindingType.TEXT_IMAGE

        return self

    def install(self):
        super().install()
        requirements_file = self.binding_dir / "requirements.txt"
        # install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])
        ASCIIColors.success("Installed successfully")

    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.
        Args:
            prompt (str): The input prompt to be tokenized.
        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        import tiktoken
        tokens_list = tiktoken.model.encoding_for_model('gpt-3.5-turbo').encode(prompt)

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
        text = tiktoken.model.encoding_for_model('gpt-3.5-turbo').decode(tokens_list)

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
                 prompt: str,
                 n_predict: int = 128,
                 callback: Callable[[str], None] = None,
                 verbose: bool = False,
                 **gpt_params):
        """Generates text out of a prompt
        Args:
            prompt (str): The prompt to use for generation
            n_predict (int, optional): Number of tokens to prodict. Defaults to 128.
            callback (Callable[[str], None], optional): A callback function that is called everytime a new text element is generated. Defaults to None.
            verbose (bool, optional): If true, the code will spit many informations about the generation process. Defaults to False.
        """
        self.binding_config.config["total_input_tokens"] += len(self.tokenize(prompt))
        self.binding_config.config["total_input_cost"] = self.binding_config.config["total_input_tokens"] * self.input_costs_by_model.get(self.config["model_name"], 0)
        try:
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
            data = {
                'model':self.config.model_name,
                'prompt': prompt,
                "stream":True,
                "temperature": float(gpt_params["temperature"]),
                "max_tokens": n_predict
            }            
            gpt_params = {**default_params, **gpt_params}
            if self.binding_config.address.strip().endswith("/"):
                self.binding_config.address = self.binding_config.address.strip()[:-1]
            url = f'{self.binding_config.address}/v1/completions'
            response = requests.post(url, headers=headers, data=json.dumps(data), stream=True, verify=self.binding_config.verify_ssl_certificate)

            if response.status_code==400:
                content = response.content.decode("utf8")
                content = json.loads(content)
                self.error(content["error"]["message"])
                return
            elif response.status_code==404:
                ASCIIColors.error(response.content.decode("utf-8", errors='ignore'))
            text = ""

            for line in response.iter_lines():
                decoded = line.decode("utf-8")
                if decoded.startswith("{"):
                    json_data = json.loads(decoded)
                    decoded = json_data["choices"][0]["text"]
                    if "error" in json_data:
                        self.error(json_data["error"]["message"])
                        break
                elif decoded.startswith("data"):
                    decoded=decoded[6:]
                    json_data = json.loads(decoded)
                    try:
                        decoded = json_data["choices"][0]["text"]
                        if "error" in json_data:
                            self.error(json_data["error"]["message"])
                            break
                    except:
                        decoded = ""
                text +=decoded
                if callback:
                    if not callback(decoded, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                            break
        except Exception as ex:
            self.error(f'Error {ex}')
            trace_exception(ex)

        self.binding_config.config["total_output_tokens"] += len(self.tokenize(text))
        self.binding_config.config["total_output_cost"] = self.binding_config.config["total_output_tokens"] * self.output_costs_by_model.get(self.config["model_name"], 0)
        self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]
        self.info(f'Consumed {self.binding_config.config["total_cost"]}$')
        self.binding_config.save()
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
        self.binding_config.config["total_input_tokens"] +=  len(self.tokenize(prompt))          
        self.binding_config.config["total_input_cost"] =  self.binding_config.config["total_input_tokens"] * self.input_costs_by_model.get(self.config["model_name"], 0)
        if not "vision" in self.config.model_name:
            raise Exception("You can not call a generate with vision on this model")
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
                                    "url": f"data:image/jpeg;base64,{encode_image(image_path)}"
                                    }                                    
                                }
                                for image_path in images
                            ]
                        }
                    ]
            
            if self.binding_config.address.strip().endswith("/"):
                self.binding_config.address = self.binding_config.address.strip()[:-1]
            url = f'{self.binding_config.address}{elf_completion_formats[self.binding_config.completion_format]}'
            response = requests.post(url, headers=headers, data=json.dumps(data), stream=True, verify=self.binding_config.verify_ssl_certificate)

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

            self.binding_config.config["total_output_tokens"] +=  len(self.tokenize(output))
            self.binding_config.config["total_output_cost"] =  self.binding_config.config["total_output_tokens"] * self.output_costs_by_model.get(self.config["model_name"], 0)
            self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]
        except Exception as ex:
            self.error(f'Error {ex}')
            trace_exception(ex)

        self.info(f'Consumed {self.binding_config.config["total_cost"]}$')
        self.binding_config.save()
        return ""

    def list_models(self):
        """Lists the models for this binding"""
        model_names = get_model_info(f'{self.binding_config.address}', self.binding_config.server_key, verify_ssl_certificate=self.binding_config.verify_ssl_certificate)
        entries=[]
        for model in model_names:
            entries.append(model["model_name"])
        return entries
                
    def get_available_models(self, app=None):
        models = get_model_info(f'{self.binding_config.address}', self.binding_config.server_key, verify_ssl_certificate=self.binding_config.verify_ssl_certificate)
        entries = []
        for model in models:
            icon_path = get_icon_path(model["model_name"])
            entry = {
                "category": "generic",
                "datasets": "unknown",
                "icon": icon_path,
                "license": "unknown",
                "model_creator": "unknown",
                "name": model["model_name"],
                "quantizer": None,
                "rank": "1.0",
                "type": "api",
                "variants": [
                    {
                        "name": model["model_name"],
                        "size": 0
                    }
                ]
            }
            entries.append(entry)

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

    oai = LiteLLM(config, lollms_paths,lollmsCom=lollms_app)
    oai.install()
    oai.binding_config.save()
    config.binding_name= "litellm"
    config.model_name=""
    config.save_config()
