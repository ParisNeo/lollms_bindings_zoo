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
from typing import Callable
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors, trace_exception
from lollms.types import MSG_TYPE
from lollms.utilities import PackageManager
import subprocess
import yaml
import re
import base64
if not PackageManager.check_package_installed("PIL"):
    PackageManager.install_package("pillow")
from PIL import Image
import io

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "OpenAIGPT"
binding_folder_name = ""

# Function to encode the image
def encode_image(image_path, max_image_width=-1):
    image = Image.open(image_path)
    width, height = image.size

    if max_image_width != -1 and width > max_image_width:
        ratio = max_image_width / width
        new_width = max_image_width
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height))

    # Check and convert image format if needed
    if image.format not in ['PNG', 'JPEG', 'GIF', 'WEBP']:
        image = image.convert('JPEG')

    # Save the image to a BytesIO object
    byte_arr = io.BytesIO()
    image.save(byte_arr, format=image.format)
    byte_arr = byte_arr.getvalue()

    return base64.b64encode(byte_arr).decode('utf-8')
  
class OpenAIGPT(LLMBinding):
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
        self.input_costs_by_model={
            "gpt-4-1106-preview":0.01,
            "gpt-4-vision-preview":0.03,
            "gpt-4":0.03,
            "gpt-4-32k":0.06,
            "gpt-3.5-turbo-1106":0.0015,
            "gpt-3.5-turbo":0.0010,
            "gpt-3.5-turbo-16k":0.003,
        }       
        self.output_costs_by_model={
            "gpt-4-1106-preview":0.03,
            "gpt-4-vision-preview":0.03,
            "gpt-4":0.06,
            "gpt-4-32k":0.12,
            "gpt-3.5-turbo-1106":0.0015,
            "gpt-3.5-turbo":0.002,
            "gpt-3.5-turbo-16k":0.004,
        }
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        # Initialization code goes here
        binding_config = TypedConfig(
            ConfigTemplate([
                {"name":"total_input_tokens","type":"float", "value":0,"help":"The total number of input tokens in $"},
                {"name":"total_output_tokens","type":"float", "value":0,"help":"The total number of output tokens in $"},
                {"name":"total_input_cost","type":"float", "value":0,"help":"The total cost caused by input tokens in $"},
                {"name":"total_output_cost","type":"float", "value":0,"help":"The total cost caused by output tokens in $"},
                {"name":"total_cost","type":"float", "value":0,"help":"The total cost in $"},
                {"name":"openai_key","type":"str","value":"","help":"A valid open AI key to generate text using open ai api"},
                {"name":"ctx_size","type":"int","value":4090, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"seed","type":"int","value":-1,"help":"Random numbers generation seed allows you to fix the generation making it dterministic. This is useful for repeatability. To make the generation random, please set seed to -1."},
                {"name":"max_image_width","type":"int","value":-1,"help":"resize the images if they have a width bigger than this (reduces cost). -1 for no change"},

            ]),
            BaseConfig(config={
                "openai_key": "",     # use avx2
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
        self.openai.api_key = self.binding_config.config["openai_key"]
        if self.openai.api_key =="":
            self.error("No API key is set!\nPlease set up your API key in the binding configuration")

        self.config.ctx_size=self.binding_config.config.ctx_size

    def build_model(self):
        import openai
        self.openai = openai

        if self.config.model_name is not None:
            if "vision" in self.config.model_name:
                self.binding_type = BindingType.TEXT_IMAGE

        self.openai.api_key = self.binding_config.config["openai_key"]
        if self.openai.api_key =="":
            self.error("No API key is set!\nPlease set up your API key in the binding configuration")
        # Do your initialization stuff
        return self

    def install(self):
        super().install()
        requirements_file = self.binding_dir / "requirements.txt"
        # install requirements
        subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])
        ASCIIColors.success("Installed successfully")
        ASCIIColors.error("----------------------")
        ASCIIColors.error("Attention please")
        ASCIIColors.error("----------------------")
        ASCIIColors.error("The chatgpt/gpt4 binding uses the openai API which is a paid service. Please create an account on the openAi website (https://platform.openai.com/) then generate a key and provide it in the configuration file.")

    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        import tiktoken
        tokens_list = tiktoken.model.encoding_for_model(self.config["model_name"]).encode(prompt)

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
        text = tiktoken.model.encoding_for_model(self.config["model_name"]).decode(tokens_list)

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
        
        self.binding_config.config["total_input_tokens"] +=  len(self.tokenize(prompt))          
        self.binding_config.config["total_input_cost"] =  self.binding_config.config["total_input_tokens"] * self.input_costs_by_model.get(self.config["model_name"],0.1) /1000
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
                                    "url": f"data:image/jpeg;base64,{encode_image(image_path, self.binding_config.max_image_width)}"
                                    }                                    
                                }
                                for image_path in images
                            ]
                        }
                    ]
            chat_completion = self.openai.chat.completions.create(
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
                    if not callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
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
        return ""       


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
            chat_completion = self.openai.chat.completions.create(
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
                    if not callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                        break
                if word:
                    output += word
                    count += 1


        except Exception as ex:
            self.error(f'Error {ex}$')
            trace_exception(ex)
        self.binding_config.config["total_output_tokens"] +=  len(self.tokenize(output))          
        self.binding_config.config["total_output_cost"] =  self.binding_config.config["total_output_tokens"] * self.output_costs_by_model[self.config["model_name"]]/1000    
        self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]
        self.info(f'Consumed {self.binding_config.config["total_output_cost"]}$')
        self.binding_config.save()
        return output

    @staticmethod
    def list_models(config:dict):
        """Lists the models for this binding
        """
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"
        

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        

        return [f["name"] for f in yaml_data]
                
    @staticmethod
    def get_available_models():
        # Create the file path relative to the child class's directory
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        return yaml_data
