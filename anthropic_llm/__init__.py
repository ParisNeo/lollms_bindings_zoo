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
from lollms.utilities import PackageManager, encode_image, get_media_type
from lollms.com import LoLLMsCom
import subprocess
import yaml
import sys
import base64
if not PackageManager.check_package_installed("PIL"):
    PackageManager.install_package("Pillow")
from PIL import Image
import io

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "AnthropicLLM"
binding_folder_name = ""


  
class AnthropicLLM(LLMBinding):
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
            "claude-3-opus-20240229":0.01,
        }       
        self.output_costs_by_model={
            "claude-3-opus-20240229":0.03,
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
                {"name":"anthropic_key","type":"str","value":"","help":"A valid open AI key to generate text using open ai api"},
                {"name":"ctx_size","type":"int","value":4090, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"seed","type":"int","value":-1,"help":"Random numbers generation seed allows you to fix the generation making it dterministic. This is useful for repeatability. To make the generation random, please set seed to -1."},
                {"name":"max_image_width","type":"int","value":-1,"help":"resize the images if they have a width bigger than this (reduces cost). -1 for no change"},

            ]),
            BaseConfig(config={
                "anthropic_key": "",     # use avx2
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
        import anthropic
        if self.binding_config.config["anthropic_key"] =="":
            self.error("No API key is set!\nPlease set up your API key in the binding configuration")
        else:
            self.client = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=self.binding_config.config["anthropic_key"],
            )        

        self.config.ctx_size=self.binding_config.config.ctx_size

    def build_model(self):
        import anthropic
        if self.binding_config.config["anthropic_key"] =="":
            self.error("No API key is set!\nPlease set up your API key in the binding configuration")
            return
        else:
            self.client = anthropic.Anthropic(
                # defaults to os.environ.get("ANTHROPIC_API_KEY")
                api_key=self.binding_config.config["anthropic_key"]
            )        
            self.binding_type = BindingType.TEXT_IMAGE
        # Do your initialization stuff
        return self

    def install(self):
        super().install()
        requirements_file = self.binding_dir / "requirements.txt"
        # install requirements
        self.ShowBlockingMessage("Installing anthropic api ...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])
            self.HideBlockingMessage()
            ASCIIColors.success("Installed successfully")
            ASCIIColors.error("----------------------")
            ASCIIColors.error("Attention please")
            ASCIIColors.error("----------------------")
            ASCIIColors.error("The anthropic binding uses the anthropic API which is a paid service. Please create an account on the anthropic website (https://console.anthropic.com/) then generate a key and provide it in the configuration of the binding.")
        except:
            self.warning("The anthropic binding uses the anthropic API which is a paid service.\nPlease create an account on the anthropic website (https://console.anthropic.com/) then generate a key and provide it in the configuration of the binding.",20)
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

            with self.client.messages.stream(
                max_tokens=n_predict,
                messages=[{"role": "user", "content": prompt}],
                model=self.config.model_name,
            ) as stream:
                for word in stream.text_stream:
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
        return output    

    def list_models(self):
        """Lists the models for this binding
        """
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"
        

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        

        return [f["name"] for f in yaml_data]
                
                
    def get_available_models(self, app:LoLLMsCom=None):
        # Create the file path relative to the child class's directory
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

    oai = AnthropicLLM(config, lollms_paths,lollmsCom=lollms_app)
    oai.install()
    oai.binding_config.save()
    config.binding_name= "anthropic"
    config.model_name="claude-3-opus-20240229"
    config.save_config()
