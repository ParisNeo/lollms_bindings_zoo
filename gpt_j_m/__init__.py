######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying binding : Abdeladim's pygptj binding
# Supported by Nomic-AI
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.

# This binding is a wrapper to marella's binding
# Follow him on his github project : https://github.com/marella/gpt4all-j 

######
from pathlib import Path
from typing import Callable
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig
from lollms.helpers import ASCIIColors
from lollms.types import MSG_TYPE
import subprocess
import yaml
import re


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "GPTJ"
binding_folder_name = "gpt_j_m"

class GPTJ(LLMBinding):
    file_extension='*.bin'
    def __init__(self, 
                config: LOLLMSConfig, 
                lollms_paths: LollmsPaths = LollmsPaths(), 
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY) -> None:
        """
        Initialize the Binding.

        Args:
            config (LOLLMSConfig): The configuration object for LOLLMS.
            lollms_paths (LollmsPaths, optional): The paths object for LOLLMS. Defaults to LollmsPaths().
            installation_option (InstallOption, optional): The installation option for LOLLMS. Defaults to InstallOption.INSTALL_IF_NECESSARY.
        """
        # Initialization code goes here

        binding_config = TypedConfig(
            ConfigTemplate([
                {"name":"gpu_layers","type":"int","value":20, "min":0},
                {"name":"use_avx2","type":"bool","value":True}
            ]),
            BaseConfig(config={
                "use_avx2": True,     # use avx2
                "gpu_layers": 20       #number of layers top offload to gpu                
            })
        )
        super().__init__(
                            Path(__file__).parent, 
                            lollms_paths, 
                            config, 
                            binding_config, 
                            installation_option
                        )
        
    def build_model(self):
        model_path = self.get_model_path()
        from gpt4allj import Model
        self.model = Model(
                model=str(model_path), avx2 = self.binding_config.config["use_avx2"]
                )
        return self
    
    def install(self):
        super().install()
        requirements_file = self.binding_dir / "requirements.txt"
        # install requirements
        subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])
        ASCIIColors.success("Installed successfully")
            
    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        return prompt.split(" ")

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        return " ".join(tokens_list)
    def generate(self, 
                 prompt:str,                  
                 n_predict: int = 128,
                 callback: Callable[[str], None] = bool,
                 verbose: bool = False,
                 **gpt_params ):
        """Generates text out of a prompt

        Args:
            prompt (str): The prompt to use for generation
            n_predict (int, optional): Number of tokens to prodict. Defaults to 128.
            callback (Callable[[str], None], optional): A callback function that is called everytime a new text element is generated. Defaults to None.
            verbose (bool, optional): If true, the code will spit many informations about the generation process. Defaults to False.
        """
        default_params = {
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.96,
            'repeat_penalty': 1.3,
            "seed":-1,
            "n_threads":8
        }
        gpt_params = {**default_params, **gpt_params}
        try:
            self.model.reset()
            output = ""
            for tok in self.model.generate(
                                            prompt, 
                                            seed=self.config['seed'],
                                            n_threads=self.config['n_threads'],
                                            n_predict=n_predict,
                                            top_k=gpt_params['top_k'],
                                            top_p=gpt_params['top_p'],
                                            temp=gpt_params["temperature"],
                                            repeat_penalty=gpt_params['repeat_penalty'],
                                            repeat_last_n=self.config['repeat_last_n'],
                                            n_batch=8,
                                            reset=True,
                                           ):
                output += tok
                if callback is not None:
                    if not callback(tok, MSG_TYPE.MSG_TYPE_CHUNK):
                        return output
        except Exception as ex:
            print(ex)
        return output
    @staticmethod
    def get_available_models():
        # Create the file path relative to the child class's directory
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        return yaml_data