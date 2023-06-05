######
# Project       : GPT4ALL-UI
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Supported by Nomic-AI
# license       : Apache 2.0
# Description   : 
# This is an interface class for GPT4All-ui bindings.

# This binding is a wrapper to the official llamacpp python bindings
# Follow him on his github project : https://github.com/abetlen/llama-cpp-python

######
from pathlib import Path
from typing import Callable
from llama_cpp import Llama
from lollms.binding import LLMBinding, LOLLMSConfig
from lollms.paths import BaseConfig
from lollms  import MSG_TYPE
from .install import Install
import yaml
import random

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "LLAMACPP"
binding_folder_name = "llama_cpp_official"

class LLAMACPP(LLMBinding):
    file_extension='*.bin'
    def __init__(self, config:LOLLMSConfig) -> None:
        """Builds a LLAMACPP binding

        Args:
            config (dict): The configuration file
        """
        super().__init__(config, False)
        
        self.models_folder = config.lollms_paths.personal_models_path / Path(__file__).parent.stem
        self.models_folder.mkdir(parents=True, exist_ok=True)

        seed = config["seed"]
        try:
            self.local_config = self.load_config_file(config.lollms_paths.personal_configuration_path/ 'binding_llamacpp_config.yaml')
        except Exception as ex:
            Install(config, force_reinstall=True)
            self.local_config = self.load_config_file(config.lollms_paths.personal_configuration_path/ 'binding_llamacpp_config.yaml')
        # if seed <=0:
        #    seed = random.randint(1, 2**31)
        if self.config.model_name.endswith(".reference"):
            with open(str(config.lollms_paths.personal_models_path/f"{binding_folder_name}/{self.config.model_name}"),'r') as f:
                model_path=f.read()
        else:
            model_path=str(config.lollms_paths.personal_models_path/f"{binding_folder_name}/{self.config.model_name}")
        
        self.model = Llama(
            model_path=model_path, 
            n_ctx=self.config["ctx_size"], 
            n_gpu_layers=self.local_config["n_gpu_layers"], 
            seed=seed)


    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        return self.model.tokenize(prompt.encode())

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        return self.model.detokenize(tokens_list).decode()

    def generate(self, 
                 prompt:str,                  
                 n_predict: int = 128,
                 callback: Callable[[str], None] = bool,
                 verbose: bool = False,
                 **gpt_params ):
        """Generates text out of a prompt

        Args:
            prompt (str): The prompt to use for generation
            n_predict (int, optional): Number of tokens to predict. Defaults to 128.
            callback (Callable[[str], None], optional): A callback function that is called every time a new text element is generated. Defaults to None.
            verbose (bool, optional): If true, the code will spit many information about the generation process. Defaults to False.
            **gpt_params: Additional parameters for GPT generation.
                temperature (float, optional): Controls the randomness of the generated text. Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.2) make it more deterministic. Defaults to 0.7 if not provided.
                top_k (int, optional): Controls the diversity of the generated text by limiting the number of possible next tokens to consider. Defaults to 0 (no limit) if not provided.
                top_p (float, optional): Controls the diversity of the generated text by truncating the least likely tokens whose cumulative probability exceeds `top_p`. Defaults to 0.0 (no truncation) if not provided.
                repeat_penalty (float, optional): Adjusts the penalty for repeating tokens in the generated text. Higher values (e.g., 2.0) make the model less likely to repeat tokens. Defaults to 1.0 if not provided.

        Returns:
            str: The generated text based on the prompt
        """
        default_params = {
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.96,
            'repeat_penalty': 1.3
        }
        gpt_params = {**default_params, **gpt_params}
        try:
            self.model.reset()
            output = ""
            tokens = self.model.tokenize(prompt.encode())
            count = 0
            for tok in self.model.generate(tokens, 
                                            temp=gpt_params["temperature"],
                                            top_k=gpt_params['top_k'],
                                            top_p=gpt_params['top_p'],
                                            repeat_penalty=gpt_params['repeat_penalty'],
                                           ):
                if count >= n_predict or (tok == self.model.token_eos()):
                    break
                try:
                    word = self.model.detokenize([tok]).decode()
                except:
                    word = ""
                if callback is not None:
                    if not callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                        break
                output += word
                count += 1
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
    

