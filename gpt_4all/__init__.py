######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Supported by Nomic-AI
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.


# This binding is a wrapper to gpt4all's official binding
# Follow him on his github project : https://github.com/ParisNeo/gpt4all

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

binding_name = "GPT4ALL"
binding_folder_name = "gpt_4all"


class GPT4ALL(LLMBinding):
    
    def __init__(self, 
                config: LOLLMSConfig, 
                lollms_paths: LollmsPaths = None, 
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY) -> None:
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
        binding_config_templete =  ConfigTemplate(
            [
                {"name":"processing_unit","type":"str","value":"gpu" if config.enable_gpu else "cpu", "options":["cpu","gpu"], "help":"The processing usit to use"},                
                {"name":"n_threads","type":"int","value":8, "min":1, "help":"Number of threads to use (make sure you don't use more threadss than your CPU can handle)"},
                {"name":"ctx_size","type":"int","value":2048, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"seed","type":"int","value":-1,"help":"Random numbers generation seed allows you to fix the generation making it dterministic. This is useful for repeatability. To make the generation random, please set seed to -1."},
            ]
            )
        binding_config = BaseConfig.from_template(binding_config_templete)
        binding_config = TypedConfig(
            binding_config_templete,
            binding_config
        )
        super().__init__(
                            Path(__file__).parent, 
                            lollms_paths, 
                            config, 
                            binding_config, 
                            installation_option,
                            supported_file_extensions=['.bin','.gguf']
                        )
        self.config.ctx_size=self.binding_config.config.ctx_size
        
    def build_model(self):        
        model_path = self.get_model_path()
        if not model_path:
            self.model = None
            return None
        
        if model_path.suffix!=".bin" and model_path.suffix!=".gguf":
            model_path = model_path.name.lower().replace("-ggml","").replace("-gguf","")
            candidates = [m for m in (self.lollms_paths.personal_models_path/self.binding_folder_name).iterdir() if model_path in m.name]
            if len(candidates)>0:
                model_path = candidates[0]
            else:
                ASCIIColors.warning(f"Model:{model_path} not found!!")
                return None

        from gpt4all import GPT4All, Embed4All

        self.model = GPT4All(
                                model_name=str(model_path.name),
                                model_path=str(model_path.parent),
                                device=self.binding_config.processing_unit
                            )
        self.model.model.set_thread_count(self.binding_config.n_threads)
        self.embedder= Embed4All()

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
    
    def embed(self, text):
        """
        Computes text embedding
        Args:
            text (str): The text to be embedded.
        Returns:
            List[float]: The embededing computed by the binding's model
        """
        return self.embedder.embed(text)
    
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
            'top_p': 0.0,
            'repeat_penalty': 1.3,
            'repeat_last_n': 10
        }
        gpt_params = {**default_params, **gpt_params}
        try:
            response_text = []
            for decoded_word in self.model.generate(prompt, 
                                            max_tokens=n_predict, 
                                            temp=gpt_params["temperature"],
                                            top_k=gpt_params['top_k'],
                                            top_p=gpt_params['top_p'],
                                            repeat_penalty=gpt_params['repeat_penalty'],
                                            repeat_last_n = self.config['repeat_last_n'],                                            
                                            streaming=True):
                response_text.append( decoded_word )
                if callback is not None:
                    if not callback(decoded_word, MSG_TYPE.MSG_TYPE_CHUNK):
                        break
        except Exception as ex:
            print(ex)
        return ''.join(response_text)

    @staticmethod
    def get_available_models():
        # Create the file path relative to the child class's directory
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        return yaml_data