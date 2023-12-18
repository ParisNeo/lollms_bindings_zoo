######
# Project       : lollms
# File          : c_transformers/__init__.py
# Author        : ParisNeo with the help of the community
# Underlying 
# engine author : marella 
# license       : Apache 2.0
# Description   : 
# This is the LLAMA_Python_CPP binding code
# This binding is a wrapper to marella's binding

######
from pathlib import Path
from typing import Callable
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig
from lollms.helpers import ASCIIColors
from lollms.com import NotificationType
from lollms.types import MSG_TYPE
from lollms.utilities import PackageManager
from lollms.utilities import AdvancedGarbageCollector
import subprocess
import yaml
import os
import sys

import platform

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "LLAMA_Python_CPP"

class LLAMA_Python_CPP(LLMBinding):
    def __init__(self, 
                config: LOLLMSConfig, 
                lollms_paths: LollmsPaths = None, 
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY,
                lollmsCom=None
                ) -> None:
        """
        Initialize the Binding.

        Args:
            config (LOLLMSConfig): The configuration object for LOLLMS.
            lollms_paths (LollmsPaths, optional): The paths object for LOLLMS. Defaults to LollmsPaths().
        """
        """
        Initialize the Binding.

        Args:
            config (LOLLMSConfig): The configuration object for LOLLMS.
            lollms_paths (LollmsPaths, optional): The paths object for LOLLMS. Defaults to LollmsPaths().
            installation_option (InstallOption, optional): The installation option for LOLLMS. Defaults to InstallOption.INSTALL_IF_NECESSARY.
        """
        self.model = None
        
        self.config = config
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
            
        # Initialization code goes here
        binding_config_template = ConfigTemplate([
            {"name":"n_threads","type":"int","value":8, "min":1},
            {"name":"batch_size","type":"int","value":1, "min":1},
            {"name":"gpu_layers","type":"int","value":20 if config.enable_gpu else 0, "min":0},
            {"name":"use_avx2","type":"bool","value":True},
            {"name":"ctx_size","type":"int","value":2048, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
            {"name":"seed","type":"int","value":-1,"help":"Random numbers generation seed allows you to fix the generation making it dterministic. This is useful for repeatability. To make the generation random, please set seed to -1."},
           
        ])
        binding_config_vals = BaseConfig.from_template(binding_config_template)

        binding_config = TypedConfig(
            binding_config_template,
            binding_config_vals
        )
        super().__init__(
                            Path(__file__).parent, 
                            lollms_paths, 
                            config, 
                            binding_config, 
                            installation_option,
                            supported_file_extensions=['.bin','.gguf'],
                            models_dir_names=["ggml","gguf"],
                            lollmsCom=lollmsCom
                        )
        self.config.ctx_size=self.binding_config.config.ctx_size

    def settings_updated(self):
        self.config.ctx_size=self.binding_config.config.ctx_size        
        

    def __del__(self):
        if self.model:
            del self.model

    def build_model(self):

        ASCIIColors.info("Building model")
        if self.config['model_name'] is None:
           ASCIIColors.error("No model is selected")
           return

        
        
        model_path = self.get_model_path()
        if not model_path:
            self.model = None
            return None

        if model_path.suffix not in self.supported_file_extensions:
            ext = model_path.suffix[1:]
            model_path = model_path.name.lower().replace(f"-{ext}","")
            candidates = [m for m in (self.lollms_paths.personal_models_path/self.binding_folder_name).iterdir() if model_path in m.name]
            if len(candidates)>0:
                model_path = candidates[0]
            

        from llama_cpp import Llama
        self.model = Llama(model_path=str(model_path))

        
        ASCIIColors.success("Model built")            
        return self
            
    def install(self):
        # free up memory
        ASCIIColors.success("freeing memory")
        AdvancedGarbageCollector.safeHardCollectMultiple(['model'],self)
        AdvancedGarbageCollector.safeHardCollectMultiple(['AutoModelForCausalLM'])
        AdvancedGarbageCollector.collect()
        ASCIIColors.success("freed memory")
        
        
        super().install()

        # INstall other requirements
        self.info("Installing requirements")
        requirements_file = self.binding_dir / "requirements.txt"
        subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])
        self.success("Requirements install done")
        current_platform = platform.system()
        
        if current_platform == 'Linux':
            subprocess.run(['pip', 'install', 'llama-cpp-python'])
        elif current_platform == 'Windows':
            subprocess.run(['pip', 'install', 'llama-cpp-python'])
        elif current_platform == 'Darwin':
            gpu_support = 'y' if subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True).stdout.decode().lower().find('gpu') != -1 else 'n'
            
            cmake_args = []
            
            if gpu_support.lower() == 'y':
                cmake_args.append("-DLLAMA_CUBLAS=on")
            
            env_vars = " ".join([f"-DCMAKE_ARGS={arg}" for arg in cmake_args])
            
            subprocess.run(['pip', 'install', 'llama-cpp-python'], env={'CMAKE_ARGS': env_vars})
        else:
            print('Unsupported platform.')
                    
        self.notify("Installed successfully")

    def uninstall(self):
        """
        UnInstallation procedure (to be implemented)
        """  
        super().uninstall()
        self.configuration_file_path.unlink()
        subprocess.run(["pip","uninstall","llama-cpp-python","-y"])

            
    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        return self.model.tokenize(prompt.encode("utf8"))

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        return self.model.detokenize(tokens_list).decode("utf8")
    
    def embed(self, text):
        """
        Computes text embedding
        Args:
            text (str): The text to be embedded.
        Returns:
            List[float]
        """
        return self.model.embed(text)
    
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
            'temperature': float(self.config.temperature),
            'top_k': int(self.config.top_k),
            'top_p': float(self.config.top_p),
            'repeat_penalty': float(self.config.repeat_penalty),
            'last_n_tokens' : int(self.config.repeat_last_n),
            "seed":int(self.binding_config.seed),
            "n_threads":self.binding_config.n_threads,
            "batch_size":self.binding_config.batch_size
        }
        gpt_params = {**default_params, **gpt_params}
        if gpt_params['seed']!=-1:
            self.seed = self.binding_config.seed
        try:
            output = ""
            # self.model.reset()
            count = 0
            
            for chunk in self.model.create_completion(
                                            prompt,
                                            max_tokens = n_predict,
                                            stream=True,
                                            top_k=int(gpt_params['top_k']),
                                            top_p=float(gpt_params['top_p']),
                                            temperature=float(gpt_params['temperature']),
                                            repeat_penalty=float(gpt_params['repeat_penalty']),
                                            # seed=int(gpt_params['seed']),
                                            #threads = int(gpt_params['n_threads']),
                                ):
                if count >= n_predict:
                    break
                try:
                    word = chunk["choices"][0]["text"]
                except Exception as ex:
                    word = ""
                if word:
                    output += word
                    count += 1
                    output=output.replace("<0x0A>","\n")
                    

                if callback is not None:
                    if not callback(output, MSG_TYPE.MSG_TYPE_FULL):
                        break
                
                
        except Exception as ex:
            print(ex)
        return output            
