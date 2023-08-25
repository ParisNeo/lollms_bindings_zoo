######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Supported by Nomic-AI
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.

# This binding is a wrapper to the official llamacpp python bindings
# Follow him on his github project : https://github.com/abetlen/llama-cpp-python

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
import os

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "LLAMACPP"

class LLAMACPP(LLMBinding):
    def __init__(self, 
                 config:LOLLMSConfig, 
                 lollms_paths:LollmsPaths = None, 
                 installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY
                ) -> None:
        """Builds a LLAMACPP binding

        Args:
            config (dict): The configuration file
        """
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        binding_config_templete =  ConfigTemplate(
            [
                {"name":"embedding","type":"bool","value":False, "help":"Activate using embeddings or not."},
                {"name":"n_threads","type":"int","value":8, "min":1, "help":"Number of threads to use (make sure you don't use more threadss than your CPU can handle)"},
                {"name":"n_gpu_layers","type":"int","value":20, "min":0, "help":"Number of layers to offload to GPU"},
                {"name":"n_parts","type":"int","value":-1, "min":-1, "help":"Number of parts to split the model into. If -1, the number of parts is automatically determined."},
                {"name":"f16_kv","type":"bool","value":True, "help":"Use half-precision for key/value cache."},
                {"name":"use_mmap","type":"bool","value":True, "help":"Use mmap if possible."},
                {"name":"use_mlock","type":"bool","value":False, "help":"Force the system to keep the model in RAM."},
                
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
        seed = self.config["seed"]
        if seed<0:
            seed = 0

        # if seed <=0:
        #    seed = random.randint(1, 2**31)
        model_path = self.get_model_path()
        if not model_path:
            self.model = None
            return None

        
        from llama_cpp import Llama

        self.model = Llama(
            model_path=str(model_path), 
            n_ctx=self.config["ctx_size"], 
            n_parts=self.binding_config.n_parts,# Number of parts to split the model into. If -1, the number of parts is automatically determined.
            f16_kv=self.binding_config.f16_kv,# Use half-precision for key/value cache.
            use_mmap=self.binding_config.use_mmap,
            n_gpu_layers=self.binding_config.n_gpu_layers,
            use_mlock = self.binding_config.use_mlock,
            n_threads=self.binding_config.n_threads,
            seed=seed,
            embedding=self.binding_config.embedding
            
            )
        return self

    def install(self):
        super().install()
        # Step 2: Install dependencies using pip from requirements.txt
        requirements_file = self.binding_dir / "requirements.txt"
        subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])
        if self.config.enable_gpu:
            ASCIIColors.yellow("This installation has enabled GPU support. Trying to install with GPU support")
            try:
                import llama_cpp
                ASCIIColors.info("Found old installation. Uninstalling.")
                self.uninstall()
            except ImportError:
                # The library is not installed
                print("The main library is not installed.")

            # Define the environment variables
            env = os.environ.copy()
            env["CMAKE_ARGS"] = "-DLLAMA_CUBLAS=on"
            env["FORCE_CMAKE"] = "1"
            result = subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "llama-cpp-python", "--no-binary", "llama-cpp-python"], env=env)

            if result.returncode != 0:
                print("Couldn't find Cuda build tools on your PC. Reverting to CPU. ")
                subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "llama-cpp-python"])

            result = subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "llama-cpp-python"])
        else:
            result = subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "llama-cpp-python"])

        ASCIIColors.success("Installed successfully")


    def uninstall(self):
        super().install()
        print("Uninstalling binding.")
        subprocess.run(["pip", "uninstall", "--yes", "llama-cpp-python"])
        ASCIIColors.success("Installed successfully")


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
    

