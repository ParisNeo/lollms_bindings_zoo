######
# Project       : lollms
# File          : c_transformers/__init__.py
# Author        : ParisNeo with the help of the community
# Underlying 
# engine author : marella 
# license       : Apache 2.0
# Description   : 
# This is the ctransformers binding code
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

if not PackageManager.check_package_installed("pyopencl"):
    PackageManager.install_package("pyopencl")
    import pyopencl as cl
else:
    import pyopencl as cl


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "CTRansformers"

class CTRansformers(LLMBinding):
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
        
    def __del__(self):
        if self.model:
            del self.model

    def build_model(self):

        ASCIIColors.info("Building model")
        if self.config['model_name'] is None:
           ASCIIColors.error("No model is selected")
           return

        if 'gpt2' in self.config['model_name']:
            model_type='gpt2'
        elif 'gptj' in self.config['model_name']:
            model_type='gptj'
        elif 'gpt_neox' in self.config['model_name'] or 'guanaco' in self.config['model_name']:
            model_type='gpt_neox'
        elif 'dolly-v2' in self.config['model_name']:
            model_type='dolly-v2'
        elif 'starcoder' in self.config['model_name'] or 'starchat-beta' in self.config['model_name'] or 'starchat-beta' in self.config['model_name'] or 'WizardCoder' in self.config['model_name']:
            model_type='starcoder'
        elif 'mpt' in self.config['model_name']:
            model_type='mpt'
        elif 'falcon' in self.config['model_name'].lower():
            model_type='falcon'
        elif 'replit' in self.config['model_name'].lower():
            model_type = 'replit'
        elif 'gptq' in self.config['model_name'].lower(): # experimental
            model_type="gptq"
        else:
            model_type='llama'

        ASCIIColors.info(f"Model type : {model_type}")
        
        
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
            

        from ctransformers import AutoModelForCausalLM

        if self.binding_config.config["use_avx2"]:
            if self.config.enable_gpu:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path), model_type=model_type,
                    gpu_layers = self.binding_config.config["gpu_layers"] if self.config.enable_gpu else 0,
                    batch_size=self.binding_config.config["batch_size"],
                    threads = self.binding_config.config["n_threads"],
                    context_length = self.binding_config.config["ctx_size"],
                    seed = self.binding_config.config["seed"],
                    reset= False
                    )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    str(model_path), model_type=model_type,
                    batch_size=self.binding_config.config["batch_size"],
                    threads = self.binding_config.config["n_threads"],
                    context_length = self.binding_config.config["ctx_size"],
                    seed = self.binding_config.config["seed"],
                    reset= False
                    )
        else:
            if self.config.enable_gpu:
                self.model = AutoModelForCausalLM.from_pretrained(
                        str(model_path), model_type=model_type, lib = "avx",
                        gpu_layers = self.binding_config.config["gpu_layers"],
                        batch_size=self.binding_config.config["batch_size"],
                        threads = self.binding_config.config["n_threads"],
                        context_length = self.binding_config.config["ctx_size"],
                        seed = self.binding_config.config["seed"],
                        reset= False
                        )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                        str(model_path), model_type=model_type, lib = "avx",
                        batch_size=self.binding_config.config["batch_size"],
                        threads = self.binding_config.config["n_threads"],
                        context_length = self.binding_config.config["ctx_size"],
                        seed = self.binding_config.config["seed"],
                        reset= False
                        )
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
        # Get the platforms
        platforms = cl.get_platforms()

        gpu_type = "None"

        # Iterate over each platform
        for platform in platforms:
            # Get the devices for the platform
            devices = platform.get_devices()
            
            # Iterate over each device
            for device in devices:
                # Check if the device is an Apple GPU
                if "Apple" in device.vendor:
                    self.info("Apple GPU detected")
                    gpu_type = "Apple"
                    break                
                # Check if the device is an NVIDIA GPU
                elif "NVIDIA" in device.vendor:
                    self.info("NVIDIA GPU detected")
                    gpu_type = "NVIDIA"
                    break
                # Check if the device is an AMD GPU
                elif "AMD" in device.vendor:
                    self.info("AMD GPU detected")
                    gpu_type = "AMD"
                    break
                # Check if the device is an Intel GPU
                elif "Intel" in device.vendor:
                    self.info("Intel GPU detected")
                    gpu_type = "Intel"
                    break        

        if sys.platform == "win32":
            # Code for Windows platform
            self.info("Running on Windows")
            if self.config.enable_gpu:
                self.info("This installation has enabled GPU support. Trying to install with GPU support")
                # Step 2: Install dependencies using pip from requirements.txt
                self.info("Trying to install a cuda enabled version of ctransformers")
                # env = os.environ.copy()
                # env["CT_CUBLAS"]="1"
                # pip install --upgrade --no-cache-dir --no-binary ctransformers
                # result = subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "ctransformers", "--no-binary", "ctransformers"], env=env) # , "--no-binary"
                if gpu_type=="NVIDIA":
                    result = subprocess.run(["pip","install","ctransformers","ctransformers[cuda]"])
                    if result.returncode != 0:
                        self.warning("Couldn't find Cuda build tools on your PC. Reverting to CPU. ")
                        # pip install --upgrade --no-cache-dir --no-binary ctransformers
                        result = subprocess.run(["pip", "install", "--upgrade", "ctransformers"])
                elif gpu_type=="AMD":
                    result = subprocess.run(["CT_HIPBLAS=1","pip","install","ctransformers", "--no-binary", "ctransformers"])
                    if result.returncode != 0:
                        self.warning("Couldn't find Rocm build tools on your PC. Reverting to CPU. ")
                        # pip install --upgrade --no-cache-dir --no-binary ctransformers
                        result = subprocess.run(["pip", "install", "--upgrade", "ctransformers"])
                elif gpu_type=="Intel":
                    self.warning("Installing for Intel.")
                    # pip install --upgrade --no-cache-dir --no-binary ctransformers
                    result = subprocess.run(["pip", "install", "--upgrade", "ctransformers"])
            else:
                self.info("Using CPU")
                # pip install --upgrade --no-cache-dir --no-binary ctransformers
                result = subprocess.run(["pip", "install", "--upgrade", "ctransformers"])

            # Add your Windows-specific code here
        elif sys.platform == "darwin":
            # Code for macOS platform
            self.info("Running on macOS")
            result = subprocess.run(["CT_METAL=1","pip","install","ctransformers", "--no-binary", "ctransformers"])
            if result.returncode != 0:
                self.warning("Couldn't find Metal build tools on your PC. Reverting to CPU. ")
                # pip install --upgrade --no-cache-dir --no-binary ctransformers
                result = subprocess.run(["pip", "install", "--upgrade", "ctransformers"])
        elif sys.platform == "linux":
            # Code for Linux platform
            self.info("Running on Linux")
            if self.config.enable_gpu:
                self.info("This installation has enabled GPU support. Trying to install with GPU support")
                # Step 2: Install dependencies using pip from requirements.txt
                # env = os.environ.copy()
                # env["CT_CUBLAS"]="1"
                # pip install --upgrade --no-cache-dir --no-binary ctransformers
                # result = subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "ctransformers", "--no-binary", "ctransformers"], env=env) # , "--no-binary"
                if gpu_type=="NVIDIA":
                    result = subprocess.run(["pip","install","ctransformers","ctransformers[cuda]"])
                    if result.returncode != 0:
                        self.notify("Couldn't find Cuda build tools on your PC. Reverting to CPU. ", notification_type=NotificationType.NOTIF_WARNING)
                        # pip install --upgrade --no-cache-dir --no-binary ctransformers
                        result = subprocess.run(["pip", "install", "--upgrade", "ctransformers"])
                elif gpu_type=="AMD":
                    result = subprocess.run(["CT_HIPBLAS=1","pip","install","ctransformers", "--no-binary", "ctransformers"])
                    if result.returncode != 0:
                        self.notify("Couldn't find Rocm build tools on your PC. Reverting to CPU. ", notification_type=NotificationType.NOTIF_WARNING)
                        # pip install --upgrade --no-cache-dir --no-binary ctransformers
                        result = subprocess.run(["pip", "install", "--upgrade", "ctransformers"])
                elif gpu_type=="Intel":
                    self.notify("Installing for Intel. ", notification_type=NotificationType.NOTIF_INFO)
                    # pip install --upgrade --no-cache-dir --no-binary ctransformers
                    result = subprocess.run(["pip", "install", "--upgrade", "ctransformers"])
            else:
                self.notify("Using CPU", notification_type=NotificationType.NOTIF_WARNING)
                # pip install --upgrade --no-cache-dir --no-binary ctransformers
                result = subprocess.run(["pip", "install", "--upgrade", "ctransformers"])

            # Add your Linux-specific code here
        else:
            # Code for other platforms
            self.notify("Unsupported platform", notification_type=NotificationType.NOTIF_ERROR)
            return
        # CT_METAL=1 pip install ctransformers --no-binary ctransformers

                    
        self.notify("Installed successfully")

    def uninstall(self):
        """
        UnInstallation procedure (to be implemented)
        """  
        super().uninstall()
        self.configuration_file_path.unlink()
        subprocess.run(["pip","uninstall","ctransformers","-y"])

            
    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        return self.model.tokenize(prompt)

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        return self.model.detokenize(tokens_list)
    
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
            for chunk in self.model(
                                            prompt,
                                            max_new_tokens = n_predict,
                                            stream=True,
                                            top_k=int(gpt_params['top_k']),
                                            top_p=float(gpt_params['top_p']),
                                            temperature=float(gpt_params['temperature']),
                                            repetition_penalty=float(gpt_params['repeat_penalty']),
                                            last_n_tokens=int(gpt_params['last_n_tokens']),
                                            seed=int(gpt_params['seed']),
                                            threads = int(gpt_params['n_threads']),
                                            batch_size= int(gpt_params['batch_size']),
                                            reset=True,
                                ):
                if chunk=="<0x0A>":
                    chunk = "\n"
                if callback is not None:
                    if not callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                        break
                output += chunk
                count += 1
                
                
        except Exception as ex:
            print(ex)
        return output            
