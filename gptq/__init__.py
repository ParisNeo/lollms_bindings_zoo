######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Supported by Nomic-AI
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.
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
__github__ = "https://github.com/ParisNeo/GPTQ_binding"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "GPTQ"
binding_folder_name = "gptq"
import os
import platform
import os
import subprocess


class GPTQ(LLMBinding):
    file_extension='*'
    def __init__(self, 
                config: LOLLMSConfig, 
                lollms_paths: LollmsPaths = None, 
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY
                ) -> None:
        """Builds a GPTQ binding

        Args:
            config (LOLLMSConfig): The configuration file
        """
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        # Initialization code goes here
        binding_config_template = ConfigTemplate([
            
            {"name":"use_triton","type":"bool","value":False, "help":"Activate triton or not"},
            {"name":"device","type":"str","value":"gpu", "options":["cpu","gpu"],"help":"Device to be used (CPU or GPU)"},
            {"name":"batch_size","type":"int","value":1, "min":1},
            {"name":"gpu_layers","type":"int","value":20, "min":0},
            {"name":"ctx_size","type":"int","value":8192, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
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
                            installation_option
                        )
        self.config.ctx_size=self.binding_config.config.ctx_size



    def build_model(self):

        from transformers import AutoTokenizer
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

        if self.config.model_name:
            
            """
            model_path = self.get_model_path()
            self.model_dir = model_path
            model_name =[f for f in Path(self.model_dir).iterdir() if f.suffix==".safetensors" or f.suffix==".pth" or f.suffix==".bin"][0]
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, device=self.binding_config.device, use_fast=True, local_files_only=True)
            use_safetensors = model_name.suffix == '.safetensors'
            model_name = model_name.stem

            if not (Path(self.model_dir) / "quantize_config.json").exists():
                quantize_config = BaseQuantizeConfig(
                    bits= 4,
                    group_size= -1,
                    desc_act=""
                )
            else:
                quantize_config = None
            # load quantized model to the first GPU
            self.model = AutoGPTQForCausalLM.from_quantized(
                self.model_dir, 
                local_files_only=True,  
                model_basename=model_name, 
                device=self.binding_config.device,
                use_triton=False,#True,
                use_safetensors=use_safetensors,
                quantize_config=quantize_config
                )
            
            """
            models_dir = self.lollms_paths.personal_models_path / "gptq"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_name = "/".join(self.config.model_name.split("/")[:-1])[1:]
            model_base_name = ".".join(self.config.model_name.split("/")[-1].split(".")[:-1])

            self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    use_fast=True,
                    cache_dir=models_dir
                    )
            # load quantized model to the first GPU
            self.model = AutoGPTQForCausalLM.from_quantized(
                model_name,
                model_basename=model_base_name, 
                use_safetensors=True,
                trust_remote_code=True,
                device_map='auto',
                cache_dir=models_dir,
                quantize_config=None
                )
            self.model.seqlen = self.binding_config.ctx_size
        else:
            ASCIIColors.error('No model selected!!')


    

    def install(self):
        super().install()
        print("This is the first time you are using this binding.")
        # Step 2: Install dependencies using pip from requirements.txt
        requirements_file = self.binding_dir / "requirements.txt"
        try:
            import llama_cpp
            ASCIIColors.info("Found old installation. Uninstalling.")
            self.uninstall()
        except ImportError:
            # The library is not installed
            print("The main library is not installed.")

        # Define the environment variables
        os_type = platform.system()
        if os_type == "Linux":
            print("Linux OS detected.")
            env = os.environ.copy()
            
            result = subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "auto_gptq-0.2.2+cu117-cp310-cp310-linux_x86_64.whl"], env=env)

            if result.returncode != 0:
                print("Couldn't find Cuda build tools on your PC. Reverting to CPU. ")
                subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "auto-gptq"])
        if os_type == "Windows":
            print("Windows OS detected.")
            env = os.environ.copy()
            result = subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "auto_gptq-0.2.2+cu117-cp310-cp310-win_amd64.whl"], env=env)

            if result.returncode != 0:
                print("Couldn't find Cuda build tools on your PC. Reverting to CPU. ")
                subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "auto-gptq"])
        else:
            subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "auto-gptq"])
        subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])

        subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])
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
        return self.tokenizer.encode(prompt)

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        return  self.tokenizer.decode(tokens_list)
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
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
            toks = self.model.generate(inputs=input_ids, temperature=gpt_params["temperature"], max_new_tokens=n_predict)

            if callback is not None:
                callback(toks, MSG_TYPE.MSG_TYPE_CHUNK)
            output = toks
        except Exception as ex:
            print(ex)
            output=""
        return output

    @staticmethod
    def download_model(repo, base_folder, callback=None):
        """
        Downloads a folder from a Hugging Face repository URL, reports the download progress using a callback function,
        and displays a progress bar.

        Args:
            repo (str): The name of the Hugging Face repository.
            base_folder (str): The base folder where the repository should be saved.
            installation_path (str): The path where the folder should be saved.
            callback (function, optional): A callback function to be called during the download
                with the progress percentage as an argument. Defaults to None.
        """
        
        from tqdm import tqdm
        import requests
        from bs4 import BeautifulSoup
        import concurrent.futures
        import wget
        import os

        dont_download = [".gitattributes"]

        url = f"https://huggingface.co/{repo}/tree/main"
        response = requests.get(url)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        file_names = []

        for a_tag in soup.find_all('a', {'class': 'group'}):
            span_tag = a_tag.find('span', {'class': 'truncate'})
            if span_tag:
                file_name = span_tag.text
                if file_name not in dont_download:
                    file_names.append(file_name)

        print(f"Repo: {repo}")
        print("Found files:")
        for file in file_names:
            print(" ", file)

        dest_dir = Path(base_folder)
        dest_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(dest_dir)

        def chunk_callback(chunk, chunk_size, total_size):
            # This function is called for each received chunk
            # Perform actions or computations on the received chunk
            # chunk: The chunk of data received
            # chunk_size: The size of each chunk in bytes
            # total_size: The total size of the file being downloaded

            # Example: Print the current progress
            downloaded = len(chunk) * chunk_size
            progress = (downloaded / total_size) * 100
            if callback:
                callback(downloaded, total_size)

        def download_file(get_file):
            filename = f"https://huggingface.co/{repo}/resolve/main/{get_file}"
            print(f"\nDownloading {filename}")
            wget.download(filename, out=str(dest_dir), bar=chunk_callback)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(download_file, file_names)


        print("Done")
    @staticmethod
    def list_models(config:dict):
        """Lists the models for this binding
        """
        # Create the file path relative to the child class's directory
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)

        return [yd['filename'] for yd in yaml_data] 
    @staticmethod
    def get_available_models():
        # Create the file path relative to the child class's directory
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        return yaml_data