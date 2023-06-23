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

import requests
from tqdm import tqdm
import os
from bs4 import BeautifulSoup
import concurrent.futures
import wget


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/GPTQ_binding"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "HuggingFace"
binding_folder_name = "hugging_face"

class HuggingFace(LLMBinding):
    file_extension='*'
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
        from transformers import AutoTokenizer, AutoModelForCausalLM
        model_name =[f for f in Path(self.model_dir).iterdir() if f.suffix==".safetensors" or f.suffix==".pth" or f.suffix==".bin"][0]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, device=self.local_config["device"], use_fast=True, local_files_only=True)
        use_safetensors = model_name.suffix == '.safetensors'
        model_name = model_name.stem


        # load quantized model to the first GPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir, 
            local_files_only=True,  
            model_basename=model_name, 
            device=self.local_config["device"],
            use_triton=True,
            use_safetensors=use_safetensors)
        return self    
    
    
    def install(self):
        super().install()
        requirements_file = self.binding_dir / "requirements.txt"
        # install requirements
        subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])
        ASCIIColors.success("Installed successfully")    

        # Example of installing py torche
        try:
            print("Checking pytorch")
            import torch
            if torch.cuda.is_available():
                print("CUDA is supported.")
            else:
                print("CUDA is not supported. Reinstalling PyTorch with CUDA support.")
                self.reinstall_pytorch_with_cuda()
        except Exception as ex:
            self.reinstall_pytorch_with_cuda()
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
        return output

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
        
        return [
            "EleutherAI/gpt-j-6b",
            "opt-125m-4bit"  
            "TheBloke/medalpaca-13B-GPTQ-4bit",
            "TheBloke/stable-vicuna-13B-GPTQ",
        ]
    @staticmethod
    def get_available_models():
        # Create the file path relative to the child class's directory
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        return yaml_data