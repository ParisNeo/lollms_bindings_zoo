######
# Project       : lollms
# File          : exllama2/__init__.py
# Author        : ParisNeo with the help from bartowski
# Underlying 
# engine author : turboderp 
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.
# Big thank you to turboderp and oobabooga for their
# paving the way with their work
######
from pathlib import Path
from typing import Callable
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig
from lollms.helpers import ASCIIColors
from lollms.types import MSG_TYPE
from lollms.helpers import trace_exception
from lollms.utilities import AdvancedGarbageCollector
from lollms.utilities import reinstall_pytorch_with_cuda, reinstall_pytorch_with_cpu, reinstall_pytorch_with_rocm

import subprocess
import yaml
import re
import urllib
import shutil
import sys
import os
import platform
from tqdm import tqdm

# sys.path.append(os.getcwd())
# pth = Path(__file__).parent/"exllamav2"
# sys.path.append(str(pth))


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "EXLLAMA2"
binding_folder_name = "exllama2"
import os
import subprocess
import torch
class EXLLAMA2(LLMBinding):
    
    def __init__(self, 
                config: LOLLMSConfig, 
                lollms_paths: LollmsPaths = None, 
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY,
                notification_callback:Callable=None
                ) -> None:
        """Builds an Exllama binding

        Args:
            config (LOLLMSConfig): The configuration file
        """
        self.model          = None
        self.tokenizer      = None
        self.cache          = None
        self.generator      = None
                
        if lollms_paths is None:
            lollms_paths = LollmsPaths()

        # Initialization code goes here
        binding_config_template = ConfigTemplate([
            
            {"name": "gpu_split", "type": "str", "value": '[24]',
                "help": "A list depicting how many layers to offload to each GPU. [gpu1,gpu2 etc]. Example [16,24]. If you have just one, the list should contain one value"},
            {"name": "ctx_size", "type": "int", "value": 4090, "min": 512,
                "help": "The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs. Reduce to save memory. Can also be increased, ideally while also using compress_pos_emn and a compatible model/LoRA"},
            {"name": "max_input_len", "type": "int", "value": 2048, "min": 512,
                "help": "Maximum length of input IDs in a single forward pass. Sequences longer than this will be processed in multiple steps"},
            {"name": "max_attention_size", "type": "int", "value": 2048**2, "min": 512,
                "help": "Increase to compress positional embeddings applied to sequence"},
            {"name": "scale_pos_emb", "type": "float", "value": 1, "min": 1, "max": 8,
                "help": "Positional embeddings compression value, set it to your ctx_size divided by 2048 when over 2048. Only set this or alpha. Increase to compress positional embeddings applied to sequence"},
            {"name": "alpha", "type": "int", "value": 1, "min": 1, "max": 32,
                "help": "Alpha value for context size extension. Only use this or scale_pos_emb. Alpha value for NTK RoPE scaling. Similar to scale_pos_emb, higher values increaste ctx but add Perplexity."},
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
                            supported_file_extensions=['.safetensors'],
                            models_dir_names=["gptq"],
                            notification_callback=notification_callback
                        )

        
        self.config.ctx_size = self.binding_config.config.ctx_size
        self.callback = None
        self.n_generated = 0
        self.n_prompt = 0

        self.skip_prompt = True
        self.decode_kwargs = {}

        # variables used in the streaming process
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True
    def settings_updated(self):
        """
        When the configuration is updated
        """
        self.config.ctx_size=self.binding_config.config.ctx_size
        

    def build_model(self):

        try:
            from torch import version as torch_version
            from exllamav2 import (
                ExLlamaV2,
                ExLlamaV2Config,
                ExLlamaV2Cache,
                ExLlamaV2Tokenizer,
            )

            from exllamav2.generator import (
                ExLlamaV2StreamingGenerator,
                ExLlamaV2Sampler
            )

        except Exception as ex:
            trace_exception(ex)
            ASCIIColors.warning("Couldn't import dependencies")
            return

        if self.config.model_name is None:
            ASCIIColors.error('No model selected!!')
            return

        if self.config.model_name:

            path = self.config.model_name
            model_path = self.get_model_path()

            if not model_path:
                self.model = None
                return None

            model_name = str(model_path).replace("\\","/")

            for ext in ['.safetensors', '.pt', '.bin']:
                found = list(model_path.glob(f"*{ext}"))
                if len(found) > 0:
                    if len(found) > 1:
                        print(
                            f'More than one {ext} model has been found. The last one will be selected. It could be wrong.')

                    model_path = found[-1]
                    break        

            config = ExLlamaV2Config()
            config.model_dir = str(model_name)
            config.prepare()
            config.max_seq_len = self.binding_config.config.ctx_size  # Reduce to save memory. Can also be increased, ideally while also using compress_pos_emn and a compatible model/LoRA
            config.max_input_len = self.binding_config.config.max_input_len  # Maximum length of input IDs in a single forward pass. Sequences longer than this will be processed in multiple steps
            config.max_attention_size = self.binding_config.config.max_attention_size  # Sequences will be processed in chunks to keep the size of the attention weights matrix <= this
            config.scale_pos_emb = self.binding_config.config.scale_pos_emb  # Increase to compress positional embeddings applied to sequence
            config.scale_alpha_value = self.binding_config.config.alpha # Alpha value for NTK RoPE scaling. Similar to scale_pos_emb, higher values increaste ctx but add Perplexity.

            if torch_version.hip:
                config.rmsnorm_no_half2 = True
                config.matmul_no_half2 = True
                config.silu_no_half2 = True
                
            ASCIIColors.success("freeing memory")
            AdvancedGarbageCollector.safeHardCollectMultiple(['model','tokenizer','cache','generator','settings'],self)
            self.model = None
            self.tokenizer = None
            self.cache = None
            self.generator = None
            self.settings = None
            
            AdvancedGarbageCollector.collect()
            self.clear_cuda()
            ASCIIColors.success("freed memory")

            ASCIIColors.red ("----------- LOLLMS EXLLAMA2 Model Information -----------------")
            ASCIIColors.magenta(f"Model name:{self.config.model_name}")
            self.print_class_attributes(config)
            ASCIIColors.red ("--------------------------------------------------------------")

            self.model = ExLlamaV2(config)
            print("Loading model: " + str(model_name))
            try:
                if self.binding_config.gpu_split.strip()[0] == "[" and self.binding_config.gpu_split.strip()[-1]=="]":
                    gpu_split = eval(self.binding_config.gpu_split)
                    ASCIIColors.success(f"GPU split:{gpu_split}")
                else:
                    gpu_split = None
            except:
                gpu_split = None
            self.model.load(gpu_split) # [16, 24]
            self.tokenizer = ExLlamaV2Tokenizer(config)
            self.cache = ExLlamaV2Cache(self.model, max_seq_len=self.binding_config.config.ctx_size)
            self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
            self.settings = ExLlamaV2Sampler.Settings()
            # Value to prepend
            self.bos = torch.tensor([[self.tokenizer.bos_token_id]])
            self.generator.warmup()

            return self

    def __del__(self):
        del self.generator
        del self.cache
        del self.tokenizer
        del self.model
        try:
            torch.cuda.empty_cache()
        except Exception as ex:
            ASCIIColors.error("Couldn't clear cuda memory")

    def install(self):
        # free up memory
        ASCIIColors.success("freeing memory")
        AdvancedGarbageCollector.safeHardCollectMultiple(['model','tokenizer','cache','generator'],self)
        AdvancedGarbageCollector.safeHardCollectMultiple(['ExLlamaGenerator','ExLlama','ExLlamaCache','ExLlamaConfig','ExLlamaTokenizer','torch_version'])
        AdvancedGarbageCollector.collect()
        self.clear_cuda()
        ASCIIColors.success("freed memory")
        
        super().install()
        print("This is the first time you are using this binding.")
                # Step 1 : install pytorch with cuda
        ASCIIColors.info("Checking pytorch")
        
        if self.config.enable_gpu:
            ASCIIColors.yellow("This installation has enabled GPU support. Trying to install with GPU support")
            ASCIIColors.info("Checking pytorch")
            try:
                import torch
                import torchvision
                if torch.cuda.is_available():
                    ASCIIColors.success(f"CUDA is supported.\nCurrent version is {torch.__version__}.")
                    if self.check_torch_version(2.1):
                        ASCIIColors.yellow("Torch version is old. Installing new version")
                        reinstall_pytorch_with_cuda()
                    else:
                        ASCIIColors.yellow("Torch OK")
                else:
                    ASCIIColors.warning("CUDA is not supported. Trying to reinstall PyTorch with CUDA support.")
                    reinstall_pytorch_with_cuda()
            except Exception as ex:
                ASCIIColors.info("Pytorch not installed. Reinstalling ...")
                reinstall_pytorch_with_cuda()    
        else:
            try:
                import torch
                import torchvision
                if self.check_torch_version(2.1):
                    ASCIIColors.warning("Torch version is too old. Trying to reinstall PyTorch with CUDA support.")
                    reinstall_pytorch_with_cpu()
            except Exception as ex:
                ASCIIColors.info("Pytorch not installed. Reinstalling ...")
                reinstall_pytorch_with_cpu() 

        # requirements_file = self.binding_dir / "requirements.txt"
        # subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])

        # Repository URL
        repo_url = "https://github.com/ParisNeo/exllamav2.git"

        # Get the path of the current script file
        script_path = Path(__file__).resolve()

        # Get the parent directory of the script file
        parent_dir = script_path.parent

        # Define the subfolder name
        subfolder_name = "exllamav2"

        # Create the full path to the subfolder
        subfolder_path = parent_dir / subfolder_name

        # Check if the subfolder exists and remove it if it does
        # if subfolder_path.exists():
        #     ASCIIColors.yellow("---------- Pulling exllama ---------")
        #     subprocess.run(["git", "pull"], cwd = str(subfolder_path), check=True)
        #     ASCIIColors.yellow("------------------------------------")

        # else:
            # Clone the repository to the subfolder
        #     subprocess.run(["git", "clone", repo_url, str(subfolder_path)])
        # Make models dir
        models_dir = self.lollms_paths.personal_models_path / "exllama2"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Install custom version of transformers
        subprocess.run(["pip", "install", "--upgrade", "transformers"])
        subprocess.run(["pip", "install", "--upgrade", "accelerate"])
        subprocess.run(["pip", "install", "--upgrade", "peft"])
        subprocess.run(["pip", "install", "--upgrade", "exllamav2"])
        current_platform = platform.system()
        # 
        ASCIIColors.success("Installed successfully")
        try:
            from torch import version as torch_version
        except:
            pass            
            


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
        t = self.tokenizer.encode(prompt)
        return t[0].tolist()

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        t = torch.IntTensor([tokens_list])
        return  self.tokenizer.decode(t)[0]
    


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
        self.callback = callback
        default_params = {
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.96,
            'repeat_penalty': 1.3,
            "seed":-1,
            "n_threads":8,
            "typical_p":0.0
        }
        self.output = ""
        self.settings.temperature = default_params['temperature']
        self.settings.top_k = default_params['top_k']
        self.settings.top_p = default_params['top_p']
        self.settings.token_repetition_penalty = default_params['repeat_penalty']

        # self.settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])
        try:
            input_ids = self.tokenizer.encode(prompt)
            # Concatenate the value to the front of the existing tensor array
            input_ids = torch.cat((self.bos, input_ids), dim=1)

            self.generator.set_stop_conditions([self.tokenizer.eos_token_id])
            self.generator.begin_stream(input_ids, self.settings, token_healing = True)
            for i in range(n_predict):
                chunk, eos, _ = self.generator.stream()
                self.output += chunk
                if  self.callback:
                    if not self.callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                        break          
                
                if eos: break      
        except Exception as ex:
            ASCIIColors.error(ex)
            trace_exception(ex)
            if callback:
                callback(str(ex),MSG_TYPE.MSG_TYPE_EXCEPTION)
        return self.output

    @staticmethod
    def get_filenames(repo):
        import requests
        from bs4 import BeautifulSoup

        dont_download = [".gitattributes"]

        blocs = repo.split("/")
        if len(blocs)==2:
            main_url = "https://huggingface.co/"+repo+"/tree/main" #f"https://huggingface.co/{}/tree/main"
        else: 
            main_url = "/".join(blocs[:-3])+"/tree/main" #f"https://huggingface.co/{}/tree/main"
        
        # https://huggingface.co/TheBloke/Spicyboros-13B-2.2-GPTQ/tree/main?not-for-all-audiences=true
        
        response = requests.get(main_url)
        html_content = response.text
        soup = BeautifulSoup(html_content, 'html.parser')

        file_names = []

        

        for a_tag in soup.find_all('a', {'class': 'group'}):
            span_tag = a_tag.find('span', {'class': 'truncate'})
            if span_tag:
                file_name = span_tag.text
                if file_name not in dont_download:
                    file_names.append(file_name)

        if len(file_names)==0:
            ASCIIColors.warning(f"No files found. This is probably a model with disclaimer. Please make sure you read the disclaimer before using the model.")
            main_url = "https://huggingface.co/"+repo+"/tree/main?not-for-all-audiences=true" #f"https://huggingface.co/{}/tree/main"
            response = requests.get(main_url)
            html_content = response.text
            soup = BeautifulSoup(html_content, 'html.parser')

            file_names = []
            for a_tag in soup.find_all('a', {'class': 'group'}):
                span_tag = a_tag.find('span', {'class': 'truncate'})
                if span_tag:
                    file_name = span_tag.text
                    if file_name not in dont_download:
                        file_names.append(file_name)
        return file_names
                    
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
        
        import wget
        import os

        blocs = repo.split("/")
        """
        if len(blocs)!=2 and len(blocs)!=4:
            raise ValueError("Bad repository path. Make sure the path is a valid hugging face path")        
        if len(blocs)==4:
        """
        if len(blocs)!=2:
            repo="/".join(blocs[-5:-3])

        file_names = EXLLAMA2.get_filenames(repo)

        dest_dir = Path(base_folder)
        dest_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(dest_dir)

        loading = ["none"]
        pbar = tqdm(total=100, desc="Downloading", unit="step")
        previous = [0]
        def chunk_callback(current, total, width=80):
            # This function is called for each received chunk
            # Perform actions or computations on the received chunk
            # chunk: The chunk of data received
            # chunk_size: The size of each chunk in bytes
            # total_size: The total size of the file being downloaded

            # Example: Print the current progress
            downloaded = current 
            progress = (current  / total) * 100
            pbar.update(progress-previous[0])  # Update the tqdm progress bar
            previous[0] = progress
            if callback and (".safetensors" in loading[0] or ".bin" in loading[0] ):
                try:
                    callback(downloaded, total)
                except:
                    callback(0, downloaded, total)

        def download_file(get_file):
            main_url = "https://huggingface.co/"+repo#f"https://huggingface.co/{}/tree/main"

            filename = f"{main_url}/resolve/main/{get_file}"
            print(f"\nDownloading {filename}")
            loading[0]=filename
            wget.download(filename, out=str(dest_dir), bar=chunk_callback)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     executor.map(download_file, file_names)
        for file_name in file_names:
            download_file(file_name)

        print("Done")

    def get_file_size(self, url):
        file_names = EXLLAMA2.get_filenames(url)
        for file_name in file_names:
            if file_name.endswith(".safetensors"):
                splt = url.split("/")
                if len(splt)==2:
                    src = f"https://huggingface.co/{url}"
                else:
                    src = "/".join(splt[:-3])
                filename = f"{src}/resolve/main/{file_name}"                
                response = urllib.request.urlopen(filename)
                
                # Extract the Content-Length header value
                file_size = response.headers.get('Content-Length')
                
                # Convert the file size to integer
                if file_size:
                    file_size = int(file_size)
                
                return file_size        
        return 4000000000

