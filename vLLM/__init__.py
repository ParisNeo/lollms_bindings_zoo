######
# Project       : lollms
# File          : hugging_face/__init__.py
# Author        : ParisNeo with the help of the community
# Underlying 
# engine author : Hugging face Inc 
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.
######
import shutil

from pathlib import Path
from typing import Callable
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors
from lollms.types import MSG_TYPE
from lollms.helpers import trace_exception
from lollms.utilities import AdvancedGarbageCollector, PackageManager
from lollms.utilities import check_and_install_torch, expand2square, load_image
import subprocess, sys
import yaml
from tqdm import tqdm
import re
import urllib
import json
if not PackageManager.check_package_installed("PIL"):
    PackageManager.install_package("Pillow")
from PIL import Image


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "vLLM"
binding_folder_name = "vLLM"
import os
import subprocess
import gc

from lollms.com import NotificationDisplayType, NotificationType



class vLLM(LLMBinding):
    
    def __init__(self, 
                config: LOLLMSConfig, 
                lollms_paths: LollmsPaths = None, 
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY,
                lollmsCom=None
                ) -> None:
        """Builds a GPTQ/AWQ binding

        Args:
            config (LOLLMSConfig): The configuration file
        """
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        # Initialization code goes here
        binding_config_template = ConfigTemplate([
            {"name":"lora_file","type":"str","value":"", "help":"If you want to load a lora on top of your model then set the path to the lora here."},
            {"name":"trust_remote_code","type":"bool","value":False, "help":"If true, remote codes found inside models ort their tokenizer are trusted and executed."},
            {"name":"device_map","type":"str","value":'auto','options':['auto','cpu','cuda:0', 'balanced', 'balanced_low_0', 'sequential'], "help":"Force using quantized version"},
            {"name":"ctx_size","type":"int","value":4090, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
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
                            supported_file_extensions=['.safetensors','.pth','.bin'],
                            models_dir_names=["transformers","gptq","awq"],
                            lollmsCom=lollmsCom
                        )
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.callback = None
        self.n_generated = 0
        self.n_prompt = 0

        self.skip_prompt = True
        self.decode_kwargs = {}

        # variables used in the streaming process
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

        self.model = None
        self.tokenizer = None
        
    def settings_updated(self):
        self.config.ctx_size = self.binding_config.config.ctx_size        
        from auto_gptq import exllama_set_max_input_length
        try:
            self.model = exllama_set_max_input_length(self.model, self.binding_config.ctx_size)
        except:
            ASCIIColors.warning("Couldn't force exllama max imput size. This is a model that doesn't support exllama.")       

    def embed(self, text):
        """
        Computes text embedding
        Args:
            text (str): The text to be embedded.
        Returns:
            List[float]
        """
        
        pass
    def __del__(self):
        import torch
        if self.tokenizer:
            del self.tokenizer
        if self.model:
            del self.model
        try:
            torch.cuda.empty_cache()
        except Exception as ex:
            ASCIIColors.error("Couldn't clear cuda memory")

    def build_model(self):
        from vllm import LLM, SamplingParams
        if not PackageManager.check_package_installed("transformers"):
            PackageManager.install_package("transformers")
        from transformers import AutoTokenizer

        if self.config.model_name:

            path = self.config.model_name
            model_path = self.get_model_path()

            if not model_path:
                self.tokenizer = None
                self.model = None
                return None

            models_dir = self.lollms_paths.personal_models_path / binding_folder_name
            models_dir.mkdir(parents=True, exist_ok=True)
            # model_path = models_dir/ path

            model_name = str(model_path).replace("\\","/")

            # Delete any old model
            if hasattr(self, "tokenizer"):
                if self.tokenizer is not None:
                    del self.model

            if hasattr(self, "model"):
                if self.model is not None:
                    del self.model

            self.tokenizer = None
            self.model = None
            gc.collect()
            ASCIIColors.info(f"Creating tokenizer {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                    str(model_name), trust_remote_code=self.binding_config.trust_remote_code
                    )
            ASCIIColors.success(f"ok")
            
            self.SamplingParams = SamplingParams
            self.model = LLM(model=model_path, revision="v1.1.8", trust_remote_code=self.binding_config.trust_remote_code)
        else:
            self.error("No model selected.\Please select a model to load")        
    def install(self):
        import platform
        if platform.system()=="Windows":
            self.InfoMessage("vllm is only supported on linux.\nPlease use the remote vllm binding instead and make sure to activate the vllm server on this pc or a remote PC.")
            return False
        
        self.info("freeing memory")
        AdvancedGarbageCollector.safeHardCollectMultiple(['model'],self)
        AdvancedGarbageCollector.safeHardCollectMultiple(['AutoModelForCausalLM'])
        AdvancedGarbageCollector.collect()
        ASCIIColors.success("freed memory")
        py_version = sys.version.split(" ")[0][:4].replace(".","")
        os.environ['VLLM_VERSION'] = '0.2.6'
        os.environ['PYTHON_VERSION'] = py_version
        try:
            import conda.cli
            conda.cli.main("install", "-c", "nvidia/label/cuda-12.1.1", "cuda-compiler", "-y")
        except Exception as ex:
            trace_exception(ex)
        super().install()
        
        try:
            subprocess.run(["pip", "install", "--upgrade", "vllm"], check=True)
            # Installation complete
            self.info("vLLM installation completed successfully!")
        except Exception as e:
            if os.path.exists("temp/vllm"):
                try:
                    self.warning("An error occurred during vLLM installation: " + str(e))
                    self.warning("Please stop the application and manually remove the 'temp/vllm' folder.")
                except:
                    pass
            else:
                self.error("An error occurred during vLLM installation: " + str(e))


    def uninstall(self):
        super().install()
        print("Uninstalling binding.")
        ASCIIColors.success("Installed successfully")



    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        return self.tokenizer.encode(prompt,add_special_tokens=False)

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        return  self.tokenizer.decode(tokens_list)
    
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
        return


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
        default_params = {
            'temperature': self.generation_config.temperature,
            'top_k': self.generation_config.top_k,
            'top_p': self.generation_config.top_p,
            'repeat_penalty': self.generation_config.repetition_penalty,
            'repeat_last_n':self.generation_config.no_repeat_ngram_size,
            "seed":-1,
            "n_threads":8,
            "begin_suppress_tokens ": self.tokenize("!")
        }
        gpt_params = {**default_params, **gpt_params}
        
        try:
            self.sampling_params = self.SamplingParams(temperature=gpt_params["temperature"], top_p=gpt_params["top_p"])
            outputs = self.model.generate([prompt], self.sampling_params)
            # Print the outputs.
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                if self.callback:
                    self.callback(generated_text, MSG_TYPE.MSG_TYPE_FULL)
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                
        except:
            pass

        return outputs
    
    @staticmethod
    def get_filenames(repo):
        import requests
        from bs4 import BeautifulSoup

        dont_download = [".gitattributes"]

        blocs = repo.split("/")
        if len(blocs)!=2:
            raise ValueError("Bad repository path")
        
        # https://huggingface.co/TheBloke/Spicyboros-13B-2.2-GPTQ/tree/main?not-for-all-audiences=true
        
        main_url = "https://huggingface.co/"+repo+"/tree/main" #f"https://huggingface.co/{}/tree/main"
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

        file_names = vLLM.get_filenames(repo)
        # if there is a safetensor then remove all bins
        nb_safe_tensors=len([f for f in file_names if ".safetensors" in str(f)])
        if nb_safe_tensors>0:
            file_names = [f for f in file_names if ".bin" not in str(f)]
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
            if total>0:
                progress = (current  / total) * 100
            else:
                progress=0
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
        
    def get_file_size(self, repo):
        blocs = repo.split("/")
        """
        if len(blocs)!=2 and len(blocs)!=4:
            raise ValueError("Bad repository path. Make sure the path is a valid hugging face path")        
        if len(blocs)==4:
        """
        if len(blocs)!=2:
            repo="/".join(blocs[-5:-3])

        file_names = vLLM.get_filenames(repo)
        for file_name in file_names:
            if file_name.endswith(".safetensors") or  file_name.endswith(".bin"):
                src = "https://huggingface.co/"+repo
                filename = f"{src}/resolve/main/{file_name}"                
                response = urllib.request.urlopen(filename)
                
                # Extract the Content-Length header value
                file_size = response.headers.get('Content-Length')
                
                # Convert the file size to integer
                if file_size:
                    file_size = int(file_size)
                
                return file_size        
        return 4000000000


    def train(self, model_name_or_path, model_basename):
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from auto_gptq.utils.peft_utils import get_gptq_peft_model, GPTQLoraConfig

        model = AutoGPTQForCausalLM.from_quantized(
            model_name_or_path,
            model_basename=model_basename,
            use_safetensors=True,
            trust_remote_code=False,
            use_triton=True,
            device="cuda:0",
            warmup_triton=False,
            trainable=True,
            inject_fused_attention=True,
            inject_fused_mlp=False,
        )
        device = model.device
        model = get_gptq_peft_model(
            model, model_id=model_name_or_path, train_mode=False
        )
