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
from lollms.helpers import trace_exception
import subprocess
import yaml
import re
import urllib
import shutil


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/GPTQ_binding"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "EXLLAMA"
binding_folder_name = "exllama"
import os
import platform
import os
import subprocess
import gc

class EXLLAMA(LLMBinding):
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
            {"name":"split_between_cpu_and_gpu","type":"bool","value":False},
            {"name":"max_gpu_mem_GB","type":"int","value":4, "min":0},
            {"name":"max_cpu_mem_GB","type":"int","value":100, "min":0},
            {"name":"automatic_context_size","type":"bool","value":True, "help":"If selected, the context size will be set automatically and the ctx_size parameter is useless."},
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
        self.callback = None
        self.n_generated = 0
        self.n_prompt = 0

        self.skip_prompt = True
        self.decode_kwargs = {}

        # variables used in the streaming process
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def build_model(self):
        from exllama.model import ExLlama, ExLlamaCache, ExLlamaConfig
        from exllama.lora import ExLlamaLora
        from exllama.tokenizer import ExLlamaTokenizer
        from exllama.generator import ExLlamaGenerator
        import exllama.model_init as model_init
        import argparse
        import torch
        import sys
        import os
        import glob
        parser = argparse.ArgumentParser(description="Simple chatbot example for ExLlama")

        # Add arguments to the parser
        parser.add_argument("-lora", "--lora", type=str, help="Path to LoRA binary to use during benchmark")
        parser.add_argument("-loracfg", "--lora_config", type=str, help="Path to LoRA config to use during benchmark")
        parser.add_argument("-ld", "--lora_dir", type=str, help="Path to LoRA config and binary to use during benchmark")

        parser.add_argument("-p", "--prompt", type=str, help="Prompt file")
        parser.add_argument("-un", "--username", type=str, help="Display name of user", default="User")
        parser.add_argument("-bn", "--botname", type=str, help="Display name of chatbot", default="Chatbort")
        parser.add_argument("-bf", "--botfirst", action="store_true", help="Start chat on bot's turn")

        parser.add_argument("-nnl", "--no_newline", action="store_true", help="Do not break bot's response on newline (allow multi-paragraph responses)")
        parser.add_argument("-temp", "--temperature", type=float, help="Temperature", default=0.95)
        parser.add_argument("-topk", "--top_k", type=int, help="Top-K", default=20)
        parser.add_argument("-topp", "--top_p", type=float, help="Top-P", default=0.65)
        parser.add_argument("-minp", "--min_p", type=float, help="Min-P", default=0.00)
        parser.add_argument("-repp", "--repetition_penalty", type=float, help="Repetition penalty", default=1.15)
        parser.add_argument("-repps", "--repetition_penalty_sustain", type=int, help="Past length for repetition penalty", default=256)
        parser.add_argument("-beams", "--beams", type=int, help="Number of beams for beam search", default=1)
        parser.add_argument("-beamlen", "--beam_length", type=int, help="Number of future tokens to consider", default=1)

        # Create the `args` object with default values
        args = parser.parse_args([])
        # Simple interactive chatbot script

        torch.set_grad_enabled(False)
        torch.cuda._lazy_init()
        config = model_init.make_config(args)

        self.model = ExLlama(config)
        cache = ExLlamaCache(self.model)
        self.tokenizer = ExLlamaTokenizer(args.tokenizer)

        model_init.print_stats(self.model)
        
        generator = ExLlamaGenerator(self.model, self.tokenizer, cache)
        generator.settings = ExLlamaGenerator.Settings()
        generator.settings.temperature = args.temperature
        generator.settings.top_k = args.top_k
        generator.settings.top_p = args.top_p
        generator.settings.min_p = args.min_p
        generator.settings.token_repetition_penalty_max = args.repetition_penalty
        generator.settings.token_repetition_penalty_sustain = args.repetition_penalty_sustain
        generator.settings.token_repetition_penalty_decay = generator.settings.token_repetition_penalty_sustain // 2
        generator.settings.beams = args.beams
        generator.settings.beam_length = args.beam_length

        generator.lora = lora

        break_on_newline = not args.no_newline

        # Be nice to Chatbort

        min_response_tokens = 4
        max_response_tokens = 256
        extra_prune = 256

        print(past, end = "")
        ids = tokenizer.encode(past)
        generator.gen_begin(ids)

        next_userprompt = username + ": "

        first_round = True        

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
            path = self.config.model_name
            models_dir = self.lollms_paths.personal_models_path / "gptq"
            models_dir.mkdir(parents=True, exist_ok=True)
            model_path = models_dir/ path

            model_name = str(model_path).replace("\\","/")
            model_base_name = [f for f in model_path.iterdir() if f.suffix==".safetensors"][0].stem
            
            if not (model_path / "quantize_config.json").exists():
                quantize_config = BaseQuantizeConfig(
                    bits=4,
                    group_size=-1,
                    desc_act=""
                )
            else:
                quantize_config = None  
                          
            self.tokenizer = None
            gc.collect()
            import os
            os.environ['TRANSFORMERS_CACHE'] = str(models_dir)

            self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    use_fast=True
                    )
            # load quantized model to the first GPU
            if self.binding_config.split_between_cpu_and_gpu:
                params = {
                    'model_basename': model_base_name,
                    'device': "cuda:0" if self.binding_config.max_gpu_mem_GB>0 else "cpu",
                    'use_triton': self.binding_config.use_triton,
                    'inject_fused_attention': True,
                    'inject_fused_mlp': True,
                    'use_safetensors': True,
                    'trust_remote_code': True,
                    'max_memory': { 0: f'{self.binding_config.max_gpu_mem_GB}GiB', 'cpu': f'{self.binding_config.max_cpu_mem_GB}GiB' },
                    'quantize_config': quantize_config,
                    'use_cuda_fp16': True,
                }
                self.model = AutoGPTQForCausalLM.from_quantized(model_path, **params)
            else:
                params = {
                    'model_basename': model_base_name,
                    'device': "cuda:0" if self.binding_config.max_gpu_mem_GB>0 else "cpu",
                    'use_triton': self.binding_config.use_triton,
                    'inject_fused_attention': True,
                    'inject_fused_mlp': True,
                    'use_safetensors': True,
                    'trust_remote_code': True,
                    'quantize_config': quantize_config,
                    'use_cuda_fp16': True,
                }
                self.model = AutoGPTQForCausalLM.from_quantized(model_path, **params)

            try:
                if not self.binding_config.automatic_context_size:
                    self.model.seqlen = self.binding_config.ctx_size
                self.config.ctx_size = self.model.seqlen
            except:
                self.model.seqlen = self.binding_config.ctx_size
                self.config.ctx_size = self.model.seqlen
            ASCIIColors.info(f"Context lenghth set to {self.model.seqlen}")
            return self
        else:
            ASCIIColors.error('No model selected!!')


    

    def install(self):
        super().install()
        print("This is the first time you are using this binding.")
                # Step 1 : install pytorch with cuda
        ASCIIColors.info("Checking pytorch")
        try:
            import torch
            import torchvision
            if torch.cuda.is_available():
                ASCIIColors.success("CUDA is supported.")
            else:
                ASCIIColors.warning("CUDA is not supported. Trying to reinstall PyTorch with CUDA support.")
                self.reinstall_pytorch_with_cuda()
        except Exception as ex:
            ASCIIColors.info("Pytorch not installed")
            self.reinstall_pytorch_with_cuda()
            
        # Repository URL
        repo_url = "https://github.com/ParisNeo/exllama.git"

        # Get the path of the current script file
        script_path = Path(__file__).resolve()

        # Get the parent directory of the script file
        parent_dir = script_path.parent

        # Define the subfolder name
        subfolder_name = "exllama"

        # Create the full path to the subfolder
        subfolder_path = parent_dir / subfolder_name

        # Check if the subfolder exists and remove it if it does
        if subfolder_path.exists():
            shutil.rmtree(subfolder_path)

        # Clone the repository to the subfolder
        subprocess.run(["git", "clone", repo_url, str(subfolder_path)])

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
    

    def put(self, value):
        """
        Recives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError("TextStreamer only supports batch size 1")
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.extend(value.tolist())
        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)

        self.output += printable_text
        if  self.callback:
            if not self.callback(printable_text, MSG_TYPE.MSG_TYPE_CHUNK):
                raise Exception("canceled")    
            
    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False
    def end(self):
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.token_cache = []
            self.print_len = 0
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        if  self.callback:
            if self.callback(printable_text, MSG_TYPE.MSG_TYPE_CHUNK):
                raise Exception("canceled")    



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
        self.callback = callback    
        try:
            self.token_cache = []
            self.print_len = 0
            self.next_tokens_are_prompt = True            
            self.n_generated = 0
            self.output = ""
            input_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.cuda()
            self.n_prompt = len(input_ids[0])
            try:
                self.model.generate(
                                            inputs=input_ids, 
                                            max_new_tokens=n_predict, 
                                            temperature=gpt_params["temperature"], 
                                            top_p=gpt_params["top_p"],
                                            repetition_penalty=gpt_params["repeat_penalty"],
                                            streamer = self,
                                            )
                
            except Exception as ex:
                if str(ex)!="canceled":
                    trace_exception(ex)
            if callback is not None:
                callback(self.output, MSG_TYPE.MSG_TYPE_FULL)
        except Exception as ex:
            ASCIIColors.error("Couldn't generate")
            trace_exception(ex)
        return self.output
    
    @staticmethod
    def get_filenames(repo):
        import requests
        from bs4 import BeautifulSoup

        dont_download = [".gitattributes"]

        main_url = '/'.join(repo.split("/")[:-3])+"/tree/main" #f"https://huggingface.co/{}/tree/main"
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

        print(f"Repo: {repo}")
        print("Found files:")
        for file in file_names:
            print(" ", file)
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

        file_names = GPTQ.get_filenames(repo)

        dest_dir = Path(base_folder)
        dest_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(dest_dir)

        loading = ["none"]
        def chunk_callback(current, total, width=80):
            # This function is called for each received chunk
            # Perform actions or computations on the received chunk
            # chunk: The chunk of data received
            # chunk_size: The size of each chunk in bytes
            # total_size: The total size of the file being downloaded

            # Example: Print the current progress
            downloaded = current 
            progress = (current  / total) * 100
            if callback and ".safetensors" in loading[0]:
                try:
                    callback(downloaded, total)
                except:
                    callback(0, downloaded, total)
        def download_file(get_file):
            src = "/".join(repo.split("/")[:-3])
            filename = f"{src}/resolve/main/{get_file}"
            print(f"\nDownloading {filename}")
            loading[0]=filename
            wget.download(filename, out=str(dest_dir), bar=chunk_callback)

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        #     executor.map(download_file, file_names)
        for file_name in file_names:
            download_file(file_name)

        print("Done")
        
    def get_file_size(self, url):
        file_names = GPTQ.get_filenames(url)
        for file_name in file_names:
            if file_name.endswith(".safetensors"):
                src = "/".join(url.split("/")[:-3])
                filename = f"{src}/resolve/main/{file_name}"                
                response = urllib.request.urlopen(filename)
                
                # Extract the Content-Length header value
                file_size = response.headers.get('Content-Length')
                
                # Convert the file size to integer
                if file_size:
                    file_size = int(file_size)
                
                return file_size        
        return 4000000000

    def list_models(self, config:dict):
        """Lists the models for this binding
        """
        models_dir:Path = self.lollms_paths.personal_models_path/config["binding_name"]  # replace with the actual path to the models folder
        return [f.name for f in models_dir.iterdir() if f.is_dir() and not f.stem.startswith(".")]

    @staticmethod
    def get_available_models():
        # Create the file path relative to the child class's directory
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        return yaml_data