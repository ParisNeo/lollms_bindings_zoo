######
# Project       : lollms
# File          : exllamav2/__init__.py
# Author        : ParisNeo with the help of the community
# Underlying 
# engine author : Hugging face Inc 
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.
######
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
import subprocess
import yaml
from tqdm import tqdm
import re
import urllib
import json
import shutil
if not PackageManager.check_package_installed("PIL"):
    PackageManager.install_package("Pillow")
from PIL import Image


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "ExLLamav2"
binding_folder_name = "bs_exllamav2"
import os
import subprocess
import gc
from datetime import datetime

from lollms.com import NotificationDisplayType, NotificationType



class ExLLamav2(LLMBinding):
    
    def __init__(self, 
                config: LOLLMSConfig, 
                lollms_paths: LollmsPaths = None, 
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY,
                lollmsCom=None
                ) -> None:
        """Builds a GPTQ/EXL2 binding

        Args:
            config (LOLLMSConfig): The configuration file
        """
        device_names = ['auto', 'cpu', 'balanced', 'balanced_low_0', 'sequential']
        try:
            import torch
    
            if torch.cuda.is_available():
                device_names.extend(['cuda:' + str(i) for i in range(torch.cuda.device_count())])
        except:
            pass
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        # Initialization code goes here
        binding_config_template = ConfigTemplate([

            {"name":"low_cpu_mem_usage","type":"bool","value":True, "help":"Low cpu memory."},
            {"name":"lora_file","type":"str","value":"", "help":"If you want to load a lora on top of your model then set the path to the lora here."},
            {"name":"trust_remote_code","type":"bool","value":False, "help":"If true, remote codes found inside models ort their tokenizer are trusted and executed."},
            {"name":"device_map","type":"str","value":'auto','options':device_names, "help":"Select how the model will be spread on multiple devices"},
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
                            models_dir_names=["exl2","gptq"],
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
        import torch
        from transformers import GenerationConfig
        import torch
        self.torch = torch
        try:

            if self.config.model_name:

                path = self.config.model_name
                self.ShowBlockingMessage(f"Building model\n{path}")
                model_path = self.get_model_path()

                if not model_path:
                    self.tokenizer = None
                    self.model = None
                    return None
                models_dir = self.lollms_paths.personal_models_path / binding_folder_name
                models_dir.mkdir(parents=True, exist_ok=True)
                # model_path = models_dir/ path

                model_name = str(model_path).replace("\\","/")

                self.destroy_model()

                gen_cfg = model_path/"generation_config.json"
                if not gen_cfg.exists():
                    with open(gen_cfg,"w") as f:
                        json.dump({
                            "_from_model_config": True,
                            "bos_token_id": 1,
                            "eos_token_id": 32000,
                            "transformers_version": "4.35.0.dev0"
                    }
                    ,f)
                import os
                os.environ['TRANSFORMERS_CACHE'] = str(models_dir)
                self.generation_config = GenerationConfig.from_pretrained(str(model_path))
                self.ShowBlockingMessage(f"Creating model {model_path}\nUsing device map: {self.binding_config.device_map}")

                
                from exllamav2 import ExLlamaV2, ExLlamaV2Config,  ExLlamaV2Cache, ExLlamaV2Tokenizer
                from exllamav2.generator import ExLlamaV2StreamingGenerator, ExLlamaV2Sampler


                config = ExLlamaV2Config()
                config.model_dir = model_name
                config.prepare()
                # config.max_seq_len = shared.args.max_seq_len
                # config.scale_pos_emb = shared.args.compress_pos_emb
                # config.scale_alpha_value = shared.args.alpha_value
                # config.no_flash_attn = shared.args.no_flash_attn
                # config.num_experts_per_token = int(shared.args.num_experts_per_token)


                self.model = ExLlamaV2(config)
                print("Loading model: " + model_name)

                self.cache = ExLlamaV2Cache(self.model, lazy = True)
                try:
                    self.model.load_autosplit(self.cache)
                except Exception as ex:
                    ASCIIColors.red("unsufficient VRAM!")
                self.ShowBlockingMessage(f"Creating tokenizer {model_path}")
                self.tokenizer = ExLlamaV2Tokenizer(config)
                self.ShowBlockingMessage(f"Recovering generation config {model_path}")

                # Initialize generator

                self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
                    
                self.ShowBlockingMessage(f"Model loaded successfully")
                self.HideBlockingMessage()
                self.settings = ExLlamaV2Sampler.Settings()


                """
                try:
                    if not self.binding_config.automatic_context_size:
                        self.model.seqlen = self.binding_config.ctx_size
                    self.config.ctx_size = self.model.seqlen
                except:
                    self.model.seqlen = self.binding_config.ctx_size
                    self.config.ctx_size = self.model.seqlen
                ASCIIColors.info(f"Context lenghth set to {self.model.seqlen}")
                
                """
                return self
            else:
                self.InfoMessage(f"No model is selected\nPlease select a model from the Models zoo to start using ExllamaV2 binding")
        except Exception as ex:
            trace_exception(ex)
            self.error(str(ex))
            self.HideBlockingMessage()

    def install(self):
        self.ShowBlockingMessage("Freeing memory...")
        ASCIIColors.success("freeing memory")
        AdvancedGarbageCollector.safeHardCollectMultiple(['model'],self)
        AdvancedGarbageCollector.safeHardCollectMultiple(['AutoModelForCausalLM'])
        AdvancedGarbageCollector.collect()
        ASCIIColors.success("freed memory")

        super().install()

        self.ShowBlockingMessage(f"Installing requirements for hardware configuration {self.config.hardware_mode}")
        try:
            if self.config.hardware_mode=="cpu-noavx":
                self.InfoMessage("Hugging face binding requires GPU, please select A GPU configuration in your hardware selection section then try again or just select another binding.")
            elif self.config.hardware_mode=="cpu":
                self.InfoMessage("Hugging face binding requires GPU, please select A GPU configuration in your hardware selection section then try again or just select another binding.")
                return
            elif self.config.hardware_mode=="amd-noavx":
                requirements_file = self.binding_dir / "requirements_amd_noavx2.txt"
            elif self.config.hardware_mode=="amd":
                requirements_file = self.binding_dir / "requirements_amd.txt"
            elif self.config.hardware_mode=="nvidia":
                requirements_file = self.binding_dir / "requirements_nvidia_no_tensorcores.txt"
                check_and_install_torch(True)
            elif self.config.hardware_mode=="nvidia-tensorcores":
                requirements_file = self.binding_dir / "requirements_nvidia.txt"
                check_and_install_torch(True)
            elif self.config.hardware_mode=="apple-intel":
                requirements_file = self.binding_dir / "requirements_apple_intel.txt"
            elif self.config.hardware_mode=="apple-silicon":
                requirements_file = self.binding_dir / "requirements_apple_silicon.txt"

            subprocess.run(["pip", "install", "--upgrade", "-r", str(requirements_file)])

            device_names = ['auto', 'cpu', 'balanced', 'balanced_low_0', 'sequential']
            import torch

            if torch.cuda.is_available():
                device_names.extend(['cuda:' + str(i) for i in range(torch.cuda.device_count())])

            # Initialization code goes here
            binding_config_template = ConfigTemplate([
                {"name":"lora_file","type":"str","value":"", "help":"If you want to load a lora on top of your model then set the path to the lora here."},
                {"name":"trust_remote_code","type":"bool","value":False, "help":"If true, remote codes found inside models ort their tokenizer are trusted and executed."},
                {"name":"device_map","type":"str","value":'auto','options':device_names, "help":"Select how the model will be spread on multiple devices"},
                {"name":"ctx_size","type":"int","value":4090, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"seed","type":"int","value":-1,"help":"Random numbers generation seed allows you to fix the generation making it dterministic. This is useful for repeatability. To make the generation random, please set seed to -1."},

            ])
            binding_config_vals = BaseConfig.from_template(binding_config_template)

            binding_config = TypedConfig(
                binding_config_template,
                binding_config_vals
            )
            self.binding_config = binding_config
            self.add_default_configurations(binding_config)
            self.sync_configuration(binding_config, self.lollms_paths)
            self.binding_config.save()
            # ASCIIColors.success("Installed successfully")
            self.success("Successfull installation")
        except Exception as ex:
            self.error(ex)
        self.HideBlockingMessage()

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
        ptt= self.tokenizer.encode(prompt)[0]
        return ptt.tolist()
    
    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        tk = self.tokenizer.decode(self.torch.tensor(tokens_list))
        return tk

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

        self.settings.temperature = float(gpt_params["temperature"])
        self.settings.top_k = int(gpt_params["top_k"])
        self.settings.top_p = float(gpt_params["top_p"])
        self.settings.top_a = 0.0
        self.settings.token_repetition_penalty = float(gpt_params["repeat_penalty"])
        # self.settings.disallow_tokens(self.tokenizer, [self.tokenizer.eos_token_id])

        self.callback = callback    
        try:
            self.output = ""
            input_ids = self.tokenizer.encode(prompt)
            prompt_tokens = input_ids.shape[-1]
            self.generator.warmup()
            self.generator.begin_stream(input_ids, self.settings)            
            try:
                generated_tokens = 0

                while generated_tokens<n_predict:
                    chunk, eos, _ = self.generator.stream()
                    if eos:
                        break
                    generated_tokens += 1
                    if callback:
                        if not callback(chunk, MSG_TYPE.MSG_TYPE_CHUNK):
                            break

            except Exception as ex:
                if str(ex)!="canceled":
                    trace_exception(ex)

        except Exception as ex:
            ASCIIColors.error("Couldn't generate")
            trace_exception(ex)
        return self.output
    
    def destroy_model(self):
        ASCIIColors.bold("Destroying model")
        # Delete any old model
        if hasattr(self, "tokenizer"):
            if self.tokenizer is not None:
                del self.model
        if hasattr(self, "cache"):
            if self.cache is not None:
                del self.cache

        if hasattr(self, "generator"):
            if self.generator is not None:
                del self.generator


        if hasattr(self, "model"):
            if self.model is not None:
                del self.model

        self.tokenizer = None
        self.model = None
        gc.collect()
        if self.config.hardware_mode=="nvidia" or self.config.hardware_mode=="nvidia-tensorcores" or self.config.hardware_mode=="nvidia-tensorcores":
            if self.model is not None:
                AdvancedGarbageCollector.safeHardCollect("model", self)
                AdvancedGarbageCollector.safeHardCollect("tokenizer", self)
                self.model = None
                self.tokenizer = None
                gc.collect()
            self.clear_cuda()
        

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
        if "bartowski" in main_url:
            main_url = main_url.replace("main","4_0")

        if "turboderp" in main_url:
            main_url = main_url.replace("main","4.0bpw")                                

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
            if "bartowski" in main_url:
                main_url = main_url.replace("main","4_0")

            if "turboderp" in main_url:
                main_url = main_url.replace("main","4.0bpw")

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

        file_names = ExLLamav2.get_filenames(repo)
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
            if "bartowski" in filename:
                filename = filename.replace("main","4_0")

            if "turboderp" in filename:
                filename = filename.replace("main","4.0bpw")            
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

        file_names = ExLLamav2.get_filenames(repo)
        for file_name in file_names:
            if file_name.endswith(".safetensors") or  file_name.endswith(".bin"):
                src = "https://huggingface.co/"+repo
                filename = f"{src}/resolve/main/{file_name}"   
                if "bartowski" in filename:
                    filename = filename.replace("main","4_0")

                if "turboderp" in filename:
                    filename = filename.replace("main","4.0bpw")                                
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


    def install_model(self, model_type:str, model_path:str, variant_name:str, client_id:int=None):
        print("Install model triggered")
        model_path = model_path.replace("\\","/")
        if "bartowski" in model_path:
            model_path = model_path.replace("main","4_0")

        if "turboderp" in model_path:
            model_path = model_path.replace("main","4.0bpw")

        if model_type.lower() in model_path.lower():
            model_type:str=model_type
        else:
            mtt = None
            for mt in self.models_dir_names:
                if mt.lower() in  model_path.lower():
                    mtt = mt
                    break
            if mtt:
                model_type = mtt
            else:
                model_type:str=self.models_dir_names[0]

        progress = 0
        installation_dir = self.searchModelParentFolder(model_path.split('/')[-1], model_type)
        parts = model_path.split("/")
        if len(parts)==2:
            filename = parts[1]
        else:
            filename = parts[4]
        installation_path = installation_dir / filename
        print("Model install requested")
        print(f"Model path : {model_path}")

        model_name = filename
        binding_folder = self.config["binding_name"]
        model_url = model_path
        signature = f"{model_name}_{binding_folder}_{model_url}"
        try:
            self.download_infos[signature]={
                "start_time":datetime.now(),
                "total_size":self.get_file_size(model_path),
                "downloaded_size":0,
                "progress":0,
                "speed":0,
                "cancel":False
            }
            
            if installation_path.exists():
                print("Error: Model already exists. please remove it first")
   
                self.lollmsCom.notify_model_install(
                            installation_path,
                            model_name,
                            binding_folder,
                            model_url,
                            self.download_infos[signature]['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                            self.download_infos[signature]['total_size'],
                            self.download_infos[signature]['downloaded_size'],
                            self.download_infos[signature]['progress'],
                            self.download_infos[signature]['speed'],
                            client_id,
                            status=True,
                            error="",
                             )

                return

            
            def callback(downloaded_size, total_size):
                progress = (downloaded_size / total_size) * 100
                now = datetime.now()
                dt = (now - self.download_infos[signature]['start_time']).total_seconds()
                speed = downloaded_size/dt
                self.download_infos[signature]['downloaded_size'] = downloaded_size
                self.download_infos[signature]['speed'] = speed

                if progress - self.download_infos[signature]['progress']>2:
                    self.download_infos[signature]['progress'] = progress
                    self.lollmsCom.notify_model_install(
                                installation_path,
                                model_name,
                                binding_folder,
                                model_url,
                                self.download_infos[signature]['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                                self.download_infos[signature]['total_size'],
                                self.download_infos[signature]['downloaded_size'],
                                self.download_infos[signature]['progress'],
                                self.download_infos[signature]['speed'],
                                client_id,
                                status=True,
                                error="",
                                )                    
                
                if self.download_infos[signature]["cancel"]:
                    raise Exception("canceled")
                    
                
            if hasattr(self, "download_model"):
                try:
                    self.download_model(model_path, installation_path, callback)
                except Exception as ex:
                    ASCIIColors.warning(str(ex))
                    trace_exception(ex)
                    self.lollmsCom.notify_model_install(
                                installation_path,
                                model_name,
                                binding_folder,
                                model_url,
                                self.download_infos[signature]['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                                self.download_infos[signature]['total_size'],
                                self.download_infos[signature]['downloaded_size'],
                                self.download_infos[signature]['progress'],
                                self.download_infos[signature]['speed'],
                                client_id,
                                status=False,
                                error="Canceled",
                                )

                    del self.download_infos[signature]
                    try:
                        if installation_path.is_dir():
                            shutil.rmtree(installation_path)
                        else:
                            installation_path.unlink()
                    except Exception as ex:
                        trace_exception(ex)
                        ASCIIColors.error(f"Couldn't delete file. Please try to remove it manually.\n{installation_path}")
                    return

            else:
                try:
                    self.download_file(model_path, installation_path, callback)
                except Exception as ex:
                    ASCIIColors.warning(str(ex))
                    trace_exception(ex)
                    self.lollmsCom.notify_model_install(
                                installation_path,
                                model_name,
                                binding_folder,
                                model_url,
                                self.download_infos[signature]['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                                self.download_infos[signature]['total_size'],
                                self.download_infos[signature]['downloaded_size'],
                                self.download_infos[signature]['progress'],
                                self.download_infos[signature]['speed'],
                                client_id,
                                status=False,
                                error="Canceled",
                                )
                    del self.download_infos[signature]
                    installation_path.unlink()
                    return    
            self.lollmsCom.notify_model_install(
                        installation_path,
                        model_name,
                        binding_folder,
                        model_url,
                        self.download_infos[signature]['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                        self.download_infos[signature]['total_size'],
                        self.download_infos[signature]['total_size'],
                        100,
                        self.download_infos[signature]['speed'],
                        client_id,
                        status=True,
                        error="",
                        )
            del self.download_infos[signature]
        except Exception as ex:
            trace_exception(ex)
            self.lollmsCom.notify_model_install(
                        installation_path,
                        model_name,
                        binding_folder,
                        model_url,
                        '',
                        0,
                        0,
                        0,
                        0,
                        client_id,
                        status=False,
                        error=str(ex),
                        )


if __name__=="__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    root_path = Path(__file__).parent
    lollms_paths = LollmsPaths.find_paths(tool_prefix="",force_local=True, custom_default_cfg_path="configs/config.yaml")
    config = LOLLMSConfig.autoload(lollms_paths)
    lollms_app = LollmsApplication("",config, lollms_paths, False, False,False, False)

    exl = ExLLamav2(config, lollms_paths,lollmsCom=lollms_app)
    exl.install()
    exl.install_model("gptq","https://huggingface.co/TheBloke/Airoboros-M-7B-3.1.2-GPTQ/resolve/main/model.safetensors","model.safetensors")
    config.binding_name= "bs_exllamav2"
    config.model_name="Airoboros-M-7B-3.1.2-GPTQ"
    config.save_config()