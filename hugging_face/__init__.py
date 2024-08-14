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
from pathlib import Path
from typing import Callable
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors
from lollms.types import MSG_OPERATION_TYPE
from lollms.helpers import trace_exception
from lollms.utilities import AdvancedGarbageCollector, PackageManager, show_yes_no_dialog
from lollms.utilities import reinstall_pytorch_with_cuda, reinstall_pytorch_with_rocm, expand2square, load_image, run_cmd
import subprocess
from datetime import datetime
from tqdm import tqdm
import sys
import urllib
import json
if not PackageManager.check_package_installed("PIL"):
    PackageManager.install_package("Pillow")

import pipmaster as pm
if not pm.is_installed("torch"):
    ASCIIColors.yellow("Diffusers: Torch not found. Installing it")
    pm.install_multiple(["torch","torchvision","torchaudio"], "https://download.pytorch.org/whl/cu121", force_reinstall=True)

import torch
if not torch.cuda.is_available():
    ASCIIColors.yellow("Diffusers: Torch not using cuda. Reinstalling it")
    pm.install_multiple(["torch","torchvision","torchaudio"], "https://download.pytorch.org/whl/cu121", force_reinstall=True)

if not PackageManager.check_package_installed("transformers"):
    PackageManager.install_or_update("transformers")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig, AutoProcessor, LlavaForConditionalGeneration    
from transformers import GPTQConfig
from transformers import AwqConfig


from PIL import Image
import shutil

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "HuggingFace"
binding_folder_name = "hugging_face"
import os
import subprocess
import gc

from lollms.com import NotificationDisplayType, NotificationType


import torch



class HuggingFace(LLMBinding):
    
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
            {"name":"max_n_predict","type":"int","value":4090, "min":512, "help":"The maximum amount of tokens to generate"},
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
        self.config.max_n_predict=self.binding_config.max_n_predict
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
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict
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

    def build_model(self, model_name=None):
        super().build_model(model_name)
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict

        
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
                os.environ['HF_HOME'] = str(models_dir)
                self.ShowBlockingMessage(f"Creating tokenizer {model_path}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                        str(model_name), trust_remote_code=self.binding_config.trust_remote_code
                        )
                

                if "llava" in str(model_path).lower() or "vision" in str(model_path).lower():
                    self.model = LlavaForConditionalGeneration.from_pretrained(str(model_path),
                                                torch_dtype=torch.float16,
                                                device_map=self.binding_config.device_map,
                                                offload_folder="offload",
                                                offload_state_dict = True, 
                                                trust_remote_code=self.binding_config.trust_remote_code,
                                                low_cpu_mem_usage=self.binding_config.low_cpu_mem_usage,
                                                )
                    self.image_rocessor = AutoProcessor.from_pretrained(str(model_path))
                    self.binding_type= BindingType.TEXT_IMAGE
                    # from transformers import pipeline
                    # self.pipe = pipeline("image-to-text", model=str(model_path))
                    # self.binding_type = BindingType.TEXT_IMAGE
                    # self.model = self.pipe.model
                elif "gptq" in str(model_path).lower():
                    self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), padding_side="left")
                    gptq_config = GPTQConfig(bits=4, tokenizer=self.tokenizer)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(model_path), quantization_config=gptq_config, 
                        device_map=self.binding_config.device_map,
                        trust_remote_code=self.binding_config.trust_remote_code,
                        low_cpu_mem_usage=self.binding_config.low_cpu_mem_usage,
                    )
                elif "awq" in str(model_path).lower():
                    self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), padding_side="left")
                    awq_config = AwqConfig(bits=4, tokenizer=self.tokenizer)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(model_path), quantization_config=awq_config, 
                        device_map=self.binding_config.device_map,
                        trust_remote_code=self.binding_config.trust_remote_code,
                        low_cpu_mem_usage=self.binding_config.low_cpu_mem_usage,
                    )                    
                print(f"Model {model_name} built successfully.")
                self.model_device = self.model.parameters().__next__().device
                self.ShowBlockingMessage(f"Model loaded successfully")
                self.HideBlockingMessage()
                self.generation_config = GenerationConfig.from_pretrained(str(model_path))
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
                self.InfoMessage(f"No model is selected\nPlease select a model from the Models zoo to start using Hugging face binding")
        except Exception as ex:
            trace_exception(ex)
            self.HideBlockingMessage()
            self.InfoMessage(f"Couldn't load the model {model_path}\nHere is the error encountered during loading:\n"+str(ex)+"\nPlease choose another model or post a request on the discord channel.")


    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def is_cuda_supported():
        return torch.cuda.is_available()

    def download_model(self, model_name):
        """
        Download a model to the specified directory.
        """
        try:
            config = AutoConfig.from_pretrained(model_name)
            config.save_pretrained(os.path.join(self.model_dir, model_name))
            AutoModelForCausalLM.from_pretrained(model_name).save_pretrained(os.path.join(self.model_dir, model_name))
            AutoTokenizer.from_pretrained(model_name).save_pretrained(os.path.join(self.model_dir, model_name))
            print(f"Model {model_name} downloaded successfully.")
        except Exception as e:
            print(f"Error downloading model: {str(e)}")

    def destroy_model(self):
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
        if self.config.hardware_mode=="nvidia" or self.config.hardware_mode=="nvidia-tensorcores" or self.config.hardware_mode=="nvidia-tensorcores":
            if self.model is not None:
                AdvancedGarbageCollector.safeHardCollect("model", self)
                AdvancedGarbageCollector.safeHardCollect("tokenizer", self)
                self.model = None
                self.tokenizer = None
                gc.collect()
            self.clear_cuda()

    def install_transformers(self):
        # Use subprocess to run the pip install command
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", self.binding_dir / "requirements.txt", "--upgrade"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed with returncode {e.returncode}")
            return False


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
                self.install_transformers()
            elif self.config.hardware_mode=="cpu":
                self.install_transformers()
            elif self.config.hardware_mode=="amd-noavx":
                if not PackageManager.check_package_installed("torch"):
                    reinstall_pytorch_with_rocm()
                else:
                    if show_yes_no_dialog("Request","Do you want to force reinstalling pytorch?"):
                        reinstall_pytorch_with_rocm()
                self.install_transformers()
            elif self.config.hardware_mode=="amd":
                if not PackageManager.check_package_installed("torch"):
                    reinstall_pytorch_with_rocm()
                else:
                    if show_yes_no_dialog("Request","Do you want to force reinstalling pytorch?"):
                        reinstall_pytorch_with_rocm()
                self.install_transformers()
            elif self.config.hardware_mode=="nvidia":
                if not PackageManager.check_package_installed("torch"):
                    reinstall_pytorch_with_cuda()
                else:
                    if show_yes_no_dialog("Request","Do you want to force reinstalling pytorch?"):
                        reinstall_pytorch_with_cuda()
                self.install_transformers()
            elif self.config.hardware_mode=="nvidia-tensorcores":
                if not PackageManager.check_package_installed("torch"):
                    reinstall_pytorch_with_cuda()
                else:
                    if show_yes_no_dialog("Request","Do you want to force reinstalling pytorch?"):
                        reinstall_pytorch_with_cuda()
                self.install_transformers()
            elif self.config.hardware_mode=="apple-intel":
                self.install_transformers()
            elif self.config.hardware_mode=="apple-silicon":
                self.install_transformers()

            device_names = ['auto', 'cpu', 'balanced', 'balanced_low_0', 'sequential']
            import torch

            if torch.cuda.is_available():
                device_names.extend(['cuda:' + str(i) for i in range(torch.cuda.device_count())])

            # Initialization code goes here
            binding_config_template = ConfigTemplate([
                {"name":"low_cpu_mem_usage","type":"bool","value":True, "help":"Low cpu memory."},
                {"name":"lora_file","type":"str","value":"", "help":"If you want to load a lora on top of your model then set the path to the lora here."},
                {"name":"trust_remote_code","type":"bool","value":False, "help":"If true, remote codes found inside models ort their tokenizer are trusted and executed."},
                {"name":"device_map","type":"str","value":'auto','options':device_names, "help":"Select how the model will be spread on multiple devices"},
                {"name":"ctx_size","type":"int","value":4090, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"max_n_predict","type":"int","value":4090, "min":512, "help":"The maximum amount of tokens to generate"},
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
            self.config.ctx_size=self.binding_config.config.ctx_size
            self.config.max_n_predict=self.binding_config.max_n_predict
            # ASCIIColors.success("Installed successfully")
            self.success("Successfull installation")
        except Exception as ex:
            self.error(ex)
        self.HideBlockingMessage()
        
    def uninstall(self):
        super().uninstall()
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
    

    def put(self, value):
        """
        Recives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if len(value.shape)==1 and (value[0] == self.tokenizer.eos_token_id or value[0] == self.tokenizer.bos_token_id):
            print("eos detected")
            return
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
            if not self.callback(printable_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
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
            if self.callback(printable_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                raise Exception("canceled")    

    def process_images(self, images, image_processor, model_cfg):
        image_aspect_ratio = model_cfg.get("image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == 'pad':
            for image in images:
                image = expand2square(image, tuple(int(x*255)
                                    for x in image_processor.image_mean))
                image = image_processor.preprocess(image, return_tensors='pt')[
                    'pixel_values'][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors='pt')['pixel_values']
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = self.torch.stack(new_images, dim=0)
        return new_images
    
    def tokenizer_image_token(self, prompt, image_token_index=None, return_tensors=None):
        if image_token_index is None:
            image_token_index = self.IMAGE_TOKEN_INDEX
            
        prompt_chunks = [
            self.tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == self.tokenizer.bos_token_id:
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        if return_tensors is not None:
            if return_tensors == 'pt':
                return self.torch.tensor(input_ids, dtype=self.torch.long)
            raise ValueError(f'Unsupported tensor type: {return_tensors}')
        return input_ids
    


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
        self.generation_config.max_new_tokens = int(n_predict)
        self.generation_config.temperature = float(gpt_params["temperature"])
        self.generation_config.top_k = int(gpt_params["top_k"])
        self.generation_config.top_p = float(gpt_params["top_p"])
        self.generation_config.repetition_penalty = float(gpt_params["repeat_penalty"])
        self.generation_config.do_sample = True if float(gpt_params["temperature"])>0 else False
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.generation_config.output_attentions = False
        self.callback = callback    
        try:
            self.token_cache = []
            self.print_len = 0
            self.next_tokens_are_prompt = True            
            self.n_generated = 0
            self.output = ""
            try:
                with self.torch.no_grad():
                    image = Image.open(images[0])
                    self.output=""
                    inputs = self.image_rocessor("<image>"+prompt, image, return_tensors='pt').to(0, self.torch.float16)

                    #self.output = self.model.generate(**inputs, max_new_tokens=200, do_sample=False)            
                    self.model.generate(
                                        **inputs, 
                                        generation_config=self.generation_config,
                                        streamer = self,
                                        )
                    
            except Exception as ex:
                if str(ex)!="canceled":
                    trace_exception(ex)

        except Exception as ex:
            ASCIIColors.error("Couldn't generate")
            trace_exception(ex)
        return self.output

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
        self.generation_config.max_new_tokens = int(n_predict)
        self.generation_config.temperature = float(gpt_params["temperature"])
        self.generation_config.top_k = int(gpt_params["top_k"])
        self.generation_config.top_p = float(gpt_params["top_p"])
        self.generation_config.repetition_penalty = float(gpt_params["repeat_penalty"])
        self.generation_config.do_sample = True if float(gpt_params["temperature"])>0 else False
        self.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.generation_config.output_attentions = False
        self.callback = callback    
        try:
            self.token_cache = []
            self.print_len = 0
            self.next_tokens_are_prompt = True            
            self.n_generated = 0
            self.output = ""
            input_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(self.model_device)
            self.n_prompt = len(input_ids[0])
            try:
                print(f"Generating text on device: {self.model.device}")
                self.model.generate(
                                    inputs=input_ids, 
                                    generation_config=self.generation_config,
                                    streamer = self,
                                    )
            except Exception as ex:
                if str(ex)!="canceled":
                    trace_exception(ex)

        except Exception as ex:
            ASCIIColors.error("Couldn't generate")
            trace_exception(ex)
        return self.output


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
    
    def download_model(self, repo, base_folder, callback=None):
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

        file_names = HuggingFace.get_filenames(repo)
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

        file_names = HuggingFace.get_filenames(repo)
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

if __name__=="__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    root_path = Path(__file__).parent
    lollms_paths = LollmsPaths.find_paths(tool_prefix="",force_local=True, custom_default_cfg_path="configs/config.yaml")
    config = LOLLMSConfig.autoload(lollms_paths)
    lollms_app = LollmsApplication("",config, lollms_paths, False, False,False, False)

    exl = HuggingFace(config, lollms_paths,lollmsCom=lollms_app)
    exl.install()
    exl.install_model("gptq","https://huggingface.co/TheBloke/Airoboros-M-7B-3.1.2-GPTQ/resolve/main/model.safetensors","model.safetensors")
    config.binding_name= "hugging_face"
    config.model_name="Airoboros-M-7B-3.1.2-GPTQ"
    config.save_config()