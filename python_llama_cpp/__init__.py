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
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors
from lollms.com import NotificationType
from lollms.types import MSG_TYPE
from lollms.utilities import PackageManager, discussion_path_to_url, show_message_dialog
from lollms.utilities import AdvancedGarbageCollector, install_cuda, install_ninja, show_yes_no_dialog
from ascii_colors import ASCIIColors, trace_exception
import subprocess
import yaml
import os
import sys
import shutil
import platform
from functools import partial

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "LLAMA_Python_CPP"

def ban_eos_logits_processor(eos_token, input_ids, logits):
    logits[eos_token] = -float('inf')
    return logits


def custom_token_ban_logits_processor(token_ids, input_ids, logits):
    for token_id in token_ids:
        logits[token_id] = -float('inf')

    return logits


def check_file_type(suffix, extensions):
    # Remove the dot from the suffix
    suffix = suffix.replace('.', '')
    
    # Check if any extension (without the dot) is a substring of the suffix
    for ext in extensions:
        if ext.replace('.', '').lower() in suffix.lower():
            return True
    return False
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
            {"name":"generation_mode","type":"str","value":"chat", "options":["chat","instruct"], "help":"generation mode can be either chat or instruct.\nChat is good but doesn't allow cooperative mode or playground use. Instruct may be subject to halucination with bad models but allow more flexibility"},
            {"name":"n_gpu_layers","type":"int","value":33 if config.hardware_mode=="nvidia" or  config.hardware_mode=="nvidia-tensorcores" or  config.hardware_mode=="amd" or  config.hardware_mode=="amd-noavx" else 0, "min":1},
            {"name":"main_gpu","type":"int","value":0, "help":"If you have more than one gpu you can select the gpu to be used here"},
            {"name":"offload_kqv","type":"bool","value":False if 'cpu' in self.config.hardware_mode or 'apple' in self.config.hardware_mode else True, "help":"If you have more than one gpu you can select the gpu to be used here"},
            {"name":"cache_capacity","type":"int","value":(2 << 30) , "help":"The size of the cache in bytes"},            
            {"name":"batch_size","type":"int","value":1, "min":1},
            {"name":"ctx_size","type":"int","value":4096, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
            {"name":"seed","type":"int","value":-1,"help":"Random numbers generation seed allows you to fix the generation making it dterministic. This is useful for repeatability. To make the generation random, please set seed to -1."},
            {"name":"lora_path","type":"str","value":"","help":"Path to a lora file to apply to the model."},
            {"name":"lora_scale","type":"float","value":1.0,"help":"Scaling to apply to the lora."},
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
                            supported_file_extensions=['.gguf', '.bin'],
                            models_dir_names=["gguf", "ggml"],
                            lollmsCom=lollmsCom
                        )
        self.config.ctx_size=self.binding_config.config.ctx_size

    def settings_updated(self):
        self.config.ctx_size=self.binding_config.config.ctx_size        
        

    def __del__(self):
        if self.model:
            del self.model



    def build_model(self):
        if self.config.hardware_mode=="nvidia":
            try:
                import llama_cpp_cuda
                self.llama_cpp = llama_cpp_cuda
            except:
                self.error("Couldn't load Llamacpp for cuda.\nReverting to CPU")
                try:
                    import llama_cpp
                except:
                    llama_cpp = None
                    self.InfoMessage("Couldn't load Llamacpp!!!\nBinding broken. Try reinstalling it")
                    return
                self.llama_cpp = llama_cpp
        elif self.config.hardware_mode=="nvidia-tensorcores":
            try:
                import llama_cpp_cuda_tensorcores
                self.llama_cpp = llama_cpp_cuda_tensorcores
            except:
                llama_cpp_cuda_tensorcores = None
                self.error("Couldn't load Llamacpp for cuda with tensorcores.\nReverting to CPU")
                try:
                    import llama_cpp
                except:
                    llama_cpp = None
                    self.InfoMessage("Couldn't load Llamacpp!!!\nBinding broken. Try reinstalling it")
                    return
                self.llama_cpp = llama_cpp
        else:
            try:
                import llama_cpp
            except:
                llama_cpp = None
                self.InfoMessage("Couldn't load Llamacpp!!!\nBinding broken. Try reinstalling it")
            self.llama_cpp = llama_cpp

        Llama = self.llama_cpp.Llama
        LlamaCache = self.llama_cpp.LlamaCache
        ASCIIColors.info("Building model")
        if self.config['model_name'] is None:
           self.InfoMessage("No model is selected\nPlease select a model from the Models zoo to start using python_llama_cpp binding")
           return

        
        
        model_path = self.get_model_path()
        if not model_path:
            self.model = None
            return None
        
        self.binding_type = BindingType.TEXT_ONLY

        if "llava" in self.config.model_name.lower() or "vision" in self.config.model_name.lower():
            mmproj_variants = [v for v in model_path.parent.iterdir() if "mmproj" in str(v)]
            if len(mmproj_variants)==0:
                self.InfoMessage("Projector file was not found. Please download it first.\nReverting to text only")

                self.model = Llama(
                                        model_path=str(model_path), 
                                        n_gpu_layers=self.binding_config.n_gpu_layers, 
                                        main_gpu=self.binding_config.main_gpu, 
                                        n_ctx=self.config.ctx_size,
                                        n_threads=self.binding_config.n_threads,
                                        n_batch=self.binding_config.batch_size,
                                        offload_kqv=self.binding_config.offload_kqv,
                                        seed=self.binding_config.seed,
                                        lora_path=self.binding_config.lora_path,
                                        lora_scale=self.binding_config.lora_scale
                                    )

            else:
                proj_file = mmproj_variants[0]
                self.binding_type = BindingType.TEXT_IMAGE
                self.chat_handler = self.llama_cpp.llama_chat_format.Llava15ChatHandler(clip_model_path=str(proj_file))
                self.model = Llama(
                                        model_path=str(model_path), 
                                        n_gpu_layers=self.binding_config.n_gpu_layers, 
                                        main_gpu=self.binding_config.main_gpu, 
                                        n_ctx=self.config.ctx_size,
                                        n_threads=self.binding_config.n_threads,
                                        n_batch=self.binding_config.batch_size,
                                        offload_kqv=self.binding_config.offload_kqv,
                                        seed=self.binding_config.seed,
                                        lora_path=self.binding_config.lora_path,
                                        lora_scale=self.binding_config.lora_scale,

                                        chat_handler=self.chat_handler,
                                        logits_all=True
                                    )
        else:
            self.model = Llama(
                                    model_path=str(model_path), 
                                    n_gpu_layers=self.binding_config.n_gpu_layers, 
                                    main_gpu=self.binding_config.main_gpu, 
                                    n_ctx=self.config.ctx_size,
                                    n_threads=self.binding_config.n_threads,
                                    n_batch=self.binding_config.batch_size,
                                    offload_kqv=self.binding_config.offload_kqv,
                                    seed=self.binding_config.seed,
                                    lora_path=self.binding_config.lora_path,
                                    lora_scale=self.binding_config.lora_scale
                                )

        # self.model.set_cache(LlamaCache(capacity_bytes=0))
        print("Testing model")
        for chunk in self.model.create_completion("question: What is 1+1\nanswer:",
                                        max_tokens = 2,
                                        stream=True):
            print(chunk["choices"][0]["text"])
        
        ASCIIColors.success("Model built")            
        return self
    
    def install_cpu(self):
        # Set the environment variable
        os.environ['CMAKE_ARGS'] = ""
        # Use subprocess to run the pip install command
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--upgrade"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed with returncode {e.returncode}")
            return False

    def install_cuda(self):
        # Set the environment variable
        os.environ['CMAKE_ARGS'] = "-DLLAMA_CUBLAS=on"
        # Use subprocess to run the pip install command
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--upgrade"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed with returncode {e.returncode}")
            return False

    def install_metal(self):
        # Set the environment variable
        os.environ['CMAKE_ARGS'] = "-DLLAMA_METAL=on"
        # Use subprocess to run the pip install command
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--upgrade"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed with returncode {e.returncode}")
            return False

    def install_rocm(self):
        # Set the environment variable
        os.environ['CMAKE_ARGS'] = "-DLLAMA_HIPBLAS=on"
        # Use subprocess to run the pip install command
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--upgrade"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed with returncode {e.returncode}")
            return False

    def install_vulkan(self):
        # Set the environment variable
        os.environ['CMAKE_ARGS'] = "-DLLAMA_VULKAN=on"
        # Use subprocess to run the pip install command
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--upgrade"], check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Subprocess failed with returncode {e.returncode}")
            return False

    def test_vs_build_tools(self):
        if platform.system() == 'Windows' and show_yes_no_dialog("info","Do you want to install vs build tools? If the install fails, it means that you need to install build tools"):
            # Download the file from the given link
            url = "https://aka.ms/vs/17/release/vs_BuildTools.exe"
            file_name = "vs_BuildTools.exe"

            # Save it to the desired folder
            download_folder = self.lollms_paths.personal_path / "tmp"
            download_folder.mkdir(exist_ok=True, parents=True)
            save_path = os.path.join(str(download_folder), file_name)

            # Open a terminal or command prompt
            os.chdir(download_folder)

            # Run the downloaded file
            subprocess.call([sys.executable, "wget", url, "-O", save_path])
            subprocess.call([save_path])

    def install(self):
        # free up memory
        ASCIIColors.success("freeing memory")
        AdvancedGarbageCollector.safeHardCollectMultiple(['model'],self)
        AdvancedGarbageCollector.safeHardCollectMultiple(['AutoModelForCausalLM'])
        AdvancedGarbageCollector.collect()
        ASCIIColors.success("freed memory")
        
        
        super().install()

        if not show_yes_no_dialog("info","This binding will be compiled on your machine.\nIt is mandatory that you have build tools installed on your system. The process may fail if you don't have them.\nOn linux, just install using sudo apt-get install build-essential\nOn windows you can install vs build tools from : https://aka.ms/vs/17/release/vs_BuildTools.exe\nDo you want to continue?"):
            return

        # INstall other requirements
        # self.info("Installing requirements")
       
        # self.success("Requirements install done")
        self.ShowBlockingMessage(f"Installing requirements for hardware configuration {self.config.hardware_mode}")

        try:
            if self.config.hardware_mode=="cpu-noavx":
                self.install_cpu()
            elif self.config.hardware_mode=="cpu":
                self.install_cpu()
            elif self.config.hardware_mode=="amd-noavx":
                if not self.install_rocm():
                    ASCIIColors.warning("Couldn't install with rocm, reverting to CPU")
                    self.install_cpu()
            elif self.config.hardware_mode=="amd":
                if not self.install_rocm():
                    ASCIIColors.warning("Couldn't install with rocm, reverting to CPU")
                    self.install_cpu()
            elif self.config.hardware_mode=="nvidia":
                install_ninja()
                install_cuda()
                if not self.install_cuda():
                    ASCIIColors.warning("Couldn't install with cuda, reverting to CPU")
                    self.install_cpu()
            elif self.config.hardware_mode=="nvidia-tensorcores":
                install_ninja()
                install_cuda()
                if not self.install_cuda():
                    ASCIIColors.warning("Couldn't install with cuda, reverting to CPU")
                    self.install_cpu()
            elif self.config.hardware_mode=="apple-intel":
                if not self.install_vulkan():
                    ASCIIColors.warning("Couldn't install with vulkan, reverting to CPU")
                    self.install_cpu()
                    
            elif self.config.hardware_mode=="apple-silicon":
                if not self.install_metal():
                    ASCIIColors.warning("Couldn't install with metal, reverting to CPU")
                    self.install_cpu()

            self.notify("Installed successfully")
        except Exception as ex:
            self.error(ex)
        self.HideBlockingMessage()
        self.InfoMessage("After installing a binding, it is often required to reboot the application.\nPlease try to reboot it first.")

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
        return self.model.detokenize(tokens_list).decode("utf8").replace("<0x0A>","")
    
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


        if self.binding_config.generation_mode=="chat":

            try:
                output = ""
                # self.model.reset()
                count = 0
                for chunk in self.model.create_chat_completion(
                                    messages = [                             
                                        {
                                            "role": "user",
                                            "content":prompt
                                        }
                                    ],
                                    max_tokens=n_predict,
                                    temperature=gpt_params["temperature"],
                                    stop=["<0x0A>"],
                                    stream=True
                                ):

                    if count >= n_predict:
                        break
                    if "content" in chunk["choices"][0]["delta"]:
                        word = chunk["choices"][0]["delta"]["content"]
                    else:
                        word = ""
                    if word:
                        output += word
                        count += 1
                        if callback is not None:
                            if not callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                                break
                    
                    
            except Exception as ex:
                print(ex)
        else:
            output = ""
            count = 0
            for chunk in self.model.create_completion(
                                    prompt,
                                    max_tokens=n_predict,
                                    temperature=gpt_params["temperature"],
                                    stop=["<0x0A>"],
                                    stream=True
                        ):
                
                word = chunk["choices"][0]["text"]

                if word:
                    output += word
                    count += 1
                    if callback is not None:
                        if not callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                            break
        return output            


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
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.96,
            'repeat_penalty': 1.3
        }
        gpt_params = {**default_params, **gpt_params}
        output = ""
        try:
            count = 0
            url_imgs = [f"http://{self.config.host}:{self.config.port}"+discussion_path_to_url(img) for img in images]
            for chunk in self.model.create_chat_completion(
                                messages = [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "image_url", "image_url": {"url": img  }}
                                            for img in url_imgs
                                        ]+[ {"type" : "text", "text": prompt}]
                                    }
                                ],
                                stop=["<0x0A>"],
                                stream=True
                            ):
                if count >= n_predict:
                    break
                try:
                    if "content" in chunk["choices"][0]["delta"]:
                        word = chunk["choices"][0]["delta"]["content"]
                    else:
                        word = ""
                except Exception as ex:
                    word = ""
                if word:
                    output += word
                    count += 1
                    if callback is not None:
                        if not callback(word, MSG_TYPE.MSG_TYPE_CHUNK):
                            break
        except Exception as ex:
            trace_exception(ex)
        return output

if __name__=="__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    root_path = Path(__file__).parent
    lollms_paths = LollmsPaths.find_paths(tool_prefix="",force_local=True, custom_default_cfg_path="configs/config.yaml")
    config = LOLLMSConfig.autoload(lollms_paths)
    lollms_app = LollmsApplication("",config, lollms_paths, False, False,False, False)

    plc = LLAMA_Python_CPP(config, lollms_paths,lollmsCom=lollms_app)
    plc.install()
    plc.install_model("gguf","https://huggingface.co/TheBloke/Airoboros-M-7B-3.1.1-GGUF/resolve/main/airoboros-m-7b-3.1.1.Q4_0.gguf","airoboros-m-7b-3.1.1.Q4_0.gguf")
    config.binding_name = "python_llama_cpp"
    config.model_name   = "Airoboros-M-7B-3.1.1-GGUF"
    config.save_config()