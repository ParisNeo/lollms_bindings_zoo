######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying 
# engine author : Open AI
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.

# This binding is a wrapper to open ai's api

######
from pathlib import Path
from typing import Callable, Any
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors, trace_exception
from lollms.types import MSG_OPERATION_TYPE
from lollms.utilities import PackageManager
from lollms.com import LoLLMsCom
import subprocess
import yaml
import sys
import base64
if not PackageManager.check_package_installed("PIL"):
    PackageManager.install_package("Pillow")
from PIL import Image
import io

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "MistralAI"
binding_folder_name = ""

# Function to encode the image
def encode_image(image_path, max_image_width=-1):
    image = Image.open(image_path)
    width, height = image.size

    if max_image_width != -1 and width > max_image_width:
        ratio = max_image_width / width
        new_width = max_image_width
        new_height = int(height * ratio)
        image = image.resize((new_width, new_height))

    # Check and convert image format if needed
    if image.format not in ['PNG', 'JPEG', 'GIF', 'WEBP']:
        image = image.convert('JPEG')

    # Save the image to a BytesIO object
    byte_arr = io.BytesIO()
    image.save(byte_arr, format=image.format)
    byte_arr = byte_arr.getvalue()

    return base64.b64encode(byte_arr).decode('utf-8')
  
class MistralAI(LLMBinding):
    def __init__(self, 
                config: LOLLMSConfig, 
                lollms_paths: LollmsPaths = None, 
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY,
                lollmsCom=None) -> None:
        """
        Initialize the Binding.

        Args:
            config (LOLLMSConfig): The configuration object for LOLLMS.
            lollms_paths (LollmsPaths, optional): The paths object for LOLLMS. Defaults to LollmsPaths().
            installation_option (InstallOption, optional): The installation option for LOLLMS. Defaults to InstallOption.INSTALL_IF_NECESSARY.
        """
        self.input_costs_by_model={
            "mistral-tiny":0.14,
            "mistral-small":0.6,
            "mistral-medium":2.5,
            "mistral-large-latest":2.5
        }       
        self.output_costs_by_model={
            "mistral-tiny":0.42,
            "mistral-small":1.8,
            "mistral-medium":7.5,
            "mistral-large-latest":2.5
        }
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        # Initialization code goes here
        binding_config = TypedConfig(
            ConfigTemplate([
                {"name":"total_input_tokens","type":"float", "value":0,"help":"The total number of input tokens in $"},
                {"name":"total_output_tokens","type":"float", "value":0,"help":"The total number of output tokens in $"},
                {"name":"total_input_cost","type":"float", "value":0,"help":"The total cost caused by input tokens in $"},
                {"name":"total_output_cost","type":"float", "value":0,"help":"The total cost caused by output tokens in $"},
                {"name":"total_cost","type":"float", "value":0,"help":"The total cost in $"},
                {"name":"mistralai_key","type":"str","value":"","help":"A valid open AI key to generate text using open ai api"},
                {"name":"ctx_size","type":"int","value":4090, "min":512, "help":"The current context size (it depends on the model you are using). Make sure the context size if correct or you may encounter bad outputs."},
                {"name":"max_n_predict","type":"int","value":4090, "min":512, "help":"The maximum amount of tokens to generate"},
                {"name":"seed","type":"int","value":-1,"help":"Random numbers generation seed allows you to fix the generation making it dterministic. This is useful for repeatability. To make the generation random, please set seed to -1."},
                {"name":"max_image_width","type":"int","value":-1,"help":"resize the images if they have a width bigger than this (reduces cost). -1 for no change"},

            ]),
            BaseConfig(config={
                "mistralai_key": "",     # use avx2
            })
        )
        super().__init__(
                            Path(__file__).parent, 
                            lollms_paths, 
                            config, 
                            binding_config, 
                            installation_option,
                            SAFE_STORE_SUPPORTED_FILE_EXTENSIONS=[''],
                            lollmsCom=lollmsCom
                        )
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict
    
    def settings_updated(self):
        if not PackageManager.check_package_installed("mistralai"):
            PackageManager.install_package("mistralai")
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        self.client = MistralClient(api_key=self.binding_config.config["mistralai_key"])
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict
    

    def build_model(self, model_name=None):
        super().build_model(model_name)
        self.config.ctx_size=self.binding_config.config.ctx_size
        self.config.max_n_predict=self.binding_config.max_n_predict
        if not PackageManager.check_package_installed("mistralai"):
            PackageManager.install_package("mistralai")
        from mistralai.client import MistralClient
        from mistralai.models.chat_completion import ChatMessage
        self.client = MistralClient(api_key=self.binding_config.config["mistralai_key"])
        self.ChatMessage = ChatMessage
        

        # Do your initialization stuff
        return self

    def install(self):
        super().install()
        requirements_file = self.binding_dir / "requirements.txt"
        # install requirements
        self.ShowBlockingMessage("Installing mistral ai api ...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])
            ASCIIColors.success("Installed successfully")
            ASCIIColors.error("----------------------")
            ASCIIColors.error("Attention please")
            ASCIIColors.error("----------------------")
            ASCIIColors.error("The mistralai binding uses the mistralai API which is a paid service. Please create an account on the mistral website (https://mistral.ai/) then generate a key and provide it in the configuration of the binding.")
        except:
            self.warning("The mistralai binding uses the openai API which is a paid service.\nPlease create an account on the openAi website (https://platform.mistral.ai/) then generate a key and provide it in the configuration of the binding.",20)
            self.HideBlockingMessage()

    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        import tiktoken
        tokens_list = tiktoken.model.encoding_for_model("gpt-3.5-turbo-1106").encode(prompt)

        return tokens_list

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        import tiktoken
        text = tiktoken.model.encoding_for_model("gpt-3.5-turbo-1106").decode(tokens_list)

        return text

    def embed(self, text):
        """
        Computes text embedding
        Args:
            text (str): The text to be embedded.
        Returns:
            List[float]
        """
        
        return self.client.embeddings(
            model="mistral-embed",
            input=[text],
        )

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
        self.error("This model do not support vision")
        return ""       


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
        if self.binding_config.config["mistralai_key"] =="":
            self.error("No API key is set!\nPlease set up your API key in the binding configuration")
            raise Exception("No API key is set!\nPlease set up your API key in the binding configuration")
        
        self.binding_config.config["total_input_tokens"] +=  len(self.tokenize(prompt))          
        self.binding_config.config["total_input_cost"] =  self.binding_config.config["total_input_tokens"] * self.input_costs_by_model.get(self.config["model_name"],0) /1000
        try:
            default_params = {
                'temperature': 0.7,
                'top_k': 50,
                'top_p': 0.96,
                'repeat_penalty': 1.3
            }
            gpt_params = {**default_params, **gpt_params}
            count = 0
            output = ""
            
            if "vision" in self.config.model_name: # For future
                messages = [
                            self.ChatMessage(role="user", content=prompt)
                        ]
            else:
                messages = [
                            self.ChatMessage(role="user", content=prompt)
                        ]
            # messages[0].model_dump()
            chat_completion = self.client.chat_stream(
                            model=self.config["model_name"],  # Choose the engine according to your OpenAI plan
                            messages=messages,
                            max_tokens=n_predict-7,  # Adjust the desired length of the generated response
                            temperature=float(gpt_params["temperature"]),  # Adjust the temperature for more or less randomness in the output
                            )
            try:
                for resp in chat_completion:
                    if count >= n_predict:
                        break
                    try:
                        word = resp.choices[0].delta.content
                    except Exception as ex:
                        word = ""
                    if callback is not None:
                        if not callback(word, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                            break
                    if word:
                        output += word
                        count += 1
            except Exception as ex:
                self.InfoMessage("The generation process failed.\nThis can happen if you exceeded your maximum spending set in your mistralai interface or if your key has been revoked.\nPlease check your mistralai acount settings.")

        except Exception as ex:
            self.error(f'Error {ex}$')
            trace_exception(ex)
        self.binding_config.config["total_output_tokens"] +=  len(self.tokenize(output))          
        self.binding_config.config["total_output_cost"] =  self.binding_config.config["total_output_tokens"] * self.output_costs_by_model[self.config["model_name"]]/1000    
        self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]
        self.info(f'Consumed {self.binding_config.config["total_output_cost"]}$')
        self.binding_config.save()
        return output

    def list_models(self):
        """Lists the models for this binding
        """
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        

        return [f["name"] for f in yaml_data]
                
                
if __name__=="__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    root_path = Path(__file__).parent
    lollms_paths = LollmsPaths.find_paths(tool_prefix="",force_local=True, custom_default_cfg_path="configs/config.yaml")
    config = LOLLMSConfig.autoload(lollms_paths)
    lollms_app = LollmsApplication("",config, lollms_paths, False, False,False, False)

    mai = MistralAI(config, lollms_paths,lollmsCom=lollms_app)
    mai.install()
    mai.binding_config.mistralai_key = input("Mistral AI Key:")
    mai.binding_config.save()
    config.binding_name= "mistral_ai"
    config.model_name="mistral-tiny"
    config.save_config()