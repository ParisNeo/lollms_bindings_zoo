######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying binding : ParisNeo's lollms remote service
# Supported by Nomic-AI
# license       : Apache 2.0
# Description   : 
# This is an interface class for lollms bindings.

# This binding is a wrapper to open ai's api

######
from pathlib import Path
from typing import Callable
from lollms.binding import LLMBinding, LOLLMSConfig
from lollms.personality import MSG_TYPE
import yaml
import re
import requests
import socketio

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, "
__license__ = "Apache 2.0"

binding_name = "LoLLMs"
binding_folder_name = ""

class LoLLMs(LLMBinding):
    # Define what is the extension of the model files supported by your binding
    # Only applicable for local models for remote models like gpt4 and others, you can keep it empty 
    # and reimplement your own list_models method
    file_extension='*.bin' 
    def __init__(self, config:LOLLMSConfig) -> None:
        """Builds a LoLLMs binding

        Args:
            config (LOLLMSConfig): The configuration file
        """
        super().__init__(config, False)
        
        # The local config can be used to store personal information that shouldn't be shared like chatgpt Key 
        # or other personal information
        # This file is never commited to the repository as it is ignored by .gitignore
        self.config = config
        self._local_config_file_path = Path(__file__).parent/"local_config.yaml"
        self.local_config = self.load_config_file(self._local_config_file_path)
        self.servers_addresses = self.local_config["servers_addresses"]

        # Ping servers
        for server_url in self.servers_addresses:
            # Create a Socket.IO client instance
            sio = socketio.Client()

            # Define the event handler for the 'connect' event
            @sio.event
            def connect():
                print('Connected to the server')

            # Define the event handler for the 'disconnect' event
            @sio.event
            def disconnect():
                print('Disconnected from the server')

            # Connect to the server
            sio.connect(server_url)

            # Disconnect from the server
            sio.disconnect()

        # Do your initialization stuff
            
    def tokenize(self, prompt:str):
        """
        Tokenizes the given prompt using the model's tokenizer.

        Args:
            prompt (str): The input prompt to be tokenized.

        Returns:
            list: A list of tokens representing the tokenized prompt.
        """
        return None

    def detokenize(self, tokens_list:list):
        """
        Detokenizes the given list of tokens using the model's tokenizer.

        Args:
            tokens_list (list): A list of tokens to be detokenized.

        Returns:
            str: The detokenized text as a string.
        """
        return None
    
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
        return ""
    
    @staticmethod
    def list_models(config:dict):
        """Lists the models for this binding
        """
        return ["ChatGpt by Open AI"]
                
    @staticmethod
    def get_available_models():
        # Create the file path relative to the child class's directory
        binding_path = Path(__file__).parent
        file_path = binding_path/"models.yaml"

        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        return yaml_data