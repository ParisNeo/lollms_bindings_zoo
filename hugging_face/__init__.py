######
# Project       : lollms
# File          : hugging_face_local/__init__.py
# Author        : ParisNeo with the help of the community
# Underlying
# engine author : Hugging Face Transformers team
# license       : Apache 2.0
# Description   :
# This binding allows Lollms to load and run Hugging Face transformer models
# locally using the 'transformers' library. It supports GPU acceleration
# (CUDA, MPS), quantization, vision models (like Gemma 3), dynamically
# fetches models from the Hub, automatically infers context size,
# and applies chat templates.
######

import io
import json
import os
import sys
import yaml
import re # Added for parsing
from datetime import datetime
from pathlib import Path
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set
import requests # For fetching images from URLs

from lollms.binding import BindingType, LLMBinding, LOLLMSConfig
from lollms.com import LoLLMsCom
from lollms.config import BaseConfig, ConfigTemplate, InstallOption, TypedConfig
from lollms.helpers import ASCIIColors, trace_exception
from lollms.paths import LollmsPaths
from lollms.types import MSG_OPERATION_TYPE
from lollms.utilities import (
    PackageManager, AdvancedGarbageCollector, get_torch_device,
    find_first_available_file_path, is_file_path, download_file # Added download_file
)
from PIL import Image
from time import perf_counter, sleep

# Set environment variable for Transformers offline mode based on config later

# Try to install necessary packages using pipmaster
if not PackageManager.check_package_installed("pipmaster"):
    PackageManager.install_package("pipmaster") # Attempt to install pipmaster if missing

try:
    import pipmaster as pm
    # Added huggingface_hub, Pillow, requests
    required_packages = ["torch", "transformers", "accelerate", "bitsandbytes", "sentencepiece", "huggingface_hub", "Pillow", "requests"]
    for entry in required_packages:
        if not pm.is_installed(entry):
            pm.install(entry)

    # Verify installation after attempt
    if not pm.is_installed("torch"): raise ImportError("PyTorch not found.")
    if not pm.is_installed("transformers"): raise ImportError("Transformers not found.")
    if not pm.is_installed("accelerate"): raise ImportError("Accelerate not found.")
    if not pm.is_installed("bitsandbytes"): raise ImportError("Bitsandbytes not found.") # Optional, but good to check
    if not pm.is_installed("huggingface_hub"): raise ImportError("huggingface_hub not found.")
    if not pm.is_installed("Pillow"): raise ImportError("Pillow (PIL) not found.") # Corrected check
    if not pm.is_installed("requests"): raise ImportError("requests not found.")


except ImportError as e:
    # Fallback or error message if pipmaster fails or isn't available
    print("Warning: pipmaster check failed or packages missing.")
    print("Please ensure torch, transformers, accelerate, bitsandbytes, sentencepiece, huggingface_hub, Pillow, and requests are installed.")
    print("Attempting to proceed, but errors may occur if packages are missing.")
    # Check again with standard import checks
    try: import torch
    except ImportError: print("Error: PyTorch is missing.")
    try: import transformers
    except ImportError: print("Error: Transformers is missing.")
    try: import accelerate
    except ImportError: print("Error: Accelerate is missing.")
    try: import bitsandbytes # Check is optional based on usage needs
    except ImportError: print("Warning: Bitsandbytes is missing (needed for 4/8-bit quantization).")
    try: import huggingface_hub
    except ImportError: print("Error: huggingface_hub is missing.")
    try: from PIL import Image
    except ImportError: print("Error: Pillow is missing.")
    try: import requests
    except ImportError: print("Error: requests is missing.")


# Now import the libraries - this assumes installation was successful or handled manually
try:
    import torch
    import transformers
    # Import specific classes needed
    try:
        from transformers import (
            AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
            BitsAndBytesConfig, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList,
            LlavaForConditionalGeneration, PaliGemmaForConditionalGeneration, # Add known VLM classes
            Gemma3ForConditionalGeneration # Import Gemma 3 explicitly
        )
    except:
        pm.install("transformers", upgrade=True)
        from transformers import (
            AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
            BitsAndBytesConfig, TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList,
            LlavaForConditionalGeneration, PaliGemmaForConditionalGeneration, # Add known VLM classes
            Gemma3ForConditionalGeneration # Import Gemma 3 explicitly
        )
    from huggingface_hub import HfApi # Added imports
    # accelerate is implicitly used by device_map='auto'
    if torch.cuda.is_available():
        try:
            import bitsandbytes as bnb # Only needed if CUDA is available for quantization
        except ImportError:
            print("Warning: bitsandbytes not found, 4/8-bit quantization will not be available.")

except ImportError as e:
    trace_exception(e)
    ASCIIColors.error("Could not import required libraries. Please ensure they are installed correctly.")
    # Set flags or raise depending on desired behavior on import failure
    # raise e # Uncomment to make import failure fatal


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023-2024, ParisNeo"
__license__ = "Apache 2.0"

binding_name = "HuggingFaceLocal"
binding_folder_name = "hugging_face"
# Define the folder for local HF models relative to lollms_paths
HF_LOCAL_MODELS_DIR = "transformers"
REFERENCE_FILE_EXTENSION = ".reference"
# Mapping from model_type/architecture name fragments to classes and processor types
# Keys should be lowercased.
KNOWN_MODEL_CLASSES = {
    # Vision Language Models (VLMs) - Need Processor
    "gemma3": (Gemma3ForConditionalGeneration, AutoProcessor),
    "paligemma": (PaliGemmaForConditionalGeneration, AutoProcessor),
    "llava": (LlavaForConditionalGeneration, AutoProcessor),
    # Add other VLMs like Idefics, MiniCPM-V etc.
    # "idefics": (IdeficsForVisionText2Text, AutoProcessor), # Example

    # Text-only Models - Need Tokenizer
    "default": (AutoModelForCausalLM, AutoTokenizer) # Fallback for text models
}

# Define the custom stopping criteria class
# It's often cleaner to define it outside the main class, but it needs access
# to the instance's flag. We pass the instance during initialization.
class StopGenerationCriteria(StoppingCriteria):
    """
    Custom StoppingCriteria that checks an external flag (`_stop_generation`).
    """
    def __init__(self, outer_instance):
        # Store a reference to the instance that holds the flag
        # Use a weakref if you're concerned about circular references,
        # but a direct reference is usually fine here.
        self.outer_instance = outer_instance

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check the flag on the outer instance at each generation step
        if self.outer_instance._stop_generation:
            if self.outer_instance.config.debug: # Optional debug log
                ASCIIColors.warning("Stopping generation internally via StoppingCriteria.")
            return True # Signal to stop generation
        return False # Signal to continue generation

class HuggingFaceLocal(LLMBinding):
    """
    Binding class for running local Hugging Face models using the Transformers library.
    Supports text and vision models, including chat template application.
    """
    def __init__(self,
                 config: LOLLMSConfig,
                 lollms_paths: LollmsPaths = None,
                 installation_option: InstallOption = InstallOption.INSTALL_IF_NECESSARY,
                 lollmsCom: Optional[LoLLMsCom] = None) -> None:
        """
        Initialize the Binding.
        """
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        self.lollms_paths = lollms_paths
        self.downloads_path = self.lollms_paths.personal_outputs_path / "downloads"
        self.downloads_path.mkdir(parents=True, exist_ok=True) # Ensure download path exists
        self.lollmsCom = lollmsCom

        # Configuration definition using ConfigTemplate
        binding_config_template = ConfigTemplate([
            # --- Model Loading ---
            {"name":"device", "type":"str", "value":"auto", "options":["auto", "cpu", "cuda", "mps"], "help":"Device to use for computation (auto detects best available: CUDA > MPS > CPU). 'auto' with accelerate enables CPU/GPU layer splitting if needed."},
            {"name":"quantization_bits", "type":"str", "value":"None", "options":["None", "4bits", "8bits"], "help":"Load model quantized in 4-bit or 8-bit. Requires CUDA and bitsandbytes. (-1 for no quantization)."},
            {"name":"use_flash_attention_2", "type":"bool", "value":False, "help":"Enable Flash Attention 2 for faster inference on compatible GPUs (requires specific hardware and torch version)."},
            {"name":"trust_remote_code", "type":"bool", "value":False, "help":"Allow executing custom code from the model's repository. Use with caution from trusted sources only."},
            {"name":"transformers_offline", "type":"bool", "value":True, "help":"Run transformers in offline mode (no internet connection needed after download)."},

            # --- Context & Generation ---
            {"name":"auto_infer_ctx_size", "type":"bool", "value":True, "help":"Automatically infer context size from the model's configuration file."},
            {"name":"ctx_size", "type":"int", "value":4096, "min": 512, "help":"Context size for the model. Used if 'auto_infer_ctx_size' is disabled or detection fails."},
            {"name":"max_n_predict", "type":"int", "value":1024, "min": 64, "help":"Maximum number of tokens to generate per response. Will be capped by the effective context size."},
            {"name":"seed", "type":"int", "value":-1, "help":"Random seed for generation (-1 for random)."},
            {"name":"apply_chat_template", "type":"bool", "value":True, "help":"Apply the model's chat template if available. Parses discussion format."},

            # --- Model Discovery ---
            {"name":"favorite_providers", "type":"str", "value":"microsoft,nvidia,mistralai,deepseek-ai,meta-llama,unsloth,ParisNeo,Bartowski", "help":"List of your favorite providers. Empty list for anyone"},
            {"name":"hub_fetch_limit", "type":"int", "value":5000, "min": 10, "max": 5000000, "help":"Maximum number of models to fetch from Hugging Face Hub for the 'available models' list."},
            {"name":"model_sorting", "type":"str", "value":"trending_score", "options": ["trending_score","created_at", "last_modified", "downloads", "likes "],"help":"Sorting criteria for models fetched from Hugging Face Hub."}, # Corrected help text
        ])
        # Default values for the configuration
        binding_config_defaults = BaseConfig(config={
            "device": "auto",
            "quantization_bits": "None",
            "use_flash_attention_2": False,
            "trust_remote_code": False,
            "transformers_offline": True,
            "auto_infer_ctx_size": True,
            "ctx_size": 4096, # Fallback value
            "max_n_predict": 1024,
            "seed": -1,
            "apply_chat_template": True,
            "favorite_providers": "microsoft,nvidia,mistralai,deepseek-ai,meta-llama,unsloth,ParisNeo,Bartowski", # Added default
            "hub_fetch_limit": 5000, # Increased default
            "model_sorting": "trending_score", # Added default
        })

        binding_config = TypedConfig(
            binding_config_template,
            binding_config_defaults
        )
        super().__init__(
            Path(__file__).parent,
            lollms_paths,
            config,
            binding_config,
            installation_option,
            supported_file_extensions=[], # Updated in build_model
            lollmsCom=lollmsCom
        )
        # Default binding type, might change in build_model
        self.binding_type = BindingType.TEXT_ONLY
        self._stop_generation = False
        self.generation_thread = None
        
        # Placeholders
        self.model = None
        self.tokenizer = None # For text models or VLM text part
        self.processor = None # For VLMs (handles text+image)
        self.device = None
        self.generation_thread: Optional[Thread] = None
        self._stop_generation = False

        # Apply offline mode setting on init
        self._apply_offline_mode()


    def _apply_offline_mode(self):
        """ Sets the TRANSFORMERS_OFFLINE environment variable based on config. """
        offline_mode = self.binding_config.config.get("transformers_offline", True)
        os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline_mode else "0"
        ASCIIColors.info(f"Transformers offline mode: {'Enabled' if offline_mode else 'Disabled'}")


    def settings_updated(self) -> None:
        """Callback triggered when binding settings are updated in the UI."""
        self._apply_offline_mode() # Re-apply offline setting
        # Rebuild if model changes or certain critical settings are updated
        # Compare current model name with potentially updated one in config
        # No need to rebuild if only non-critical settings like hub_fetch_limit change
        # Ideally, LoLLMs core would tell us *what* changed, but for now, rebuild if model changes.
        # Assuming the core updates self.config.model_name before calling this
        # if self.model is None or self.config.model_name != self._loaded_model_name: # Need to track loaded name
        # For simplicity, let's rebuild if any setting relevant to loading changes
        self.build_model(self.config.model_name) # Rebuild with the current model name

    def stop_generation(self):
        """Sets the stop flag and waits for the generation thread to finish."""
        self._stop_generation = True
        if self.generation_thread and self.generation_thread.is_alive():
            try:
                # The StoppingCriteria should handle the actual stop.
                # Joining ensures we wait for the thread to exit cleanly.
                self.generation_thread.join()
                if self.config.debug: ASCIIColors.info("Generation thread joined successfully.")
            except Exception as e:
                self.error(f"Error joining generation thread: {e}")
                trace_exception(e)
        self.generation_thread = None
        # It's often good practice to reset the flag *before* the next generation starts,
        # which is already handled in the generate method's thread setup.

    def get_local_hf_model_path(self, model_name: str) -> Optional[Path]:
        """Resolves the full path to a local Hugging Face model directory."""
        if not model_name:
            return None
        # Handle potential relative paths or just folder names
        if '/' in model_name or '\\' in model_name:
             # Assumes path is like 'author/model' or potentially a full path fragment
             model_full_path = self.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR / model_name
        else:
             # Assume it's just the model folder name directly under HF_LOCAL_MODELS_DIR
             model_full_path = self.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR / model_name

        # Normalize the path
        model_full_path = model_full_path.resolve()
        return model_full_path


    def build_model(self, model_name: Optional[str] = None) -> LLMBinding:
        """
        Loads the specified Hugging Face model and tokenizer/processor.
        Handles unloading of previous models and resource cleanup.
        Determines if the model is multimodal and attempts to infer context size.
        """
        super().build_model(model_name) # Sets self.config.model_name

        current_model_name = self.config.model_name
        if not current_model_name:
            self.error("No model selected in LoLLMs configuration.")
             # Ensure cleanup even if no new model is selected
            if self.model is not None: self._unload_model()
            return self

        model_full_path = self.get_local_hf_model_path(current_model_name)

        if not model_full_path or not model_full_path.exists() or not model_full_path.is_dir():
            self.error(f"Model folder not found: {model_full_path}")
            self.error(f"Please ensure the model '{current_model_name}' is downloaded correctly into the '{self.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR}' directory.")
            # Ensure cleanup if the target model folder is invalid
            if self.model is not None: self._unload_model()
            if self.lollmsCom: self.lollmsCom.InfoMessage(f"HuggingFaceLocal Error: Model folder '{current_model_name}' not found.")
            return self

        # --- Unload Previous Model ---
        # Check if a model is currently loaded *before* trying to load the new one
        if self.model is not None:
            self._unload_model() # Use helper function for clarity

        # --- Start Loading New Model ---
        ASCIIColors.info(f"Loading model: {current_model_name} from {model_full_path}")
        self.ShowBlockingMessage(f"Loading {current_model_name}...\nPlease wait.")

        try:
            # --- Determine Device ---
            requested_device = self.binding_config.config.get("device", "auto")
            if requested_device == "auto":
                # Use get_torch_device to find the best *single* device for potential offloading/CPU parts
                # The actual model placement is primarily handled by device_map="auto" below
                self.device = get_torch_device()
                ASCIIColors.info(f"Auto-detected primary device: {self.device}. Using device_map='auto' for potential multi-device loading.")
                device_map_strategy: Union[str, Dict] = "auto" # Let accelerate handle splitting
            elif requested_device in ["cuda", "mps", "cpu"]:
                self.device = requested_device
                # If a specific device is forced, we might still use 'auto' map for large models,
                # or force everything to that device if possible. Let's stick with 'auto' map
                # as it's generally more robust for potentially large models.
                # User forcing 'cpu' will likely result in 'auto' placing everything on CPU anyway.
                device_map_strategy = "auto"
                # device_map_strategy = self.device # Alternative: Force to the single device (might OOM)
                ASCIIColors.info(f"Using configured device: {self.device} with device_map='auto'.")
            else:
                self.warning(f"Invalid device '{requested_device}' requested. Falling back to 'auto'.")
                self.device = get_torch_device()
                device_map_strategy = "auto"

            if "cuda" not in self.device and self.binding_config.quantization_bits in ["4bits", "8bits"]:
                self.warning("Quantization requires CUDA. Disabling quantization.")
                self.binding_config.config["quantization_bits"] = "None" # Update runtime config

            # --- Load Model Config First ---
            ASCIIColors.info(f"Loading config from: {model_full_path}")
            trust_code = self.binding_config.config.get("trust_remote_code", False)
            model_config = AutoConfig.from_pretrained(model_full_path, trust_remote_code=trust_code)

            # --- Determine Model Class and Processor/Tokenizer ---
            ModelClass = AutoModelForCausalLM; ProcessorTokenizerClass = AutoTokenizer; is_vision_model = False
            model_type_str = getattr(model_config, "model_type", "").lower()
            architectures = [arch.lower() for arch in getattr(model_config, "architectures", [])]
            selected_class_key = "default"
            for key, (model_cls, proc_tok_cls) in KNOWN_MODEL_CLASSES.items():
                if key != "default" and (model_type_str == key or any(key in arch for arch in architectures)):
                    selected_class_key = key; ModelClass = model_cls; ProcessorTokenizerClass = proc_tok_cls
                    is_vision_model = ProcessorTokenizerClass == AutoProcessor # Assume VLM if Processor is needed
                    ASCIIColors.info(f"Detected matching model type: {key}")
                    break

            if is_vision_model:
                 self.binding_type = BindingType.TEXT_IMAGE
                 self.supported_file_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
                 ASCIIColors.info(f"Loading as Vision Model using {ModelClass.__name__} and {ProcessorTokenizerClass.__name__}")
            else:
                 self.binding_type = BindingType.TEXT_ONLY
                 self.supported_file_extensions = []
                 ASCIIColors.info(f"Loading as Text Model using {ModelClass.__name__} and {ProcessorTokenizerClass.__name__}")


            # --- Prepare Loading Arguments ---
            kwargs: Dict[str, Any] = {
                "trust_remote_code": trust_code,
                "device_map": device_map_strategy  # Use the determined strategy ('auto' or specific device)
            }
            quantization_bits = self.binding_config.config.get("quantization_bits", "None")
            if quantization_bits in ["4bits", "8bits"] and torch.cuda.is_available(): # Check CUDA availability again
                try:
                    compute_dtype = torch.bfloat16 # Common compute type for 4-bit
                    torch_dtype = torch.bfloat16 # Load weights in bfloat16 for 4-bit
                    if quantization_bits == "8bits":
                        # 8-bit usually doesn't need a specific compute_dtype set this way, torch_dtype controls load type
                         torch_dtype = torch.float16 # Or keep None? Check BNB docs. Let's try float16

                    bnb_config = BitsAndBytesConfig(
                        load_in_8bit=(quantization_bits == "8bits"),
                        load_in_4bit=(quantization_bits == "4bits"),
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=compute_dtype if quantization_bits == "4bits" else None # Only for 4-bit
                    )
                    kwargs["quantization_config"] = bnb_config
                    kwargs["torch_dtype"] = torch_dtype # Set loading dtype
                    ASCIIColors.info(f"Applying {quantization_bits} quantization.")
                except Exception as bnb_ex:
                    self.error(f"Failed to create BitsAndBytesConfig: {bnb_ex}. Disabling quantization.")
                    trace_exception(bnb_ex)
                    self.binding_config.config["quantization_bits"] = "None" # Update runtime config
                    if "quantization_config" in kwargs: del kwargs["quantization_config"]
                    if "torch_dtype" in kwargs: del kwargs["torch_dtype"]

            # Set torch_dtype if not using quantization, based on primary device
            if "quantization_config" not in kwargs:
                if self.device == "cuda": kwargs["torch_dtype"] = torch.float16
                elif self.device == "mps": kwargs["torch_dtype"] = torch.float16 # MPS often benefits from float16
                else: kwargs["torch_dtype"] = torch.float32 # CPU default

            use_flash_attention = self.binding_config.config.get("use_flash_attention_2", False)
            if use_flash_attention and "cuda" in self.device:
                 try:
                     major, minor = map(int, transformers.__version__.split('.')[:2])
                     if major >= 4 and minor >= 34: # Check transformers version
                         kwargs["attn_implementation"] = "flash_attention_2"
                         # Note: torch_dtype might need to be float16 or bfloat16 for FA2
                         if kwargs.get("torch_dtype") == torch.float32:
                              kwargs["torch_dtype"] = torch.float16 # Prefer float16 if not set
                              ASCIIColors.warning("Flash Attention 2 typically requires float16/bfloat16. Setting torch_dtype to float16.")
                         ASCIIColors.info("Attempting Flash Attention 2 implementation.")
                     else:
                         ASCIIColors.warning("Transformers version might be older than 4.34. Sticking to default attention mechanism.")
                 except Exception as fa_ex:
                     ASCIIColors.warning(f"Couldn't check/apply Flash Attention 2 setting: {fa_ex}")


            # --- Load Processor or Tokenizer ---
            ASCIIColors.info(f"Loading {ProcessorTokenizerClass.__name__} from: {model_full_path}")
            processor_tokenizer_instance = ProcessorTokenizerClass.from_pretrained(model_full_path, trust_remote_code=trust_code)
            if is_vision_model:
                self.processor = processor_tokenizer_instance
                self.tokenizer = getattr(self.processor, 'tokenizer', None)
                if not self.tokenizer:
                    self.warning("Could not find 'tokenizer' attribute on the processor. Text tokenization might rely solely on the processor.")
                    if callable(getattr(self.processor, "encode", None)) and callable(getattr(self.processor, "decode", None)):
                        self.tokenizer = self.processor
                        ASCIIColors.info("Using the processor itself as a fallback tokenizer.")
                    else:
                        self.warning("Processor cannot be used as a fallback tokenizer (missing encode/decode).")
            else:
                self.tokenizer = processor_tokenizer_instance; self.processor = None

            # --- Handle Missing Pad Token ---
            tokenizer_to_check = self._get_tokenizer_or_processor()
            pad_token_to_add = None
            if tokenizer_to_check and getattr(tokenizer_to_check, 'pad_token_id', None) is None:
                 pad_token = getattr(tokenizer_to_check,'pad_token',None)
                 if pad_token is None:
                     eos_token_id = getattr(tokenizer_to_check, 'eos_token_id', None)
                     eos_token = getattr(tokenizer_to_check, 'eos_token', None)
                     if eos_token_id is not None and eos_token is not None:
                        try:
                            tokenizer_to_check.pad_token_id = eos_token_id
                            tokenizer_to_check.pad_token = eos_token
                            ASCIIColors.warning("Tokenizer/Processor missing pad_token, setting to eos_token.")
                        except Exception as pad_ex:
                            ASCIIColors.warning(f"Could not set pad_token to eos_token: {pad_ex}")
                     else:
                        # Add a default pad token if both pad and eos are missing
                        pad_token_to_add = '[PAD]'
                        if pad_token_to_add not in tokenizer_to_check.get_vocab():
                            try:
                                num_added = tokenizer_to_check.add_special_tokens({'pad_token': pad_token_to_add})
                                if num_added > 0:
                                    ASCIIColors.warning(f"Added special token '{pad_token_to_add}' as pad_token.")
                                    # Resize model embeddings later after model load
                                else:
                                    ASCIIColors.warning(f"Tried adding '{pad_token_to_add}', but it might already exist without being set.")
                                # Explicitly set after adding
                                tokenizer_to_check.pad_token = pad_token_to_add
                            except Exception as add_tok_ex:
                                ASCIIColors.error(f"Failed to add or set pad token '{pad_token_to_add}': {add_tok_ex}")
                        else:
                            try:
                                tokenizer_to_check.pad_token = pad_token_to_add
                                ASCIIColors.warning(f"'{pad_token_to_add}' token exists but wasn't set as pad token. Setting it now.")
                            except Exception as pad_ex:
                                ASCIIColors.warning(f"Could not set existing token '{pad_token_to_add}' as pad_token: {pad_ex}")

                     # Final check if pad_token_id got set
                     if getattr(tokenizer_to_check, 'pad_token_id', None) is None:
                         self.warning("Could not determine or set a pad_token_id. Generation might fail for batching/padding.")

            # --- Load Model ---
            self.info(f"Loading model using {ModelClass.__name__} from: {model_full_path} with config: {kwargs}")
            load_start_time = perf_counter()
            self.model = ModelClass.from_pretrained(model_full_path, **kwargs)
            self.info(f"Model loaded in {perf_counter() - load_start_time:.2f} seconds.")

            # Report device map if using 'auto'
            if device_map_strategy == "auto" and hasattr(self.model, 'hf_device_map'):
                self.info(f"Model device map (accelerate): {self.model.hf_device_map}")
            elif device_map_strategy != "auto":
                 self.info(f"Model loaded on specified device: {device_map_strategy}")


            # --- Resize Embeddings if Pad Token was Added ---
            if pad_token_to_add and hasattr(self.model, 'resize_token_embeddings') and tokenizer_to_check:
                 current_vocab_size = getattr(tokenizer_to_check, 'vocab_size', len(tokenizer_to_check))
                 model_embedding_size = self.model.get_input_embeddings().weight.shape[0]
                 if model_embedding_size < current_vocab_size:
                     self.info(f"Resizing model token embeddings from {model_embedding_size} to match tokenizer size: {current_vocab_size}")
                     self.model.resize_token_embeddings(current_vocab_size)
                     # Re-check/set pad_token_id after resize, crucial if it wasn't set before
                     if getattr(tokenizer_to_check, 'pad_token_id', None) is None and tokenizer_to_check.pad_token:
                          pad_id = tokenizer_to_check.convert_tokens_to_ids(tokenizer_to_check.pad_token)
                          if isinstance(pad_id, int):
                              try:
                                 tokenizer_to_check.pad_token_id = pad_id
                                 ASCIIColors.info(f"Set pad_token_id to {pad_id} after resizing.")
                              except Exception as pad_id_ex:
                                  ASCIIColors.warning(f"Failed to set pad_token_id after resize: {pad_id_ex}")
                 else:
                     self.info("Token embeddings size already matches tokenizer vocab size after potential token addition.")


            # --- Infer/Set Context Size ---
            effective_ctx_size = 4096 # Default fallback
            if self.binding_config.config.get("auto_infer_ctx_size", True):
                detected_ctx_size = None
                possible_keys = ['max_position_embeddings', 'n_positions', 'model_max_length', 'seq_length'] # Added model_max_length
                for key in possible_keys:
                    ctx_val = getattr(model_config, key, None)
                    if isinstance(ctx_val, int) and ctx_val > 0:
                        detected_ctx_size = ctx_val
                        ASCIIColors.info(f"Auto-detected context size ({key}): {detected_ctx_size}")
                        break
                # Fallback check on tokenizer/processor if config fails
                if not detected_ctx_size and tokenizer_to_check:
                     ctx_val = getattr(tokenizer_to_check, 'model_max_length', None)
                     if isinstance(ctx_val, int) and ctx_val > 512: # Avoid unrealistically small values
                         detected_ctx_size = ctx_val
                         ASCIIColors.info(f"Auto-detected context size (tokenizer.model_max_length): {detected_ctx_size}")

                if detected_ctx_size:
                    effective_ctx_size = detected_ctx_size
                else:
                    effective_ctx_size = self.binding_config.config.get("ctx_size", 4096)
                    ASCIIColors.warning(f"Could not auto-detect context size. Using configured/default: {effective_ctx_size}")
            else:
                 effective_ctx_size = self.binding_config.config.get("ctx_size", 4096)
                 ASCIIColors.info(f"Using manually configured context size: {effective_ctx_size}")

            self.config.ctx_size = effective_ctx_size
            # Ensure the binding's config reflects the potentially updated value
            self.binding_config.config["ctx_size"] = effective_ctx_size


            # --- Validate and Set Max Prediction Tokens ---
            configured_max_predict = self.binding_config.config.get("max_n_predict", 1024)
            effective_max_predict = configured_max_predict
            # Ensure max_predict doesn't exceed context size minus a buffer (e.g., 10 tokens for safety)
            buffer = 10
            if effective_max_predict >= effective_ctx_size - buffer:
                 capped_predict = max(64, effective_ctx_size - buffer) # Keep a minimum generation capability
                 self.warning(f"Configured max_n_predict ({effective_max_predict}) too close to effective_ctx_size ({effective_ctx_size}). Capping to {capped_predict}.")
                 effective_max_predict = capped_predict
            elif effective_max_predict < 64: # Ensure a reasonable minimum
                self.warning(f"Configured max_n_predict ({effective_max_predict}) is very low. Setting to minimum 64.")
                effective_max_predict = 64

            self.config.max_n_predict = effective_max_predict
             # Ensure the binding's config reflects the potentially updated value
            self.binding_config.config["max_n_predict"] = effective_max_predict


            # --- Check if Chat Template Exists ---
            tokenizer_or_processor = self._get_tokenizer_or_processor()
            if tokenizer_or_processor and hasattr(tokenizer_or_processor, 'chat_template') and tokenizer_or_processor.chat_template:
                 ASCIIColors.info("Model has a chat template defined.")
                 self.binding_config.config["apply_chat_template"] = True # Ensure it's enabled if found
            else:
                 ASCIIColors.warning("Model does not have a chat template defined in its tokenizer/processor config. Will use raw prompt input.")
                 self.binding_config.config["apply_chat_template"] = False # Disable if not found

            ASCIIColors.success(f"Model {current_model_name} loaded successfully.")
            ASCIIColors.success(f"Effective Ctx: {self.config.ctx_size}, Max Gen: {self.config.max_n_predict}. Type: {self.binding_type.name}")
            self.HideBlockingMessage()

        except ImportError as e:
            self.error(f"Import error while loading {current_model_name}: {e}")
            self.error("Ensure required libraries are installed: torch, transformers, accelerate, bitsandbytes, huggingface_hub, Pillow, requests")
            trace_exception(e); self._unload_model(); self.HideBlockingMessage() # Ensure cleanup on failure
        except Exception as e:
            self.error(f"Failed to load model {current_model_name}: {e}")
            trace_exception(e); self._unload_model(); self.HideBlockingMessage() # Ensure cleanup on failure
            if "out of memory" in str(e).lower(): self.error("CUDA OOM Error."); self.lollmsCom.InfoMessage("HuggingFaceLocal Error: CUDA Out of Memory. Try lower quantization or smaller model.") if self.lollmsCom else None
            elif "requires the pytorch library" in str(e).lower() and "but it was not found" in str(e).lower(): self.error("PyTorch not found error during loading.")
            elif self.lollmsCom: self.lollmsCom.InfoMessage(f"HuggingFaceLocal Error: Failed to load model.\n{e}")

        return self

    def _unload_model(self):
        """ Safely unloads the model and associated components, freeing memory. """
        if self.model is not None:
            self.info("Unloading previous model from memory...")
            try:
                del self.model
            except Exception as ex:
                self.warning(f"Exception during model deletion: {ex}")
            self.model = None
        if self.tokenizer is not None:
            try:
                del self.tokenizer
            except Exception as ex:
                self.warning(f"Exception during tokenizer deletion: {ex}")
            self.tokenizer = None
        if self.processor is not None:
            try:
                del self.processor
            except Exception as ex:
                self.warning(f"Exception during processor deletion: {ex}")
            self.processor = None

        # Explicitly collect garbage and clear CUDA cache if available
        self.info("Running garbage collection and clearing CUDA cache (if applicable)...")
        AdvancedGarbageCollector.collect() # Assumes this calls gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.info("CUDA cache cleared.")
        else:
            self.info("CUDA not available, skipping cache clearing.")

        # Reset binding type and supported extensions
        self.binding_type = BindingType.TEXT_ONLY
        self.supported_file_extensions = []
        self.info("Previous model unloaded and resources potentially freed.")


    def install(self) -> None:
        """Installs necessary Python packages using pipmaster."""
        super().install()
        self.ShowBlockingMessage("Installing Hugging Face Transformers requirements...")
        try:
            import pipmaster as pm
            pm.install_multiple(["torch","torchvision","torchaudio"], "https://download.pytorch.org/whl/cu124", force_reinstall=True)

            # Core requirements
            pm.install_multiple(["transformers", "accelerate", "sentencepiece", "huggingface_hub", "Pillow", "requests"], force_reinstall=True)
            # Optional but highly recommended for features
            pm.install_multiple(["bitsandbytes"])
            self.HideBlockingMessage()
            ASCIIColors.success("Hugging Face requirements installation process finished.")

            ASCIIColors.info("----------------------\nAttention:\n----------------------")
            ASCIIColors.info("This binding requires manual download of Hugging Face models.")
            ASCIIColors.info(f"1. Find models on Hugging Face Hub (https://huggingface.co/models). Filter for 'transformers' compatible models.")
            ASCIIColors.info(f"2. Use tools like 'git lfs clone' or 'huggingface-cli download model_id --local-dir path/to/models/{HF_LOCAL_MODELS_DIR}/model_id' to download.")
            ASCIIColors.info(f"3. Ensure the final model folder structure is like: {self.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR}/<author_name>/<model_name>")
            ASCIIColors.info(f"   Example: models/{HF_LOCAL_MODELS_DIR}/google/gemma-1.1-2b-it")
            ASCIIColors.info(f"   Example: models/{HF_LOCAL_MODELS_DIR}/llava-hf/llava-1.5-7b-hf")
            ASCIIColors.info(f"4. Select the model folder name (e.g., 'google/gemma-1.1-2b-it') in LoLLMs settings.")
        except ImportError: self.HideBlockingMessage(); self.error("pipmaster not found. Please install it manually (`pip install pipmaster`) and retry installation.")
        except Exception as e: self.error(f"Installation failed: {e}"); trace_exception(e); self.HideBlockingMessage()


    def _get_tokenizer_or_processor(self) -> Optional[Union[AutoTokenizer, AutoProcessor]]:
        """Returns the processor if available, otherwise the tokenizer."""
        # Prefer processor if it exists, as it often wraps the tokenizer for VLMs
        if self.processor:
            return self.processor
        elif self.tokenizer:
            return self.tokenizer
        else:
            return None


    def tokenize(self, prompt: str) -> List[int]:
        """Tokenizes the given prompt."""
        tokenizer_to_use = self._get_tokenizer_or_processor()
        if tokenizer_to_use:
            try:
                # Use encode method which is common to both Tokenizer and Processor (usually)
                return tokenizer_to_use.encode(prompt)
            except Exception as e: self.error(f"Tokenization error: {e}"); trace_exception(e); return []
        else: self.error("Tokenizer or Processor not loaded."); return []


    def detokenize(self, tokens_list: List[int]) -> str:
        """Detokenizes the given list of tokens."""
        tokenizer_to_use = self._get_tokenizer_or_processor()
        if tokenizer_to_use:
            try:
                # Use decode method
                return tokenizer_to_use.decode(tokens_list, skip_special_tokens=True)
            except Exception as e: self.error(f"Detokenization error: {e}"); trace_exception(e); return ""
        else: self.error("Tokenizer or Processor not loaded."); return ""


    def _prepare_common_generation_kwargs(self, n_predict: int, gpt_params: dict) -> dict:
        """ Helper to prepare common generation arguments using effective n_predict. """
        # n_predict here is already the effective value (potentially capped by build_model)
        effective_max_n_predict = n_predict

        # Start with LoLLMs-provided parameters mapped to HF names
        default_params = {
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            # Add other potential default lollms params here if needed
            # e.g., repetition_penalty if self.config has it
        }
        # Override defaults with any specific gpt_params passed for this call
        gen_kwargs = {**default_params, **gpt_params}

        tokenizer_to_use = self._get_tokenizer_or_processor()
        if not tokenizer_to_use:
            raise RuntimeError("Tokenizer/Processor not available for generation config.")

        # --- Build final kwargs for transformers generate() ---
        final_gen_kwargs = {
            "max_new_tokens": effective_max_n_predict,
            "pad_token_id": getattr(tokenizer_to_use, 'pad_token_id', None),
            # Use eos_token_id from tokenizer/processor if available
            "eos_token_id": getattr(tokenizer_to_use, 'eos_token_id', None),
             # Set stopping criteria using eos_token_id if available
             # Note: some models might need multiple EOS tokens. Handle manually if needed.
            "do_sample": True # Default to sampling
        }

        # Handle specific parameter conversions/validations
        temp = float(gen_kwargs.get('temperature', 0.8)) # Default temp slightly higher
        if temp <= 0.01: # Consider temps close to 0 as greedy
            final_gen_kwargs["do_sample"] = False
            # For greedy, sometimes setting top_k=1 is more reliable than low temp
            final_gen_kwargs["temperature"] = 1.0 # Set temp to 1 for greedy (often ignored but good practice)
            final_gen_kwargs["top_k"] = 1
            final_gen_kwargs["top_p"] = 1.0 # Usually ignored when do_sample=False
        else:
            final_gen_kwargs["do_sample"] = True
            final_gen_kwargs["temperature"] = max(0.01, temp) # Ensure temp is slightly above 0
            top_p = float(gen_kwargs.get('top_p', 0.95)) # Common default top_p
            top_k = int(gen_kwargs.get('top_k', 50)) # Common default top_k
            # Validate and apply top_p and top_k only if sampling
            if 0.0 < top_p <= 1.0: final_gen_kwargs["top_p"] = top_p
            else: final_gen_kwargs["top_p"] = 0.95; self.warning(f"Invalid top_p {top_p}, using 0.95")
            if top_k > 0: final_gen_kwargs["top_k"] = top_k
            else: final_gen_kwargs["top_k"] = 50; self.warning(f"Invalid top_k {top_k}, using 50")

        # Add other parameters if present in gpt_params (e.g., repetition_penalty)
        if 'repetition_penalty' in gen_kwargs:
            rp = float(gen_kwargs['repetition_penalty'])
            if rp > 0: final_gen_kwargs['repetition_penalty'] = rp

        # --- Seed ---
        seed = int(self.binding_config.config.get("seed", -1)) # Ensure seed is int
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            ASCIIColors.info(f"Generation seed set to: {seed}")
        else:
             # Explicitly unset seed? No, just don't set it in kwargs. Transformers handles random seed internally if not specified.
             pass

        # --- Check for missing critical tokens ---
        if final_gen_kwargs["pad_token_id"] is None: self.warning("pad_token_id is None. Padding during generation might fail, especially with batching.")
        if final_gen_kwargs["eos_token_id"] is None: self.warning("eos_token_id is None. Model might not stop generating naturally and rely solely on max_new_tokens.")

        return final_gen_kwargs




    def _generation_thread_runner(self, **kwargs):
        """ Helper function to run model.generate in a thread and catch exceptions. """
        try:
            with torch.no_grad():
                # The stopping_criteria passed in kwargs will now be used by generate
                self.model.generate(**kwargs)
        except Exception as e:
            # Don't check _stop_generation here, as it might be a legitimate stop
            # Only log if it's an actual *unexpected* error
            if not self._stop_generation: # Log only if stop wasn't requested
                self.error(f"Error within generation thread: {e}")
                trace_exception(e)
                if self.lollmsCom: self.lollmsCom.notify_callback(f"Gen Thread Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            else:
                 if self.config.debug: ASCIIColors.info("Generation thread stopped as requested.") # Or potentially a different info log
        finally:
            # Ensure the streamer knows the generation is done, even if stopped early.
            # This might happen implicitly when generate exits, but doesn't hurt.
            # streamer = kwargs.get("streamer")
            # if streamer:
            #     streamer.end() # Often not needed as generate exiting does this
            pass # Clean exit


    def generate(self,
                 prompt: str,
                 n_predict: Optional[int] = None,
                 callback: Optional[Callable[[str, int], bool]] = None, # Adjusted callback sig
                 verbose: bool = False,
                 **gpt_params) -> str:
        """ Generates text using the loaded text-only model, applying chat template if enabled. """
        effective_n_predict = self._validate_n_predict(n_predict)
        tokenizer_or_processor = self._get_tokenizer_or_processor()
        if not self.model or not tokenizer_or_processor:
            self.error("Model or Tokenizer/Processor not loaded.")
            return "[Error: Model not ready]"

        self.stop_generation() # Stop any previous generation and clear thread/flag

        try:
            final_gen_kwargs = self._prepare_common_generation_kwargs(effective_n_predict, gpt_params)
            if self.config.debug or verbose: ASCIIColors.debug(f"Text Gen raw params: {gpt_params}")
            if self.config.debug or verbose: ASCIIColors.debug(f"Text Gen effective params: {final_gen_kwargs}")

            apply_template = self.binding_config.config.get("apply_chat_template", True)
            input_ids = None

            # --- Prepare Inputs (Apply Template if enabled) ---
            # (Keep your existing template application logic here)
            # ... (your code for applying template or using raw prompt) ...
            if apply_template and hasattr(tokenizer_or_processor, 'chat_template') and tokenizer_or_processor.chat_template:
                try:
                    messages = self.parse_lollms_discussion(prompt)
                    if not messages:
                        self.warning("Prompt parsing resulted in empty message list. Using raw prompt.")
                        input_ids = tokenizer_or_processor.encode(prompt, return_tensors="pt").to(self.model.device if hasattr(self.model,'device') else self.device)
                    else:
                        if self.config.debug or verbose: ASCIIColors.debug(f"Parsed messages for template: {messages}")
                        templated_inputs = tokenizer_or_processor.apply_chat_template(
                            messages, add_generation_prompt=True, return_tensors="pt"
                        )
                        input_ids = templated_inputs.to(self.model.device if hasattr(self.model,'device') else self.device)
                        if self.config.debug or verbose: ASCIIColors.debug(f"Formatted prompt via template (tokens): {input_ids.shape}")
                except Exception as template_ex:
                    self.error(f"Failed to apply chat template: {template_ex}. Falling back to raw prompt.")
                    trace_exception(template_ex)
                    input_ids = tokenizer_or_processor.encode(prompt, return_tensors="pt").to(self.model.device if hasattr(self.model,'device') else self.device)
            else:
                if apply_template: self.warning("Chat template application requested but no template found or disabled. Using raw prompt.")
                input_ids = tokenizer_or_processor.encode(prompt, return_tensors="pt").to(self.model.device if hasattr(self.model,'device') else self.device)
            # --- End Input Preparation ---


            input_token_count = input_ids.shape[1]
            if input_token_count >= self.config.ctx_size:
                raise ValueError(f"Prompt too long ({input_token_count} tokens vs context {self.config.ctx_size}). Please shorten your input or increase model context.")

            if input_token_count + final_gen_kwargs["max_new_tokens"] > self.config.ctx_size:
                 self.warning(f"Potential context overflow: Input ({input_token_count}) + Max New ({final_gen_kwargs['max_new_tokens']}) > Context ({self.config.ctx_size}). Output might be truncated.")
                 # Optional adjustment here if needed

        except ValueError as ve:
             self.error(f"Input Error: {ve}")
             if callback: callback(f"Input Error: {ve}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return f"[Input Error: {ve}]"
        except Exception as e:
            self.error(f"Input Preparation Error: {e}")
            trace_exception(e)
            if callback: callback(f"Input Prep Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return f"[Input Preparation Error: {e}]"

        streamer = TextIteratorStreamer(
            tokenizer_or_processor, skip_prompt=True, skip_special_tokens=True
        )

        # *** Add the custom stopping criteria ***
        stop_criterion = StopGenerationCriteria(self)
        # Check if user provided their own criteria list
        existing_criteria = final_gen_kwargs.pop('stopping_criteria', None) # Remove if accidentally prepared
        stopping_criteria_list = StoppingCriteriaList()
        if isinstance(existing_criteria, StoppingCriteriaList):
            stopping_criteria_list.extend(existing_criteria) # Keep user's criteria
        elif isinstance(existing_criteria, StoppingCriteria):
             stopping_criteria_list.append(existing_criteria) # Handle single criterion

        stopping_criteria_list.append(stop_criterion) # Add our custom stop checker


        generation_kwargs_for_thread = {
             "input_ids": input_ids,
             **final_gen_kwargs,
             "streamer": streamer,
             "stopping_criteria": stopping_criteria_list # Pass the combined list
        }

        output_buffer = ""
        start_time = perf_counter()

        try:
            # Reset stop flag *before* starting the thread
            self._stop_generation = False
            self.generation_thread = Thread(target=self._generation_thread_runner, kwargs=generation_kwargs_for_thread)
            self.generation_thread.start()
            if self.config.debug or verbose: ASCIIColors.info("Starting text generation stream...")

            for new_text in streamer:
                # No need to check self._stop_generation here explicitly for breaking the loop,
                # as the streamer will stop yielding when the underlying generate call stops.
                # However, checking it *can* provide a slightly faster exit from this loop
                # *if* the flag is set between streamer yields. It also handles the callback logic.
                if self._stop_generation:
                    ASCIIColors.warning("Stop generation requested (detected in main loop).")
                    break # Exit loop even if streamer hasn't technically finished

                if new_text:
                    output_buffer += new_text
                    if callback:
                        try:
                            continue_generating = callback(new_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK)
                            if continue_generating is False: # Check for explicit False
                                self._stop_generation = True # Signal the thread to stop
                                ASCIIColors.warning("Generation stopped by callback returning False.")
                                # Don't break immediately, let the StoppingCriteria handle it
                                # or let the next streamer iteration check the flag.
                                # Breaking here might miss the very last token chunk.
                                # Consider if you want immediate break vs graceful stop via criteria.
                                # For now, let the criteria handle the actual stop.
                                # If immediate stop is desired: break
                        except Exception as cb_ex:
                             self.error(f"Callback exception: {cb_ex}")
                             trace_exception(cb_ex)
                             self._stop_generation = True # Signal stop on callback error
                             break # Exit loop on error

            # Wait for thread to finish *after* the streamer is exhausted or loop broken
            if self.generation_thread and self.generation_thread.is_alive():
                 if self.config.debug or verbose: ASCIIColors.info("Waiting for generation thread to complete...")
                 self.generation_thread.join() # Wait for the thread runner to exit
                 if self.config.debug or verbose: ASCIIColors.info("Generation thread finished.")



        except Exception as e:
             self.error(f"Generation Error (Stream Loop): {e}")
             trace_exception(e)
             output_buffer += f"\n[Error during generation stream: {e}]"
             if callback: callback(f"Generation Stream Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             # Ensure stop is attempted if stream fails
             self.stop_generation() # Request stop and join
        finally:
            # Thread cleanup is now handled within stop_generation or naturally when joined
            self.generation_thread = None # Clear reference after join/stop
            end_time = perf_counter()
            total_time = end_time - start_time
            # Token counting might be slightly off if stopped early, but gives an idea
            num_tokens_generated = len(self.tokenize(output_buffer.replace("\n[STOPPED]",""))) # Exclude stop message
            tokens_per_sec = (num_tokens_generated / total_time) if total_time > 0 else 0
            if self.config.debug or verbose:
                ASCIIColors.info(f"Generation {'stopped' if self._stop_generation else 'finished'} in {total_time:.2f} seconds.")
                ASCIIColors.info(f"Approx tokens generated: {num_tokens_generated}, Tokens/sec: {tokens_per_sec:.2f}")


        return output_buffer

    def generate_with_images(self,
                             prompt: str,
                             images: List[str],
                             n_predict: Optional[int] = None,
                             callback: Optional[Callable[[str, int], bool]] = None, # Adjusted callback sig
                             verbose: bool = False,
                             **gpt_params) -> str:
        """ Generates text using prompt and images (multimodal), applying chat template if available. """
        # Validate n_predict
        effective_n_predict = self._validate_n_predict(n_predict)

        # Check if the loaded model is actually a vision model
        if self.binding_type != BindingType.TEXT_IMAGE or not self.processor:
            self.warning("generate_with_images called, but the current model is not a vision model or processor is missing.")
            # Fallback to text-only generation using the prompt, ignoring images
            self.warning("Ignoring images and falling back to text-only generation.")
            return self.generate(prompt, effective_n_predict, callback, verbose=verbose, **gpt_params)

        # Check for essential components
        if not self.model:
            self.error("Vision model not loaded.")
            return "[Error: Model not ready]"

        if not images:
            self.warning("No images provided to generate_with_images. Falling back to text-only generation.")
            return self.generate(prompt, effective_n_predict, callback, verbose=verbose, **gpt_params)

        # Stop any ongoing generation
        self.stop_generation() # Use helper method

        loaded_pil_images: List[Image.Image] = []
        try:
            # --- Prepare Generation Arguments ---
            final_gen_kwargs = self._prepare_common_generation_kwargs(effective_n_predict, gpt_params)
            if self.config.debug or verbose: ASCIIColors.debug(f"Vision Gen raw params: {gpt_params}")
            if self.config.debug or verbose: ASCIIColors.debug(f"Vision Gen effective params: {final_gen_kwargs}")

            # --- Load Images ---
            failed_images = []
            for img_path_or_url in images:
                pil_img = self._process_image_argument(img_path_or_url) # Handles download/load/convert
                if pil_img:
                    loaded_pil_images.append(pil_img)
                else:
                    failed_images.append(img_path_or_url)

            if not loaded_pil_images:
                # If all images failed, fallback to text generation
                self.error("Failed to load any valid images. Falling back to text generation.")
                return self.generate(prompt, effective_n_predict, callback, verbose=verbose, **gpt_params)

            if failed_images:
                self.warning(f"Skipped loading {len(failed_images)} invalid/missing images: {failed_images}")

            # --- Prepare Inputs (Apply Template if enabled) ---
            apply_template = self.binding_config.config.get("apply_chat_template", True)
            inputs = None # This will hold the final model input (e.g., dict with input_ids, pixel_values)

            if apply_template and hasattr(self.processor, 'chat_template') and self.processor.chat_template:
                try:
                    # Parse the prompt using LoLLMs format
                    messages = self.parse_lollms_discussion(prompt)
                    if not messages:
                         self.warning("Prompt parsing resulted in empty message list. Constructing basic user message.")
                         # Create a user message containing the prompt and images
                         content_list = [{"type": "text", "text": prompt}]
                         for pil_img in loaded_pil_images:
                             content_list.append({"type": "image", "image": pil_img}) # Pass PIL Image objects
                         messages = [{"role": "user", "content": content_list}]
                    else:
                        # Find the last user message to attach images and potentially text
                        last_user_msg_index = -1
                        for i in range(len(messages) - 1, -1, -1):
                            if messages[i]['role'] == 'user':
                                last_user_msg_index = i
                                break

                        if last_user_msg_index == -1:
                            # No user message found, add one at the end
                            self.warning("No user message found in parsed discussion. Appending images and prompt to a new user message.")
                            content_list = [{"type": "text", "text": prompt}] # Start with the full prompt text
                            for pil_img in loaded_pil_images: content_list.append({"type": "image", "image": pil_img})
                            messages.append({"role": "user", "content": content_list})
                        else:
                            # Attach images to the *last* user message found
                            last_content = messages[last_user_msg_index]['content']
                            # Ensure content is a list for multimodal input
                            if isinstance(last_content, str):
                                messages[last_user_msg_index]['content'] = [{"type": "text", "text": last_content}]
                            elif not isinstance(last_content, list):
                                self.warning(f"Unexpected content type ({type(last_content)}) in last user message. Wrapping it.")
                                messages[last_user_msg_index]['content'] = [{"type": "text", "text": str(last_content)}]

                            # Append images (pass PIL Image objects directly to template processor)
                            # Decide where to insert images - often at the beginning or end of user text
                            # Let's append them after the text for simplicity
                            for pil_img in loaded_pil_images:
                                messages[last_user_msg_index]['content'].append({"type": "image", "image": pil_img})
                            if self.config.debug or verbose: ASCIIColors.debug(f"Appended images to last user message (index {last_user_msg_index}).")

                    if self.config.debug or verbose: ASCIIColors.debug(f"Messages prepared for vision template: {messages}") # Careful logging PIL

                    # Apply template - Processor handles image conversion and tokenization
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True, # Format for model response
                        return_tensors="pt"
                    ).to(self.model.device if hasattr(self.model,'device') else self.device) # Send to model's device
                    if self.config.debug or verbose: ASCIIColors.info("Applied vision chat template successfully.")

                except Exception as template_ex:
                    self.error(f"Failed to apply vision chat template: {template_ex}. Falling back to processor direct call.")
                    trace_exception(template_ex)
                    # Fallback: Use processor directly with text and PIL images
                    inputs = self.processor(text=prompt, images=loaded_pil_images, return_tensors="pt").to(self.model.device if hasattr(self.model,'device') else self.device)

            else: # Template not available or disabled
                if apply_template:
                     self.warning("Vision chat template application requested but no template found/supported. Using processor direct call.")
                # Use processor directly with text and PIL images
                inputs = self.processor(text=prompt, images=loaded_pil_images, return_tensors="pt").to(self.model.device if hasattr(self.model,'device') else self.device)

            # --- Validate Input Length (if possible) ---
            input_token_count = 0
            if isinstance(inputs, dict) and 'input_ids' in inputs:
                 input_token_count = inputs['input_ids'].shape[-1]
            elif hasattr(inputs, 'input_ids'): # Handle cases where processor might return object with attributes
                 input_token_count = inputs.input_ids.shape[-1]

            if input_token_count > 0 :
                if input_token_count >= self.config.ctx_size:
                    self.error(f"Combined text/image input is too long ({input_token_count} tokens) for context size ({self.config.ctx_size}).")
                    raise ValueError(f"Input too long ({input_token_count} tokens vs context {self.config.ctx_size}).")
                elif input_token_count + final_gen_kwargs["max_new_tokens"] > self.config.ctx_size:
                    self.warning(f"Potential context overflow: Input ({input_token_count}) + Max New ({final_gen_kwargs['max_new_tokens']}) > Context ({self.config.ctx_size}). Output might be truncated.")
            else:
                 self.warning("Could not determine input token count from processor output.")


        except ValueError as ve: # Catch specific value errors like input too long
             self.error(f"Input Error (Vision): {ve}")
             if callback: callback(f"Input Error (Vision): {ve}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             # Close images even on error
             for img in loaded_pil_images: img.close()
             return f"[Input Error: {ve}]"
        except Exception as e:
            self.error(f"Input Preparation Error (Vision): {e}")
            trace_exception(e)
            if callback: callback(f"Input Prep Error (Vision): {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             # Close images even on error
            for img in loaded_pil_images: img.close()
            return f"[Input Preparation Error: {e}]"
        finally:
            # Close the PIL images after processing *unless* they are needed by the model directly (unlikely with HF)
            for img in loaded_pil_images:
                try: img.close()
                except Exception: pass # Ignore errors closing already closed images

        # --- Setup Streaming ---
        # Use the processor's tokenizer (or fallback) for decoding the stream
        stream_tokenizer = self.tokenizer if self.tokenizer else self.processor # Use processor if tokenizer not distinct/available
        if not hasattr(stream_tokenizer, 'decode'):
            self.error("Cannot stream output: No valid tokenizer/decoder found for vision model.")
            return "[Error: Decoder not available]"

        streamer = TextIteratorStreamer(
            stream_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Prepare arguments for the generation thread
        # Input 'inputs' should be the dict returned by processor or apply_chat_template
        if not isinstance(inputs, dict):
             # This shouldn't happen with standard HF processors/templates, but add a safeguard
             self.error("Inputs are not in the expected dictionary format for vision model generation.")
             return "[Error: Invalid input format for generation]"

        # Combine the processed inputs (input_ids, pixel_values etc.) with generation kwargs
        generation_kwargs_for_thread = {**inputs, **final_gen_kwargs, "streamer": streamer}
        output_buffer = ""
        start_time = perf_counter()

        try:
            # --- Start Generation Thread ---
            self.generation_thread = Thread(target=self._generation_thread_runner, kwargs=generation_kwargs_for_thread)
            self._stop_generation = False # Reset stop flag
            self.generation_thread.start()
            if self.config.debug or verbose: ASCIIColors.info("Starting vision model generation stream...")

            # --- Consume Stream ---
            for new_text in streamer:
                if self._stop_generation: # Check stop flag
                    ASCIIColors.warning("Stop generation requested.")
                    break
                if new_text:
                    output_buffer += new_text
                    if callback:
                        try:
                            continue_generating = callback(new_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK)
                            if continue_generating is False:
                                self._stop_generation = True
                                ASCIIColors.warning("Generation stopped by callback returning False.")
                                break
                        except Exception as cb_ex:
                             self.error(f"Callback exception: {cb_ex}")
                             trace_exception(cb_ex)
                             self._stop_generation = True # Stop if callback fails
                             break

            # Wait for thread to finish
            if self.config.debug or verbose: ASCIIColors.info("Vision generation stream finished.")

        except Exception as e:
             self.error(f"Generation Error (Vision Stream): {e}")
             trace_exception(e)
             output_buffer += f"\n[Error during vision generation stream: {e}]"
             if callback: callback(f"Gen Stream Error (Vision): {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             self.stop_generation() # Ensure cleanup
        finally:
            self.generation_thread = None # Clear thread reference
            # self._stop_generation = False # Reset handled by stop_generation() or next start
            end_time = perf_counter()
            total_time = end_time - start_time
            num_tokens_generated = len(self.tokenize(output_buffer)) # Approx token count
            tokens_per_sec = (num_tokens_generated / total_time) if total_time > 0 else 0
            if self.config.debug or verbose:
                ASCIIColors.info(f"Vision generation finished in {total_time:.2f} seconds.")
                ASCIIColors.info(f"Approx tokens generated: {num_tokens_generated}, Tokens/sec: {tokens_per_sec:.2f}")


        return output_buffer



    def stop_generation(self):
        """Requests the generation thread to stop."""
        self._stop_generation = True
        if self.generation_thread and self.generation_thread.is_alive():
            self.info("Requesting generation stop...")
            # No direct way to interrupt model.generate, relies on checking _stop_generation in streamer loop
            # We just join to wait for it to potentially finish or timeout
            self.generation_thread.join(timeout=1.0) # Wait briefly
            if self.generation_thread.is_alive():
                self.warning("Generation thread did not stop quickly. It might finish current step.")
            else:
                 self.info("Generation thread stopped.")
        self.generation_thread = None # Clear reference after joining/timeout
        self._stop_generation = False # Reset flag for next generation

    def _process_image_argument(self, image_path_or_url: str) -> Optional[Image.Image]:
        """ Loads an image from a local path or URL. Downloads URL if necessary. """
        try:
            pil_image = None
            if is_file_path(image_path_or_url):
                # Try absolute path first
                path_obj = Path(image_path_or_url)
                if path_obj.is_file() and path_obj.exists():
                    pil_image = Image.open(path_obj)
                else:
                    # Try relative paths (uploads, shared etc.)
                    found_path = find_first_available_file_path([
                        self.lollms_paths.personal_uploads_path / image_path_or_url,
                        self.lollms_paths.shared_uploads_path / image_path_or_url,
                        self.downloads_path / image_path_or_url, # Check downloads too
                    ])
                    if found_path and found_path.is_file():
                        pil_image = Image.open(found_path)
                    else:
                        self.warning(f"Local image file not found: {image_path_or_url}")
                        return None
            elif image_path_or_url.startswith(("http://", "https://")):
                # Handle URL
                self.info(f"Processing image URL: {image_path_or_url}")
                try:
                    # Generate a safe filename from URL hash for caching
                    filename = f"image_{hash(image_path_or_url)}.png" # Use hash for uniqueness, png assumed common
                    local_filepath = self.downloads_path / filename

                    if local_filepath.exists():
                        self.info(f"Using cached image: {local_filepath}")
                        pil_image = Image.open(local_filepath)
                    else:
                        # Download the image
                        self.info(f"Downloading image to {local_filepath}...")
                        # Use requests with stream=True for potentially large images
                        headers = {'User-Agent': 'Lollms_HuggingFaceLocal_Binding/1.0'} # Be polite
                        response = requests.get(image_path_or_url, stream=True, timeout=30, headers=headers)
                        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                        # Try to open directly from stream first (memory efficient)
                        try:
                             pil_image = Image.open(response.raw)
                             # Save to cache after successful open
                             pil_image.save(local_filepath) # Save in a common format like PNG
                             self.info(f"Image downloaded and cached successfully.")
                        except Exception as img_open_ex:
                            self.error(f"Failed to open image stream from {image_path_or_url}: {img_open_ex}. Trying download then open.")
                            # Fallback: download completely then open
                            local_filepath.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
                            with open(local_filepath, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            if local_filepath.exists():
                                pil_image = Image.open(local_filepath)
                                self.info(f"Image downloaded and opened from file.")
                            else:
                                raise IOError("Failed to save downloaded image file.")

                except requests.exceptions.RequestException as req_ex:
                    self.error(f"Error downloading image {image_path_or_url}: {req_ex}")
                    trace_exception(req_ex)
                    return None
                except Exception as dl_ex:
                     self.error(f"Error processing image URL {image_path_or_url}: {dl_ex}")
                     trace_exception(dl_ex)
                     return None
            else:
                self.warning(f"Invalid image format or path: {image_path_or_url}. Expecting local path or HTTP/HTTPS URL.")
                return None

            # Ensure image is RGB and return a copy (safer)
            if pil_image:
                 return pil_image.convert("RGB").copy() # Return a copy to avoid issues if original is closed elsewhere
            else:
                 return None # Should have returned earlier if failed

        except Exception as e:
            self.error(f"Failed to load or process image '{image_path_or_url}': {e}")
            trace_exception(e)
            return None


    def _validate_n_predict(self, n_predict: Optional[int]) -> int:
        """ Validates n_predict against configuration and context size. """
        # Use binding config's max_n_predict as the default/cap
        default_max_predict = self.binding_config.config.get("max_n_predict", 1024)

        if n_predict is None:
            n_predict = default_max_predict # Use configured default if not provided
        elif not isinstance(n_predict, int) or n_predict <= 0:
             self.warning(f"Invalid n_predict value ({n_predict}). Using default: {default_max_predict}")
             n_predict = default_max_predict

        # Ensure n_predict does not exceed the effective max_n_predict derived during build_model
        if n_predict > default_max_predict:
            self.warning(f"Requested n_predict ({n_predict}) exceeds effective maximum ({default_max_predict}). Capping.")
            n_predict = default_max_predict

        # Final sanity check against absolute minimum
        n_predict = max(1, n_predict) # Ensure at least 1 token prediction is requested

        return n_predict
    def _is_valid_model_dir(self, model_path: Path) -> bool:
        """
        Checks if a specific directory likely contains Hugging Face model files.
        It should NOT return true for an 'author' directory that only contains model subdirs.
        """
        if not model_path.is_dir():
            return False
        # Check for common indicators of a model directory *directly within this path*
        has_config = (model_path / "config.json").exists()
        # Check for *any* common model weight/definition files *directly within this path*
        # Using list comprehension and next() for efficiency (stops at first find)
        has_weights = next((True for pattern in [
            "*.safetensors",
            "*.bin",
            "*.pth",
            "*.gguf",
            "*.ggml",
            "tf_model.h5",
            "pytorch_model.bin",
            "flax_model.msgpack"
        ] if any(model_path.glob(pattern))), False) # Check if *any* files match patterns

        # A directory is considered a *model directory* if it has config OR weights directly inside.
        # An author directory usually only contains subdirectories.
        return has_config or has_weights

    def list_models(self) -> List[str]:
        """
        Lists locally available models. Handles:
        1. Models directly under HF_LOCAL_MODELS_DIR (e.g., hf_models/model-A).
        2. Models nested under an author folder (e.g., hf_models/authorB/model-C).
        3. Reference files (.reference) pointing to model directories elsewhere.
        """
        local_hf_root = self.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR
        if not local_hf_root.exists() or not local_hf_root.is_dir():
            self.info(f"Local HF models directory not found or is not a directory: {local_hf_root}")
            return []

        model_identifiers: Set[str] = set()

        try:
            for item in local_hf_root.iterdir():
                item_path = local_hf_root / item # Get the full path

                # 1. Handle reference files
                if item_path.is_file() and item.name.lower().endswith(REFERENCE_FILE_EXTENSION):
                    model_name_from_file = item.stem # e.g., "my-model" from "my-model.reference"
                    try:
                        with open(item_path, 'r', encoding='utf-8') as f:
                            target_path_str = f.read().strip()

                        if not target_path_str:
                            self.warning(f"Reference file '{item.name}' is empty. Skipping.")
                            continue

                        target_path = Path(target_path_str)

                        # IMPORTANT: Validate the *target* path using _is_valid_model_dir
                        if self._is_valid_model_dir(target_path):
                            # Use the reference file name as the identifier
                            model_identifier = model_name_from_file.replace("\\", "/")
                            model_identifiers.add(model_identifier)
                            self.info(f"Found valid reference '{model_identifier}' pointing to model dir: {target_path}")
                        else:
                            self.warning(f"Reference file '{item.name}' points to an invalid or non-model directory: {target_path}. Skipping.")

                    except Exception as e:
                         self.error(f"Error processing reference file {item.name}: {e}")
                         trace_exception(e) # Uncomment if you have this helper

                # 2. Handle directories
                elif item_path.is_dir():
                    # Check if 'item' itself is a model directory (e.g., hf_models/model-A)
                    if self._is_valid_model_dir(item_path):
                        # Yes, it's a model directory directly under local_hf_root
                        model_identifier = item.name.replace("\\", "/")
                        model_identifiers.add(model_identifier)
                        # If it's a direct model, we don't need to look inside it for more models
                        # (standard HF format doesn't nest models within models)
                    else:
                        # No, 'item' is not a model directory itself.
                        # Check if it contains subdirectories that *are* model directories
                        # (i.e., treat 'item' as a potential author folder like hf_models/TheBloke)
                        is_author_folder = False
                        for sub_item in item_path.iterdir():
                            sub_item_path = item_path / sub_item
                            # Check if the sub-item is a directory AND a valid model directory
                            if sub_item_path.is_dir() and self._is_valid_model_dir(sub_item_path):
                                # Found a nested model like hf_models/authorB/model-C
                                model_identifier = f"{item.name}/{sub_item.name}".replace("\\", "/")
                                model_identifiers.add(model_identifier)
                                is_author_folder = True # Mark that 'item' acted as an author folder

                        # Optional: Log if a directory was neither a model nor a valid author folder containing models
                        # if not self._is_valid_model_dir(item_path) and not is_author_folder:
                        #    self.info(f"Directory '{item.name}' is not a model and contains no valid model subdirectories.")


        except Exception as e:
            self.error(f"Error scanning models directory {local_hf_root}: {e}")
            trace_exception(e) # Uncomment if you have this helper
            return [] # Return empty list on major scanning error

        # Convert the set to a sorted list for consistent output
        sorted_models = sorted(list(model_identifiers))

        # Use ASCIIColors or self.info for final count
        try:
            ASCIIColors.info(f"Found {len(sorted_models)} potential local models (incl. references).")
        except NameError:
             self.info(f"Found {len(sorted_models)} potential local models (incl. references).")

        return sorted_models


    def get_available_models(self, app: Optional[LoLLMsCom] = None) -> List[dict]:
        """ Gets available models: local + fetched from Hub. """
        lollms_models = []
        local_model_names = set(self.list_models()) # Get names like 'author/model' or 'model'
        local_hf_root = self.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR
        binding_folder = binding_folder_name if binding_folder_name else binding_name.lower()
        default_icon = f"/bindings/{binding_folder}/logo.png"

        # Add Local Models
        self.info("Processing locally available models...")
        for model_id in sorted(list(local_model_names)):
            model_path = local_hf_root / model_id # Path is relative to local_hf_root
            author = "Unknown"
            model_name_only = model_id
            if '/' in model_id:
                parts = model_id.split('/', 1)
                author = parts[0]
                model_name_only = parts[1]

            model_info = {
                "category": "local",
                "datasets": "Unknown",
                "icon": default_icon,
                "last_commit_time": None,
                "license": "Unknown",
                "model_creator": author,
                "model_creator_link": f"https://huggingface.co/{author}" if author != "Unknown" else "https://huggingface.co/",
                "name": model_id, # Use the full ID 'author/model' or 'model' as the unique name/identifier
                "display_name": model_name_only, # For cleaner display perhaps
                "provider": "local",
                "rank": 5.0, # High rank for local models
                "type": "model",
                "variants": [{"name": model_id, "size": -1, "is_local": True}], # Size -1 means unknown/not fetched, mark local
                "description": "Locally available Hugging Face model."
            }
            # Try to get more info if possible
            try:
                if model_path.exists():
                    model_info["last_commit_time"] = datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

                config_path = model_path / "config.json"
                is_vision = False
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f: config_data = json.load(f)
                    model_type = config_data.get("model_type", "").lower()
                    architectures = [a.lower() for a in config_data.get("architectures", [])]
                    # Check against known VLM keys
                    is_vision = any(key in model_type or any(key in arch for arch in architectures)
                                    for key in KNOWN_MODEL_CLASSES if key != "default" and KNOWN_MODEL_CLASSES[key][1] == AutoProcessor)
                    # Also check pipeline tag if present
                    if not is_vision and 'pipeline_tag' in config_data:
                        pipeline = config_data['pipeline_tag'].lower()
                        if any(p in pipeline for p in ["image-to-text", "visual-question-answering","image-text-to-text"]):
                            is_vision = True

                    if is_vision:
                        model_info["description"] += " (Vision Capable)"
                        model_info["category"] = "local_vision" # Add a subcategory
                    else:
                         model_info["category"] = "local_text"

                    # Try to get license if available
                    model_info["license"] = config_data.get("license", "Unknown")

            except Exception as local_info_ex:
                 self.warning(f"Could not get extra info for local model {model_id}: {local_info_ex}")

            lollms_models.append(model_info)

        # Fetch Models from Hub
        filtered_hub_count = 0
        favorite_providers_list = [p.strip() for p in self.binding_config.favorite_providers.split(",") if p.strip()] # Clean list
        if not favorite_providers_list: favorite_providers_list = [None] # Search all if empty

        try:
            self.info("Fetching models from Hugging Face Hub...")
            
            api = HfApi()
            limit_per_provider = self.binding_config.config.get("hub_fetch_limit", 5000) // max(1, len(favorite_providers_list)) # Distribute limit
            limit_per_provider = max(10, limit_per_provider) # Ensure a minimum fetch
            sort_by = self.binding_config.config.get("model_sorting", "trending_score") # Use configured sorting, default 'trending'
            # Map sorting UI options to Hub API options
            sort_mapping = {
                "trending_score": "trending", # Assuming this is what 'trending_score' meant
                "created_at": "created_at",
                "last_modified": "lastModified",
                "downloads": "downloads",
                "likes": "likes"
            }
            hub_sort_key = sort_mapping.get(sort_by.strip(), "trending_score") # Fallback to trending


            # Fetch potentially relevant models: text-gen, text2text, image-to-text etc.
            hub_models_list = []
            # Prioritize pipelines most relevant to this binding
            relevant_tasks = ["text-generation", "image-text-to-text"]
            # Add general 'transformers' tag search as fallback? Maybe too broad.

            seen_ids = set(local_model_names) # Keep track of seen models (including local)

            for provider in favorite_providers_list:
                self.info(f"Fetching models for provider: {provider if provider else 'All Providers'} (sort: {hub_sort_key}, limit per type: {limit_per_provider})")
                provider_models = set() # Track models found for this provider to avoid duplicates across pipelines
                # Search by tasks first
                for task in relevant_tasks:
                    try:
                        model_iterator = api.list_models(
                            author=provider if provider else None,
                            task=task,
                            sort=hub_sort_key,
                            direction=-1, # Most popular/recent first
                            limit=limit_per_provider,
                            cardData=True # Fetch card data for license etc.
                            )
                        count = 0
                        models= list(model_iterator)
                        for model in models:
                             if model.modelId not in seen_ids:
                                 hub_models_list.append(model)
                                 seen_ids.add(model.modelId)
                                 provider_models.add(model.modelId)
                                 count += 1
                        if count > 0: self.info(f" Found {count} models for task '{task}'")

                    except Exception as pipe_ex:
                        ASCIIColors.warning(f"Could not fetch models for provider '{provider}' task '{task}': {pipe_ex}")

                 # Optional: Add a broader search for the provider if few results found?
                 # if not provider_models and provider:
                 #     # Maybe search just by author tag 'transformers'?
                 #     pass

            ASCIIColors.info(f"Fetched {len(hub_models_list)} unique potential models from Hub.")


            # Process fetched Hub models
            for model in hub_models_list:
                try:
                    model_id = model.modelId
                    # Skip if already local (double check)
                    if model_id in local_model_names: continue

                    # Filter out formats typically not handled by this binding directly
                    # Check model ID first, then tags for confirmation
                    skip_keywords = ["gguf", "ggml", "awq", "gptq", "exl2", "onnx"]
                    if any(kw in model_id.lower() for kw in skip_keywords):
                        # Check tags to see if it ALSO has 'transformers' - might be base + quant
                        if 'transformers' not in (model.tags or []):
                            continue # Skip if only quant tag and not transformers tag

                    # Filter based on tags (more reliable)
                    format_tags = {'gguf', 'ggml', 'awq', 'gptq', 'onnx', 'exl2'}
                    model_tags = set(model.tags or [])
                    # Skip if it has a quant/format tag AND doesn't have a core 'transformers' or 'pytorch' tag
                    is_quant_only = format_tags.intersection(model_tags) and not {'transformers', 'pytorch', 'jax', 'safetensors'}.intersection(model_tags)
                    if is_quant_only:
                        continue

                    # Determine category based on pipeline tag primarily
                    pipeline = model.pipeline_tag.lower() if model.pipeline_tag else ""
                    category = "hub_other"
                    if any(p in pipeline for p in ["image-to-text", "visual-question-answering", "image-text-to-text"]): category = "hub_vision"
                    elif any(p in pipeline for p in ["text-generation", "text2text-generation", "conversational", "summarization"]): category = "hub_text"
                    # Refine category based on tags if pipeline is missing/generic
                    elif 'text-generation' in model_tags or 'text2text-generation' in model_tags: category = "hub_text"
                    elif 'image-to-text' in model_tags: category = "hub_vision"


                    # Extract info from cardData if available
                    license_info = "Check card"
                    datasets_info = "Check card"
                    if model.cardData:
                        license_info = model.cardData.get('license', "Check card")
                        # Simplify common licenses
                        if isinstance(license_info, list): license_info = license_info[0] # Take first if list
                        if isinstance(license_info, str):
                            license_info=license_info.replace("apache-2.0","Apache 2.0").replace("mit","MIT") # Prettify common ones
                            if len(license_info)>20: license_info=license_info[:17]+"..." # Truncate long names

                        datasets_info = model.cardData.get('datasets', "Check card")
                        if isinstance(datasets_info, list): datasets_info = ", ".join(datasets_info) # Join if list
                        if isinstance(datasets_info, str) and len(datasets_info)>30: datasets_info = datasets_info[:27]+"..." # Truncate

                    # Build description string
                    description_parts = []
                    if model.downloads is not None: description_parts.append(f"Dl: {model.downloads:,}")
                    if model.likes is not None: description_parts.append(f"Likes: {model.likes:,}")
                    if model.lastModified: description_parts.append(f"Upd: {model.lastModified.split('T')[0]}")
                    description = ", ".join(description_parts)


                    author_hub = model.author or "Unknown"
                    model_name_only_hub = model_id.split('/')[-1] # Get last part as name

                    entry = {
                        "category": category,
                        "datasets": datasets_info,
                        "icon": default_icon,
                        "last_commit_time": model.lastModified,
                        "license": license_info,
                        "model_creator": author_hub,
                        "model_creator_link": f"https://huggingface.co/{author_hub}" if author_hub != "Unknown" else "https://huggingface.co/",
                        "name": model_id, # Full ID is the unique identifier
                        "display_name": model_name_only_hub, # For display
                        "provider": author_hub, # Indicate source provider/author
                        "rank": model.likes or model.downloads or 0, # Rank by likes or downloads
                        "type": "downloadable", # Mark as needing download
                        "description": description,
                        "link": f"https://huggingface.co/{model_id}",
                        "variants": [{"name": model_id, "size": -1, "is_local": False}], # Size unknown, mark not local
                    }
                    lollms_models.append(entry)
                    filtered_hub_count += 1
                except Exception as ex:
                    ASCIIColors.debug(f"Error processing Hub model {model.modelId}: {ex}")

            ASCIIColors.info(f"Added {filtered_hub_count} Hugging Face Hub models after filtering.")

        except ImportError: self.error("huggingface_hub library not found. Cannot fetch Hub models.")
        except ConnectionError: pass # Already warned about offline mode
        except Exception as e: self.error(f"Failed to fetch models from Hugging Face Hub: {e}"); trace_exception(e)

        # Add fallbacks only if Hub fetch failed or yielded zero results
        if filtered_hub_count == 0 and not self.binding_config.transformers_offline:
             self.warning("Hub fetch yielded no results. Adding fallback examples.")
             # Define fallbacks using the same structure
             fallback_models = [
                 {"category": "hub_text", "name": "google/gemma-1.1-2b-it", "display_name": "gemma-1.1-2b-it", "model_creator":"google", "description":"(Fallback) Google Gemma 1.1 2B IT", "icon": default_icon, "rank": 1500, "type":"downloadable", "provider":"google", "variants":[{"name":"google/gemma-1.1-2b-it", "size":-1, "is_local": False}], "link":"https://huggingface.co/google/gemma-1.1-2b-it", "license":"Gemma", "datasets": "Check card", "last_commit_time":None, "model_creator_link":"https://huggingface.co/google"},
                 {"category": "hub_text", "name": "meta-llama/Meta-Llama-3-8B-Instruct", "display_name": "Meta-Llama-3-8B-Instruct", "model_creator":"meta-llama", "description":"(Fallback) Meta Llama 3 8B Instruct", "icon": default_icon, "rank": 5000, "type":"downloadable", "provider":"meta-llama", "variants":[{"name":"meta-llama/Meta-Llama-3-8B-Instruct", "size":-1, "is_local": False}], "link":"https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct", "license":"Llama3", "datasets": "Check card", "last_commit_time":None, "model_creator_link":"https://huggingface.co/meta-llama"},
                 {"category": "hub_vision", "name": "google/paligemma-3b-mix-448", "display_name": "paligemma-3b-mix-448", "model_creator":"google", "description":"(Fallback) Google PaliGemma 3B Mix", "icon": default_icon, "rank": 1000, "type":"downloadable", "provider":"google", "variants":[{"name":"google/paligemma-3b-mix-448", "size":-1, "is_local": False}], "link":"https://huggingface.co/google/paligemma-3b-mix-448", "license":"Gemma", "datasets": "Check card", "last_commit_time":None, "model_creator_link":"https://huggingface.co/google"},
                 {"category": "hub_vision", "name": "llava-hf/llava-1.5-7b-hf", "display_name": "llava-1.5-7b-hf", "model_creator":"llava-hf", "description":"(Fallback) LLaVA 1.5 7B HF", "icon": default_icon, "rank": 2000, "type":"downloadable", "provider":"llava-hf", "variants":[{"name":"llava-hf/llava-1.5-7b-hf", "size":-1, "is_local": False}], "link":"https://huggingface.co/llava-hf/llava-1.5-7b-hf", "license":"Llama2", "datasets": "Check card", "last_commit_time":None, "model_creator_link":"https://huggingface.co/llava-hf"},
                ]
             added_fb = 0
             current_names = {m['name'] for m in lollms_models} # Get names already added (local + hub)
             for fm in fallback_models:
                 if fm["name"] not in current_names:
                     lollms_models.append(fm)
                     added_fb += 1
             if added_fb > 0: self.info(f"Added {added_fb} fallback examples.")

        # Sort models: Local first, then by rank (descending), then by name (ascending)
        lollms_models.sort(key=lambda x: (
             0 if x.get('provider') == 'local' else 1, # Local first
            -float(x.get('rank', 0) if isinstance(x.get('rank'), (int, float)) else 0), # Higher rank first (convert rank to float)
            x['name'] # Alphabetical by name as tie-breaker
        ))
        ASCIIColors.success(f"Prepared {len(lollms_models)} models for Lollms UI.")
        return lollms_models


# --- Main execution block for basic testing ---
if __name__ == "__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    # from lollms.app import LollmsApplication # Not needed directly for binding test
    from pathlib import Path
    from lollms.types import MSG_OPERATION_TYPE

    print("Initializing LoLLMs environment for HF Local testing...")
    # Use a temporary directory for testing to avoid interfering with user setup
    lollms_paths = LollmsPaths.find_paths(force_local=True, tool_prefix="test_hf_local_", create_dirs=True)
    # Ensure the models directory exists within the temp paths
    (lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"Using test paths: {lollms_paths.paths}")
    print(f"Expected models folder: {lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR}")

    config = LOLLMSConfig.autoload(lollms_paths)
    # Dummy LoLLMsCom for testing callbacks and messages
    class TestCom(LoLLMsCom):
        def notify_callback(self, chunk: str, msg_type: MSG_OPERATION_TYPE):
            # Simulate the behavior of the actual LoLLMsCom callback notification
             if hasattr(self, '_callback') and self._callback:
                 try:
                     self._callback(chunk, msg_type)
                 except Exception as e:
                     print(f"Error in test callback handler: {e}")
             else:
                 # Default print if no callback attached externally
                 prefix = f"[{msg_type.name}]"
                 if msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK: prefix = "" # No prefix for chunks
                 print(f"{prefix}{chunk}", end="" if msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK else "\n", flush=True)

        # Implement other LoLLMsCom methods used by the binding
        def InfoMessage(self, msg): print(f"\nInfo: {msg}")
        def WarningMessage(self, msg): print(f"\nWarning: {msg}")
        def ErrorMessage(self, msg): print(f"\nError: {msg}")
        def ExceptionMessage(self, msg, ex): print(f"\nException: {msg}\n{ex}") # Corrected signature
        def ShowBlockingMessage(self, msg): print(f"\nBlocking Msg: {msg}")
        def HideBlockingMessage(self): print("\nHide Blocking Msg")
        # Add a way to set the test callback
        def set_callback(self, callback_func): self._callback = callback_func

    lollms_app_com = TestCom()

    print("Creating HuggingFaceLocal binding instance...")
    # Set install option to avoid automatic installs during testing
    hf_binding = HuggingFaceLocal(config, lollms_paths, installation_option=InstallOption.NEVER_INSTALL, lollmsCom=lollms_app_com)

    # --- Test Installation Logic (optional, requires pipmaster) ---
    # print("\n--- Testing installation command ---")
    # hf_binding.install() # Uncomment to test installation flow

    # --- Test Listing ---
    print("\n--- Listing locally available models ---")
    # Create dummy model folders for testing list_models
    # IMPORTANT: For actual testing, download real models to the 'test_hf_local_paths/models/transformers' folder
    dummy_model_path1 = lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR / "dummy_author" / "dummy_text_model"
    dummy_model_path2 = lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR / "dummy_vision_model"
    dummy_model_path1.mkdir(parents=True, exist_ok=True)
    (dummy_model_path1 / "config.json").touch()
    (dummy_model_path1 / "model.safetensors").touch()
    dummy_model_path2.mkdir(parents=True, exist_ok=True)
    (dummy_model_path2 / "config.json").touch() # Simulate config for detection

    print(f"Check dummy path 1: {dummy_model_path1}")
    print(f"Check dummy path 2: {dummy_model_path2}")

    local_models = hf_binding.list_models()
    if local_models: print("Found models:\n" + "\n".join(f"- {m}" for m in local_models))
    else: print(f"No models found in {lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR}. Please download a model for load/gen tests.")

    print("\n--- Getting combined models list for UI (local + hub) ---")
    # Temporarily disable offline mode for hub fetch during test
    original_offline = hf_binding.binding_config.config.get("transformers_offline", True)
    hf_binding.binding_config.config["transformers_offline"] = False
    hf_binding._apply_offline_mode()

    available_models_ui = hf_binding.get_available_models()

    # Restore offline mode setting
    hf_binding.binding_config.config["transformers_offline"] = original_offline
    hf_binding._apply_offline_mode()

    if available_models_ui:
        print(f"Total models listed for UI: {len(available_models_ui)}")
        print("Showing sample entries (local first):")
        count = 0
        for model_info in available_models_ui:
            if count < 5 or model_info.get('provider') == 'local':
                 cat = model_info.get('category', 'N/A')
                 rank = model_info.get('rank', 0)
                 dtype = model_info.get('type', 'N/A')
                 dname = model_info.get('display_name', model_info['name'])
                 print(f"- {model_info['name']} (Display: {dname}, Cat: {cat}, Rank: {rank}, Type: {dtype})")
                 count +=1
            elif count == 5: print("  ...") ; count+=1 # Avoid printing ellipsis repeatedly
        if len(available_models_ui)>count: print(f"  ... ({len(available_models_ui)-count} more)")

    else: print("Failed to get model list for UI.")

    # --- Test Callback Function ---
    def test_callback_func(chunk: str, msg_type: int, metadata: Optional[Dict]=None) -> bool:
        type_name = MSG_OPERATION_TYPE(msg_type).name if isinstance(msg_type, int) else str(msg_type)
        if type_name == 'MSG_OPERATION_TYPE_ADD_CHUNK': print(chunk, end="", flush=True)
        elif type_name == 'MSG_OPERATION_TYPE_EXCEPTION': print(f"\n## EXC: {chunk} ##"); return False # Stop on exception
        else: print(f"\n## {type_name}: {chunk} ##")
        # Return True to continue generation, False to stop
        return True

    # Set the callback on the test comm object
    lollms_app_com.set_callback(test_callback_func)


    # --- Test Loading and Generation (if a REAL local model exists) ---
    # Replace 'expected_model_id' with a model you have downloaded locally for testing
    # e.g., "google/gemma-1.1-2b-it" or "llava-hf/llava-1.5-7b-hf"
    expected_model_id = "google/gemma-1.1-2b-it" # <--- CHANGE THIS TO YOUR DOWNLOADED MODEL ID

    real_model_path = lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR / expected_model_id
    if real_model_path.exists() and real_model_path.is_dir() and (real_model_path / "config.json").exists():
        print(f"\n--- Found real model for testing: {expected_model_id} ---")
        config.model_name = expected_model_id

        # Configure settings for testing (use defaults or override)
        # hf_binding.binding_config.config["quantization_bits"] = "4bits" # Test 4-bit if CUDA available & bitsandbytes installed
        hf_binding.binding_config.config["device"] = "auto"      # Use auto device selection
        hf_binding.binding_config.config["apply_chat_template"] = True # Attempt template use
        hf_binding.binding_config.config["trust_remote_code"] = False # Set True ONLY if required by your test model and you trust it

        hf_binding.settings_updated() # Apply changes and trigger build_model
        sleep(2) # Give time for rebuilding messages (if any)

        if hf_binding.model and hf_binding._get_tokenizer_or_processor():
            print(f"\n--- Model {expected_model_id} loaded (Type: {hf_binding.binding_type.name}) ---")
            print(f"Effective Ctx: {hf_binding.config.ctx_size}, Effective Max Gen: {hf_binding.config.max_n_predict}")
            tokenizer_proc = hf_binding._get_tokenizer_or_processor()
            print(f"Tokenizer/Processor class: {type(tokenizer_proc).__name__}")
            tpl = getattr(tokenizer_proc, 'chat_template', None)
            print(f"Has chat template: {'Yes' if tpl else 'No'}")
            # print(f"Chat template:\n{tpl}") # Uncomment to see template

            # --- Test Text Generation (using LoLLMs format) ---
            print("\n--- Testing Text Generation (with Template Formatting) ---")
            prompt_text_lollms = """!@>system:
You are a helpful AI assistant. Be concise and informative.
!@>user:
Explain the concept of "attention" in transformer models in one or two sentences.
!@>assistant:
Attention mechanisms allow transformer models to weigh the importance of different input tokens when producing an output token, focusing on relevant parts of the input sequence.
!@>user:
What is quantization in the context of LLMs?
"""
            print(f"Prompt (LoLLMs format):\n{prompt_text_lollms}\nResponse:")
            try:
                start = perf_counter()
                # Use the LoLLMs formatted prompt
                full_response = hf_binding.generate(prompt_text_lollms, n_predict=100, callback=test_callback_func, verbose=True) # Use verbose=True for more debug info
                print(f"\n--- Text Gen Done ({perf_counter() - start:.2f}s) ---")
            except Exception as e: print(f"\nText Gen Failed: {e}"); trace_exception(e)

            # --- Test Vision Generation (if applicable) ---
            if hf_binding.binding_type == BindingType.TEXT_IMAGE and hf_binding.processor:
                print("\n--- Testing Vision Generation (with Template Formatting) ---")
                # Use a known accessible image URL or a local path within the test env
                # image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat-dog.jpg"
                # Create a dummy image file for testing local paths
                dummy_image_path = lollms_paths.personal_uploads_path / "test_image.png"
                try:
                    dummy_img = Image.new('RGB', (60, 30), color = 'red')
                    dummy_img.save(dummy_image_path)
                    print(f"Created dummy image at: {dummy_image_path}")
                    image_ref = str(dummy_image_path) # Use local path
                except Exception as img_ex:
                    print(f"Could not create dummy image: {img_ex}. Vision test might fail.")
                    image_ref = None # Fallback

                if image_ref:
                    prompt_vision_lollms = f"""!@>system:
You are an expert image analyst. Describe the image content.
!@>user:
Describe the main subject of the provided image.
"""
                    print(f"Image Ref: {image_ref}")
                    print(f"Prompt (LoLLMs format):\n{prompt_vision_lollms}\nResponse:")
                    try:
                        start = perf_counter()
                        full_response = hf_binding.generate_with_images(
                            prompt_vision_lollms,
                            [image_ref], # Pass image path in list
                            n_predict=80,
                            callback=test_callback_func,
                            verbose=True
                        )
                        print(f"\n--- Vision Gen Done ({perf_counter() - start:.2f}s) ---")
                    except Exception as e: print(f"\nVision Gen Failed: {e}"); trace_exception(e)
                else:
                     print("\n--- Skipping Vision Test (Could not prepare image reference) ---")

            elif hf_binding.binding_type == BindingType.TEXT_IMAGE:
                 print("\n--- Skipping Vision Test (Model identified as vision, but failed to load processor or correct binding type) ---")
            else:
                print("\n--- Skipping Vision Test (Model detected as Text-Only) ---")

            # --- Test unloading ---
            print("\n--- Testing Model Unloading ---")
            hf_binding._unload_model()
            if hf_binding.model is None and hf_binding.tokenizer is None and hf_binding.processor is None:
                print("Model unload successful (references set to None). Check GPU memory if applicable.")
            else:
                print("Model unload failed (references still exist).")

        else:
            print(f"\n--- Skipping generation tests: Failed to load model or tokenizer/processor for {expected_model_id} ---")
            print(f"--- Ensure the model is correctly downloaded to: {real_model_path.parent} ---")
    else:
        print(f"\n--- Skipping loading and generation tests: Real model '{expected_model_id}' not found ---")
        print(f"--- Please download it to: {real_model_path.parent} ---")
        print(f"--- You can use: huggingface-cli download {expected_model_id} --local-dir {real_model_path.parent} --local-dir-use-symlinks False ---")

    print("\nScript finished.")