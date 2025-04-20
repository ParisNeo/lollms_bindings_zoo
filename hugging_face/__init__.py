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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
    if not pm.is_installed("bitsandbytes"): raise ImportError("Bitsandbytes not found.")
    if not pm.is_installed("huggingface_hub"): raise ImportError("huggingface_hub not found.")
    if not pm.is_installed("PIL"): raise ImportError("Pillow not found.")
    if not pm.is_installed("requests"): raise ImportError("requests not found.")


except ImportError as e:
    # Fallback or error message if pipmaster fails or isn't available
    print("Warning: pipmaster check failed or packages missing.")
    print("Please ensure torch, transformers, accelerate, bitsandbytes, sentencepiece, huggingface_hub, Pillow, and requests are installed.")
    print("Attempting to proceed, but errors may occur if packages are missing.")
    # Check again with standard import checks
    # Use standard import checks for clarity
    try: import torch
    except ImportError: print("Error: PyTorch is missing.")
    try: import transformers
    except ImportError: print("Error: Transformers is missing.")
    try: import accelerate
    except ImportError: print("Error: Accelerate is missing.")
    try: import bitsandbytes
    except ImportError: print("Error: Bitsandbytes is missing.")
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
    from transformers import (
        AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoProcessor,
        BitsAndBytesConfig, TextIteratorStreamer,
        LlavaForConditionalGeneration, PaliGemmaForConditionalGeneration, # Add known VLM classes
        Gemma3ForConditionalGeneration # Import Gemma 3 explicitly
    )
    from huggingface_hub import HfApi # Added imports
    # from accelerate import Accelerator, dispatch_model, infer_auto_device_map - Let transformers handle device_map='auto'
    if torch.cuda.is_available():
        try:
            import bitsandbytes as bnb # Only needed if CUDA is available for quantization
        except ImportError:
            print("Warning: bitsandbytes not found, 4/8-bit quantization will not be available.")

except ImportError as e:
    trace_exception(e)
    ASCIIColors.error("Could not import required libraries. Please ensure they are installed correctly.")
    # Decide whether to raise or allow graceful failure
    # raise e # Or set flags and handle later


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023-2024, ParisNeo"
__license__ = "Apache 2.0"

binding_name = "HuggingFaceLocal"
binding_folder_name = "hugging_face"
# Define the folder for local HF models relative to lollms_paths
HF_LOCAL_MODELS_DIR = "transformers"

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
            {"name":"device", "type":"str", "value":"auto", "options":["auto", "cpu", "cuda", "mps"], "help":"Device to use for computation (auto detects best available: CUDA > MPS > CPU)."},
            {"name":"quantization_bits", "type":"int", "value":-1, "options":[-1, 4, 8], "help":"Load model quantized in 4-bit or 8-bit. Requires CUDA and bitsandbytes. (-1 for no quantization)."},
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
            {"name":"favorite_providers", "type":"str", "value":"microsoft,nvidia,mistralai,deepseek-ai,meta-llama,unsloth,ParisNeo,Bartowski", "help":"List of your bets providers. Empty list for anyone"},
            {"name":"hub_fetch_limit", "type":"int", "value":5000, "min": 10, "max": 5000000, "help":"Maximum number of models to fetch from Hugging Face Hub for the 'available models' list."},
            {"name":"model_sorting", "type":"str", "value":"trending_score", "options": ["trending_score","created_at", "last_modified", "downloads", "likes "],"help":"Maximum number of models to fetch from Hugging Face Hub for the 'available models' list."},
        ])
        # Default values for the configuration
        binding_config_defaults = BaseConfig(config={
            "device": "auto",
            "quantization_bits": -1,
            "use_flash_attention_2": False,
            "trust_remote_code": False,
            "transformers_offline": True,
            "auto_infer_ctx_size": True,
            "ctx_size": 4096, # Fallback value
            "max_n_predict": 1024,
            "seed": -1,
            "apply_chat_template": True,
            "hub_fetch_limit": 100,
            "model_sorting": "trending_score",
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
        self.build_model(self.config.model_name) # Rebuild with the current model name


    def get_local_hf_model_path(self, model_name: str) -> Optional[Path]:
        """Resolves the full path to a local Hugging Face model directory."""
        if not model_name:
            return None
        model_full_path = self.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR / model_name
        return model_full_path


    def build_model(self, model_name: Optional[str] = None) -> LLMBinding:
        """
        Loads the specified Hugging Face model and tokenizer/processor.
        Determines if the model is multimodal and attempts to infer context size.
        """
        super().build_model(model_name) # Sets self.config.model_name

        current_model_name = self.config.model_name
        if not current_model_name:
            # Reset state if no model is selected
            self.model = None; self.tokenizer = None; self.processor = None
            self.binding_type = BindingType.TEXT_ONLY; self.supported_file_extensions = []
            self.error("No model selected in LoLLMs configuration.")
            return self

        model_full_path = self.get_local_hf_model_path(current_model_name)

        if not model_full_path or not model_full_path.exists() or not model_full_path.is_dir():
            self.error(f"Model folder not found: {model_full_path}")
            self.error(f"Please ensure the model '{current_model_name}' is downloaded correctly into the '{HF_LOCAL_MODELS_DIR}' directory.")
            self.model = None; self.tokenizer = None; self.processor = None
            self.binding_type = BindingType.TEXT_ONLY; self.supported_file_extensions = []
            if self.lollmsCom: self.lollmsCom.InfoMessage(f"HuggingFaceLocal Error: Model folder '{current_model_name}' not found.")
            return self

        # Clear previous model from memory
        if self.model is not None: self.info("Unloading previous model..."); del self.model; self.model = None
        if self.tokenizer is not None: del self.tokenizer; self.tokenizer = None
        if self.processor is not None: del self.processor; self.processor = None
        AdvancedGarbageCollector.collect()

        self.info(f"Loading model: {current_model_name}")
        self.ShowBlockingMessage(f"Loading {current_model_name}...\nPlease wait.")

        try:
            # --- Determine Device ---
            requested_device = self.binding_config.config.get("device", "auto")
            self.device = get_torch_device() if requested_device == "auto" else requested_device
            ASCIIColors.info(f"Selected device: {self.device}")
            if "cuda" not in self.device and self.binding_config.quantization_bits in [4, 8]:
                self.warning("Quantization requires CUDA. Disabling quantization.")
                self.binding_config.config["quantization_bits"] = -1

            # --- Load Model Config First ---
            self.info(f"Loading config from: {model_full_path}")
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
                 self.info(f"Loading as Vision Model using {ModelClass.__name__} and {ProcessorTokenizerClass.__name__}")
            else:
                 self.binding_type = BindingType.TEXT_ONLY
                 self.supported_file_extensions = []
                 self.info(f"Loading as Text Model using {ModelClass.__name__} and {ProcessorTokenizerClass.__name__}")


            # --- Prepare Loading Arguments ---
            kwargs: Dict[str, Any] = {"trust_remote_code": trust_code, "device_map": "auto"}
            quantization_bits = self.binding_config.config.get("quantization_bits", -1)
            if quantization_bits in [4, 8] and "cuda" in self.device:
                try:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=(quantization_bits == 8), load_in_4bit=(quantization_bits == 4),
                                                    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                    kwargs["quantization_config"] = bnb_config; kwargs["torch_dtype"] = torch.bfloat16; ASCIIColors.info(f"Applying {quantization_bits}-bit quantization.")
                except Exception as bnb_ex:
                    self.error(f"Failed to create BitsAndBytesConfig: {bnb_ex}. Disabling quantization.")
                    self.binding_config.config["quantization_bits"] = -1
            elif self.device == "cuda": kwargs["torch_dtype"] = torch.float16
            elif self.device == "mps": kwargs["torch_dtype"] = torch.float16 # MPS often benefits from float16
            else: kwargs["torch_dtype"] = torch.float32

            use_flash_attention = self.binding_config.config.get("use_flash_attention_2", False)
            if use_flash_attention and "cuda" in self.device:
                 try:
                     major, minor = map(int, transformers.__version__.split('.')[:2])
                     # Flash Attention 2 integration might vary; check if 'attn_implementation' is supported
                     # >= 4.34 recommended for stable FA2 support in transformers
                     if major >= 4 and minor >= 34: # Be a bit conservative with version check
                         kwargs["attn_implementation"] = "flash_attention_2"
                         ASCIIColors.info("Attempting Flash Attention 2 implementation.")
                     else:
                         # Try legacy use_flash_attention_2 if available, might depend on specific model class
                         # Note: This is less standard now. device_map='auto' + FA2 can be tricky.
                         # kwargs["use_flash_attention_2"] = True # Older way, less likely to work well
                         ASCIIColors.warning("Transformers version might be older than ideal for seamless Flash Attention 2. Sticking to default attn.")
                 except Exception as fa_ex:
                     ASCIIColors.warning(f"Couldn't check/apply Flash Attention 2 setting: {fa_ex}")


            # --- Load Processor or Tokenizer ---
            self.info(f"Loading {ProcessorTokenizerClass.__name__} from: {model_full_path}")
            processor_tokenizer_instance = ProcessorTokenizerClass.from_pretrained(model_full_path, trust_remote_code=trust_code)
            if is_vision_model:
                self.processor = processor_tokenizer_instance
                # Try to get the underlying tokenizer for VLM text parts
                self.tokenizer = getattr(self.processor, 'tokenizer', None)
                if not self.tokenizer:
                    self.warning("Could not find 'tokenizer' attribute on the processor. Text tokenization might rely solely on the processor.")
                    # Fallback: Use the processor itself if it behaves like a tokenizer (some do)
                    if callable(getattr(self.processor, "encode", None)) and callable(getattr(self.processor, "decode", None)):
                        self.tokenizer = self.processor # Use processor as tokenizer if it has encode/decode
                    else:
                        self.warning("Processor cannot be used as a fallback tokenizer.")

            else:
                self.tokenizer = processor_tokenizer_instance; self.processor = None

            # --- Handle Missing Pad Token ---
            # Use the combined _get_tokenizer_or_processor helper
            tokenizer_to_check = self._get_tokenizer_or_processor()
            pad_token_to_add = None
            if tokenizer_to_check and tokenizer_to_check.pad_token_id is None:
                if tokenizer_to_check.eos_token_id is not None:
                    tokenizer_to_check.pad_token_id = tokenizer_to_check.eos_token_id
                    tokenizer_to_check.pad_token = tokenizer_to_check.eos_token
                    ASCIIColors.warning("Tokenizer/Processor missing pad_token, setting to eos_token.")
                else:
                    # Add a default pad token if both pad and eos are missing
                    pad_token_to_add = '[PAD]'
                    # Check if it exists before adding
                    if pad_token_to_add not in tokenizer_to_check.get_vocab():
                         num_added = tokenizer_to_check.add_special_tokens({'pad_token': pad_token_to_add})
                         if num_added > 0:
                             ASCIIColors.warning(f"Added special token '{pad_token_to_add}' to tokenizer/processor.")
                             # Resize model embeddings later after model load
                         else:
                             ASCIIColors.warning(f"Special token '{pad_token_to_add}' may already exist, but pad_token_id is still None.")
                             # Try to set it manually if it exists now
                             try:
                                 tokenizer_to_check.pad_token = pad_token_to_add
                                 # Note: pad_token_id might still require explicit setting if lookup fails
                             except Exception: pass
                    else:
                        tokenizer_to_check.pad_token = pad_token_to_add
                        ASCIIColors.warning(f"'{pad_token_to_add}' token exists but wasn't set as pad token. Setting it now.")

                    if tokenizer_to_check.pad_token_id is None:
                        self.warning("Could not determine or set a pad_token_id. Generation might fail for batching/padding.")


            # --- Load Model ---
            self.info(f"Loading model using {ModelClass.__name__} from: {model_full_path} with config: {kwargs}")
            self.model = ModelClass.from_pretrained(model_full_path, **kwargs)

            # --- Resize Embeddings if Pad Token was Added ---
            if pad_token_to_add and hasattr(self.model, 'resize_token_embeddings'):
                 # Check if resize is actually needed (vocab size might already match)
                 current_vocab_size = getattr(tokenizer_to_check, 'vocab_size', len(tokenizer_to_check))
                 if self.model.get_input_embeddings().weight.shape[0] < current_vocab_size:
                     self.info(f"Resizing model token embeddings to match added token(s). New size: {current_vocab_size}")
                     self.model.resize_token_embeddings(current_vocab_size)
                     # Re-check pad_token_id after resize, sometimes needed
                     if tokenizer_to_check and tokenizer_to_check.pad_token_id is None and tokenizer_to_check.pad_token:
                          pad_id = tokenizer_to_check.convert_tokens_to_ids(tokenizer_to_check.pad_token)
                          if isinstance(pad_id, int): tokenizer_to_check.pad_token_id = pad_id

                 else:
                     self.info("Token embeddings size already matches tokenizer vocab size after potential token addition.")


            # --- Infer/Set Context Size ---
            effective_ctx_size = 4096 # Default fallback
            if self.binding_config.config.get("auto_infer_ctx_size", True):
                detected_ctx_size = None; possible_keys = ['max_position_embeddings', 'n_positions', 'seq_length']
                for key in possible_keys:
                    ctx_val = getattr(model_config, key, None)
                    if isinstance(ctx_val, int) and ctx_val > 0:
                        detected_ctx_size = ctx_val; ASCIIColors.info(f"Auto-detected context size ({key}): {detected_ctx_size}"); break
                if detected_ctx_size: effective_ctx_size = detected_ctx_size
                else: effective_ctx_size = self.binding_config.config.get("ctx_size", 4096); ASCIIColors.warning(f"Could not auto-detect context size. Using configured: {effective_ctx_size}")
            else:
                 effective_ctx_size = self.binding_config.config.get("ctx_size", 4096); ASCIIColors.info(f"Using manually configured context size: {effective_ctx_size}")

            self.config.ctx_size = effective_ctx_size
            self.binding_config.config["ctx_size"] = effective_ctx_size # Update binding config view


            # --- Validate and Set Max Prediction Tokens ---
            configured_max_predict = self.binding_config.config.get("max_n_predict", 1024)
            effective_max_predict = configured_max_predict
            # Ensure max_predict doesn't exceed context size minus some buffer (e.g., 5 tokens)
            if effective_max_predict >= effective_ctx_size:
                 # Cap prediction length to context size minus a small buffer
                 capped_predict = max(64, effective_ctx_size - 5) # Keep a minimum generation capability
                 self.warning(f"Configured max_n_predict ({effective_max_predict}) >= effective_ctx_size ({effective_ctx_size}). Capping to {capped_predict}.")
                 effective_max_predict = capped_predict
            elif effective_max_predict < 64: # Ensure a reasonable minimum
                self.warning(f"Configured max_n_predict ({effective_max_predict}) is very low. Setting to minimum 64.")
                effective_max_predict = 64


            self.config.max_n_predict = effective_max_predict
            self.binding_config.config["max_n_predict"] = effective_max_predict # Update binding config view

            # --- Check if Chat Template Exists ---
            tokenizer_or_processor = self._get_tokenizer_or_processor()
            if tokenizer_or_processor and hasattr(tokenizer_or_processor, 'chat_template') and tokenizer_or_processor.chat_template:
                 ASCIIColors.info("Model has a chat template defined.")
            else:
                 ASCIIColors.warning("Model does not have a chat template defined in its tokenizer/processor config. Will use raw prompt input.")
                 self.binding_config.config["apply_chat_template"] = False # Disable if not found

            ASCIIColors.success(f"Model {current_model_name} loaded. Effective Ctx: {self.config.ctx_size}, Max Gen: {self.config.max_n_predict}. Type: {self.binding_type.name}")
            self.HideBlockingMessage()

        except ImportError as e:
            self.error(f"Import error while loading: {e}")
            self.error("Ensure required libraries are installed: torch, transformers, accelerate, bitsandbytes, huggingface_hub, Pillow, requests")
            trace_exception(e); self.model = None; self.tokenizer = None; self.processor = None; self.HideBlockingMessage()
        except Exception as e:
            self.error(f"Failed to load model {current_model_name}: {e}")
            trace_exception(e); self.model = None; self.tokenizer = None; self.processor = None; self.HideBlockingMessage()
            if "out of memory" in str(e).lower(): self.error("CUDA OOM."); self.lollmsCom.InfoMessage("HuggingFaceLocal Error: CUDA Out of Memory.") if self.lollmsCom else None
            elif self.lollmsCom: self.lollmsCom.InfoMessage(f"HuggingFaceLocal Error: Failed to load model.\n{e}")

        return self


    def install(self) -> None:
        """Installs necessary Python packages using pipmaster."""
        super().install()
        self.ShowBlockingMessage("Installing Hugging Face Transformers requirements...")
        try:
            import pipmaster as pm
            requirements = ["torch", "transformers", "accelerate", "bitsandbytes", "sentencepiece", "huggingface_hub", "Pillow", "requests"]
            for req in requirements:
                if not pm.is_installed(req): self.info(f"Installing {req}..."); pm.install(req)
                else: self.info(f"{req} already installed.")
            self.HideBlockingMessage()
            ASCIIColors.success("Hugging Face requirements installed successfully.")
            ASCIIColors.info("----------------------\nAttention:\n----------------------")
            ASCIIColors.info("This binding requires manual download of Hugging Face models.")
            ASCIIColors.info(f"1. Find models on Hugging Face Hub (https://huggingface.co/models).")
            ASCIIColors.info(f"2. Download using 'git clone' or 'huggingface-cli download'.")
            ASCIIColors.info(f"3. Place the model folder into: {self.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR}")
            ASCIIColors.info(f"   Example: models/{HF_LOCAL_MODELS_DIR}/google/gemma-3-4b-it")
            ASCIIColors.info("4. Select the model folder name in LoLLMs settings.")
        except ImportError: self.HideBlockingMessage(); self.error("pipmaster not found. Install manually.")
        except Exception as e: self.error(f"Installation failed: {e}"); trace_exception(e); self.HideBlockingMessage()


    def _get_tokenizer_or_processor(self) -> Optional[Union[AutoTokenizer, AutoProcessor]]:
        """Returns the processor if available, otherwise the tokenizer."""
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
                # Use encode method which is common to both Tokenizer and Processor
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
        # n_predict here is already the effective value (potentially capped)
        effective_max_n_predict = n_predict

        # Start with LoLLMs-provided parameters
        default_params = {
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            # Add other potential default lollms params here if needed
        }
        # Override defaults with any specific gpt_params passed for this call
        gen_kwargs = {**default_params, **gpt_params}

        tokenizer_to_use = self._get_tokenizer_or_processor()
        if not tokenizer_to_use:
            raise RuntimeError("Tokenizer/Processor not available for generation config.")

        # --- Build final kwargs for transformers ---
        final_gen_kwargs = {
            "max_new_tokens": effective_max_n_predict,
            "pad_token_id": tokenizer_to_use.pad_token_id,
            # Use eos_token_id from tokenizer/processor if available
            "eos_token_id": tokenizer_to_use.eos_token_id,
             # Set stopping criteria using eos_token_id if available
             # Note: some models might need multiple EOS tokens. Handle manually if needed.
            "do_sample": True # Default to sampling
        }

        # Handle specific parameter conversions/validations
        temp = float(gen_kwargs.get('temperature', 0.7))
        if temp <= 0.01: # Consider temps close to 0 as greedy
            final_gen_kwargs["do_sample"] = False
            # Some models behave better with temp=1.0, top_k=1 for greedy
            final_gen_kwargs["temperature"] = 1.0
            final_gen_kwargs["top_k"] = 1
            final_gen_kwargs["top_p"] = 1.0 # Usually ignored when do_sample=False, but set for clarity
        else:
            final_gen_kwargs["temperature"] = temp
            top_p = float(gen_kwargs.get('top_p', 1.0))
            top_k = int(gen_kwargs.get('top_k', 50)) # Often called top_k in transformers
            # Validate and apply top_p and top_k only if sampling
            if 0.0 < top_p <= 1.0: final_gen_kwargs["top_p"] = top_p
            else: final_gen_kwargs["top_p"] = 1.0 # Use default if invalid
            if top_k > 0: final_gen_kwargs["top_k"] = top_k
            else: final_gen_kwargs["top_k"] = 50 # Use default if invalid



        # --- Seed ---
        seed = int(self.binding_config.config.get("seed", -1)) # Ensure seed is int
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            ASCIIColors.info(f"Generation seed set to: {seed}")

        # --- Check for missing critical tokens ---
        if final_gen_kwargs["pad_token_id"] is None: self.warning("pad_token_id is None. Padding during generation might fail.")
        if final_gen_kwargs["eos_token_id"] is None: self.warning("eos_token_id is None. Model might not stop generating naturally.")

        return final_gen_kwargs


    def generate(self,
                 prompt: str,
                 n_predict: Optional[int] = None,
                 callback: Optional[Callable[[str, int, Optional[Dict]], bool]] = None, # Adjusted callback sig
                 verbose: bool = False,
                 **gpt_params) -> str:
        """ Generates text using the loaded text-only model, applying chat template if enabled. """
        # Ensure n_predict is valid and respects model context
        effective_n_predict = self._validate_n_predict(n_predict)

        # Handle incompatible model type
        if self.binding_type != BindingType.TEXT_ONLY:
            self.warning("generate() called on a vision model. Use generate_with_images() or expect potential errors.")
            # Optionally, could try to proceed if a tokenizer exists, but might be unreliable.
            # For now, let's return an error message.
            # return "[Error: generate() should not be called on vision models. Use generate_with_images().]"
            # Or proceed cautiously:
            if not self.tokenizer:
                 self.error("Vision model loaded, but no accessible tokenizer found for text generation fallback.")
                 return "[Error: Model is vision-capable but lacks a text tokenizer.]"


        # Check model and tokenizer/processor readiness
        tokenizer_or_processor = self._get_tokenizer_or_processor()
        if not self.model or not tokenizer_or_processor:
            self.error("Model or Tokenizer/Processor not loaded.")
            return "[Error: Model not ready]"

        # Stop any ongoing generation
        if self.generation_thread and self.generation_thread.is_alive():
            self.warning("Stopping previous generation thread.")
            self._stop_generation = True
            try: self.generation_thread.join(timeout=5) # Wait up to 5 seconds
            except Exception as join_ex: self.error(f"Error joining previous thread: {join_ex}")
            if self.generation_thread.is_alive(): self.error("Previous generation thread did not stop!")
        self._stop_generation = False

        try:
            # --- Prepare Generation Arguments ---
            final_gen_kwargs = self._prepare_common_generation_kwargs(effective_n_predict, gpt_params)
            if verbose: ASCIIColors.verbose(f"Text Gen raw params: {gpt_params}")
            if verbose: ASCIIColors.verbose(f"Text Gen effective params: {final_gen_kwargs}")

            # --- Prepare Inputs (Apply Template if enabled) ---
            apply_template = self.binding_config.config.get("apply_chat_template", True)
            inputs = None

            if apply_template and hasattr(tokenizer_or_processor, 'chat_template') and tokenizer_or_processor.chat_template:
                try:
                    messages = self.parse_lollms_discussion(prompt)
                    if not messages: raise ValueError("Prompt parsing resulted in empty message list.")
                    if self.config.debug: ASCIIColors.debug(f"Parsed messages for template: {messages}")

                    # Apply template - DO NOT tokenize here, let generate handle it for streaming
                    # Tokenize=False returns a string
                    if self.config.debug:
                        formatted_prompt = tokenizer_or_processor.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True # Crucial for instructing model to generate next turn
                        )
                        ASCIIColors.debug(f"Formatted prompt via template:\n{formatted_prompt}")
                        inputs = tokenizer_or_processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True, # Adds the prompt for the assistant's turn
                            return_tensors="pt"
                        ).to(self.device)
                    else:
                        # --- Let generate handle tokenization directly from messages for better efficiency/streaming ---
                        # This seems to be the more modern way for HF generate with chat inputs
                        inputs = tokenizer_or_processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True, # Adds the prompt for the assistant's turn
                            return_tensors="pt"
                        ).to(self.device)
                        if self.config.debug: ASCIIColors.info("Applied chat template successfully.")

                except Exception as template_ex:
                    self.error(f"Failed to apply chat template: {template_ex}. Falling back to raw prompt.")
                    trace_exception(template_ex)
                    # Fallback: Tokenize the raw prompt string
                    inputs = tokenizer_or_processor(prompt, return_tensors="pt").to(self.device)
            else:
                if apply_template:
                    self.warning("Chat template application requested but no template found. Using raw prompt.")
                # Tokenize the raw prompt string if template not applied/available
                inputs = tokenizer_or_processor(prompt, return_tensors="pt").to(self.device)

            # --- Validate Input Length ---
            input_token_count = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else inputs.shape[1]
            if input_token_count >= self.config.ctx_size:
                # Try to recover by truncating (inform user)
                # This is tricky - ideally truncation should happen *before* templating or tokenizing.
                # For now, just raise error.
                self.error(f"Input prompt is too long ({input_token_count} tokens) for the model's context size ({self.config.ctx_size}).")
                raise ValueError(f"Prompt too long ({input_token_count} tokens vs context {self.config.ctx_size}). Please shorten your input.")
            if input_token_count + final_gen_kwargs["max_new_tokens"] > self.config.ctx_size:
                 self.warning(f"Potential context overflow: Input ({input_token_count}) + Max New ({final_gen_kwargs['max_new_tokens']}) > Context ({self.config.ctx_size}). Output might be truncated.")
                 # Optionally adjust max_new_tokens dynamically:
                 # final_gen_kwargs['max_new_tokens'] = self.config.ctx_size - input_token_count - 5 # Add buffer

        except Exception as e:
            self.error(f"Input Preparation Error: {e}")
            trace_exception(e)
            if callback: callback(f"Input Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return f"[Input Error: {e}]"

        # --- Setup Streaming ---
        # Use the primary tokenizer for streaming decoding
        streamer = TextIteratorStreamer(
            tokenizer_or_processor,
            skip_prompt=True, # Skip the input prompt part from the output stream
            skip_special_tokens=True
        )

        # Prepare arguments for the generation thread
        # Handle two input types: dict from apply_chat_template or tensors directly
        if isinstance(inputs, dict):
             generation_kwargs_for_thread = {**inputs, **final_gen_kwargs, "streamer": streamer}
        else: # Assuming raw tensor input
             generation_kwargs_for_thread = {"input_ids": inputs, **final_gen_kwargs, "streamer": streamer}

        output_buffer = ""
        start_time = perf_counter() # For performance metric

        try:
            # --- Start Generation Thread ---
            self.generation_thread = Thread(target=self.model.generate, kwargs=generation_kwargs_for_thread)
            self.generation_thread.start()
            if self.config.debug: ASCIIColors.info("Starting text generation stream...")

            # --- Consume Stream ---
            for new_text in streamer:
                if self._stop_generation:
                    ASCIIColors.warning("Stop generation requested.")
                    break
                if new_text:
                    output_buffer += new_text
                    # Use the new callback signature
                    if callback:
                         continue_generating = callback(new_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK)
                         if continue_generating is False: # Check for explicit False to stop
                              self._stop_generation = True
                              ASCIIColors.warning("Generation stopped by callback returning False.")
                              break

            # Ensure thread finishes
            self.generation_thread.join()
            if self.config.debug: ASCIIColors.info("Text generation stream finished.")

        except Exception as e:
             self.error(f"Generation Error: {e}")
             trace_exception(e)
             output_buffer += f"\n[Error during generation: {e}]"
             if callback: callback(f"Generation Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             # Ensure thread is cleaned up even if it failed mid-stream
             if self.generation_thread and self.generation_thread.is_alive():
                 self.warning("Attempting to join failed generation thread.")
                 self.generation_thread.join(timeout=1)
        finally:
            self.generation_thread = None
            self._stop_generation = False
            end_time = perf_counter()
            if self.config.debug: ASCIIColors.info(f"Generation finished in {end_time - start_time:.2f} seconds.")

        # Final callback with full response (optional, depending on LoLLMs standards)
        # if callback: callback(output_buffer, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_FULL_RESPONSE)

        return output_buffer


    def _process_image_argument(self, image_path_or_url: str) -> Optional[Image.Image]:
        """ Loads an image from a local path or URL. Downloads URL if necessary. """
        try:
            if is_file_path(image_path_or_url):
                # Check if it's a valid path first
                path_obj = Path(image_path_or_url)
                if path_obj.exists() and path_obj.is_file():
                    return Image.open(path_obj).convert("RGB")
                else:
                    # Try finding relative to known paths if it's not absolute
                    found_path = find_first_available_file_path([
                        self.lollms_paths.personal_uploads_path / image_path_or_url,
                        self.lollms_paths.shared_uploads_path / image_path_or_url,
                        # Add other potential relative locations if needed
                    ])
                    if found_path:
                        return Image.open(found_path).convert("RGB")
                    else:
                        self.warning(f"Local image file not found: {image_path_or_url}")
                        return None
            elif image_path_or_url.startswith(("http://", "https://")):
                # Download the image
                self.info(f"Downloading image from: {image_path_or_url}")
                # Generate a safe filename based on URL
                # Be careful with long URLs or special chars
                filename = Path(image_path_or_url).name
                if not filename: filename = "downloaded_image_" + str(hash(image_path_or_url)) + ".jpg" # Fallback name
                # Basic sanitization
                filename = re.sub(r'[\\/*?:"<>|]',"", filename)
                local_filepath = self.downloads_path / filename

                # Check if already downloaded
                if local_filepath.exists():
                     self.info(f"Using cached image: {local_filepath}")
                     return Image.open(local_filepath).convert("RGB")

                # Perform download
                try:
                    success = download_file(image_path_or_url, local_filepath, self.lollmsCom.InfoMessage if self.lollmsCom else print)
                    if success and local_filepath.exists():
                        return Image.open(local_filepath).convert("RGB")
                    else:
                        self.warning(f"Failed to download image from URL: {image_path_or_url}")
                        return None
                except Exception as dl_ex:
                     self.error(f"Error downloading image {image_path_or_url}: {dl_ex}")
                     trace_exception(dl_ex)
                     return None
            else:
                self.warning(f"Invalid image format or path: {image_path_or_url}. Expecting local path or HTTP/HTTPS URL.")
                return None
        except Exception as e:
            self.error(f"Failed to load or process image '{image_path_or_url}': {e}")
            trace_exception(e)
            return None

    def _validate_n_predict(self, n_predict: Optional[int]) -> int:
        """ Validates n_predict against configuration and context size. """
        if n_predict is None:
            n_predict = self.config.max_n_predict # Use configured default
        elif not isinstance(n_predict, int) or n_predict <= 0:
             self.warning(f"Invalid n_predict value ({n_predict}). Using default: {self.config.max_n_predict}")
             n_predict = self.config.max_n_predict

        # Ensure n_predict does not exceed the effective max_n_predict derived during build_model
        if n_predict > self.config.max_n_predict:
            self.warning(f"Requested n_predict ({n_predict}) exceeds effective maximum ({self.config.max_n_predict}). Capping.")
            n_predict = self.config.max_n_predict

        return n_predict

    def generate_with_images(self,
                             prompt: str,
                             images: List[str],
                             n_predict: Optional[int] = None,
                             callback: Optional[Callable[[str, int, Optional[Dict]], bool]] = None, # Adjusted callback sig
                             verbose: bool = False,
                             **gpt_params) -> str:
        """ Generates text using prompt and images (multimodal), applying chat template if available. """
        # Validate n_predict
        effective_n_predict = self._validate_n_predict(n_predict)

        # Check if the loaded model is actually a vision model
        if self.binding_type != BindingType.TEXT_IMAGE or not self.processor:
            self.warning("generate_with_images called, but the current model is not a vision model or processor is missing.")
            # Fallback to text-only generation using the prompt
            return self.generate(prompt, effective_n_predict, callback, verbose, **gpt_params)

        # Check for essential components
        if not self.model:
            self.error("Vision model not loaded.")
            return "[Error: Model not ready]"

        if not images:
            self.warning("No images provided to generate_with_images. Falling back to text-only generation.")
            return self.generate(prompt, effective_n_predict, callback, verbose, **gpt_params)

        # Stop any ongoing generation
        if self.generation_thread and self.generation_thread.is_alive():
            self.warning("Stopping previous generation thread.")
            self._stop_generation = True
            try: self.generation_thread.join(timeout=5)
            except Exception as join_ex: self.error(f"Error joining previous thread: {join_ex}")
            if self.generation_thread.is_alive(): self.error("Previous generation thread did not stop!")
        self._stop_generation = False

        loaded_pil_images: List[Image.Image] = []
        try:
            # --- Prepare Generation Arguments ---
            final_gen_kwargs = self._prepare_common_generation_kwargs(effective_n_predict, gpt_params)
            if self.config.debug: ASCIIColors.debug(f"Vision Gen raw params: {gpt_params}")
            if self.config.debug: ASCIIColors.debug(f"Vision Gen effective params: {final_gen_kwargs}")

            # --- Load Images ---
            failed_images = []
            for img_path_or_url in images:
                pil_img = self._process_image_argument(img_path_or_url)
                if pil_img:
                    loaded_pil_images.append(pil_img)
                else:
                    failed_images.append(img_path_or_url)

            if not loaded_pil_images:
                raise ValueError("Failed to load any valid images.")
            if failed_images:
                self.warning(f"Skipped loading {len(failed_images)} invalid/missing images: {failed_images}")

            # --- Prepare Inputs (Apply Template if enabled) ---
            apply_template = self.binding_config.config.get("apply_chat_template", True)
            inputs = None

            if apply_template and hasattr(self.processor, 'chat_template') and self.processor.chat_template:
                try:
                    messages = self.parse_lollms_discussion(prompt)
                    if not messages: raise ValueError("Prompt parsing resulted in empty message list.")

                    # Find the last user message to attach images to
                    last_user_msg_index = -1
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i]['role'] == 'user':
                            last_user_msg_index = i
                            break

                    if last_user_msg_index == -1:
                         # No user message found, create one to hold images and maybe prompt?
                         self.warning("No user message in parsed prompt to attach images. Creating one.")
                         # If prompt wasn't parsed into messages (e.g., only system prompt), use it here.
                         content_list = [{"type": "text", "text": prompt if not messages else "Image context:"}]
                         for pil_img in loaded_pil_images: content_list.append(pil_img) # Pass PIL directly
                         messages.append({"role": "user", "content": content_list})
                    else:
                        # Ensure the content of the last user message is a list
                        last_content = messages[last_user_msg_index]['content']
                        if isinstance(last_content, str):
                             messages[last_user_msg_index]['content'] = [{"type": "text", "text": last_content}]
                        elif not isinstance(last_content, list):
                             self.warning(f"Unexpected content type in last user message: {type(last_content)}. Replacing.")
                             messages[last_user_msg_index]['content'] = [{"type": "text", "text": str(last_content)}] # Fallback

                        # Append images (pass PIL Image objects directly)
                        for pil_img in loaded_pil_images:
                             messages[last_user_msg_index]['content'].append(pil_img)

                    if self.config.debug: ASCIIColors.debug(f"Messages prepared for vision template: {messages}") # Careful logging PIL

                    # Apply template and tokenize directly
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(self.device)
                    if self.config.debug: ASCIIColors.info("Applied vision chat template successfully.")

                except Exception as template_ex:
                    self.error(f"Failed to apply vision chat template: {template_ex}. Falling back to processor direct call.")
                    trace_exception(template_ex)
                    # Fallback: Use processor directly with text and images
                    inputs = self.processor(text=prompt, images=loaded_pil_images, return_tensors="pt").to(self.device)

            else: # Template not available or disabled
                if apply_template:
                     self.warning("Vision chat template application requested but no template found/supported. Using processor direct call.")
                # Use processor directly with text and images
                inputs = self.processor(text=prompt, images=loaded_pil_images, return_tensors="pt").to(self.device)

            # --- Validate Input Length ---
            input_token_count = inputs.input_ids.shape[-1] if hasattr(inputs, 'input_ids') else 0 # Check last dim for tokens
            if input_token_count == 0: self.warning("Processor returned inputs with 0 tokens.")
            elif input_token_count >= self.config.ctx_size:
                self.error(f"Combined text/image input is too long ({input_token_count} tokens) for context size ({self.config.ctx_size}).")
                raise ValueError(f"Input too long ({input_token_count} tokens vs context {self.config.ctx_size}).")
            elif input_token_count + final_gen_kwargs["max_new_tokens"] > self.config.ctx_size:
                 self.warning(f"Potential context overflow: Input ({input_token_count}) + Max New ({final_gen_kwargs['max_new_tokens']}) > Context ({self.config.ctx_size}).")

        except Exception as e:
            self.error(f"Input Preparation Error (Vision): {e}")
            trace_exception(e)
            if callback: callback(f"Input Error (Vision): {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return f"[Input Error: {e}]"
        finally:
            # Close the PIL images after processing
            for img in loaded_pil_images:
                img.close()

        # --- Setup Streaming ---
        # Use the processor's tokenizer (or fallback) for decoding the stream
        stream_tokenizer = self.tokenizer or self.processor # Use processor if tokenizer not distinct
        if not hasattr(stream_tokenizer, 'decode'):
            self.error("Cannot stream output: No valid tokenizer/decoder found.")
            return "[Error: Decoder not available]"

        streamer = TextIteratorStreamer(
            stream_tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        # Prepare arguments for the generation thread
        # Input is likely a dict from processor or apply_chat_template
        if not isinstance(inputs, dict):
             # Should typically be a dict containing 'input_ids', 'pixel_values', etc.
             self.warning("Inputs are not in the expected dictionary format. Trying to adapt.")
             inputs = {"input_ids": inputs} # Simplistic adaptation, might fail

        generation_kwargs_for_thread = {**inputs, **final_gen_kwargs, "streamer": streamer}
        output_buffer = ""
        start_time = perf_counter()

        try:
            # --- Start Generation Thread ---
            self.generation_thread = Thread(target=self.model.generate, kwargs=generation_kwargs_for_thread)
            self.generation_thread.start()
            if self.config.debug: ASCIIColors.info("Starting vision model generation stream...")

            # --- Consume Stream ---
            for new_text in streamer:
                if self._stop_generation:
                    ASCIIColors.warning("Stop generation requested.")
                    break
                if new_text:
                    output_buffer += new_text
                    if callback:
                        continue_generating = callback(new_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK)
                        if continue_generating is False:
                             self._stop_generation = True
                             ASCIIColors.warning("Generation stopped by callback returning False.")
                             break

            # Ensure thread finishes
            self.generation_thread.join()
            if self.config.debug: ASCIIColors.info("Vision generation stream finished.")

        except Exception as e:
             self.error(f"Generation Error (Vision): {e}")
             trace_exception(e)
             output_buffer += f"\n[Error during vision generation: {e}]"
             if callback: callback(f"Gen Error (Vision): {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             if self.generation_thread and self.generation_thread.is_alive():
                 self.warning("Attempting to join failed vision generation thread.")
                 self.generation_thread.join(timeout=1)
        finally:
            self.generation_thread = None
            self._stop_generation = False
            end_time = perf_counter()
            if self.config.debug: ASCIIColors.info(f"Vision generation finished in {end_time - start_time:.2f} seconds.")

        return output_buffer


    def list_models(self) -> List[str]:
        """ Lists locally downloaded models. """
        local_hf_root = self.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR
        if not local_hf_root.exists() or not local_hf_root.is_dir():
            self.info(f"Local HF directory not found: {local_hf_root}")
            return []
        model_folders = []
        try:
            # Iterate through top-level items (potential author folders like 'google')
            for author_item in local_hf_root.iterdir():
                if author_item.is_dir():
                    # Iterate through items inside author folder (actual model folders like 'gemma-3-4b-it')
                    for model_item in author_item.iterdir():
                        if model_item.is_dir():
                            # Basic check for a model directory structure
                            is_model_dir = ( (model_item / "config.json").exists() or
                                             list(model_item.glob("*.safetensors")) or
                                             list(model_item.glob("*.bin")) or
                                             list(model_item.glob("*.pth")) # Include pytorch weights
                                           )
                            if is_model_dir:
                                # Store as 'author/model_name'
                                model_folders.append(f"{author_item.name}/{model_item.name}".replace("\\", "/"))
                # Also check for models directly under HF_LOCAL_MODELS_DIR (e.g., 'my_custom_model')
                elif author_item.is_dir(): # Re-check top level items if they weren't author folders
                     is_model_dir = ( (author_item / "config.json").exists() or
                                      list(author_item.glob("*.safetensors")) or
                                      list(author_item.glob("*.bin")) or
                                      list(author_item.glob("*.pth"))
                                    )
                     if is_model_dir:
                         model_folders.append(author_item.name.replace("\\", "/"))


        except Exception as e:
            self.error(f"Error scanning models directory {local_hf_root}: {e}")
            trace_exception(e)
            return []

        model_folders.sort()
        ASCIIColors.info(f"Found {len(model_folders)} potential local HF models.")
        return model_folders


    def get_available_models(self, app: Optional[LoLLMsCom] = None) -> List[dict]:
        """ Gets available models: local + fetched from Hub. """
        lollms_models = []
        local_model_names = set(self.list_models())
        local_hf_root = self.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR
        binding_folder = binding_folder_name if binding_folder_name else binding_name.lower() # Use lowercase
        default_icon = f"/bindings/{binding_folder}/logo.png"

        # Add Local Models
        for model_name in sorted(list(local_model_names)):
            model_path = local_hf_root / model_name
            model_info = {
                "category": "local",
                "datasets": "Unknown",
                "icon": default_icon,
                "last_commit_time": None,
                "license": "Unknown",
                "model_creator": "Unknown",
                "model_creator_link": "https://huggingface.co/",
                "name": model_name,
                "provider": "local",
                "rank": 5.0, # High rank for local models
                "type": "model",
                "variants": [{"name": model_name + " (Local)", "size": -1}], # Size -1 means unknown/not fetched
                "description": "Locally downloaded Hugging Face model."
            }
            # Try to get more info if possible
            try:
                model_info["last_commit_time"] = datetime.fromtimestamp(model_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                # Simple creator inference from path 'author/model'
                if '/' in model_name:
                    creator = model_name.split('/')[0]
                    model_info["model_creator"] = creator
                    model_info["model_creator_link"] = f"https://huggingface.co/{creator}"

                # Basic check for vision capability based on known class types in config (if loaded)
                # This is limited as the model isn't fully loaded here
                config_path = model_path / "config.json"
                if config_path.exists():
                    with open(config_path, 'r') as f: config_data = json.load(f)
                    model_type = config_data.get("model_type", "").lower()
                    architectures = [a.lower() for a in config_data.get("architectures", [])]
                    is_vision = any(key in model_type or any(key in arch for arch in architectures)
                                    for key in KNOWN_MODEL_CLASSES if key != "default" and KNOWN_MODEL_CLASSES[key][1] == AutoProcessor)
                    if is_vision:
                        model_info["description"] += " (Likely Vision Capable)"
                        model_info["category"] = "local_vision" # Add a subcategory

            except Exception as local_info_ex:
                 self.warning(f"Could not get extra info for local model {model_name}: {local_info_ex}")

            lollms_models.append(model_info)

        # Fetch Models from Hub
        filtered_hub_count = 0
        favorite_providers = self.binding_config.favorite_providers.split(",")

        try:
            ASCIIColors.info("Fetching models from Hugging Face Hub...")
            api = HfApi()
            limit = self.binding_config.config.get("hub_fetch_limit", 100)
            sort_by = self.binding_config.config.get("model_sorting", "trending_score") # Use configured sorting
            # Fetch potentially relevant models: text-gen, text2text, image-to-text etc.
            hub_models_list = []
            relevant_pipelines = ["text-generation", "Image-Text-to-Text"] # Expand pipelines
            for provider in favorite_providers:
                for pipeline in relevant_pipelines:
                    try:
                        model_iterator = api.list_models(
                            task=pipeline, # Use ModelFilter for task
                            author=provider if provider else None,
                            sort=sort_by,
                            direction=-1, # Most popular/recent first
                            limit=limit # Limit per pipeline to avoid overwhelming results
                            )
                        hub_models_list.extend(list(model_iterator))
                        # Simple deduplication based on modelId
                        seen_ids = set()
                        deduped_list = []
                        for model in hub_models_list:
                            if model.modelId not in seen_ids:
                                deduped_list.append(model)
                                seen_ids.add(model.modelId)
                        hub_models_list = deduped_list

                    except Exception as pipe_ex:
                        ASCIIColors.warning(f"Could not fetch models for provider {provider}: {pipe_ex}")

            ASCIIColors.info(f"Fetched {len(hub_models_list)} unique models from Hub across relevant pipelines (before filtering).")


            for model in hub_models_list:
                try:
                    model_id = model.modelId
                    if model_id in local_model_names: continue # Skip if already local

                    # Filter out formats typically not handled by this binding directly
                    skip_keywords = ["gguf", "ggml", "-awq", "-gptq", "-exl2", "-onnx"] # More specific exclusion
                    if any(kw in model_id.lower() for kw in skip_keywords): continue

                    # Filter based on tags if present (more reliable than keywords)
                    format_tags = {'gguf', 'ggml', 'awq', 'gptq', 'onnx', 'exl2'}
                    model_tags = set(model.tags or [])
                    # Skip if it has a quantized format tag AND doesn't have a core 'transformers' or 'pytorch' tag
                    if format_tags.intersection(model_tags) and not {'transformers', 'pytorch', 'jax'}.intersection(model_tags):
                        continue

                    # Determine category based on pipeline tag
                    pipeline = model.pipeline_tag or ""
                    if any(p in pipeline for p in ["image-to-text", "visual-question-answering"]): category = "hub_vision"
                    elif any(p in pipeline for p in ["text-generation", "text2text-generation"]): category = "hub_text"
                    else: category = "hub_other" # Or skip if category is uncertain

                    # Build description string
                    description_parts = []
                    if model.downloads is not None: description_parts.append(f"Dl: {model.downloads:,}")
                    if model.likes is not None: description_parts.append(f"Likes: {model.likes:,}")
                    if model.lastModified: description_parts.append(f"Upd: {model.lastModified.split('T')[0]}")
                    description = ", ".join(description_parts)

                    entry = {
                        "category": category,
                        "datasets": "Check card", # Placeholder
                        "icon": default_icon,
                        "last_commit_time": model.lastModified,
                        "license": "Check card", # Placeholder
                        "model_creator": model.author or "Unknown",
                        "model_creator_link": f"https://huggingface.co/{model.author}" if model.author else "https://huggingface.co/",
                        "name": model_id.split("/")[1],
                        "provider": model_id.split("/")[0], # Indicate source
                        "rank": model.likes or 0, # Rank primarily by likes, fallback downloads? Use configured sort?
                        "type": "downloadable", # Mark as needing download
                        "description": description,
                        "link": f"https://huggingface.co/{model_id}",
                        "variants": [{"name": model_id + " (Hub)", "size": -1}], # Size unknown
                    }
                    lollms_models.append(entry)
                    filtered_hub_count += 1
                except Exception as ex:
                    ASCIIColors.debug(f"Model error: {ex}")

            ASCIIColors.info(f"Added {filtered_hub_count} Hub models after filtering.")

        except ImportError: self.error("huggingface_hub library not found. Cannot fetch Hub models.")
        except Exception as e: self.error(f"Failed to fetch models from Hugging Face Hub: {e}"); trace_exception(e)

        # Add fallbacks if Hub fetch failed or yielded very few results
        if filtered_hub_count < 5: # Arbitrary threshold
             self.warning(f"Hub fetch resulted in few models ({filtered_hub_count}). Adding fallback examples.")
             fallback_models = [
                 {"category": "hub_text", "name": "google/gemma-1.1-2b-it", "description":"(Fallback) Google Gemma 1.1 2B IT", "icon": default_icon, "rank": 1500, "type":"downloadable", "variants":[{"name":"google/gemma-1.1-2b-it (Hub)", "size":-1}]},
                 {"category": "hub_text", "name": "meta-llama/Meta-Llama-3-8B-Instruct", "description":"(Fallback) Meta Llama 3 8B Instruct", "icon": default_icon, "rank": 5000, "type":"downloadable", "variants":[{"name":"meta-llama/Meta-Llama-3-8B-Instruct (Hub)", "size":-1}]},
                 {"category": "hub_vision", "name": "google/paligemma-3b-mix-448", "description":"(Fallback) Google PaliGemma 3B Mix", "icon": default_icon, "rank": 1000, "type":"downloadable", "variants":[{"name":"google/paligemma-3b-mix-448 (Hub)", "size":-1}]},
                 {"category": "hub_vision", "name": "llava-hf/llava-1.5-7b-hf", "description":"(Fallback) LLaVA 1.5 7B HF", "icon": default_icon, "rank": 2000, "type":"downloadable", "variants":[{"name":"llava-hf/llava-1.5-7b-hf (Hub)", "size":-1}]},
                ]
             added_fb = 0
             for fm in fallback_models:
                 if fm["name"] not in local_model_names and fm["name"] not in {m['name'] for m in lollms_models}:
                     lollms_models.append(fm)
                     added_fb += 1
             if added_fb > 0: self.info(f"Added {added_fb} fallback examples.")

        # Sort models: local first, then rank (descending), then name (ascending)
        lollms_models.sort(key=lambda x: (
             0 if x.get('category', '').startswith('local') else 1, # Local first
            -x.get('rank', 0), # Higher rank first
            x['name'] # Alphabetical by name as tie-breaker
        ))
        ASCIIColors.success(f"Prepared {len(lollms_models)} models for Lollms UI.")
        return lollms_models


# --- Main execution block for basic testing ---
if __name__ == "__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    from lollms.types import MSG_OPERATION_TYPE

    print("Initializing LoLLMs environment for HF Local testing...")
    lollms_paths = LollmsPaths.find_paths(force_local=True, tool_prefix="test_hf_local_")
    config = LOLLMSConfig.autoload(lollms_paths)
    # Dummy LoLLMsCom for testing callbacks
    class TestCom(LoLLMsCom):
        def InfoMessage(self, msg): print(f"Info: {msg}")
        def WarningMessage(self, msg): print(f"Warning: {msg}")
        def ErrorMessage(self, msg): print(f"Error: {msg}")
        def ExceptionMessage(self, msg): print(f"Exception: {msg}")
        def ShowBlockingMessage(self, msg): print(f"Blocking Msg: {msg}")
        def HideBlockingMessage(self): print("Hide Blocking Msg")

    lollms_app_com = TestCom() # Use the dummy Com

    print("Creating HuggingFaceLocal binding instance...")
    hf_binding = HuggingFaceLocal(config, lollms_paths, installation_option=InstallOption.INSTALL_IF_NECESSARY, lollmsCom=lollms_app_com)

    # --- Test Listing ---
    print("\nListing locally available models:")
    local_models = hf_binding.list_models()
    if local_models: print("\n".join(f"- {m}" for m in local_models))
    else: print("No local models found. Please download a model (e.g., 'google/gemma-1.1-2b-it' or 'llava-hf/llava-1.5-7b-hf') into models/transformers/")

    print("\nGetting combined models list for UI (local + hub):")
    available_models_ui = hf_binding.get_available_models()
    if available_models_ui:
        print(f"Total models listed for UI: {len(available_models_ui)}")
        print("Showing top 5 and bottom 5 entries:")
        for i, model_info in enumerate(available_models_ui):
             if i < 5 or i >= len(available_models_ui) - 5:
                 cat = model_info.get('category', 'N/A')
                 rank = model_info.get('rank', 0)
                 dtype = model_info.get('type', 'N/A')
                 print(f"- {model_info['name']} (Cat: {cat}, Rank: {rank}, Type: {dtype})")
             elif i == 5: print("  ...")
    else: print("Failed to get model list for UI.")

    # --- Test Callback ---
    def test_callback(chunk: str, msg_type: int) -> bool:
        if msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK: print(chunk, end="", flush=True)
        elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION: print(f"\n## EXC: {chunk} ##"); return False # Stop on exception
        elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING: print(f"\n## WARN: {chunk} ##")
        elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_INFO: print(f"\n## INFO: {chunk} ##")
        # Return True to continue generation, False to stop
        return True

    # --- Test Loading and Generation (if a local model exists) ---
    if local_models:
        # --- Select a model to test ---
        # Prioritize vision models if available
        test_model_name = next((m for m in local_models if any(vlm in m.lower() for vlm in ['gemma-3', 'llava', 'paligemma', 'idefics'])), None)
        is_vision_test = test_model_name is not None

        if not test_model_name:
             # Fallback to a text model (try Gemma or Llama first)
             test_model_name = next((m for m in local_models if 'gemma' in m.lower() or 'llama' in m.lower()), None)
             if not test_model_name:
                 test_model_name = local_models[0] # Absolute fallback
             is_vision_test = False


        print(f"\n--- Attempting to load local model: {test_model_name} ---")
        config.model_name = test_model_name
        # Optional: Configure settings for testing
        # hf_binding.binding_config.config["quantization_bits"] = 4 # Test 4-bit if CUDA available
        # hf_binding.binding_config.config["device"] = "cuda"      # Force CUDA if desired
        hf_binding.binding_config.config["apply_chat_template"] = True # Ensure template is attempted
        hf_binding.binding_config.config["trust_remote_code"] = True # Necessary for some models like llava, USE WITH CAUTION

        hf_binding.settings_updated() # Apply changes and rebuild
        sleep(2) # Give time for rebuilding messages

        if hf_binding.model and hf_binding._get_tokenizer_or_processor():
            print(f"\n--- Model {test_model_name} loaded (Type: {hf_binding.binding_type.name}) ---")
            print(f"Effective Ctx: {hf_binding.config.ctx_size}, Effective Max Gen: {hf_binding.config.max_n_predict}")
            tokenizer_proc = hf_binding._get_tokenizer_or_processor()
            print(f"Tokenizer/Processor class: {type(tokenizer_proc).__name__}")
            print(f"Has chat template: {'Yes' if hasattr(tokenizer_proc, 'chat_template') and tokenizer_proc.chat_template else 'No'}")


            # --- Test Text Generation (using LoLLMs format) ---
            print("\n--- Testing Text Generation (with Template Formatting) ---")
            # Simple LoLLMs format prompt
            prompt_text_lollms = """!@>system:
You are a helpful AI assistant. Be concise.
!@>discussion:
!@>user:
Explain the theory of relativity in one sentence.
!@>lollms:
It states that the laws of physics are the same for all non-accelerating observers, and that the speed of light in a vacuum is constant regardless of the observer or source motion.
!@>user:
Now explain quantum entanglement simply.
"""
            # Raw prompt for comparison if template fails
            prompt_text_raw = "Explain quantum entanglement simply."

            print(f"Prompt (LoLLMs format):\n{prompt_text_lollms}\nResponse:")
            try:
                start = perf_counter()
                # Use the LoLLMs formatted prompt
                full_response = hf_binding.generate(prompt_text_lollms, n_predict=150, callback=test_callback, verbose=True)
                print(f"\n--- Text Gen Done ({perf_counter() - start:.2f}s) ---")
                # print(f"Full response received:\n{full_response}") # Already printed by callback
            except Exception as e: print(f"\nText Gen Failed: {e}"); trace_exception(e)


            # --- Test Vision Generation (if applicable) ---
            if hf_binding.binding_type == BindingType.TEXT_IMAGE and hf_binding.processor:
                print("\n--- Testing Vision Generation (with Template Formatting) ---")
                # Use a known accessible image URL
                image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat-dog.jpg"
                # Vision prompt in LoLLMs format
                prompt_vision_lollms = f"""!@>system:
You are an expert image analyst. Describe the image content accurately.
!@>discussion:
!@>user:
Describe the animals visible in the provided image. What are they doing?
"""
                print(f"Image URL: {image_url}")
                print(f"Prompt (LoLLMs format):\n{prompt_vision_lollms}\nResponse:")
                try:
                    start = perf_counter()
                    # Pass image URL(s) in the list
                    full_response = hf_binding.generate_with_images(
                        prompt_vision_lollms,
                        [image_url],
                        n_predict=100,
                        callback=test_callback,
                        verbose=True
                    )
                    print(f"\n--- Vision Gen Done ({perf_counter() - start:.2f}s) ---")
                except Exception as e: print(f"\nVision Gen Failed: {e}"); trace_exception(e)
            elif is_vision_test:
                 print("\n--- Skipping Vision Test (Model identified as vision, but failed to load processor or correct binding type) ---")
            else:
                print("\n--- Skipping Vision Test (Model detected as Text-Only) ---")
        else:
            print(f"\n--- Skipping generation tests: Failed to load model or tokenizer/processor for {test_model_name} ---")
    else:
        print("\n--- Skipping loading and generation tests: No local models found ---")
        print(f"--- Please download models to: {hf_binding.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR} ---")

    print("\nScript finished.")