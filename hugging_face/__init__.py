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
# fetches models from the Hub, and automatically infers context size.
######

import io
import json
import os
import sys
import yaml
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
    find_first_available_file_path, is_file_path
)
from PIL import Image

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
    if not pm.is_installed("torch"): print("Error: PyTorch is missing.")
    if not pm.is_installed("transformers"): print("Error: Transformers is missing.")
    if not pm.is_installed("accelerate"): print("Error: Accelerate is missing.")
    if not pm.is_installed("bitsandbytes"): print("Error: Bitsandbytes is missing.")
    if not pm.is_installed("huggingface_hub"): print("Error: huggingface_hub is missing.")
    if not pm.is_installed("PIL"): print("Error: Pillow is missing.")
    if not pm.is_installed("requests"): print("Error: requests is missing.")


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
    from huggingface_hub import HfApi, ModelFilter # Added imports
    # from accelerate import Accelerator, dispatch_model, infer_auto_device_map - Let transformers handle device_map='auto'
    if torch.cuda.is_available():
        import bitsandbytes as bnb # Only needed if CUDA is available for quantization
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
binding_folder_name = "hugging_face_local"
# Define the folder for local HF models relative to lollms_paths
HF_LOCAL_MODELS_DIR = "hugging_face"

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
    Supports text and vision models.
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

            # --- Model Discovery ---
            {"name":"hub_fetch_limit", "type":"int", "value":100, "min": 10, "max": 5000, "help":"Maximum number of models to fetch from Hugging Face Hub for the 'available models' list."},
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
            "hub_fetch_limit": 100,
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
        ASCIIColors.info("HuggingFaceLocal settings updated. Rebuilding model if necessary.")
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
        AdvancedGarbageCollector.safe_collect()

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
                    is_vision_model = True
                    ASCIIColors.info(f"Detected vision model type: {key}")
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
                bnb_config = BitsAndBytesConfig(load_in_8bit=(quantization_bits == 8), load_in_4bit=(quantization_bits == 4),
                                                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                kwargs["quantization_config"] = bnb_config; kwargs["torch_dtype"] = torch.bfloat16; ASCIIColors.info(f"Applying {quantization_bits}-bit quantization.")
            elif self.device == "cuda": kwargs["torch_dtype"] = torch.float16
            elif self.device == "mps": kwargs["torch_dtype"] = torch.float16
            else: kwargs["torch_dtype"] = torch.float32

            use_flash_attention = self.binding_config.config.get("use_flash_attention_2", False)
            if use_flash_attention and "cuda" in self.device:
                 try:
                     major, minor, _ = map(int, transformers.__version__.split('.')[:3])
                     if major >= 4 and minor >= 37: kwargs["attn_implementation"] = "flash_attention_2"; ASCIIColors.info("Attempting Flash Attention 2.")
                     else: ASCIIColors.warning("Transformers version might be too old for Flash Attention 2.")
                 except Exception: ASCIIColors.warning("Couldn't check transformers version for Flash Attention 2.")


            # --- Load Processor or Tokenizer ---
            self.info(f"Loading {ProcessorTokenizerClass.__name__} from: {model_full_path}")
            processor_tokenizer_instance = ProcessorTokenizerClass.from_pretrained(model_full_path, trust_remote_code=trust_code)
            if is_vision_model:
                self.processor = processor_tokenizer_instance
                self.tokenizer = getattr(self.processor, 'tokenizer', None)
                if not self.tokenizer: self.warning("Could not find 'tokenizer' on processor.")
            else:
                self.tokenizer = processor_tokenizer_instance; self.processor = None

            # --- Handle Missing Pad Token ---
            pad_token_to_add = None
            if self.tokenizer and self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token; ASCIIColors.warning("Tokenizer missing pad_token, setting to eos_token.")
                else:
                    pad_token_to_add = '[PAD]'
                    self.tokenizer.pad_token = pad_token_to_add
                    ASCIIColors.warning("Tokenizer missing pad_token and eos_token. Will add '[PAD]' if needed.")


            # --- Load Model ---
            self.info(f"Loading model using {ModelClass.__name__} from: {model_full_path} with config: {kwargs}")
            self.model = ModelClass.from_pretrained(model_full_path, **kwargs)

            # --- Add Pad Token and Resize Embeddings (if necessary) ---
            if pad_token_to_add and hasattr(self.model, 'resize_token_embeddings'):
                 self.warning(f"Adding special token '{pad_token_to_add}' to tokenizer and resizing model embeddings.")
                 num_added = self.tokenizer.add_special_tokens({'pad_token': pad_token_to_add})
                 if num_added > 0: self.model.resize_token_embeddings(len(self.tokenizer))
                 else: self.warning("Special token '[PAD]' may already exist.")
                 if self.tokenizer.pad_token_id is None: self.error("Failed to set pad_token_id after adding token!")


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
            if effective_max_predict >= effective_ctx_size:
                 capped_predict = max(64, effective_ctx_size - 5) # Ensure buffer and min value
                 self.warning(f"Configured max_n_predict ({effective_max_predict}) >= effective_ctx_size ({effective_ctx_size}). Capping to {capped_predict}.")
                 effective_max_predict = capped_predict

            self.config.max_n_predict = effective_max_predict
            self.binding_config.config["max_n_predict"] = effective_max_predict # Update binding config view

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


    def tokenize(self, prompt: str) -> List[int]:
        """Tokenizes the given prompt."""
        tokenizer_to_use = self.tokenizer or getattr(self.processor, 'tokenizer', None)
        if tokenizer_to_use:
            try: return tokenizer_to_use.encode(prompt)
            except Exception as e: self.error(f"Tokenization error: {e}"); trace_exception(e); return []
        else: self.error("Tokenizer or Processor not loaded."); return []


    def detokenize(self, tokens_list: List[int]) -> str:
        """Detokenizes the given list of tokens."""
        tokenizer_to_use = self.tokenizer or getattr(self.processor, 'tokenizer', None)
        if tokenizer_to_use:
            try: return tokenizer_to_use.decode(tokens_list, skip_special_tokens=True)
            except Exception as e: self.error(f"Detokenization error: {e}"); trace_exception(e); return ""
        else: self.error("Tokenizer or Processor not loaded."); return ""


    def _prepare_common_generation_kwargs(self, n_predict: int, gpt_params: dict) -> dict:
        """ Helper to prepare common generation arguments using effective n_predict. """
        # n_predict here is already the effective value (potentially capped)
        effective_max_n_predict = n_predict

        default_params = {'temperature': self.config.temperature, 'top_p': self.config.top_p,
                          'top_k': self.config.top_k, 'repetition_penalty': self.config.repetition_penalty}
        gen_kwargs = {**default_params, **gpt_params}

        tokenizer_to_use = self.tokenizer or getattr(self.processor, 'tokenizer', None)
        if not tokenizer_to_use: raise RuntimeError("Tokenizer/Processor not available for generation config.")

        final_gen_kwargs = {"max_new_tokens": effective_max_n_predict,
                            "pad_token_id": tokenizer_to_use.pad_token_id,
                            "eos_token_id": tokenizer_to_use.eos_token_id, "do_sample": True}

        if final_gen_kwargs["pad_token_id"] is None: self.warning("pad_token_id is None.")
        if final_gen_kwargs["eos_token_id"] is None: self.warning("eos_token_id is None.")

        temp = float(gen_kwargs.get('temperature', 0.7))
        if temp <= 0.01: final_gen_kwargs["do_sample"] = False; final_gen_kwargs["temperature"] = 1.0; final_gen_kwargs["top_k"] = 1
        else:
            final_gen_kwargs["temperature"] = temp
            top_p = float(gen_kwargs.get('top_p', 1.0)); top_k = int(gen_kwargs.get('top_k', 50))
            if 0.0 < top_p <= 1.0: final_gen_kwargs["top_p"] = top_p
            if top_k > 0: final_gen_kwargs["top_k"] = top_k

        rep_penalty = float(gen_kwargs.get('repetition_penalty', 1.0))
        if rep_penalty != 1.0: final_gen_kwargs["repetition_penalty"] = rep_penalty

        seed = self.binding_config.config.get("seed", -1)
        if seed != -1: torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None

        return final_gen_kwargs


    def generate(self,
                 prompt: str,
                 n_predict: Optional[int] = None,
                 callback: Optional[Callable[[str, int, dict], bool]] = None,
                 verbose: bool = False,
                 **gpt_params) -> str:
        """ Generates text (text-only focus). """
        effective_n_predict = n_predict if n_predict is not None else self.config.max_n_predict

        if self.binding_type != BindingType.TEXT_ONLY: self.warning("generate() called on vision model.")
        if not self.model or not self.tokenizer: self.error("Model/Tokenizer not loaded."); return ""

        if self.generation_thread and self.generation_thread.is_alive():
            self.warning("Stopping previous generation."); self._stop_generation = True; self.generation_thread.join(5)
        self._stop_generation = False

        try:
            final_gen_kwargs = self._prepare_common_generation_kwargs(effective_n_predict, gpt_params)
            if verbose: ASCIIColors.verbose(f"Text Gen params: {final_gen_kwargs}")

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_token_count = inputs.input_ids.shape[1]
            if input_token_count >= self.config.ctx_size - 5: raise ValueError(f"Prompt too long ({input_token_count})")
            if input_token_count + final_gen_kwargs["max_new_tokens"] > self.config.ctx_size: self.warning(f"Input+MaxNew may exceed Ctx.")
        except Exception as e: self.error(f"Input Error: {e}"); trace_exception(e); return ""

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs_for_thread = {**inputs, **final_gen_kwargs, "streamer": streamer}
        output_buffer = ""

        try: # Generation thread
            self.generation_thread = Thread(target=self.model.generate, kwargs=generation_kwargs_for_thread)
            self.generation_thread.start()
            if verbose: ASCIIColors.info("Starting text generation stream...")
            for new_text in streamer:
                if self._stop_generation: ASCIIColors.warning("Stop requested."); break
                if new_text:
                     output_buffer += new_text
                     if callback and not callback(new_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK, {}):
                         self._stop_generation = True; ASCIIColors.warning("Stopped by callback."); break
            self.generation_thread.join()
            if verbose: ASCIIColors.info("Text generation stream finished.")
        except Exception as e:
             self.error(f"Generation Error: {e}"); trace_exception(e); output_buffer += f"\n[Error: {e}]"
             if callback: callback(f"Gen Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION, {})
             if self.generation_thread and self.generation_thread.is_alive(): self.generation_thread.join(1)
        finally: self.generation_thread = None; self._stop_generation = False

        return output_buffer


    def _process_image_argument(self, image_path_or_url: str) -> Optional[Image.Image]:
        """ Loads an image from a local path or URL. """
        try:
            if is_file_path(image_path_or_url):
                valid_path = find_first_available_file_path([Path(image_path_or_url)])
                if valid_path and valid_path.exists(): return Image.open(valid_path).convert("RGB")
                else: self.warning(f"Local image not found: {image_path_or_url}"); return None
            elif image_path_or_url.startswith(("http://", "https://")):
                local_path = self.lollms_paths.personal_outputs_path/"downloads"
                if local_path and local_path.exists(): return Image.open(local_path).convert("RGB")
                else: self.warning(f"Failed download/access URL: {image_path_or_url}"); return None
            else: self.warning(f"Invalid image format: {image_path_or_url}"); return None
        except Exception as e: self.error(f"Failed load image '{image_path_or_url}': {e}"); trace_exception(e); return None


    def generate_with_images(self,
                             prompt: str,
                             images: List[str],
                             n_predict: Optional[int] = None,
                             callback: Optional[Callable[[str, int, dict], bool]] = None,
                             verbose: bool = False,
                             **gpt_params) -> str:
        """ Generates text using prompt and images (multimodal). """
        effective_n_predict = n_predict if n_predict is not None else self.config.max_n_predict

        if self.binding_type != BindingType.TEXT_IMAGE: self.warning("generate_with_images called on non-vision model."); return self.generate(prompt, effective_n_predict, callback, verbose, **gpt_params)
        if not self.model or not self.processor: self.error("Vision Model/Processor not loaded."); return ""
        if not images: self.warning("No images provided. Falling back to text-only."); return self.generate(prompt, effective_n_predict, callback, verbose, **gpt_params)

        if self.generation_thread and self.generation_thread.is_alive():
            self.warning("Stopping previous generation."); self._stop_generation = True; self.generation_thread.join(5)
        self._stop_generation = False

        loaded_pil_images: List[Image.Image] = []
        try: # Prepare inputs
            final_gen_kwargs = self._prepare_common_generation_kwargs(effective_n_predict, gpt_params)
            if verbose: ASCIIColors.verbose(f"Vision Gen params: {final_gen_kwargs}")

            failed_images = []
            for img_path in images:
                 pil_img = self._process_image_argument(img_path)
                 if pil_img: loaded_pil_images.append(pil_img)
                 else: failed_images.append(img_path)
            if not loaded_pil_images: raise ValueError("Failed to load any valid images.")
            if failed_images: self.warning(f"Skipped {len(failed_images)} images")

            # Prepare messages for processor (simple interleave)
            template_messages = [{"role": "user", "content": []}]
            if prompt: template_messages[0]["content"].append({"type": "text", "text": prompt})
            for pil_img in loaded_pil_images: template_messages[0]["content"].append(pil_img) # Pass PIL

            try: # Use apply_chat_template if possible
                inputs = self.processor.apply_chat_template(template_messages, add_generation_prompt=True,
                                                            tokenize=True, return_dict=True, return_tensors="pt").to(self.device)
            except Exception as template_ex: # Fallback to manual processing
                self.warning(f"apply_chat_template failed ({template_ex}). Processing manually.")
                inputs = self.processor(text=prompt, images=loaded_pil_images, return_tensors="pt").to(self.device)

            input_token_count = inputs.get("input_ids", torch.tensor([[]])).shape[1]
            if input_token_count == 0: self.warning("Processor returned no input_ids.")
            elif input_token_count >= self.config.ctx_size - 5: raise ValueError(f"Input too long ({input_token_count})")
            elif input_token_count + final_gen_kwargs["max_new_tokens"] > self.config.ctx_size: self.warning(f"Input+MaxNew may exceed Ctx.")

        except Exception as e:
            self.error(f"Input Error (Vision): {e}"); trace_exception(e)
            if callback: callback(f"Input Error (Vision): {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION, {})
            return ""
        finally:
            for img in loaded_pil_images: img.close()

        # --- Streaming Setup ---
        stream_tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.tokenizer
        if not stream_tokenizer: self.error("Tokenizer missing for streaming."); return ""
        streamer = TextIteratorStreamer(stream_tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs_for_thread = {**inputs, **final_gen_kwargs, "streamer": streamer}
        output_buffer = ""

        try: # Generation thread
            self.generation_thread = Thread(target=self.model.generate, kwargs=generation_kwargs_for_thread)
            self.generation_thread.start()
            if verbose: ASCIIColors.info("Starting vision generation stream...")
            for new_text in streamer:
                if self._stop_generation: ASCIIColors.warning("Stop requested."); break
                if new_text:
                     output_buffer += new_text
                     if callback and not callback(new_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK, {}):
                         self._stop_generation = True; ASCIIColors.warning("Stopped by callback."); break
            self.generation_thread.join()
            if verbose: ASCIIColors.info("Vision generation stream finished.")
        except Exception as e:
             self.error(f"Generation Error (Vision): {e}"); trace_exception(e); output_buffer += f"\n[Error: {e}]"
             if callback: callback(f"Gen Error (Vision): {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION, {})
             if self.generation_thread and self.generation_thread.is_alive(): self.generation_thread.join(1)
        finally: self.generation_thread = None; self._stop_generation = False

        return output_buffer


    def list_models(self) -> List[str]:
        """ Lists locally downloaded models. """
        local_hf_root = self.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR
        if not local_hf_root.exists() or not local_hf_root.is_dir(): return []
        model_folders = []
        try:
            for item in local_hf_root.iterdir():
                if item.is_dir() and ( (item / "config.json").exists() or list(item.glob("*.safetensors")) or list(item.glob("*.bin")) ):
                    model_folders.append(str(item.relative_to(local_hf_root)).replace("\\", "/"))
        except Exception as e: self.error(f"Error scanning models dir {local_hf_root}: {e}"); trace_exception(e); return []
        model_folders.sort(); ASCIIColors.info(f"Found {len(model_folders)} local HF models.")
        return model_folders


    def get_available_models(self, app: Optional[LoLLMsCom] = None) -> List[dict]:
        """ Gets available models: local + fetched from Hub. """
        lollms_models = []
        local_model_names = set(self.list_models())
        local_hf_root = self.lollms_paths.personal_models_path / HF_LOCAL_MODELS_DIR
        binding_folder = binding_folder_name if binding_folder_name else binding_name
        default_icon = f"/bindings/{binding_folder.lower()}/logo.png"

        # Add Local Models
        for model_name in sorted(list(local_model_names)):
            entry = {"category": "local", "datasets": "Unknown", "icon": default_icon, "last_commit_time": None,
                     "license": "Unknown", "model_creator": "Unknown", "name": model_name, "quantizer": None,
                     "rank": 5.0, "type": "model", "variants": [{"name": model_name + " (Local)", "size": -1}],
                     "model_creator_link": f"https://huggingface.co/{model_name.split('/')[0]}" if '/' in model_name else "https://huggingface.co/"}
            try: entry["last_commit_time"] = datetime.fromtimestamp((local_hf_root / model_name).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            except Exception: pass
            lollms_models.append(entry)

        # Fetch Models from Hub
        filtered_hub_count = 0
        try:
            self.info("Fetching models from Hugging Face Hub...")
            api = HfApi(); limit = self.binding_config.config.get("hub_fetch_limit", 100)
            hub_models = api.list_models(filter=["text-generation", "image-to-text","image-text-to-text"], sort=self.binding_config.config.get("model_sorting", "trending_score"), direction=-1, limit=limit)
            
            self.info(f"Fetched models from Hub (before filtering).")

            for model in hub_models:
                model_id = model.modelId
                if model_id in local_model_names: continue
                skip_keywords = ["gguf", "gptq", "awq", "ggml", "-onnx"]
                if any(kw in model_id.lower() for kw in skip_keywords): continue
                format_tags = {'gguf', 'gptq', 'awq', 'ggml'}; model_tags = set(model.tags or [])
                if format_tags.intersection(model_tags) and not {'transformers', 'pytorch'}.intersection(model_tags): continue

                category = "hub_vision" if "image-to-text" in (model.pipeline_tag or "") else "hub_text"
                description = f"Downloads: {model.downloads or 'N/A'}" + (f", Updated: {model.lastModified.split('T')[0]}" if model.lastModified else "")
                entry = {"category": category, "datasets": "Check card", "icon": default_icon, "last_commit_time": model.lastModified,
                         "license": "Check card", "model_creator": model.author or "Unknown", "name": model_id, "quantizer": None,
                         "rank": 1.0 + (model.downloads / 1e7 if model.downloads else 0), "type": "downloadable", "description": description,
                         "link": f"https://huggingface.co/{model_id}", "variants": [{"name": model_id + " (Hub)", "size": -1}],
                         "model_creator_link": f"https://huggingface.co/{model.author}" if model.author else "https://huggingface.co/"}
                lollms_models.append(entry); filtered_hub_count += 1
            self.info(f"Added {filtered_hub_count} Hub models after filtering.")

        except ImportError: self.error("huggingface_hub not found.")
        except Exception as e: self.error(f"Failed fetch Hub models: {e}"); trace_exception(e)

        # Add fallbacks if Hub fetch failed/yielded few results
        if filtered_hub_count == 0:
             fallback_models = [
                 {"category": "hub_text", "name": "google/gemma-1.1-2b-it", "description":"(Fallback) Google Gemma 2B IT", "icon": default_icon, "rank": 1.5, "type":"downloadable", "variants":[{"name":"gemma-1.1-2b-it (Hub)", "size":-1}]},
                 {"category": "hub_text", "name": "meta-llama/Meta-Llama-3-8B-Instruct", "description":"(Fallback) Meta Llama 3 8B Instruct", "icon": default_icon, "rank": 1.4, "type":"downloadable", "variants":[{"name":"Meta-Llama-3-8B-Instruct (Hub)", "size":-1}]},
                 {"category": "hub_vision", "name": "google/gemma-3-4b-it", "description":"(Fallback) Google Gemma 3 4B Vision IT", "icon": default_icon, "rank": 1.6, "type":"downloadable", "variants":[{"name":"gemma-3-4b-it (Hub)", "size":-1}]},
                 {"category": "hub_vision", "name": "Salesforce/blip-image-captioning-large", "description":"(Fallback) Salesforce BLIP Captioning", "icon": default_icon, "rank": 1.2, "type":"downloadable", "variants":[{"name":"blip-image-captioning-large (Hub)", "size":-1}]}, ]
             added_fb = sum(1 for fm in fallback_models if fm["name"] not in local_model_names and (lollms_models.append(fm) or True))
             if added_fb > 0: self.warning(f"Hub fetch failed/sparse. Added {added_fb} fallback examples.")

        # Sort models: local first, then rank, then name
        lollms_models.sort(key=lambda x: (x.get('category') != 'local', -x.get('rank', 1.0), x['name']))
        ASCIIColors.success(f"Formatted {len(lollms_models)} models for Lollms UI.")
        return lollms_models


# --- Main execution block for basic testing ---
if __name__ == "__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    from lollms.types import MSG_OPERATION_TYPE
    from time import perf_counter

    print("Initializing LoLLMs environment for HF Local testing...")
    lollms_paths = LollmsPaths.find_paths(force_local=True, tool_prefix="test_hf_local_")
    config = LOLLMSConfig.autoload(lollms_paths)
    lollms_app = LollmsApplication("TestApp", config, lollms_paths, load_bindings=False, load_personalities=False, load_models=False)

    print("Creating HuggingFaceLocal binding instance...")
    hf_binding = HuggingFaceLocal(config, lollms_paths, installation_option=InstallOption.INSTALL_IF_NECESSARY, lollmsCom=lollms_app.com)

    # --- Test Listing ---
    print("\nListing locally available models:")
    local_models = hf_binding.list_models()
    if local_models: print("\n".join(f"- {m}" for m in local_models))
    else: print("No local models found.")

    print("\nGetting combined models list for UI (local + hub):")
    available_models_ui = hf_binding.get_available_models()
    if available_models_ui:
        print(f"Total models listed for UI: {len(available_models_ui)}")
        print("Showing top 10 entries:")
        for model_info in available_models_ui[:10]:
             cat = model_info.get('category', 'N/A')
             rank = model_info.get('rank', 'N/A')
             print(f"- {model_info['name']} (Cat: {cat}, Rank: {rank:.2f})")
    else: print("Failed to get model list for UI.")

    # --- Test Callback ---
    def test_callback(chunk: str, msg_type: int, metadata: dict) -> bool:
        if msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK: print(chunk, end="", flush=True)
        elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION: print(f"\n## EXC: {chunk} ##"); return False
        elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING: print(f"\n## WARN: {chunk} ##")
        return True

    # --- Test Loading and Generation (if a local model exists) ---
    if local_models:
        # Try to find a vision model first for more comprehensive test
        test_model_name = next((m for m in local_models if any(vlm in m.lower() for vlm in ['gemma-3', 'llava', 'paligemma'])), None)
        if not test_model_name:
            test_model_name = local_models[0] # Fallback to first local model

        print(f"\n--- Attempting to load local model: {test_model_name} ---")
        config.model_name = test_model_name
        # Optional: Configure quantization/device for testing
        # hf_binding.binding_config.config["quantization_bits"] = 4
        # hf_binding.binding_config.config["device"] = "cuda"
        # hf_binding.binding_config.config["auto_infer_ctx_size"] = True # Ensure auto-infer is on

        hf_binding.build_model()

        if hf_binding.model:
            print(f"\n--- Model {test_model_name} loaded (Type: {hf_binding.binding_type.name}) ---")
            print(f"Effective Ctx: {hf_binding.config.ctx_size}, Effective Max Gen: {hf_binding.config.max_n_predict}")

            # --- Test Text Generation ---
            print("\n--- Testing Text Generation ---")
            prompt_text = "Explain quantum entanglement in simple terms."
            print(f"Prompt: {prompt_text}\nResponse:")
            try:
                start = perf_counter()
                hf_binding.generate(prompt_text, n_predict=150, callback=test_callback, verbose=True)
                print(f"\n--- Text Gen Done ({perf_counter() - start:.2f}s) ---")
            except Exception as e: print(f"\nText Gen Failed: {e}"); trace_exception(e)

            # --- Test Vision Generation (if model supports it) ---
            if hf_binding.binding_type == BindingType.TEXT_IMAGE:
                print("\n--- Testing Vision Generation ---")
                image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/cat-dog.jpg"
                prompt_vision = "Describe the animals in the image."
                print(f"Image URL: {image_url}")
                print(f"Prompt: {prompt_vision}\nResponse:")
                try:
                    start = perf_counter()
                    hf_binding.generate_with_images(prompt_vision, [image_url], n_predict=100, callback=test_callback, verbose=True)
                    print(f"\n--- Vision Gen Done ({perf_counter() - start:.2f}s) ---")
                except Exception as e: print(f"\nVision Gen Failed: {e}"); trace_exception(e)
            else:
                print("\n--- Skipping Vision Test (Model detected as Text-Only) ---")
        else:
            print(f"\n--- Skipping generation tests: Failed to load model {test_model_name} ---")
    else:
        print("\n--- Skipping loading and generation tests: No local models found ---")

    print("\nScript finished.")