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
from typing import Callable, Any, List
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors, trace_exception
from lollms.types import MSG_OPERATION_TYPE
from lollms.utilities import (
    AdvancedGarbageCollector,
    run_cmd,
)
import subprocess
from datetime import datetime
from tqdm import tqdm
import sys
import urllib
import json
import os
import gc
import pipmaster as pm

# Suppress warnings from huggingface_hub
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

pm.install_if_missing("Pillow")
from PIL import Image

# Check and Install/Upgrade Core Libraries
# Consider moving pipmaster/direct pip installs to the `install` method for cleaner separation.
try:
    import pipmaster as pm
    if not pm.is_installed("torch"):
        ASCIIColors.yellow("HuggingFace: PyTorch not found. Installing...")
        # Attempt auto-detection or ask user? For now, default to CUDA 12.1 if GPU detected.
        if torch.cuda.is_available():
            pm.install_multiple(["torch","torchvision","torchaudio"], "https://download.pytorch.org/whl/cu121", force_reinstall=True)
        else:
             pm.install_multiple(["torch","torchvision","torchaudio"], force_reinstall=True) # Install CPU version

    import torch
    if torch.cuda.is_available() and not str(torch.version.cuda).startswith("12."):
         ASCIIColors.warning(f"HuggingFace: Detected CUDA version {torch.version.cuda} but recommending 12.x.")
         # Optionally prompt for reinstall:
         # if show_yes_no_dialog("CUDA Mismatch", "Recommended CUDA version is 12.x, but found {torch.version.cuda}. Reinstall PyTorch with CUDA 12.1?"):
         #     reinstall_pytorch_with_cuda() # Assumes cu121

except ImportError:
    ASCIIColors.error("HuggingFace: pipmaster not found. Cannot auto-install PyTorch.")
    # Fallback or guide user
except Exception as e:
    ASCIIColors.error(f"HuggingFace: Error during PyTorch check/install: {e}")
    # Fallback or guide user

# Check other core dependencies
pm.install_if_missing("transformers")
pm.install_if_missing("accelerate")
pm.install_if_missing("bitsandbytes")
pm.install_if_missing("huggingface_hub")
pm.install_if_missing("safetensors")


# Conditional import for Flash Attention
try:
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        if not pm.is_installed("flash-attn"):
             # Recommend manual install due to potential build complexities
            ASCIIColors.yellow("HuggingFace: Flash Attention 2 not installed. For optimal performance on Ampere+ GPUs, consider installing it manually (pip install flash-attn --no-build-isolation).")
            _flash_attn_available = False
        else:
             import flash_attn
             _flash_attn_available = True
    else:
        _flash_attn_available = False
except ImportError:
    _flash_attn_available = False
except Exception as e:
    ASCIIColors.warning(f"HuggingFace: Couldn't check/import flash_attn: {e}")
    _flash_attn_available = False


# Now import the heavy libraries
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        GenerationConfig,
        AutoConfig,
        AutoProcessor,
        BitsAndBytesConfig
    )
    # Conditional import based on model type check later
    # from transformers import LlavaForConditionalGeneration
    # from transformers import GPTQConfig # If needed explicitly
    # from transformers import AwqConfig # If needed explicitly
    from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files

except ImportError as e:
    ASCIIColors.error(f"HuggingFace: Failed to import core libraries: {e}")
    ASCIIColors.error("Please ensure PyTorch, transformers, accelerate, bitsandbytes, and huggingface_hub are installed correctly.")
    # Optionally trigger installation here if possible, or raise the error.
    raise e


import shutil

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, ParisNeo"
__license__ = "Apache 2.0"

binding_name = "HuggingFace"
binding_folder_name = "hugging_face"

# ================================== Helper Functions ==================================
def get_device_map_options():
    """Generates the list of device map options."""
    options = ['auto', 'cpu', 'balanced', 'balanced_low_0', 'sequential']
    try:
        import torch
        if torch.cuda.is_available():
            options.extend([f'cuda:{i}' for i in range(torch.cuda.device_count())])
        if torch.backends.mps.is_available():
            options.append('mps') # Add MPS if available
    except Exception as e:
        ASCIIColors.warning(f"Could not detect GPU devices: {e}")
    return options

def get_torch_dtype_options():
    """Gets available torch data types."""
    options = ["auto", "float32"]
    try:
        import torch
        if torch.cuda.is_available():
            options.append("float16")
            if torch.cuda.is_bf16_supported():
                options.append("bfloat16")
    except:
        pass
    return options

def get_attn_implementation_options():
    """Gets available attention implementation options."""
    options = ["eager", "sdpa"] # sdpa is generally good and widely available
    if _flash_attn_available:
        options.append("flash_attention_2")
    return options

# ================================== Binding Class ==================================
class HuggingFace(LLMBinding):

    def __init__(self,
                 config: LOLLMSConfig,
                 lollms_paths: LollmsPaths = None,
                 installation_option: InstallOption = InstallOption.INSTALL_IF_NECESSARY,
                 lollmsCom=None) -> None:
        """
        Builds a HuggingFace binding.

        Args:
            config (LOLLMSConfig): The configuration file.
            lollms_paths (LollmsPaths, optional): Paths configuration. Defaults to None.
            installation_option (InstallOption, optional): Installation behavior. Defaults to InstallOption.INSTALL_IF_NECESSARY.
            lollmsCom (LollmsCom, optional): Communication object. Defaults to None.
        """
        if lollms_paths is None:
            lollms_paths = LollmsPaths()

        # Set environment variable for memory optimization early
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True' # Alternative to max_split_size_mb

        # Configuration Template
        binding_config_template = ConfigTemplate([
            # Model Loading Strategy
            {"name": "quantization_mode", "type": "str", "value": "none", "options": ["none", "8bit", "4bit"], "help": "Load model quantized to 8 or 4 bits using bitsandbytes. Select 'none' for full precision (requires more VRAM)."},
            {"name": "torch_dtype", "type": "str", "value": "auto", "options": get_torch_dtype_options(), "help": "Torch data type (e.g., float16, bfloat16). 'auto' selects the best available."},
            {"name": "trust_remote_code", "type": "bool", "value": False, "help": "Trust and execute remote code from model authors (use with caution)."},
            {"name": "device_map", "type": "str", "value": 'auto', "options": get_device_map_options(), "help": "Device mapping strategy for distributing model layers across devices."},
            {"name": "attn_implementation", "type": "str", "value": "sdpa", "options": get_attn_implementation_options(), "help": "Attention implementation. 'flash_attention_2' is fastest on compatible GPUs."},
            # {"name": "low_cpu_mem_usage", "type": "bool", "value": True, "help": "Attempt to reduce CPU memory usage during model loading (may be slower)."}, # This is often handled by device_map='auto' and accelerate

            # LoRA (Keep commented out or add proper implementation later)
            # {"name":"lora_file","type":"str","value":"", "help":"Path to LoRA adapter file (requires peft library). Experimental."},

            # Generation Parameters
            {"name": "ctx_size", "type": "int", "value": 4096, "min": 512, "help": "Model context window size. Check the model card for its supported size."},
            {"name": "max_n_predict", "type": "int", "value": 1024, "min": 64, "help": "Maximum number of tokens to generate per request."},
            {"name": "seed", "type": "int", "value": -1, "help": "Random seed for generation (-1 for random)."},

            # Installation/Model Management
            {"name": "hf_repo_id", "type": "str", "value": "google/gemma-1.1-2b-it", "help": "Hugging Face repository ID (e.g., 'google/gemma-1.1-2b-it', 'TheBloke/Mistral-7B-Instruct-v0.2-GPTQ'). This will be used if no specific model is selected in the main LoLLMs settings."},
            # You might still need model_name in the main config, hf_repo_id is a fallback/default here

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
            supported_file_extensions=['.safetensors', '.bin'], # Focus on standard HF formats
            models_dir_names=["transformers", "gptq", "awq"], # Keep these for legacy/structure
            lollmsCom=lollmsCom
        )
        # Sync context/prediction sizes with main config
        self.config.ctx_size = self.binding_config.config.ctx_size
        self.config.max_n_predict = self.binding_config.max_n_predict

        # Initialization state
        self.model = None
        self.tokenizer = None
        self.processor = None # For multi-modal models
        self.generation_config = None
        self.model_device = 'cpu'

        # Streaming state
        self.callback = None
        self.streaming_buffer = ""
        self.first_token_generated = False


    def settings_updated(self):
        """Called when binding settings are updated."""
        self.config.ctx_size = self.binding_config.ctx_size
        self.config.max_n_predict = self.binding_config.max_n_predict
        # Potentially trigger model reload if significant settings changed (device_map, quantization)
        if self.model:
             ASCIIColors.info("HuggingFace binding settings updated. Model reload might be necessary for some changes to take effect.")
             # Example: Force reload if quantization changed
             # current_quant = ... # Need to store quantization state during load
             # if self.binding_config.quantization_mode != current_quant:
             #     self.destroy_model() # Or implement a more graceful update if possible
        # Note: AutoGPTQ max length update was specific, general HF models usually adapt

    def embed(self, text: str) -> List[float]:
        """
        Computes text embedding. (Basic Implementation - Not Optimized)

        Args:
            text (str): The text to be embedded.

        Returns:
            List[float]: The embedding vector.

        Raises:
            NotImplementedError: If the model or tokenizer is not loaded, or if embedding logic needs refinement.
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be loaded before embedding.")

        try:
            # Use pooler output or mean of last hidden states
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.model_device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Option 1: Use last hidden state mean pooling
                embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().tolist()
                # Option 2: If the model has a pooler (common in BERT-like models, less in GPT-like)
                # if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                #     embedding = outputs.pooler_output.squeeze().cpu().tolist()
            return embedding
        except Exception as e:
            trace_exception(e)
            raise NotImplementedError(f"Embedding failed: {e}. This model might not be suitable for simple embedding extraction, or an error occurred.")


    def __del__(self):
        """Destructor: Clean up resources."""
        self.destroy_model()

    def destroy_model(self):
        """Safely clean up the model and tokenizer resources."""
        ASCIIColors.info("Destroying HuggingFace model and tokenizer...")
        if hasattr(self, "tokenizer") and self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        if hasattr(self, "model") and self.model:
            del self.model
            self.model = None
        if hasattr(self, "processor") and self.processor:
            del self.processor
            self.processor = None

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                ASCIIColors.info("CUDA cache cleared.")
            except Exception as e:
                ASCIIColors.error(f"Failed to clear CUDA cache: {e}")
        # Hard collection can be risky, use sparingly if needed
        # AdvancedGarbageCollector.safeHardCollectMultiple(['model', 'tokenizer', 'processor'], self)

    def build_model(self, model_name=None):
        """Builds the Hugging Face model and tokenizer."""
        super().build_model(model_name) # Handles selection priority (user > binding default)

        # Use the model name selected by the user via main config, or fallback to binding default
        effective_model_name = self.config.model_name or self.binding_config.hf_repo_id
        if not effective_model_name:
            self.Error("No model name specified in main configuration or binding default.")
            return None

        ASCIIColors.info(f"Building HuggingFace model: {effective_model_name}")
        self.ShowBlockingMessage(f"Loading model: {effective_model_name}\nPlease wait...")

        # Determine the actual path (Hugging Face ID or local path)
        model_path_obj = self.get_model_path() # Resolves local paths if available
        if model_path_obj and model_path_obj.exists():
            model_identifier = str(model_path_obj)
            ASCIIColors.info(f"Found local model path: {model_identifier}")
        else:
            model_identifier = effective_model_name # Use HF repo ID
            ASCIIColors.info(f"Using Hugging Face repository ID: {model_identifier}")
            # Check if we need to download it (snapshot_download handles this implicitly)
            # but we might want to show progress earlier if needed.

        # Clean up any previous model
        self.destroy_model()

        try:
            # --- Configuration ---
            hf_config = AutoConfig.from_pretrained(
                model_identifier,
                trust_remote_code=self.binding_config.trust_remote_code,
            )

            # --- Determine Model Type (Basic check) ---
            is_vision_model = "vision" in hf_config.model_type.lower() or \
                              any("llava" in arch.lower() for arch in getattr(hf_config, "architectures", []))

            # --- Tokenizer / Processor ---
            ASCIIColors.info("Loading tokenizer/processor...")
            if is_vision_model:
                # Use AutoProcessor for multi-modal models
                self.processor = AutoProcessor.from_pretrained(
                    model_identifier,
                    trust_remote_code=self.binding_config.trust_remote_code
                )
                self.tokenizer = self.processor.tokenizer # Often the processor wraps the tokenizer
                self.binding_type = BindingType.TEXT_IMAGE
                ASCIIColors.info("Loaded AutoProcessor for Vision model.")
            else:
                # Standard tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_identifier,
                    trust_remote_code=self.binding_config.trust_remote_code,
                    padding_side="left" # Important for generation
                )
                self.binding_type = BindingType.TEXT_ONLY
                ASCIIColors.info("Loaded AutoTokenizer.")

            # --- Quantization Configuration ---
            quantization_config = None
            load_in_8bit = False
            load_in_4bit = False
            if self.binding_config.quantization_mode == "8bit":
                 # Requires bitsandbytes
                if pm.is_installed("bitsandbytes"):
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    load_in_8bit = True # Also set flag for AutoModel
                    ASCIIColors.info("Using 8-bit quantization (bitsandbytes).")
                else:
                    self.warning("8-bit quantization requested, but 'bitsandbytes' is not installed. Loading in full precision.")
            elif self.binding_config.quantization_mode == "4bit":
                if pm.is_installed("bitsandbytes"):
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 if self.binding_config.torch_dtype == "bfloat16" else torch.float16, # Match compute dtype
                        bnb_4bit_quant_type="nf4", # Common default
                        bnb_4bit_use_double_quant=True, # Common default
                    )
                    load_in_4bit = True # Also set flag for AutoModel
                    ASCIIColors.info("Using 4-bit quantization (bitsandbytes).")
                else:
                    self.warning("4-bit quantization requested, but 'bitsandbytes' is not installed. Loading in full precision.")

            # --- Torch Dtype ---
            torch_dtype = getattr(torch, self.binding_config.torch_dtype, None) if self.binding_config.torch_dtype != "auto" else None
            if torch_dtype is None and self.binding_config.torch_dtype == "auto":
                 # Auto select best available type
                 if torch.cuda.is_available():
                     if torch.cuda.is_bf16_supported():
                         torch_dtype = torch.bfloat16
                         ASCIIColors.info("Auto-selected torch dtype: bfloat16")
                     else:
                         torch_dtype = torch.float16
                         ASCIIColors.info("Auto-selected torch dtype: float16")
                 else:
                     torch_dtype = torch.float32
                     ASCIIColors.info("Auto-selected torch dtype: float32 (CPU)")


            # --- Attention Implementation ---
            attn_implementation = self.binding_config.attn_implementation
            if attn_implementation == "flash_attention_2" and not _flash_attn_available:
                ASCIIColors.warning("Flash Attention 2 requested but not available/functional. Falling back to 'sdpa'.")
                attn_implementation = "sdpa"

            # --- Model Loading ---
            ASCIIColors.info(f"Loading model weights ({self.binding_config.quantization_mode}, {self.binding_config.torch_dtype}, {attn_implementation})...")
            model_load_args = {
                "pretrained_model_name_or_path": model_identifier,
                "config": hf_config, # Pass the loaded config
                "quantization_config": quantization_config,
                "device_map": self.binding_config.device_map if self.binding_config.device_map != "cpu" else None, # device_map='cpu' can cause issues
                "trust_remote_code": self.binding_config.trust_remote_code,
                # "low_cpu_mem_usage": self.binding_config.low_cpu_mem_usage, # Often handled by accelerate
                "torch_dtype": torch_dtype,
                # Only add attn_implementation if not None or default 'eager'
                **({"attn_implementation": attn_implementation} if attn_implementation != "eager" else {})
            }

            # Handle device explicitly if not using device_map
            target_device = torch.device("cuda" if torch.cuda.is_available() and self.binding_config.device_map != 'cpu' else "cpu")
            if self.binding_config.device_map == "cpu":
                 model_load_args['device_map'] = None # Don't use device_map for CPU-only

            # Load appropriate AutoModel class
            if is_vision_model:
                # Dynamically import Llava if needed
                try:
                    from transformers import LlavaForConditionalGeneration
                    self.model = LlavaForConditionalGeneration.from_pretrained(**model_load_args)
                    ASCIIColors.info("Loaded LlavaForConditionalGeneration.")
                except ImportError:
                    self.error("Failed to import LlavaForConditionalGeneration. Is 'transformers' fully updated?")
                    raise
                except Exception as e: # Catch potential loading errors specific to Llava
                     self.error(f"Failed loading vision model ({e.__class__.__name__}): {e}")
                     raise
            elif "gptq" in model_identifier.lower() or getattr(hf_config, "quantization_config", {}).get("quant_method") == "gptq":
                # GPTQ requires specific handling (often device_map and quantization_config interaction)
                 try:
                    # GPTQ models usually don't use BitsAndBytes config directly
                    model_load_args.pop("quantization_config", None)
                    # device_map is crucial for GPTQ
                    if not model_load_args.get("device_map"):
                        model_load_args["device_map"] = "auto"
                    self.model = AutoModelForCausalLM.from_pretrained(**model_load_args)
                    ASCIIColors.info("Loaded AutoModelForCausalLM for GPTQ model.")
                 except Exception as e:
                     self.error(f"Failed loading GPTQ model ({e.__class__.__name__}): {e}")
                     self.warning("Hints: Ensure 'auto-gptq' might be needed (install manually if required). Check device_map setting.")
                     raise
            elif "awq" in model_identifier.lower() or getattr(hf_config, "quantization_config", {}).get("quant_method") == "awq":
                # AWQ specific handling
                try:
                     # AWQ might need specific config or library, handled by AutoModel
                    model_load_args.pop("quantization_config", None) # AutoAWQ might handle it internally
                    if not model_load_args.get("device_map"):
                        model_load_args["device_map"] = "auto"
                    self.model = AutoModelForCausalLM.from_pretrained(**model_load_args)
                    ASCIIColors.info("Loaded AutoModelForCausalLM for AWQ model.")
                except Exception as e:
                    self.error(f"Failed loading AWQ model ({e.__class__.__name__}): {e}")
                    self.warning("Hints: Ensure 'autoawq' might be needed (install manually if required). Check device_map setting.")
                    raise
            else:
                # Standard Causal LM
                self.model = AutoModelForCausalLM.from_pretrained(**model_load_args)
                ASCIIColors.info("Loaded AutoModelForCausalLM.")

            # If not using device_map, explicitly move model to device
            if self.binding_config.device_map == "cpu":
                 self.model.to(target_device)
                 self.model_device = str(target_device)
                 ASCIIColors.info(f"Model moved to CPU explicitly.")
            elif not model_load_args.get("device_map"): # If device_map wasn't used (e.g., single GPU)
                try:
                    self.model.to(target_device)
                    self.model_device = str(target_device)
                    ASCIIColors.info(f"Model moved to {self.model_device} explicitly.")
                except Exception as e:
                     ASCIIColors.warning(f"Could not explicitly move model to {target_device}: {e}. Relying on internal placement.")
                     try:
                        self.model_device = str(next(self.model.parameters()).device)
                     except:
                        self.model_device = 'unknown'


            else:
                 # Get device from the first parameter if using device_map
                 try:
                     self.model_device = str(next(self.model.parameters()).device)
                     ASCIIColors.info(f"Model device detected from parameters: {self.model_device} (using device_map='{self.binding_config.device_map}')")
                 except Exception as e:
                     ASCIIColors.warning(f"Could not detect model device from parameters: {e}")
                     self.model_device = 'unknown' # Or default to cuda:0 if expected

            # --- Generation Config ---
            try:
                # Try loading from model path first, then from identifier
                gen_config_path = model_path_obj / "generation_config.json" if model_path_obj and model_path_obj.exists() else None
                if gen_config_path and gen_config_path.exists():
                    self.generation_config = GenerationConfig.from_pretrained(str(model_path_obj))
                    ASCIIColors.info("Loaded generation config from local path.")
                else:
                    self.generation_config = GenerationConfig.from_pretrained(model_identifier)
                    ASCIIColors.info("Loaded generation config from Hugging Face repo.")
            except Exception as e:
                ASCIIColors.warning(f"Could not load generation config ({e}). Using default.")
                self.generation_config = GenerationConfig(
                    max_new_tokens=self.binding_config.max_n_predict,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id, # Handle missing pad token
                    do_sample=True, # Default to sampling
                    temperature=0.7, # Default temp
                    top_p=0.9, # Default top_p
                    # Add other defaults as needed
                )
            # Ensure pad_token is set if missing (important for batching/generation)
            if self.tokenizer.pad_token_id is None:
                 self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                 self.model.config.pad_token_id = self.tokenizer.eos_token_id
                 self.generation_config.pad_token_id = self.tokenizer.eos_token_id
                 ASCIIColors.warning("Tokenizer missing pad_token_id. Setting to eos_token_id.")


            ASCIIColors.success(f"Model {effective_model_name} built successfully. Device: {self.model_device}")
            self.HideBlockingMessage()
            return self

        except Exception as e:
            trace_exception(e)
            self.error(f"Error building model {effective_model_name}: {e}")
            self.error("Possible causes: Incorrect model name, network issues, insufficient VRAM/RAM, missing dependencies (bitsandbytes, accelerate), or incompatible model configuration.")
            self.destroy_model() # Clean up partial load
            self.HideBlockingMessage()
            return None

    # ================================= Installation ==================================

    def install(self):
        """Installs necessary dependencies for the Hugging Face binding."""
        self.ShowBlockingMessage("Installing HuggingFace requirements...")
        ASCIIColors.info("Checking and installing HuggingFace dependencies...")

        # Free memory before installations
        self.destroy_model() # Ensure model is unloaded
        AdvancedGarbageCollector.collect()

        requirements_file = self.binding_dir / "requirements.txt"
        if requirements_file.exists():
             try:
                ASCIIColors.info(f"Installing dependencies from {requirements_file}")
                # Use -qq for quieter output, capture errors if needed
                run_cmd([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)],
                        f"Install {binding_name} dependencies",
                        show_stdout=True) # Show output for user feedback
                ASCIIColors.success("Dependencies installed successfully.")
             except subprocess.CalledProcessError as e:
                 self.error(f"Failed to install dependencies: {e}")
                 self.ShowBlockingMessage(f"Error installing dependencies.\nPlease check the console log.\nTry installing manually:\n pip install --upgrade --no-cache-dir -r {requirements_file}")
                 # Consider raising the error or returning False
                 self.HideBlockingMessage()
                 return False
             except Exception as e:
                 self.error(f"An unexpected error occurred during dependency installation: {e}")
                 trace_exception(e)
                 self.HideBlockingMessage()
                 return False
        else:
            ASCIIColors.warning("requirements.txt not found. Skipping pip install.")
            self.warning("Binding may not function correctly without required packages.")

        # Conditional PyTorch Reinstallation (Optional - Keep or Remove based on desired behavior)
        self.ShowBlockingMessage("Checking PyTorch installation...")
        try:
            import torch
            if torch.cuda.is_available():
                 ASCIIColors.info(f"PyTorch with CUDA {torch.version.cuda} found.")
                 # Example: Prompt user to reinstall if CUDA version mismatch detected
                 # Install CUDA toolkit version based on torch cuda version
                 major, minor = torch.version.cuda.split(".")[:2]
                 cuda_version = f"{major}.{minor}"

            elif torch.backends.mps.is_available():
                 ASCIIColors.info("PyTorch with MPS (Apple Silicon) found.")
            else:
                 ASCIIColors.info("PyTorch CPU version found.")
            # Add checks/reinstall options for ROCm if needed
            # elif platform.system() == "Linux" and is_rocm_available(): # Need is_rocm_available helper
            #     if show_yes_no_dialog("PyTorch ROCm", "Reinstall PyTorch with latest ROCm support?"):
            #         reinstall_pytorch_with_rocm()

        except ImportError:
            self.error("PyTorch not found after installation attempt. Please install it manually.")
            self.ShowBlockingMessage("PyTorch installation failed. Please check logs and install manually.")
            self.HideBlockingMessage()
            return False
        except Exception as e:
             self.error(f"Error during PyTorch check/CUDA setup: {e}")
             trace_exception(e)
             self.ShowBlockingMessage(f"Error during PyTorch/CUDA check: {e}")
             self.HideBlockingMessage()
             # Decide whether to proceed or fail installation

        # Update configuration options after install (device list might change)
        self.binding_config.template.find("device_map").options = get_device_map_options()
        self.binding_config.template.find("torch_dtype").options = get_torch_dtype_options()
        self.binding_config.template.find("attn_implementation").options = get_attn_implementation_options()
        self.binding_config.config.device_map = 'auto' # Reset to auto after install
        self.binding_config.config.attn_implementation = "sdpa" if "sdpa" in get_attn_implementation_options() else "eager" # Sensible default
        self.binding_config.save() # Save updated template/config

        ASCIIColors.success(f"{binding_name} binding installed successfully.")
        self.HideBlockingMessage()
        return True

    def uninstall(self):
        """Uninstalls the binding (placeholder)."""
        super().uninstall()
        ASCIIColors.info(f"Uninstalling {binding_name} binding...")
        # Add any specific cleanup if necessary, like removing large cached files
        ASCIIColors.success(f"{binding_name} binding uninstalled (dependency removal is manual).")


    # ================================== Model Download ==================================
    def get_file_size(self, hf_repo_id: str) -> int:
        """
        Gets the total size of model files to be downloaded for a Hugging Face repository.

        Args:
            hf_repo_id (str): The Hugging Face repository ID.

        Returns:
            int: Total size in bytes, or 0 if unable to determine.
        """
        try:
            files_info = list_repo_files(hf_repo_id, repo_type="model")
            total_size = 0
            # Sum size of likely model weights files
            weight_extensions = (".safetensors", ".bin", ".pt", ".pth")
            # Filter out common non-weight files if necessary (e.g., README, config)
            ignore_patterns = ["README", ".gitattributes", "config.json", "tokenizer", "vocab", "merges", "LICENSE", ".md"]

            # Prioritize safetensors if available
            has_safetensors = any(f.lower().endswith(".safetensors") for f in files_info)

            for filename in files_info:
                # Skip common non-model files
                if any(pattern in filename for pattern in ignore_patterns):
                    continue

                # If safetensors exist, skip .bin files
                if has_safetensors and filename.lower().endswith(".bin"):
                    continue

                # Consider only likely weight files (can be refined)
                if filename.lower().endswith(weight_extensions):
                     try:
                         # Get specific file info (more reliable for size)
                         file_info = hf_hub_download(repo_id=hf_repo_id, filename=filename, repo_type="model", resume_download=True, etag_timeout=60, local_files_only=True) # Try local first
                         # If not local, need to fetch info - hf_hub_download doesn't easily give size without downloading.
                         # Alternative: use requests HEAD on the URL - more complex
                         # Let's estimate based on list_repo_files (less accurate) - requires adding size info if available
                         # Or just report the size of the largest file as a rough estimate?

                         # Using a placeholder size for now as getting exact total size is tricky without download/API calls
                         # Placeholder: Assume a large file size for progress display.
                         # A better approach requires parsing the repo page or using a library feature if available.
                         # For now, returning a fixed large number or size of largest file.
                         if file_info: # If file info was available locally (not standard use)
                              total_size += os.path.getsize(file_info)
                         else:
                              # Crude fallback: find largest file size from URL (if possible)
                              # This part is complex and often not needed; snapshot_download shows progress.
                              # Let's return a representative size (e.g., 4GB for a 7B model)
                              if "7b" in hf_repo_id.lower(): return 4 * 1024**3
                              if "13b" in hf_repo_id.lower(): return 8 * 1024**3
                              return 5 * 1024**3 # Default guess

                     except Exception:
                          pass # Ignore errors fetching size for individual files here

            # If size calculation failed, return estimate
            if total_size == 0:
                if "7b" in hf_repo_id.lower(): return 4 * 1024**3
                if "13b" in hf_repo_id.lower(): return 8 * 1024**3
                return 5 * 1024**3

            return total_size

        except Exception as e:
            ASCIIColors.warning(f"Could not determine model size for {hf_repo_id}: {e}")
            return 0 # Indicate failure

    def install_model(self, model_type: str, model_path: str, variant_name: str, client_id: int = None):
        """
        Installs a model from a Hugging Face repository ID.

        Args:
            model_type (str): The type of model (e.g., 'transformers', 'gptq'). Used to determine subdirectory.
            model_path (str): The Hugging Face repository ID (e.g., 'google/gemma-1.1-2b-it').
            variant_name (str): The filename or variant identifier (often derived from repo ID).
            client_id (int, optional): Client ID for notifications. Defaults to None.
        """
        ASCIIColors.info(f"Install model triggered for repo: {model_path}")
        hf_repo_id = model_path # Assume model_path is the repo ID

        # Determine installation directory based on type
        if model_type.lower() not in self.models_dir_names:
             # Infer type if possible, otherwise default
             inferred_type = self.models_dir_names[0] # Default to 'transformers'
             for t in self.models_dir_names:
                 if t.lower() in hf_repo_id.lower():
                     inferred_type = t
                     break
             model_type = inferred_type
             ASCIIColors.info(f"Inferred model type: {model_type}")


        # Get the actual local directory name (often the repo name part)
        local_model_name = hf_repo_id.split('/')[-1]
        installation_dir = self.lollms_paths.personal_models_path / binding_folder_name / model_type
        installation_path = installation_dir / local_model_name # Install into a subdirectory named after the repo

        ASCIIColors.info(f"Target installation path: {installation_path}")

        if installation_path.exists():
            self.warning(f"Model directory already exists: {installation_path}")
            # Optionally: Add logic to check integrity or update
            self.lollmsCom.notify_model_install(
                installation_path=str(installation_path),
                model_name=local_model_name,
                binding_folder=binding_folder_name,
                model_url=hf_repo_id, # Use repo ID as URL identifier
                start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                total_size=0, # Can't easily get total size beforehand
                downloaded_size=0,
                progress=100, # Indicate completion (already exists)
                speed=0,
                client_id=client_id,
                status=True, # True = finished/exists
                error="Model already installed.",
                notification_type=NotificationType.NOTIFICATION_MODEL_INSTALLATION_FINISHED # Use finished type
            )
            return

        # --- Prepare for Download ---
        signature = f"{local_model_name}_{binding_folder_name}_{hf_repo_id}"
        start_time = datetime.now()
        total_size = self.get_file_size(hf_repo_id) # Estimate size for initial notification

        self.download_infos[signature] = {
            "start_time": start_time,
            "total_size": total_size,
            "downloaded_size": 0,
            "progress": 0,
            "speed": 0,
            "cancel": False,
            "thread": None # To store the download thread
        }

        # --- Define Progress Callback for snapshot_download (if possible) ---
        # snapshot_download's progress is mainly via tqdm. Hooking into it requires custom tqdm class.
        # Simpler: Notify start, use tqdm for console, notify end.

        try:
            # Notify start
            self.lollmsCom.notify_model_install(
                installation_path=str(installation_path), model_name=local_model_name, binding_folder=binding_folder_name, model_url=hf_repo_id,
                start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"), total_size=total_size, downloaded_size=0, progress=0, speed=0,
                client_id=client_id, status=True, error="", notification_type=NotificationType.NOTIFICATION_MODEL_INSTALLATION_STARTED
            )

            # --- Perform Download using snapshot_download ---
            ASCIIColors.info(f"Downloading repository {hf_repo_id} to {installation_path}...")

            # Filter files - prioritize safetensors, ignore flax/tf usually
            ignore_patterns = ["*.msgpack", "*.h5", "*.ot", "*.flax", "*.ckpt", "*.pt"] # Ignore common non-pytorch weights
            allow_patterns = ["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"] # Allow necessary files

             # Check if safetensors exist in the repo
            repo_files = list_repo_files(hf_repo_id, repo_type="model")
            has_safetensors = any(f.lower().endswith(".safetensors") for f in repo_files)
            if has_safetensors:
                ignore_patterns.append("*.bin") # Ignore .bin if .safetensors are present
                ASCIIColors.info("Prioritizing .safetensors files.")


            # Use snapshot_download with tqdm for console progress
            download_kwargs = {
                "repo_id": hf_repo_id,
                "local_dir": installation_path,
                "repo_type": "model",
                "local_dir_use_symlinks": False, # Avoid symlinks for portability
                "resume_download": True,
                "ignore_patterns": ignore_patterns,
                #"allow_patterns": allow_patterns, # Use allow_patterns OR ignore_patterns
                "tqdm_class": tqdm, # Integrate with console tqdm
                "etag_timeout": 60, # Increase timeout for large files
            }

            try:
                 snapshot_download(**download_kwargs)
                 # TODO: Find a way to monitor cancellation for snapshot_download if needed
                 # This might involve running it in a thread and having a flag checked by the main thread.
            except Exception as download_ex:
                 # Check if it's a cancellation-like error if cancellation is implemented
                 if "cancel" in str(download_ex).lower(): # Basic check
                     self.warning(f"Download cancelled for {hf_repo_id}.")
                     raise Exception("canceled") from download_ex
                 else:
                     raise download_ex # Re-raise other download errors


            # --- Final Notification ---
            download_duration = (datetime.now() - start_time).total_seconds()
            # Use actual directory size after download
            actual_size = sum(f.stat().st_size for f in installation_path.glob('**/*') if f.is_file())
            speed = actual_size / download_duration if download_duration > 0 else 0

            self.lollmsCom.notify_model_install(
                installation_path=str(installation_path), model_name=local_model_name, binding_folder=binding_folder_name, model_url=hf_repo_id,
                start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"), total_size=actual_size, downloaded_size=actual_size, progress=100, speed=speed,
                client_id=client_id, status=True, error="", notification_type=NotificationType.NOTIFICATION_MODEL_INSTALLATION_FINISHED
            )
            ASCIIColors.success(f"Model {hf_repo_id} downloaded successfully.")

        except Exception as e:
            trace_exception(e)
            error_message = str(e) if str(e) != "canceled" else "Download Canceled"
            status = False if error_message != "Download Canceled" else True # Mark canceled as 'not an error' status

            self.lollmsCom.notify_model_install(
                installation_path=str(installation_path), model_name=local_model_name, binding_folder=binding_folder_name, model_url=hf_repo_id,
                start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
                total_size=self.download_infos[signature]['total_size'], # Keep original estimate
                downloaded_size=self.download_infos[signature]['downloaded_size'], # Last known downloaded
                progress=self.download_infos[signature]['progress'], # Last known progress
                speed=0,
                client_id=client_id, status=status, error=error_message,
                notification_type=NotificationType.NOTIFICATION_MODEL_INSTALLATION_FAILED if status is False else NotificationType.NOTIFICATION_MODEL_INSTALLATION_CANCELLED
            )
            # Clean up partially downloaded folder on error (but not on cancel)
            if status is False and installation_path.exists():
                 try:
                     ASCIIColors.info(f"Cleaning up partially downloaded folder: {installation_path}")
                     shutil.rmtree(installation_path)
                 except Exception as cleanup_error:
                     ASCIIColors.error(f"Could not remove partial download: {cleanup_error}")

        finally:
             # Clean up download info entry
             if signature in self.download_infos:
                 del self.download_infos[signature]


    def cancel_model_install(self, client_id: int, model_url: str):
         """Cancels an ongoing model installation."""
         model_name = model_url.split('/')[-1] # Assuming URL is repo ID
         binding_folder = binding_folder_name
         signature = f"{model_name}_{binding_folder}_{model_url}"

         if signature in self.download_infos:
             self.download_infos[signature]["cancel"] = True
             # TODO: If using threads for download, need to signal the thread to stop.
             # snapshot_download doesn't directly support cancellation token.
             # A more complex setup involving threading and checking the cancel flag
             # periodically would be needed for immediate cancellation.
             # For now, this flag mainly prevents further progress updates and marks failure.
             ASCIIColors.info(f"Cancellation requested for {model_url}. Download may continue until next check or completion.")
             # Notify cancellation attempt
             self.lollmsCom.notify_model_install(
                 installation_path="", model_name=model_name, binding_folder=binding_folder, model_url=model_url,
                 start_time=self.download_infos[signature]['start_time'].strftime("%Y-%m-%d %H:%M:%S"),
                 total_size=self.download_infos[signature]['total_size'],
                 downloaded_size=self.download_infos[signature]['downloaded_size'],
                 progress=self.download_infos[signature]['progress'],
                 speed=0,
                 client_id=client_id, status=True, error="Cancellation requested",
                 notification_type=NotificationType.NOTIFICATION_MODEL_INSTALLATION_CANCELLED
             )
         else:
             ASCIIColors.warning(f"No active download found for signature: {signature}")


    # ============================ Generation Functions ============================
    def tokenize(self, prompt: str) -> List[int]:
        """Tokenizes the given prompt."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded.")
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def detokenize(self, tokens_list: List[int]) -> str:
        """Detokenizes the given list of tokens."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not loaded.")
        return self.tokenizer.decode(tokens_list)

    # --- Custom Streamer ---
    # Using a simple callback function approach instead of a full streamer class
    def _streaming_callback(self, token_ids: List[int], **kwargs):
        """Internal callback for generation, handles streaming."""
        if self.callback is None:
            return True # Continue generation

        try:
            # Detokenize the new tokens
            # Need to handle potential leading space if it's not the first token
            new_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)

            # Simple buffer approach to avoid partial words (optional)
            self.streaming_buffer += new_text
            if self.first_token_generated:
                # If not the first token, check for space to split words
                if " " in self.streaming_buffer or "\n" in self.streaming_buffer:
                     parts = self.streaming_buffer.rsplit(" ", 1)
                     if len(parts) > 1: # Found a space
                         chunk_to_send = parts[0] + " "
                         self.streaming_buffer = parts[1] # Keep the rest
                     elif "\n" in self.streaming_buffer:
                          parts = self.streaming_buffer.split("\n",1)
                          chunk_to_send = parts[0]+"\n"
                          self.streaming_buffer = parts[1]
                     else: # No space or newline yet, wait for more tokens
                          chunk_to_send = ""
                else: # No space/newline, keep buffering
                    chunk_to_send = ""
            else: # First token(s)
                 chunk_to_send = self.streaming_buffer
                 self.streaming_buffer = "" # Clear buffer after sending first part
                 self.first_token_generated = True


            if chunk_to_send:
                 # ASCIIColors.yellow(f"Sending chunk: '{chunk_to_send}'") # Debug
                 if not self.callback(chunk_to_send, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                     ASCIIColors.warning("Generation cancelled by callback.")
                     # How to stop the generator? Raise an exception? Set a flag?
                     # Returning False from streamer usually stops it.
                     return False # Signal generator to stop

            return True # Continue generation

        except Exception as e:
            ASCIIColors.error(f"Error in streaming callback: {e}")
            trace_exception(e)
            return False # Stop generation on error

    def _streaming_end_callback(self):
        """Called at the end of streaming to flush buffer."""
        if self.callback and self.streaming_buffer:
             # ASCIIColors.yellow(f"Sending final buffer: '{self.streaming_buffer}'") # Debug
             try:
                 self.callback(self.streaming_buffer, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK)
             except Exception as e:
                 ASCIIColors.error(f"Error sending final buffer: {e}")
        self.streaming_buffer = "" # Clear buffer
        self.first_token_generated = False # Reset flag


    def generate(self,
                 prompt: str,
                 n_predict: int = None,
                 callback: Callable[[str, int, dict], bool] = None,
                 verbose: bool = False,
                 **gpt_params) -> str:
        """
        Generates text based on a prompt using the loaded Hugging Face model.

        Args:
            prompt (str): The input prompt.
            n_predict (int, optional): Max tokens to generate. Uses binding default if None.
            callback (Callable[[str, int, dict], bool], optional): Callback for streaming.
            verbose (bool, optional): Enables verbose output (currently unused here).
            **gpt_params: Additional generation parameters.

        Returns:
            str: The generated text.
        """
        if self.model is None or self.tokenizer is None:
            self.error("Model is not built yet. Please build the model first.")
            return ""

        self.callback = callback # Store callback for streaming
        self.streaming_buffer = "" # Reset buffer
        self.first_token_generated = False # Reset flag

        # --- Prepare Generation Parameters ---
        current_generation_config = self.generation_config.copy() # Work on a copy

        # Override with user-provided params
        current_generation_config.max_new_tokens = n_predict if n_predict is not None else self.binding_config.max_n_predict
        current_generation_config.temperature = float(gpt_params.get('temperature', current_generation_config.temperature))
        current_generation_config.top_k = int(gpt_params.get('top_k', current_generation_config.top_k))
        current_generation_config.top_p = float(gpt_params.get('top_p', current_generation_config.top_p))
        current_generation_config.repetition_penalty = float(gpt_params.get('repeat_penalty', getattr(current_generation_config, 'repetition_penalty', 1.0))) # Handle missing attribute
        # current_generation_config.no_repeat_ngram_size = int(gpt_params.get('repeat_last_n', getattr(current_generation_config, 'no_repeat_ngram_size', 0)))

        seed = int(gpt_params.get('seed', self.binding_config.seed))
        if seed != -1:
            torch.manual_seed(seed)
            ASCIIColors.info(f"Set generation seed to: {seed}")

        # Ensure sampling is enabled if temp > 0, disabled otherwise
        current_generation_config.do_sample = True if current_generation_config.temperature > 0 else False

        # Force pad token id if needed
        if current_generation_config.pad_token_id is None:
             current_generation_config.pad_token_id = self.tokenizer.eos_token_id


        ASCIIColors.info(f"Generating with config: {current_generation_config}")

        # --- Tokenize Input ---
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model_device)
            input_ids = inputs["input_ids"]
            # attention_mask = inputs["attention_mask"] # Might be needed for some models/padding
        except Exception as e:
            self.error(f"Error tokenizing prompt: {e}")
            trace_exception(e)
            return ""

        # --- Generation ---
        output_text = ""
        try:
            with torch.no_grad():
                # Use generate method with the custom callback for streaming
                outputs = self.model.generate(
                    inputs=input_ids,
                    # attention_mask=attention_mask, # Pass mask if generated
                    generation_config=current_generation_config,
                    streamer=self if callback else None, # Use the binding itself as the streamer if callback is provided
                    # Add other relevant args if needed
                )

                # If not streaming, decode the full output
                if not callback:
                     # outputs will contain the full sequence (prompt + generation)
                     # Need to decode only the generated part
                     generated_ids = outputs[0][input_ids.shape[-1]:]
                     output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        except Exception as e:
            # Check if it was a cancellation via callback returning False
            if "returned False" in str(e): # Heuristic check
                 ASCIIColors.info("Generation stopped by callback.")
            else:
                 self.error(f"Error during generation: {e}")
                 trace_exception(e)
                 # Send error notification?
            output_text = "" # Ensure empty output on error/cancel during non-stream

        finally:
            # If streaming, flush the final buffer via the end callback
            if callback:
                output_text = self.streaming_buffer # The buffer contains the full streamed text
                self._streaming_end_callback() # Flush anything remaining
            self.callback = None # Clear callback

        return output_text


    def generate_with_images(self,
                             prompt: str,
                             images: List[str],
                             n_predict: int = None,
                             callback: Callable[[str, int, dict], bool] = None,
                             verbose: bool = False,
                             **gpt_params) -> str:
        """
        Generates text based on a prompt and images using a multi-modal model.

        Args:
            prompt (str): The input prompt (should include image placeholders like '<image>').
            images (List[str]): List of paths to image files.
            n_predict (int, optional): Max tokens to generate. Uses binding default if None.
            callback (Callable[[str, int, dict], bool], optional): Callback for streaming.
            verbose (bool, optional): Enables verbose output.
            **gpt_params: Additional generation parameters.

        Returns:
            str: The generated text.
        """
        if self.model is None or self.processor is None or self.tokenizer is None:
            self.error("Multi-modal model/processor not built. Please load a suitable model (e.g., Llava).")
            return ""
        if not images:
            self.warning("generate_with_images called with no images. Falling back to text-only generation.")
            return self.generate(prompt, n_predict, callback, verbose, **gpt_params)

        self.callback = callback
        self.streaming_buffer = ""
        self.first_token_generated = False

        # --- Prepare Generation Parameters (similar to text-only) ---
        current_generation_config = self.generation_config.copy()
        current_generation_config.max_new_tokens = n_predict if n_predict is not None else self.binding_config.max_n_predict
        current_generation_config.temperature = float(gpt_params.get('temperature', current_generation_config.temperature))
        # ... (copy other param settings from generate)
        seed = int(gpt_params.get('seed', self.binding_config.seed))
        if seed != -1: torch.manual_seed(seed)
        current_generation_config.do_sample = True if current_generation_config.temperature > 0 else False
        if current_generation_config.pad_token_id is None:
             current_generation_config.pad_token_id = self.tokenizer.eos_token_id

        ASCIIColors.info(f"Generating with images and config: {current_generation_config}")

        # --- Process Inputs ---
        try:
            raw_images = [Image.open(img_path).convert('RGB') for img_path in images]
            # AutoProcessor should handle text and images together
            inputs = self.processor(text=prompt, images=raw_images, return_tensors="pt").to(self.model_device, dtype=torch.float16 if self.model_device != 'cpu' else torch.float32) # Use float16 on GPU
            input_ids = inputs["input_ids"]

        except Exception as e:
            self.error(f"Error processing prompt/images: {e}")
            trace_exception(e)
            return ""

        # --- Generation ---
        output_text = ""
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, # Pass processed inputs directly
                    generation_config=current_generation_config,
                    streamer=self if callback else None,
                )
                if not callback:
                     generated_ids = outputs[0][input_ids.shape[-1]:]
                     output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip() # Use processor for decoding

        except Exception as e:
            if "returned False" in str(e):
                 ASCIIColors.info("Generation stopped by callback.")
            else:
                 self.error(f"Error during multi-modal generation: {e}")
                 trace_exception(e)
            output_text = ""
        finally:
            if callback:
                output_text = self.streaming_buffer # Full streamed text
                self._streaming_end_callback()
            self.callback = None

        return output_text

    # --- Streamer Interface Methods ---
    # These methods are called by `model.generate` when `streamer=self`
    def put(self, value):
        """Receives token IDs from the generator during streaming."""
        # Ensure value is on CPU and is a list/1D tensor
        if isinstance(value, torch.Tensor):
             if value.numel() > 1 and value.shape[0]>1: # Batch size > 1 not supported here
                value = value[0] # Take first batch element
             value = value.cpu().tolist() # Move to CPU and convert to list

        # Filter out pad/eos tokens if necessary (though decode usually handles them)
        # value = [tok for tok in value if tok not in [self.tokenizer.pad_token_id, self.tokenizer.eos_token_id]]

        if value:
             if not self._streaming_callback(value): # Call our internal handler
                 # If callback returned False, we need to signal generate to stop.
                 # Raising an exception is one way, though maybe not the cleanest.
                 raise Exception("Generation stopped by streamer callback returned False") # Signal stop


    def end(self):
        """Called by the generator when streaming finishes."""
        # The _streaming_end_callback handles flushing the buffer
        # This method is part of the streamer interface but logic is in the other callback
        pass


    @staticmethod
    def list_models(config: LOLLMSConfig) -> List[str]:
        """Lists models available for the binding."""
        # This static method should ideally list models from the configured directories
        # It is called by the LoLLMs UI to populate the model selection list
        binding_path = Path(__file__).parent
        lollms_paths = LollmsPaths(config.lollms_path) # Get paths based on main config
        models_dir = lollms_paths.personal_models_path / binding_folder_name
        model_list = []

        for model_type_dir in HuggingFace.models_dir_names:
            type_path = models_dir / model_type_dir
            if type_path.exists():
                for model_folder in type_path.iterdir():
                    if model_folder.is_dir(): # Each model is expected to be a directory
                        # Basic check for a config file to verify it's likely a model folder
                        if (model_folder / "config.json").exists():
                             # Construct the identifier Lollms expects (usually relative path/name)
                             # Format: category/model_name (e.g., transformers/gemma-1.1-2b-it)
                             model_list.append(f"{model_type_dir}/{model_folder.name}")

        # You might also want to add hardcoded popular models or fetch from HF Hub here
        # Example: model_list.extend(["transformers/google/gemma-1.1-2b-it", "gptq/TheBloke/Mistral-7B-Instruct-v0.2-GPTQ"])

        return model_list

    @staticmethod
    def get_available_models(config: LOLLMSConfig):
        # This static method should return a list of models available for installation
        # This is used by the model zoo UI.
        # It should ideally fetch popular/recommended models from Hugging Face Hub.

        # Example fetching (simplified):
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            # Search for popular models - adjust query/sort/limit as needed
            models = api.list_models(
                filter="text-generation", # Filter for text generation models
                sort="downloads",
                direction=-1,
                limit=50 # Get top 50 most downloaded
            )

            # Format for Lollms zoo (needs category, name, description etc.)
            zoo_models = []
            for model in models:
                # Basic filtering (e.g., ignore private models, specific authors)
                if model.private: continue

                # Infer category (basic)
                category = "transformers" # Default
                if "gptq" in model.modelId.lower(): category = "gptq"
                elif "awq" in model.modelId.lower(): category = "awq"

                zoo_models.append({
                    "category": category,
                    "name": model.modelId,
                    "author": model.author or "N/A",
                    "description": f"Downloads: {model.downloads}, Last Modified: {model.lastModified}\nTags: {', '.join(model.tags)}",
                    "icon": '/bindings/hugging_face/logo.png', # Binding icon
                    "license": "Check model card", # Encourage checking license
                    "model_type": "causal-lm", # Or infer based on tags
                    "installation_size": "Varies", # Can't easily predict size
                    "link": f"https://huggingface.co/{model.modelId}",
                    # Add other fields required by the zoo format
                })
            return zoo_models

        except Exception as e:
            ASCIIColors.error(f"Failed to fetch models from Hugging Face Hub: {e}")
            # Return a default list or empty list
            return [
                 {"category": "transformers", "name": "google/gemma-1.1-2b-it", "description":"Recommended default: Google Gemma 2B IT", "icon": '/bindings/hugging_face/logo.png'},
                 {"category": "transformers", "name": "meta-llama/Meta-Llama-3-8B-Instruct", "description":"Meta Llama 3 8B Instruct", "icon": '/bindings/hugging_face/logo.png'},
                 {"category": "gptq", "name": "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", "description":"Popular Mistral 7B Instruct GPTQ by TheBloke", "icon": '/bindings/hugging_face/logo.png'},
                 # Add more curated suggestions
            ]

# ========================== Main Execution (for testing) ==========================
if __name__ == "__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    # from lollms.app import LollmsApplication # Not strictly needed for basic binding test
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Test HuggingFace Binding")
    parser.add_argument("--config", type=str, default=None, help="Path to LoLLMs config file (config.yaml)")
    parser.add_argument("--model", type=str, default="google/gemma-1.1-2b-it", help="HF Model ID to test")
    parser.add_argument("--prompt", type=str, default="Explain the concept of Large Language Models in simple terms:", help="Prompt for generation")
    parser.add_argument("--max_tokens", type=int, default=150, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Generation temperature")

    args = parser.parse_args()

    # --- Setup LoLLMs ---
    # Use find_paths to locate the configuration directory correctly
    lollms_paths = LollmsPaths.find_paths(force_local=True, custom_default_cfg_path=args.config)
    # Ensure a configuration object exists, even if minimal
    config = LOLLMSConfig.autoload(lollms_paths)
    # Override binding and model from args for testing
    config.binding_name = binding_name
    config.model_name = args.model # Use the model specified in args
    # config.save_config() # Avoid saving changes during testing unless intended

    # --- Initialize Binding ---
    # Pass a placeholder lollmsCom if not running full LollmsApplication
    class MockLollmsCom:
        def notify_model_install(self, *args, **kwargs): print(f"Mock Notify Install: {args}, {kwargs}")
        def ShowBlockingMessage(self, msg): print(f"Mock Blocking Msg: {msg}")
        def HideBlockingMessage(self): print("Mock Hide Blocking Msg")
        # Add other methods if the binding calls them during init/build
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARN: {msg}")
        def error(self, msg): print(f"ERR: {msg}")
        def success(self, msg): print(f"SUCCESS: {msg}")

    hf_binding = HuggingFace(config, lollms_paths, lollmsCom=MockLollmsCom())

    # --- Build Model ---
    print(f"\nBuilding model: {args.model}")
    # Use the build_model method of the instance
    model_built_successfully = hf_binding.build_model() # build_model returns self on success, None on failure

    if model_built_successfully:
        print("\n--- Starting Generation Test ---")
        print(f"Prompt: {args.prompt}")
        print("Response:")

        full_response = ""
        def test_callback(chunk, type, metadata=None):
            # nonlocal full_response # <--- REMOVE THIS LINE
            global full_response # Use global if you absolutely must, but modifying outer scope variable is fine here
            print(chunk, end="", flush=True)
            full_response += chunk
            return True # Continue generation

        try:
            # Call generate on the binding instance
            hf_binding.generate(
                args.prompt,
                n_predict=args.max_tokens,
                callback=test_callback,
                gpt_params={'temperature': args.temp}, # Pass params in gpt_params dict
                # Add other params if needed
            )
            print("\n\n--- Generation Complete ---")
            # print(f"\nFull Response:\n{full_response}") # Already printed by callback

        except Exception as e:
            print(f"\n--- Generation Failed ---")
            print(f"Error: {e}")
            trace_exception(e)
        finally:
            print("\n--- Cleaning up ---")
            hf_binding.destroy_model()
            print("Model destroyed.")
    else:
        print("Failed to build the model. Exiting.")