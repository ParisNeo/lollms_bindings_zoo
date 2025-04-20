######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying
# engine author : Anthropic, PBC
# license       : Apache 2.0
# Description   :
# This binding provides an interface to Anthropic's Claude models using their official API.
# It supports text generation and vision capabilities for multimodal models.
# It includes auto-detection for model context size and max output tokens based on known values.
# It dynamically lists available models using the Anthropic API endpoint.
# Update date   : 2024-07-26
######
import io
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from lollms.binding import BindingType, LLMBinding, LOLLMSConfig
from lollms.com import LoLLMsCom
from lollms.config import BaseConfig, ConfigTemplate, InstallOption, TypedConfig
from lollms.helpers import ASCIIColors, trace_exception
from lollms.paths import LollmsPaths
from lollms.types import MSG_OPERATION_TYPE
from lollms.utilities import (
    PackageManager,
    encode_image,
    find_first_available_file_path,
    is_file_path,
)
from PIL import Image

# Try to install necessary packages using PackageManager
if not PackageManager.check_package_installed("anthropic"):
    PackageManager.install_package("anthropic")
if not PackageManager.check_package_installed("requests"):
    PackageManager.install_package("requests")


# Import required libraries
try:
    import anthropic
    import requests # Added for direct API call
    from anthropic.types import Message, MessageParam, TextBlockParam, ImageBlockParam, Usage
except ImportError as e:
    # Distinguish between missing packages
    missing_pkg = ""
    if "No module named 'anthropic'" in str(e):
        missing_pkg = "anthropic"
    elif "No module named 'requests'" in str(e):
        missing_pkg = "requests"

    if missing_pkg:
        print(f"Error importing required library: {e}")
        print(f"Please ensure '{missing_pkg}' is installed (`pip install {missing_pkg}`)")
        # Decide how to handle: either raise or set flags
        if missing_pkg == 'anthropic':
            anthropic = None
        requests = None # Assume requests might also be missing if anthropic fails import sometimes
    else:
        # Unexpected import error
        raise e from None


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023-2024, ParisNeo"
__license__ = "Apache 2.0"

binding_name = "Anthropic"
# Make sure this matches the actual folder name if it's different from binding_name
# User corrected this in their prompt. Keeping it as `anthropic_llm`
binding_folder_name = "anthropic_llm"

# API Endpoint
ANTHROPIC_API_BASE_URL = "https://api.anthropic.com/v1"

# ================= Known Model Limits (Update with Anthropic documentation) =================
# These serve as fallbacks if the API doesn't provide the limits directly or if the API call fails.
MODEL_CONTEXT_SIZES: Dict[str, int] = {
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-2.1": 200000,
    "claude-2.0": 100000,
    "claude-instant-1.2": 100000,
    "claude-3": 200000, # General family fallback
    "default": 100000,
}

MODEL_MAX_OUTPUT_TOKENS: Dict[str, int] = {
    "claude-3-opus-20240229": 4096,
    "claude-3-sonnet-20240229": 4096,
    "claude-3-haiku-20240307": 4096,
    "claude-3-5-sonnet-20240620": 4096,
    "claude-2.1": 4096,
    "claude-2.0": 4096,
    "claude-instant-1.2": 4096,
    "claude-3": 4096, # General family fallback
    "default": 4096,
}

# ================= Pricing Information (Update with latest Anthropic prices) =================
INPUT_COSTS_BY_MODEL: Dict[str, float] = {
    "claude-3-opus-20240229": 15.00 / 1_000_000,
    "claude-3-sonnet-20240229": 3.00 / 1_000_000,
    "claude-3-haiku-20240307": 0.25 / 1_000_000,
    "claude-3-5-sonnet-20240620": 3.00 / 1_000_000,
    "claude-2.1": 8.00 / 1_000_000,
    "claude-2.0": 8.00 / 1_000_000,
    "claude-instant-1.2": 0.80 / 1_000_000,
    "default": 3.00 / 1_000_000,
}

OUTPUT_COSTS_BY_MODEL: Dict[str, float] = {
    "claude-3-opus-20240229": 75.00 / 1_000_000,
    "claude-3-sonnet-20240229": 15.00 / 1_000_000,
    "claude-3-haiku-20240307": 1.25 / 1_000_000,
    "claude-3-5-sonnet-20240620": 15.00 / 1_000_000,
    "claude-2.1": 24.00 / 1_000_000,
    "claude-2.0": 24.00 / 1_000_000,
    "claude-instant-1.2": 2.40 / 1_000_000,
    "default": 15.00 / 1_000_000,
}

# Static list removed, will be populated by API call
# KNOWN_MODELS = [...]

class Anthropic(LLMBinding):
    """
    Binding class for interacting with the Anthropic Claude API.
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
        self.lollmsCom = lollmsCom

        # Ensure dependencies are available
        if anthropic is None:
            raise ImportError("Anthropic library not found. Please install `anthropic`.")
        if requests is None:
            raise ImportError("Requests library not found. Please install `requests`.")

        # Configuration definition using ConfigTemplate
        binding_config_template = ConfigTemplate([
            # --- General ---
            {"name": "anthropic_api_key", "type": "str", "value": "", "help": "Your Anthropic API key. Found at https://console.anthropic.com/settings/keys"},
            {"name": "auto_detect_limits", "type": "bool", "value": True, "help": "Automatically detect and use the selected model's context size and max output tokens based on known values. If false, uses the manually set values below.", "requires_restart": False},
            {"name": "ctx_size", "type": "int", "value": 200000, "min": 1000, "help": "Model's maximum context size (input tokens). Automatically updated based on the selected model if 'auto_detect_limits' is enabled. Default shown is for Claude 3 family."},
            {"name": "max_n_predict", "type": "int", "value": 4096, "min": 1, "help": "Maximum number of tokens to generate per response (maps to max_tokens in API). Automatically updated based on the selected model if 'auto_detect_limits' is enabled. Default shown is common for Claude models."},

            # --- Vision ---
            {"name": "max_image_width", "type": "int", "value": -1, "help": "Resize images if wider than this before sending to vision models (reduces cost/tokens). -1 for no change."},

            # --- API Versioning (Optional) ---
            {"name": "anthropic_version_header", "type": "str", "value": "2023-06-01", "help": "API Version header. Use '2023-06-01' for stable features. Check Anthropic docs for beta versions if needed."},

            # --- Cost Estimation ---
            {"name": "turn_on_cost_estimation", "type": "bool", "value": False, "help": "Estimate query costs based on input/output text tokens."},
            {"name": "total_input_tokens", "type": "float", "value": 0, "help": "Accumulated input tokens (for cost estimation)."},
            {"name": "total_output_tokens", "type": "float", "value": 0, "help": "Accumulated output tokens (for cost estimation)."},
            {"name": "total_input_cost", "type": "float", "value": 0, "help": "Accumulated input cost ($)."},
            {"name": "total_output_cost", "type": "float", "value": 0, "help": "Accumulated output cost ($)."},
            {"name": "total_cost", "type": "float", "value": 0, "help": "Total accumulated cost ($)."},
        ])
        # Default values for the configuration
        binding_config_defaults = BaseConfig(config={
            "anthropic_api_key": "",
            "auto_detect_limits": True,
            "ctx_size": 200000,
            "max_n_predict": 4096,
            "max_image_width": -1,
            "anthropic_version_header": "2023-06-01",
            "turn_on_cost_estimation": False,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_input_cost": 0,
            "total_output_cost": 0,
            "total_cost": 0,
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
            supported_file_extensions=['.png', '.jpg', '.jpeg', '.webp', '.gif'],
            lollmsCom=lollmsCom
        )
        self.client: Optional[anthropic.Anthropic] = None
        self.available_models: List[str] = [] # Populated by API call
        self.fetched_api_models: List[Dict[str, Any]] = [] # Store full API response data

        self.settings_updated() # Attempt initial client setup and model fetch


    def _update_anthropic_key(self) -> bool:
        """Initializes the Anthropic client."""
        api_key = None
        source = None
        config_key = self.binding_config.config.get("anthropic_api_key", "")

        if config_key and config_key.strip():
            api_key = config_key
            source = "configuration"
        else:
            api_key_env = os.getenv('ANTHROPIC_API_KEY')
            if api_key_env:
                api_key = api_key_env
                source = "environment variable"

        if api_key:
            try:
                version_header = self.binding_config.config.get("anthropic_version_header", "2023-06-01")
                self.client = anthropic.Anthropic(
                    api_key=api_key,
                    default_headers={"anthropic-version": version_header}
                )
                ASCIIColors.info(f"Using Anthropic API key from {source}.")
                return True
            except anthropic.AuthenticationError:
                self.error(f"Invalid Anthropic API key from {source}. Please check your key.")
            except Exception as e:
                self.error(f"Failed to initialize Anthropic client: {e}")
                trace_exception(e)
            self.client = None
            return False
        else:
            self.warning("No Anthropic API key found. Binding will not function.")
            self.client = None
            return False

    def _update_available_models(self) -> None:
        """
        Fetches the list of available models from the Anthropic API using a direct HTTP request.
        Updates self.available_models (list of IDs) and self.fetched_api_models (list of dicts).
        """
        if not self.client:
            self.warning("Anthropic client not initialized (no API key?). Cannot fetch models.")
            self.available_models = []
            self.fetched_api_models = []
            return

        api_key = self.binding_config.config.get("anthropic_api_key", os.getenv('ANTHROPIC_API_KEY'))
        version_header = self.binding_config.config.get("anthropic_version_header", "2023-06-01")

        if not api_key: # Double check here although client init should handle it
            self.error("API Key missing, cannot fetch models.")
            self.available_models = []
            self.fetched_api_models = []
            return

        headers = {
            "x-api-key": api_key,
            "anthropic-version": version_header,
            "accept": "application/json"
        }
        url = f"{ANTHROPIC_API_BASE_URL}/models"
        models_data = []
        model_ids = []

        try:
            ASCIIColors.info("Fetching available models from Anthropic API...")
            response = requests.get(url, headers=headers, timeout=15) # Add timeout
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            data = response.json()

            if "data" in data and isinstance(data["data"], list):
                models_data = data["data"]
                # Extract model IDs - Assuming model object has an 'id' field
                model_ids = sorted([model.get("id") for model in models_data if model.get("id")])
                ASCIIColors.success(f"Successfully fetched {len(model_ids)} models from Anthropic API.")
            else:
                self.warning("API response received, but 'data' field is missing or not a list.")
                ASCIIColors.warning(f"Raw response sample: {str(data)[:200]}...") # Log sample for debugging

        except requests.exceptions.RequestException as e:
            self.error(f"Network error fetching Anthropic models: {e}")
            trace_exception(e)
        except json.JSONDecodeError as e:
             self.error(f"Error decoding JSON response from Anthropic API: {e}")
             self.error(f"Received text: {response.text[:200]}...") # Log raw text
        except Exception as e:
            self.error(f"Failed to fetch or process Anthropic models: {e}")
            trace_exception(e)
            # Attempt to get status code if it's an HTTPError
            status_code = getattr(e.response, 'status_code', None)
            if status_code == 401:
                 self.error("Authentication Error (401): Invalid Anthropic API key.")
            elif status_code:
                 self.error(f"API request failed with status code: {status_code}")

        # Update instance variables whether successful or not (will be empty lists on failure)
        self.fetched_api_models = models_data
        self.available_models = model_ids

        if not self.available_models:
             self.warning("Could not retrieve model list from API. Model selection might be limited or empty.")


    def settings_updated(self) -> None:
        """Callback triggered when binding settings are updated in the UI."""
        ASCIIColors.info("Anthropic settings updated.")
        key_changed = self._update_anthropic_key()

        if self.client:
            # Fetch models if the key changed or if the list is currently empty
            if key_changed or not self.available_models:
                self._update_available_models()

            # Rebuild the model to apply potential changes in limits/settings
            if self.config.model_name:
                self.info("Rebuilding model due to settings update...")
                self.build_model(self.config.model_name)
            else:
                self.info("No model currently selected, skipping model rebuild after settings update.")
        else:
            # Key became invalid or was removed
            self.available_models = []
            self.fetched_api_models = []
            self.config.ctx_size = self.binding_config.config.ctx_size
            self.config.max_n_predict = self.binding_config.config.max_n_predict
            if self.lollmsCom:
                self.lollmsCom.InfoMessage("Anthropic Error: API Key is missing or invalid. Please configure it.")


    def get_model_limits(self, model_name: str, model_data: Optional[Dict] = None) -> Tuple[Optional[int], Optional[int]]:
        """
        Tries to find context size and max output tokens for a model.
        Prioritizes data from the API response (`model_data`) if provided,
        then falls back to hardcoded dictionaries.

        Args:
            model_name: The name/ID of the model.
            model_data: Optional dictionary containing data fetched from the API for this model.

        Returns:
            A tuple (context_size, max_output_tokens). Values are None if not determinable.
        """
        api_ctx = None
        api_max_out = None

        # 1. Try getting limits from the fetched API data for this specific model
        if model_data:
            # Adjust keys based on actual API response structure
            api_ctx = model_data.get("context_length")
            api_max_out = model_data.get("max_output_tokens")
            # Ensure they are integers if found
            try: api_ctx = int(api_ctx) if api_ctx is not None else None
            except ValueError: api_ctx = None
            try: api_max_out = int(api_max_out) if api_max_out is not None else None
            except ValueError: api_max_out = None

        # 2. If API didn't provide them, fall back to static dictionaries
        fallback_ctx, fallback_max_out = None, None
        if model_name:
            # Use direct lookup first
            fallback_ctx = MODEL_CONTEXT_SIZES.get(model_name)
            fallback_max_out = MODEL_MAX_OUTPUT_TOKENS.get(model_name)
            # Try prefix matching if direct lookup failed
            if fallback_ctx is None:
                best_match_len = 0
                for prefix, size in MODEL_CONTEXT_SIZES.items():
                    if model_name.startswith(prefix) and len(prefix) > best_match_len:
                        fallback_ctx = size; best_match_len = len(prefix)
            if fallback_max_out is None:
                best_match_len = 0
                for prefix, size in MODEL_MAX_OUTPUT_TOKENS.items():
                    if model_name.startswith(prefix) and len(prefix) > best_match_len:
                        fallback_max_out = size; best_match_len = len(prefix)
            # Use default if still not found
            if fallback_ctx is None: fallback_ctx = MODEL_CONTEXT_SIZES.get("default")
            if fallback_max_out is None: fallback_max_out = MODEL_MAX_OUTPUT_TOKENS.get("default")

        # Prioritize API values if they exist, otherwise use fallbacks
        final_ctx = api_ctx if api_ctx is not None else fallback_ctx
        final_max_out = api_max_out if api_max_out is not None else fallback_max_out

        return final_ctx, final_max_out


    def build_model(self, model_name: Optional[str] = None) -> LLMBinding:
        """Builds model state using selected name and fetched/static data."""
        super().build_model(model_name) # Sets self.config.model_name

        if not self.client:
            if not self._update_anthropic_key():
                self.available_models = []
                self.fetched_api_models = []
                self.error("Model build failed: Anthropic API key is missing or invalid.")
                if self.lollmsCom: self.lollmsCom.InfoMessage("Anthropic Error: Cannot build model. API Key missing or invalid.")
                return self

        # Ensure model list is fetched if empty
        if not self.available_models and not self.fetched_api_models:
            self._update_available_models()

        current_model_name = self.config.model_name or ""
        if not current_model_name:
            self.warning("No model name selected. Cannot determine limits or capabilities.")
            self.config.ctx_size = self.binding_config.config.ctx_size
            self.config.max_n_predict = self.binding_config.config.max_n_predict
            return self

        # Find the corresponding model data from the fetched list
        current_model_data = next((m for m in self.fetched_api_models if m.get("id") == current_model_name), None)
        if not current_model_data and self.fetched_api_models: # Log if model selected but not found in API list
             self.warning(f"Selected model '{current_model_name}' not found in the list fetched from API. Limits will rely on static fallbacks.")

        # --- Determine Effective Limits ---
        detected_ctx_size, detected_max_output = self.get_model_limits(current_model_name, current_model_data)

        effective_ctx_size = self.binding_config.config["ctx_size"]
        effective_max_output = self.binding_config.config["max_n_predict"]

        limit_source_ctx = "manual/default"
        limit_source_max_out = "manual/default"

        if self.binding_config.auto_detect_limits:
            ASCIIColors.info(f"Auto-detect limits enabled for {current_model_name}.")
            if detected_ctx_size is not None:
                effective_ctx_size = detected_ctx_size
                limit_source_ctx = "API" if current_model_data and current_model_data.get("context_length") else "static fallback"
                ASCIIColors.success(f"  Using context size: {effective_ctx_size} (from {limit_source_ctx})")
            else:
                ASCIIColors.warning(f"  Could not determine context size. Using manual/default: {effective_ctx_size}")

            if detected_max_output is not None:
                effective_max_output = detected_max_output
                limit_source_max_out = "API" if current_model_data and current_model_data.get("max_output_tokens") else "static fallback"
                ASCIIColors.success(f"  Using max output tokens: {effective_max_output} (from {limit_source_max_out})")
            else:
                 ASCIIColors.warning(f"  Could not determine max output tokens. Using manual/default: {effective_max_output}")
        else:
            ASCIIColors.info("Auto-detect limits disabled. Using manually configured limits.")
            limit_source_ctx = "manual"
            limit_source_max_out = "manual"
            # Optionally warn if manual settings exceed detected limits
            if detected_ctx_size is not None and effective_ctx_size > detected_ctx_size:
                self.warning(f"Manually set ctx_size ({effective_ctx_size}) exceeds limit ({detected_ctx_size}).")
            if detected_max_output is not None and effective_max_output > detected_max_output:
                 self.warning(f"Manually set max_n_predict ({effective_max_output}) exceeds limit ({detected_max_output}).")

        # --- Update Configurations ---
        self.binding_config.config["ctx_size"] = effective_ctx_size
        self.binding_config.config["max_n_predict"] = effective_max_output
        self.config.ctx_size = effective_ctx_size
        self.config.max_n_predict = effective_max_output

        ASCIIColors.info(f"Effective limits set: ctx_size={effective_ctx_size} (from {limit_source_ctx}), max_n_predict={effective_max_output} (from {limit_source_max_out})")

        # --- Determine Binding Type ---
        is_vision_capable = False
        if current_model_data:
            # Check for a specific field indicating vision support (adjust key if needed)
            is_vision_capable = current_model_data.get("supports_vision", False) # Example key
        else:
            # Fallback to name checking if API data is missing
            is_vision_capable = current_model_name.startswith("claude-3")

        if is_vision_capable:
            self.binding_type = BindingType.TEXT_IMAGE
            self.supported_file_extensions=['.png', '.jpg', '.jpeg', '.webp', '.gif']
            ASCIIColors.info(f"Model {current_model_name} determined as vision capable. Binding type: TEXT_IMAGE.")
        else:
            self.binding_type = BindingType.TEXT_ONLY
            self.supported_file_extensions=[]
            ASCIIColors.info(f"Model {current_model_name} determined as text-only. Binding type: TEXT_ONLY.")

        ASCIIColors.success(f"Anthropic binding built successfully. Model: {current_model_name}.")
        return self


    def install(self) -> None:
        """Installs the necessary anthropic and requests packages."""
        super().install()
        self.ShowBlockingMessage("Installing Anthropic API client requirements...")
        try:
            requirements = ["anthropic", "requests"]
            for req in requirements:
                if not PackageManager.check_package_installed(req):
                    PackageManager.install_package(req)
                else:
                    self.info(f"{req} package already installed.")
            self.HideBlockingMessage()
            ASCIIColors.success("Anthropic client requirements installed successfully.")
            # Add user guidance messages
            ASCIIColors.info("----------------------")
            ASCIIColors.info("Attention:")
            ASCIIColors.info("----------------------")
            ASCIIColors.info("The Anthropic binding uses the Claude API, which is a paid service.")
            ASCIIColors.info("1. Create an account at https://www.anthropic.com/")
            ASCIIColors.info("2. Generate an API key from https://console.anthropic.com/settings/keys")
            ASCIIColors.info("3. Provide the key in the binding settings or set the ANTHROPIC_API_KEY environment variable.")
        except Exception as e:
            self.HideBlockingMessage()
            self.error(f"Installation failed: {e}")
            trace_exception(e)
            self.warning("Installation failed. Please ensure you have pip installed and internet access.", 20)

    def count_tokens(self, prompt):
        if prompt:
            return self.client.messages.count_tokens(model=self.config.model_name, messages=[
            {"role": "user", "content": prompt}
            ]).input_tokens
        else:
            return 0
    # --- Tokenizer methods remain unchanged as they rely on client.count_tokens ---
    def tokenize(self, prompt: str) -> List[int]:
        """Returns token count as a list: [count]"""
        return prompt.split(" ")

    def detokenize(self, tokens_list: List[int]) -> str:
        """Not supported."""
        return " ".join(tokens_list)

    def embed(self, text: Union[str, List[str]]) -> Optional[List[List[float]]]:
        """Not supported."""
        self.error("Anthropic API does not support embeddings."); return None

    # --- Message preparation method remains unchanged ---
    def _prepare_anthropic_messages(self, prompt: str, images: Optional[List[str]] = None) -> Tuple[List[MessageParam], int]:
        """Prepares messages list for API call."""
        content: List[Union[TextBlockParam, ImageBlockParam]] = []
        processed_image_count = 0
        invalid_image_paths = []

        if prompt:
            content.append({"type": "text", "text": prompt})

        if images:
            max_image_width = self.binding_config.config.get("max_image_width", -1)
            for image_path_str in images:
                image_path = Path(image_path_str)
                valid_image_path = find_first_available_file_path([image_path])
                if not valid_image_path or not is_file_path(valid_image_path) or not valid_image_path.exists():
                    self.warning(f"Image path not found/invalid: {image_path_str}. Skipping."); invalid_image_paths.append(image_path_str); continue
                try:
                    encoded_image_tuple = encode_image(str(valid_image_path), max_image_width, return_format=True)
                    if encoded_image_tuple:
                        encoded_image_str, media_type = encoded_image_tuple
                        if not media_type.startswith("image/"):
                            self.warning(f"Unsupported image type '{media_type}' for {valid_image_path}. Skipping."); invalid_image_paths.append(image_path_str); continue
                        content.append({ "type": "image", "source": { "type": "base64", "media_type": media_type, "data": encoded_image_str }})
                        processed_image_count += 1; ASCIIColors.info(f"Prepared image: {valid_image_path} ({media_type})")
                    else:
                        self.warning(f"Could not encode image: {valid_image_path}"); invalid_image_paths.append(image_path_str)
                except Exception as img_ex:
                    self.error(f"Error processing image {valid_image_path}: {img_ex}"); invalid_image_paths.append(image_path_str); trace_exception(img_ex)

        if invalid_image_paths: self.warning(f"Skipped {len(invalid_image_paths)} invalid/unloadable images.")
        messages: List[MessageParam] = [{"role": "user", "content": content}]
        return messages, processed_image_count

    # --- API call processing method remains largely unchanged ---
    def _process_api_call(self, prompt: str, images: Optional[List[str]] = None, n_predict: Optional[int] = None, callback: Optional[Callable[[str, int, dict], bool]] = None, verbose: bool = False, **claude_params) -> str:
        """Internal method to handle the API call and streaming."""
        if not self.client:
            self.error("Anthropic client not initialized."); return ""
        model_name = self.config.model_name
        if not model_name: self.error("No model selected."); return ""

        effective_max_tokens = self.config.max_n_predict
        if n_predict is not None:
             if 0 < n_predict <= self.config.max_n_predict: effective_max_tokens = n_predict
             elif n_predict > self.config.max_n_predict: self.warning(f"n_predict ({n_predict}) > model limit ({self.config.max_n_predict}). Capping."); effective_max_tokens = self.config.max_n_predict
        if verbose: ASCIIColors.verbose(f"Effective max_tokens: {effective_max_tokens}")

        default_params = {'temperature': self.config.temperature}
        final_params = {**default_params, **claude_params}
        api_temperature = float(final_params.get("temperature", 1.0))

        messages, _ = self._prepare_anthropic_messages(prompt, images)
        if not messages[0]['content']: self.error("Empty prompt and no valid images."); return ""

        input_tokens = 0; prompt_text_only = prompt
        if self.binding_config.config.turn_on_cost_estimation:
            try:
                input_tokens = self.count_tokens(prompt_text_only) if prompt_text_only else 0
                self.binding_config.config["total_input_tokens"] += input_tokens
                input_cost_rate = INPUT_COSTS_BY_MODEL.get(model_name, INPUT_COSTS_BY_MODEL.get("default", 0))
                input_cost = input_tokens * input_cost_rate
                self.binding_config.config["total_input_cost"] += input_cost
            except Exception as count_ex: self.warning(f"Could not count input tokens: {count_ex}")

        output = ""; total_output_tokens = 0; final_usage: Optional[Usage] = None; finish_reason: Optional[str] = None; metadata = {}; start_time = perf_counter()
        try:
            api_call_params = {"model": model_name, "messages": messages, "max_tokens": effective_max_tokens}
            if api_temperature != 1.0: api_call_params["temperature"] = api_temperature
            if verbose: ASCIIColors.verbose(f"Calling Anthropic API. Params: {api_call_params}")

            with self.client.messages.stream(**api_call_params) as stream:
                stream_finished = False
                for event in stream:
                    if event.type == "message_start":
                         metadata["message_id"] = event.message.id; metadata["model"] = event.message.model
                    elif event.type == "content_block_delta" and event.delta.type == "text_delta":
                        chunk_text = event.delta.text; output += chunk_text
                        if callback and not callback(chunk_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                            self.info("Generation stopped by callback."); stream_finished = True; break
                    elif event.type == "message_delta":
                         if event.usage: final_usage = event.usage
                         if event.delta and event.delta.stop_reason: finish_reason = event.delta.stop_reason
                    elif event.type == "message_stop":
                         finish_reason = event.message.stop_reason; final_usage = event.message.usage; stream_finished = True; break
                if not stream_finished and finish_reason is None: self.info("Stream loop finished without stop reason.")

        except anthropic.APIError as e: self.error(f'Anthropic API Error: {e}'); trace_exception(e); callback(f"API Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION) if callback else None; return ""
        except Exception as e: self.error(f'Error during generation: {e}'); trace_exception(e); callback(f"Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION) if callback else None; return ""
        finally:
            generation_time = perf_counter() - start_time; ASCIIColors.info(f"Generation finished in {generation_time:.2f}s.")
            if finish_reason: metadata["finish_reason"] = finish_reason
            if final_usage: metadata["usage"] = final_usage

        if self.binding_config.config.turn_on_cost_estimation:
            if final_usage and final_usage.output_tokens is not None:
                total_output_tokens = final_usage.output_tokens
                if final_usage.input_tokens is not None: # Correct input cost with API data
                    self.binding_config.config["total_input_tokens"] -= input_tokens # Subtract initial estimate
                    self.binding_config.config["total_input_tokens"] += final_usage.input_tokens
                    input_cost_rate = INPUT_COSTS_BY_MODEL.get(model_name, INPUT_COSTS_BY_MODEL.get("default", 0))
                    self.binding_config.config["total_input_cost"] = self.binding_config.config["total_input_tokens"] * input_cost_rate
            elif output: # Fallback token counting
                self.warning("API did not provide usage stats. Estimating output tokens."); total_output_tokens = self.count_tokens(output)
            self.binding_config.config["total_output_tokens"] += total_output_tokens
            output_cost_rate = OUTPUT_COSTS_BY_MODEL.get(model_name, OUTPUT_COSTS_BY_MODEL.get("default", 0))
            output_cost = total_output_tokens * output_cost_rate
            self.binding_config.config["total_output_cost"] += output_cost
            self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]
            self.info(f'Accumulated cost: ${self.binding_config.config["total_cost"]:.6f}')
            self.binding_config.save()
        return output

    # --- generate and generate_with_images remain wrappers around _process_api_call ---
    def generate_with_images(self, prompt: str, images: List[str], n_predict: Optional[int] = None, callback: Optional[Callable[[str, int, dict], bool]] = None, verbose: bool = False, **claude_params) -> str:
        """Generates text using prompt and images."""
        if self.binding_type != BindingType.TEXT_IMAGE:
            self.error(f"Model '{self.config.model_name}' doesn't support image input."); return ""
        if not images:
            self.warning("generate_with_images called with no images. Using text-only generate."); return self.generate(prompt, n_predict, callback, verbose, **claude_params)
        return self._process_api_call(prompt, images, n_predict, callback, verbose, **claude_params)

    def generate(self, prompt: str, n_predict: Optional[int] = None, callback: Optional[Callable[[str, int, dict], bool]] = None, verbose: bool = False, **claude_params) -> str:
        """Generates text using prompt."""
        return self._process_api_call(prompt, None, n_predict, callback, verbose, **claude_params)

    # --- list_models uses the fetched list ---
    def list_models(self) -> List[str]:
        """Lists available model IDs fetched from API."""
        if not self.available_models and self.client:
            # Attempt to fetch if list is empty and client exists
            ASCIIColors.warning("Model list empty, attempting fetch...")
            self._update_available_models()
        return self.available_models

    # --- get_available_models uses the fetched full data ---
    def get_available_models(self, app: Optional[LoLLMsCom] = None) -> List[dict]:
        """Gets model list formatted for LoLLMs UI using fetched API data."""
        lollms_com = self.lollmsCom or app
        if not self.client:
            self.error("Cannot get models for UI, client not initialized.")
            if lollms_com: lollms_com.InfoMessage("Anthropic Error: Cannot fetch models. API Key missing/invalid.")
            return []

        # Ensure model list is populated
        if not self.fetched_api_models:
             self._update_available_models()
             if not self.fetched_api_models:
                  self.error("No models available to display after fetch attempt.")
                  if lollms_com: lollms_com.InfoMessage("Anthropic Error: Failed to retrieve models from API.")
                  return []

        models_info: List[Dict[str, Any]] = []
        # Corrected folder name usage
        binding_folder = binding_folder_name or binding_name.lower() # Fallback to binding_name if folder_name is empty
        ui_path_prefix = f"/bindings/{binding_folder}/" # Use the potentially corrected folder name
        default_icon = ui_path_prefix + "logo.png"

        base_metadata = { "author": "Anthropic, PBC", "license": "Commercial API", "creation_date": None, "category": "API Model", "datasets": "Proprietary Anthropic Datasets", "commercial_use": True, "icon": default_icon, "model_creator": "Anthropic", "model_creator_link": "https://www.anthropic.com/", "provider": None, "type": "api" }

        for model_data in self.fetched_api_models:
            model_id = model_data.get("id")
            if not model_id: continue # Skip if no ID

            ctx_size, max_output = self.get_model_limits(model_id, model_data)

            # Determine category based on API data or name fallback
            is_vision_capable = model_data.get("supports_vision", model_id.startswith("claude-3"))
            category = "multimodal" if is_vision_capable else "text"

            # Ranking logic
            rank = 1.0
            if "opus" in model_id: rank = 4.0
            elif "claude-3-5-sonnet" in model_id: rank = 3.5
            elif "sonnet" in model_id: rank = 3.0
            elif "haiku" in model_id: rank = 2.5
            elif "claude-2" in model_id: rank = 2.0
            elif "instant" in model_id: rank = 1.5

            size_proxy = ctx_size if ctx_size is not None else -1

            model_entry = {
                **base_metadata,
                "name": model_id,
                "display_name": model_data.get("name", model_id.replace("-", " ").title()), # Use API name if available
                "category": category,
                "rank": rank,
                "description": model_data.get("description", f"Anthropic Claude API model. Context: {ctx_size or '?'}. Max Output: {max_output or '?'}."),
                "ctx_size": ctx_size if ctx_size is not None else -1,
                "variants": [{"name": model_id, "size": size_proxy}]
            }
            # Add raw API data for potential future use/debugging
            model_entry["api_details"] = model_data
            models_info.append(model_entry)

        models_info.sort(key=lambda x: (-x['rank'], x['name']))
        ASCIIColors.success(f"Formatted {len(models_info)} Anthropic models for Lollms UI.")
        return models_info


# --- Main execution block remains for testing ---
if __name__ == "__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    from lollms.types import MSG_OPERATION_TYPE
    from time import perf_counter

    print("Initializing LoLLMs environment for testing...")
    lollms_paths = LollmsPaths.find_paths(force_local=True, tool_prefix="test_anthropic_")
    config = LOLLMSConfig.autoload(lollms_paths)
    lollms_app = LollmsApplication("TestApp", config, lollms_paths, load_bindings=False, load_personalities=False, load_models=False)

    print("Creating Anthropic binding instance...")
    try:
        # Note: Corrected class name used here
        anthropic_binding = Anthropic(config, lollms_paths, installation_option=InstallOption.INSTALL_IF_NECESSARY, lollmsCom=lollms_app.com)

        # --- API Key Setup ---
        if not os.getenv('ANTHROPIC_API_KEY') and not anthropic_binding.binding_config.config.get("anthropic_api_key"):
            try:
                key_input = input("Enter Anthropic API Key for testing: ").strip()
                if key_input:
                    anthropic_binding.binding_config.config["anthropic_api_key"] = key_input
                    print("API Key set for this session.")
                else:
                    print("No API key provided. Tests requiring API access will fail.")
            except EOFError:
                print("No API key input detected.")

        print("\nUpdating settings (initializes client and fetches models)...")
        anthropic_binding.settings_updated() # This now also calls _update_available_models

        available_models = anthropic_binding.list_models() # Should now use the fetched list
        if available_models:
            print(f"\nAvailable Anthropic Models ({len(available_models)} fetched from API):")
            # Print subset if list is long
            limit = 10
            if len(available_models) > limit:
                for m in available_models[:limit//2]: print(f"- {m}")
                print("  ...")
                for m in available_models[-limit//2:]: print(f"- {m}")
            else:
                for m in available_models: print(f"- {m}")

            # --- Test Setup ---
            test_model = "claude-3-haiku-20240307"
            if test_model not in available_models:
                print(f"\nWarning: Test model '{test_model}' not found in API list. Trying fallback.")
                test_model = "claude-3-5-sonnet-20240620"
                if test_model not in available_models:
                    if available_models: test_model = available_models[0]
                    else: test_model = None; print("Error: No models available from API to test.")

            if test_model:
                print(f"\nSelecting model for testing: {test_model}")
                anthropic_binding.config.model_name = test_model
                anthropic_binding.build_model() # Applies limits based on fetched/static data
                print(f"Effective limits for {test_model}: Context={anthropic_binding.config.ctx_size}, Max Output={anthropic_binding.config.max_n_predict}")

                def print_callback(chunk: str, msg_type: int, metadata: dict) -> bool:
                    # Simplified callback from previous example
                    type_str = MSG_OPERATION_TYPE(msg_type).name if msg_type in MSG_OPERATION_TYPE._value2member_map_ else f"UNKNOWN({msg_type})"
                    if msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK: print(chunk, end="", flush=True)
                    elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION: print(f"\n## [{type_str}] EXCEPTION: {chunk} ##"); return False
                    elif msg_type in [MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_INFO, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING]: print(f"\n## [{type_str}]: {chunk} ##")
                    return True

                # --- Test Case 1: Standard Text Generation (Streaming) ---
                print("\n" + "="*20 + " Test 1: Standard Text (Streaming) " + "="*20)
                prompt1 = "Explain the difference between a large language model and a multimodal model in simple terms."
                print(f"Prompt: {prompt1}\nResponse Stream:")
                try:
                    anthropic_binding.generate(prompt1, n_predict=200, callback=print_callback, verbose=True)
                    print("\n-- Test 1 Complete --")
                except Exception as e: print(f"\nTest 1 failed: {e}"); trace_exception(e)

                # --- Test Case 2: Vision Generation (Streaming) ---
                print("\n" + "="*20 + " Test 2: Vision Input (Streaming) " + "="*20)
                dummy_image_path = lollms_paths.personal_outputs_path / "test_anthropic_image.png"
                try:
                    from PIL import Image, ImageDraw # Keep import here for testing scope
                    img = Image.new('RGB', (120, 50), color = 'darkblue'); d = ImageDraw.Draw(img); d.text((10,10), "Claude Test", fill='white'); img.save(dummy_image_path)
                    print(f"Created dummy image: {dummy_image_path}")
                    prompt2 = "Describe the image provided."; images2 = [str(dummy_image_path)]
                    print(f"Prompt: {prompt2}\nImage: {images2[0]}\nResponse Stream:")
                    try:
                        anthropic_binding.generate_with_images(prompt2, images2, n_predict=100, callback=print_callback, verbose=True)
                        print("\n-- Test 2 Complete --")
                    except Exception as e: print(f"\nTest 2 failed: {e}"); trace_exception(e)
                    finally:
                        if dummy_image_path.exists(): dummy_image_path.unlink()
                except ImportError: print("PIL/Pillow not installed. Skipping vision test.")
                except Exception as e: print(f"Error during image creation/test: {e}")

                # --- Test Case 3: Cost Estimation ---
                print("\n" + "="*20 + " Test 3: Cost Estimation Check " + "="*20)
                anthropic_binding.binding_config.config["turn_on_cost_estimation"] = True
                prompt3 = "Write a very short poem about AI."; print(f"Prompt: {prompt3}\nGenerating...")
                try:
                    anthropic_binding.generate(prompt3, n_predict=50, callback=None, verbose=False)
                    print("Response Received.")
                    print(f"Final Accumulated Cost: ${anthropic_binding.binding_config.config['total_cost']:.6f}")
                    print(f"Total Input Tokens: {anthropic_binding.binding_config.config['total_input_tokens']}")
                    print(f"Total Output Tokens: {anthropic_binding.binding_config.config['total_output_tokens']}")
                    print("\n-- Test 3 Complete --")
                except Exception as e: print(f"\nTest 3 failed: {e}"); trace_exception(e)
                finally: anthropic_binding.binding_config.config["turn_on_cost_estimation"] = False
            else:
                print("\nSkipping generation tests as no suitable model could be selected/available.")
        else:
             print("\nCould not retrieve model list from API. Check API key and network connection.")
             print("Skipping tests.")

    except ImportError as e:
        print(f"Initialization failed due to missing import: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during initialization or testing: {e}")
        trace_exception(e)

    print("\nScript finished.")