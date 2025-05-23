######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying
# engine author : OpenAI
# license       : Apache 2.0
# Description   :
# This binding provides an interface to OpenAI's various GPT models,
# including text generation, vision capabilities, web search, and file search tools.
# It automatically attempts to configure context size and max output tokens
# based on known model specifications.
# Update date   : 19/07/2024
######
from pathlib import Path
from typing import Callable, Any, Optional, List, Dict, Union, Tuple
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors, trace_exception
from lollms.types import MSG_OPERATION_TYPE
from lollms.utilities import PackageManager, encode_image, find_first_available_file_path, is_file_path

from lollms.com import LoLLMsCom
import subprocess
import yaml
import sys
from PIL import Image
import io
import os
import json

# Try to install necessary packages using pipmaster
try:
    import pipmaster as pm
    pm.ensure_packages({
                        "openai":">=1.77.0",
                        "tiktoken":""
                    })
except ImportError:
    print("Warning: pipmaster not found. Please install required packages manually: pip install openai tiktoken")
    # Attempt direct import, assuming they might already be installed
    pass

# Import required libraries
try:
    import openai
    import tiktoken
    # Check for version compatibility if needed (optional)
    # from packaging import version
    # if version.parse(openai.__version__) < version.parse("1.0.0"):
    #     raise ImportError("Please upgrade the OpenAI library to version 1.0.0 or higher: pip install --upgrade openai")

except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure 'openai' and 'tiktoken' are installed (`pip install openai tiktoken`)")
    raise e # Re-raise the exception to prevent the script from continuing without dependencies


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023-2024, ParisNeo"
__license__ = "Apache 2.0"

binding_name = "OpenAIGPT"
binding_folder_name = "open_ai" # Assumed folder name in lollms_bindings_zoo


# ================= Known Model Limits =================
# Reference: User provided data + OpenAI documentation patterns
# These are used when auto_detect_limits is enabled and serve as fallbacks
# If the exact model isn't listed, we try to find the longest matching prefix.

MODEL_CONTEXT_SIZES: Dict[str, int] = {
    "gpt-4o-mini": 200000,         # From user data
    "gpt-4o": 128000,               # Official Docs
    "gpt-4.1-mini": 1047576,       # From user data
    "gpt-4.1": 1047576,            # From user data
    "gpt-4-turbo": 128000,         # Official Docs (includes -preview, -1106, -vision)
    "gpt-4-turbo-preview": 128000,
    "gpt-4-1106-preview": 128000,
    "gpt-4-vision-preview": 128000,
    "gpt-4-32k": 32768,            # Official Docs
    "gpt-4": 8192,                 # Official Docs (Base GPT-4)
    "gpt-3.5-turbo-0125": 16385,    # Official Docs
    "gpt-3.5-turbo-1106": 16385,    # Older, same context
    "gpt-3.5-turbo-instruct": 4096,# Legacy instruct model
    "gpt-3.5-turbo": 16385,        # Default alias, assume latest context
    # --- O-Series ---
    "o4-mini": 200000,             # From user data
    "o3-mini": 200000,             # From user data
    "o3": 200000,                  # From user data
    "o1-pro": 200000,              # From user data
    "o1-mini": 128000,             # From user data
    # --- Fallbacks (Broader categories) ---
    "gpt-4": 8192,                 # General GPT-4 fallback
    "gpt-3.5-turbo": 16385,        # General GPT-3.5 fallback
}

MODEL_MAX_OUTPUT_TOKENS: Dict[str, int] = {
    "gpt-4o-mini": 100000,         # From user data
    "gpt-4o": 16384,               # From user data (but some docs say 4096 for vision?) - Using user data
    "gpt-4.1-mini": 32768,         # From user data
    "gpt-4.1": 32768,              # From user data
    "gpt-4-turbo": 4096,           # Official Docs (includes -preview, -1106, -vision)
    "gpt-4-turbo-preview": 4096,
    "gpt-4-1106-preview": 4096,
    "gpt-4-vision-preview": 4096,
    "gpt-4-32k": 32768,            # Assuming it can output its full context? (Needs verification, often less) - Capping lower for safety
    "gpt-4": 8192,                 # Often capped lower in practice, e.g., 4096
    "gpt-3.5-turbo-0125": 4096,    # Official Docs
    "gpt-3.5-turbo-1106": 4096,    # Older
    "gpt-3.5-turbo-instruct": 4096,# Legacy instruct model
    "gpt-3.5-turbo": 4096,         # Default alias
    # --- O-Series ---
    "o4-mini": 100000,             # From user data
    "o3-mini": 100000,             # From user data
    "o3": 100000,                  # From user data
    "o1-pro": 100000,              # From user data
    "o1-mini": 65536,              # From user data
    # --- Fallbacks ---
    "gpt-4": 4096,                 # General GPT-4 fallback (safer than 8k)
    "gpt-3.5-turbo": 4096,         # General GPT-3.5 fallback
}


class OpenAIGPT(LLMBinding):
    """
    Binding class for interacting with the OpenAI API.

    Handles communication with various OpenAI models using different API endpoints
    (Chat Completions, Responses API for tools).
    Supports features like automatic context/output limit detection (using known values),
    vision input, web search, file search, cost estimation, and model listing.
    """
    def __init__(self,
                config: LOLLMSConfig,
                lollms_paths: LollmsPaths = None,
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY,
                lollmsCom:Optional[LoLLMsCom]=None) -> None:
        """
        Initialize the Binding.

        Args:
            config: The main LoLLMs configuration object.
            lollms_paths: An object containing important LoLLMs paths.
            installation_option: Enum defining installation behavior.
            lollmsCom: Communication object for interacting with LoLLMs UI (optional).
        """
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        self.lollmsCom = lollmsCom

        # --- Pricing Dictionaries (Update with latest OpenAI prices) ---
        self.input_costs_by_model: Dict[str, float] = {
            # Format: "model_id": cost_per_million_input_tokens / 1_000_000
            "o3":           10.00 / 1_000_000,
            "o4-mini":       1.100 / 1_000_000,
            "gpt-4.1":       2.00 / 1_000_000,
            "gpt-4.1-mini":  0.40 / 1_000_000,
            "gpt-4.1-nano":  0.100 / 1_000_000, # Assuming nano exists
            "gpt-4o":        5.00 / 1_000_000,  # Updated price based on docs ($2.5?) - keeping user data for now
            "gpt-4o-mini":   0.15 / 1_000_000,
            "gpt-4-turbo":          10.00 / 1_000_000,
            "gpt-4-turbo-preview":  10.00 / 1_000_000,
            "gpt-4-1106-preview":   10.00 / 1_000_000,
            "gpt-4-vision-preview": 10.00 / 1_000_000,
            "gpt-4":                30.00 / 1_000_000,
            "gpt-4-32k":            60.00 / 1_000_000,
            "gpt-3.5-turbo-0125":   0.50 / 1_000_000,
            "gpt-3.5-turbo":        0.50 / 1_000_000, # Alias assumes latest pricing
            "gpt-3.5-turbo-1106":   1.00 / 1_000_000, # Older pricing
            "gpt-3.5-turbo-instruct": 1.50 / 1_000_000,
            "gpt-3.5-turbo-16k":    3.00 / 1_000_000, # Older model
            # Add other models or update as needed
            "o1-mini": 1.1 / 1_000_000, # From user data
            "o1-pro": 150.0 / 1_000_000, # From user data
            # Fallbacks
            "default": 1.00 / 1_000_000 # Arbitrary fallback
        }
        self.output_costs_by_model: Dict[str, float] = {
            # Format: "model_id": cost_per_million_output_tokens / 1_000_000
            "o3":           40.00 / 1_000_000,
            "o4-mini":       4.400 / 1_000_000,
            "gpt-4.1":       8.00 / 1_000_000,
            "gpt-4.1-mini":  1.60 / 1_000_000,
            "gpt-4.1-nano":  0.400 / 1_000_000, # Assuming nano exists
            "gpt-4o":       15.00 / 1_000_000, # Updated price based on docs ($10?) - keeping user data
            "gpt-4o-mini":   0.60 / 1_000_000,
            "gpt-4-turbo":          30.00 / 1_000_000,
            "gpt-4-turbo-preview":  30.00 / 1_000_000,
            "gpt-4-1106-preview":   30.00 / 1_000_000,
            "gpt-4-vision-preview": 30.00 / 1_000_000,
            "gpt-4":                60.00 / 1_000_000,
            "gpt-4-32k":            120.00 / 1_000_000,
            "gpt-3.5-turbo-0125":   1.50 / 1_000_000,
            "gpt-3.5-turbo":        1.50 / 1_000_000, # Alias assumes latest
            "gpt-3.5-turbo-1106":   2.00 / 1_000_000, # Older pricing
            "gpt-3.5-turbo-instruct": 2.00 / 1_000_000,
            "gpt-3.5-turbo-16k":    4.00 / 1_000_000, # Older model
            # Add other models or update as needed
            "o1-mini": 4.4 / 1_000_000, # From user data
            "o1-pro": 600.0 / 1_000_000, # From user data
            # Fallbacks
            "default": 3.00 / 1_000_000 # Arbitrary fallback
        }
        # Note: Costs for tools (web search, file search) are separate and not included in the above.

        # Configuration definition using ConfigTemplate
        binding_config_template = ConfigTemplate([
            # --- General ---
            {"name":"openai_key","type":"str","value":"","help":"Your OpenAI API key. Found at https://platform.openai.com/api-keys"},
            {"name": "auto_detect_limits", "type": "bool", "value": True, "help": "Automatically detect and use the selected model's context size and max output tokens based on known values. If false, uses the manually set values below.", "requires_restart": False}, # Requires model rebuild, not full restart
            {"name":"ctx_size","type":"int","value":128000, "min":512, "help":"Model's maximum context size (input + output tokens). Automatically updated based on the selected model if 'auto_detect_limits' is enabled. Default shown is for GPT-4 Turbo/GPT-4o."},
            {"name":"max_n_predict","type":"int","value":4096, "min":1, "help":"Maximum number of tokens to generate per response. Automatically updated based on the selected model if 'auto_detect_limits' is enabled. Should be <= model's max output tokens. Default shown is common for many models."},
            {"name":"seed","type":"int","value":-1,"help":"Random seed for generation (-1 for random). Note: Seed support varies by model and API endpoint."},

            # --- Vision ---
            {"name":"max_image_width","type":"int","value":-1,"help":"Resize images if wider than this before sending to vision models (reduces cost). -1 for no change."},

            # --- Tools ---
            {"name":"enable_web_search", "type":"bool", "value":False, "help":"Enable web search using the Responses API (requires compatible model like gpt-4.1, gpt-4o). Disables token streaming."},
            {"name":"web_search_force", "type":"bool", "value":False, "help":"Force the use of the web search tool (if enabled)."},
            {"name":"web_search_context_size", "type":"str", "value":"medium", "options":["low", "medium", "high"], "help":"Context size for web search (affects cost, quality, latency)."},
            {"name":"web_search_user_location_country", "type":"str", "value":"", "help":"Approximate user country (ISO code, e.g., US) for localized web search."},
            {"name":"web_search_user_location_city", "type":"str", "value":"", "help":"Approximate user city for localized web search."},
            {"name":"web_search_user_location_region", "type":"str", "value":"", "help":"Approximate user region/state for localized web search."},
            {"name":"web_search_user_location_timezone", "type":"str", "value":"", "help":"Approximate user timezone (IANA, e.g., America/Chicago) for localized web search."},

            {"name":"enable_file_search", "type":"bool", "value":False, "help":"Enable file search using the Responses API (requires compatible model and a Vector Store ID). Disables token streaming."},
            {"name":"file_search_vector_store_id", "type":"str", "value":"", "help":"The ID of the OpenAI Vector Store containing files to search. Create Vector Stores at https://platform.openai.com/vector-stores"},
            {"name":"file_search_force", "type":"bool", "value":False, "help":"Force the use of the file search tool (if enabled and Vector Store ID is provided)."},
            {"name":"file_search_max_num_results", "type":"int", "value":20, "min":1, "max":50, "help":"Maximum number of file search results to retrieve."},

            # --- Cost Estimation ---
            {"name":"turn_on_cost_estimation","type":"bool", "value":False,"help":"Estimate query costs based on input/output text tokens (Note: Tool usage costs are separate and NOT included)."},
            {"name":"total_input_tokens","type":"float", "value":0,"help":"Accumulated input tokens (for cost estimation)."},
            {"name":"total_output_tokens","type":"float", "value":0,"help":"Accumulated output tokens (for cost estimation)."},
            {"name":"total_input_cost","type":"float", "value":0,"help":"Accumulated input cost ($) (text tokens only)."},
            {"name":"total_output_cost","type":"float", "value":0,"help":"Accumulated output cost ($) (text tokens only)."},
            {"name":"total_cost","type":"float", "value":0,"help":"Total accumulated cost ($) (text tokens only)."},
        ])
        # Default values for the configuration
        binding_config_defaults = BaseConfig(config={
            "openai_key": "",
            "auto_detect_limits": True,
            "ctx_size": 128000,        # Fallback if detection fails or is off
            "max_n_predict": 4096,    # Fallback if detection fails or is off
            "seed": -1,
            "max_image_width": -1,
            "enable_web_search": False,
            "web_search_force": False,
            "web_search_context_size": "medium",
            "web_search_user_location_country": "",
            "web_search_user_location_city": "",
            "web_search_user_location_region": "",
            "web_search_user_location_timezone": "",
            "enable_file_search": False,
            "file_search_vector_store_id": "",
            "file_search_force": False,
            "file_search_max_num_results": 20,
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
                            SAFE_STORE_SUPPORTED_FILE_EXTENSIONS=['.png', '.jpg', '.jpeg', '.webp', '.gif'], # Supported by vision models
                            lollmsCom=lollmsCom
                        )
        # Defer setting self.config context/prediction limits until build_model
        self.available_models: List[str] = []
        self.client: Optional[openai.OpenAI] = None


    def _update_openai_key(self) -> bool:
        """
        Sets the openai.api_key and initializes the OpenAI client
        based on configuration or environment variables.

        Returns:
            True if an API key was found and client initialized, False otherwise.
        """
        api_key = None
        source = None
        config_key = self.binding_config.config.get("openai_key", "")

        if config_key and config_key.strip():
            api_key = config_key
            source = "configuration"
        else:
            api_key_env = os.getenv('OPENAI_API_KEY')
            if api_key_env:
                api_key = api_key_env
                source = "environment variable"

        if api_key:
            # Avoid setting the global openai.api_key if possible, prefer client instantiation
            try:
                self.client = openai.OpenAI(api_key=api_key)
                # Test the key with a simple call (optional, but recommended)
                # self.client.models.list(limit=1)
                ASCIIColors.info(f"Using OpenAI API key from {source}.")
                return True
            except openai.AuthenticationError:
                 self.error(f"Invalid OpenAI API key provided from {source}. Please check your key.")
                 self.client = None
                 return False
            except Exception as e:
                self.error(f"Failed to initialize OpenAI client: {e}")
                trace_exception(e)
                self.client = None
                return False
        else:
            self.warning("No OpenAI API key found in configuration or environment variables. OpenAI binding will not function.")
            self.client = None
            return False

    def _update_available_models(self) -> None:
        """
        Fetches the list of available models from the OpenAI API, filters for relevant models,
        and updates self.available_models. Requires the client to be initialized.
        """
        if not self.client:
            self.warning("OpenAI client not initialized. Cannot fetch models.")
            self.available_models = []
            return

        try:
            ASCIIColors.info("Fetching available models from OpenAI API...")
            models_response = self.client.models.list()
            raw_models = [model.id for model in models_response.data]
            ASCIIColors.info(f"Fetched {len(raw_models)} total models. Filtering...")

            # --- Filtering Logic ---
            filtered_models: List[str] = []
            # Prefixes for models to EXCLUDE (non-generative tasks)
            excluded_prefixes: List[str] = ['dall-e', 'tts-', 'whisper-']
            # Keywords for models to EXCLUDE (specific functionalities)
            excluded_keywords: List[str] = ['embedding', 'moderation', 'similarity', 'edit']

            for model_id in raw_models:
                is_excluded = False
                # Check excluded prefixes
                if any(model_id.startswith(prefix) for prefix in excluded_prefixes):
                    is_excluded = True

                # Check excluded keywords *unless* it's a core family model
                if not is_excluded and not model_id.startswith(('gpt-', 'o1', 'o3', 'o4')):
                    if any(keyword in model_id for keyword in excluded_keywords):
                        is_excluded = True

                # Specific exclusion for base models often used for fine-tuning
                if not is_excluded and model_id in ['ada', 'babbage', 'curie', 'davinci'] and not model_id.startswith('text-davinci-'):
                    is_excluded = True

                if not is_excluded:
                    filtered_models.append(model_id)
            # --- End Filtering Logic ---

            self.available_models = sorted(filtered_models)

            ASCIIColors.success(f"Found {len(self.available_models)} relevant models after filtering.")
            if not self.available_models:
                 self.warning("API call succeeded but returned no relevant models after filtering.")

        except openai.AuthenticationError:
            self.error("Authentication Error: Invalid OpenAI API key. Cannot fetch models.")
            self.available_models = []
        except Exception as e:
            self.error(f"Failed to fetch models from OpenAI API: {e}")
            trace_exception(e)
            self.available_models = []

    def settings_updated(self) -> None:
        """Callback triggered when binding settings are updated in the UI."""
        ASCIIColors.info("OpenAI settings updated.")
        key_changed = self._update_openai_key()

        # If the key is valid (or was already valid), try to rebuild the model
        # This applies changes like toggling auto_detect_limits or manual context sizes
        if self.client:
            if key_changed: # Refresh models only if the key itself changed
                self._update_available_models()
            # Rebuild the model to apply potential changes in limits/settings
            if self.config.model_name:
                self.info("Rebuilding model due to settings update...")
                self.build_model(self.config.model_name)
            else:
                 self.info("No model currently selected, skipping model rebuild after settings update.")
        else:
            # If the key became invalid or was removed
            self.available_models = []
            self.config.ctx_size = self.binding_config.config.ctx_size # Reset to default/manual
            self.config.max_n_predict = self.binding_config.config.max_n_predict # Reset to default/manual
            if self.lollmsCom:
                 self.lollmsCom.InfoMessage("OpenAI Error: API Key is missing or invalid. Please configure it.")


    def get_model_limits(self, model_name: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Tries to find the context size and max output tokens for a given model name
        using the hardcoded dictionaries.

        Args:
            model_name: The name of the model.

        Returns:
            A tuple (context_size, max_output_tokens). Values are None if not found.
        """
        if not model_name:
            return None, None

        found_ctx = None
        found_max_out = None
        best_match_len = 0

        # Find the longest prefix match in context sizes
        for prefix, size in MODEL_CONTEXT_SIZES.items():
            if model_name.startswith(prefix):
                if len(prefix) > best_match_len:
                    found_ctx = size
                    best_match_len = len(prefix)

        best_match_len = 0
        # Find the longest prefix match in max output tokens
        for prefix, size in MODEL_MAX_OUTPUT_TOKENS.items():
            if model_name.startswith(prefix):
                if len(prefix) > best_match_len:
                    found_max_out = size
                    best_match_len = len(prefix)

        return found_ctx, found_max_out


    def build_model(self, model_name: Optional[str] = None) -> LLMBinding:
        """
        Sets up the binding for the selected model.
        This involves ensuring the API key is set, fetching model lists if needed,
        and setting the context/output limits based on the auto_detect setting.

        Args:
            model_name: The name of the model to potentially load (used by parent).

        Returns:
            The instance of the binding.

        Raises:
            RuntimeError: If the API key is missing or invalid after attempting setup.
        """
        super().build_model(model_name) # Sets self.config.model_name from argument or config

        if not self.client:
             if not self._update_openai_key():
                 self.available_models = []
                 self.error("Model build failed: OpenAI API key is missing or invalid.")
                 # Maybe notify UI if possible?
                 if self.lollmsCom:
                     self.lollmsCom.InfoMessage("OpenAI Error: Cannot build model. API Key is missing or invalid.")
                 # Returning self allows LoLLMs to potentially handle the error state
                 return self

        # Fetch models if list is empty (might have been fetched by settings_updated)
        if not self.available_models:
            self._update_available_models()

        current_model_name = self.config.model_name or ""
        if not current_model_name:
            self.warning("No model name selected. Cannot determine limits or capabilities.")
            # Keep defaults from binding_config
            self.config.ctx_size = self.binding_config.config.ctx_size
            self.config.max_n_predict = self.binding_config.config.max_n_predict
            return self

        # --- Determine Effective Limits ---
        detected_ctx_size, detected_max_output = self.get_model_limits(current_model_name)

        effective_ctx_size = self.binding_config.config["ctx_size"] # Start with manual/default
        effective_max_output = self.binding_config.config["max_n_predict"] # Start with manual/default

        if self.binding_config.auto_detect_limits:
            ASCIIColors.info(f"Auto-detect limits enabled for {current_model_name}.")
            if detected_ctx_size is not None:
                effective_ctx_size = detected_ctx_size
                ASCIIColors.success(f"  Detected context size: {effective_ctx_size}")
            else:
                ASCIIColors.warning(f"  Could not auto-detect context size for {current_model_name}. Using manual/default value: {effective_ctx_size}")

            if detected_max_output is not None:
                effective_max_output = detected_max_output
                ASCIIColors.success(f"  Detected max output tokens: {effective_max_output}")
            else:
                 ASCIIColors.warning(f"  Could not auto-detect max output tokens for {current_model_name}. Using manual/default value: {effective_max_output}")
        else:
            ASCIIColors.info("Auto-detect limits disabled. Using manually configured limits.")
            # effective_ctx_size/effective_max_output are already set to manual values
            # Optionally warn if manual settings exceed detected limits (if detection worked)
            if detected_ctx_size is not None and effective_ctx_size > detected_ctx_size:
                self.warning(f"Manually set ctx_size ({effective_ctx_size}) exceeds the detected model limit ({detected_ctx_size}). This may cause errors.")
            if detected_max_output is not None and effective_max_output > detected_max_output:
                 self.warning(f"Manually set max_n_predict ({effective_max_output}) exceeds the detected model limit ({detected_max_output}). The API might cap the output or error.")

        # --- Update Configurations ---
        # Update binding config reflects the effective values used (for saving state)
        self.binding_config.config["ctx_size"] = effective_ctx_size
        self.binding_config.config["max_n_predict"] = effective_max_output
        # Update main lollms config for global access and UI reflection
        self.config.ctx_size = effective_ctx_size
        self.config.max_n_predict = effective_max_output

        # Log the final state (UI read-only status is just informational here)
        ASCIIColors.info(f"Effective limits set: ctx_size={effective_ctx_size}, max_n_predict={effective_max_output}")
        ASCIIColors.info(f"Context size and Max Output Tokens fields are {'managed by auto-detect' if self.binding_config.auto_detect_limits else 'editable (manual)'}.")

        # --- Determine Binding Type and Log API Info ---
        if "vision" in current_model_name or "4o" in current_model_name: # gpt-4o is multimodal
            self.binding_type = BindingType.TEXT_IMAGE
            self.SAFE_STORE_SUPPORTED_FILE_EXTENSIONS=['.png', '.jpg', '.jpeg', '.webp', '.gif'] # Enable image extensions
            ASCIIColors.info(f"Model {current_model_name} supports vision. Binding type: TEXT_IMAGE.")
        else:
            self.binding_type = BindingType.TEXT_ONLY
            self.SAFE_STORE_SUPPORTED_FILE_EXTENSIONS=[] # Disable file extensions for non-vision models
            # Log expected API usage based on name (actual choice happens in generate)
            legacy_indicators = ["instruct", "davinci", "curie", "babbage", "ada"]
            is_legacy = any(ind in current_model_name for ind in legacy_indicators) and "turbo" not in current_model_name and "gpt-3.5" not in current_model_name
            if is_legacy: ASCIIColors.info(f"Model {current_model_name} appears legacy/instruct. Will use Completions API if tools disabled.")
            else: ASCIIColors.info(f"Model {current_model_name} assumed Chat. Will use Chat API if tools disabled.")
            ASCIIColors.info(f"Binding type set to TEXT_ONLY.")

        ws_enabled = self.binding_config.config.get("enable_web_search", False)
        fs_enabled = self.binding_config.config.get("enable_file_search", False)
        ASCIIColors.success(f"OpenAI binding built successfully. Model: {current_model_name}. Web Search: {ws_enabled}. File Search: {fs_enabled}.")
        return self


    def install(self) -> None:
        """Installs necessary Python packages using pipmaster."""
        super().install()
        self.ShowBlockingMessage("Installing OpenAI API client requirements...")
        try:
            # Use pipmaster if available
            import pipmaster as pm
            requirements = ["openai", "tiktoken"]
            for req in requirements:
                if not pm.is_installed(req):
                    self.info(f"Installing {req}...")
                    pm.install(req)
                else:
                    self.info(f"{req} already installed.")
            self.HideBlockingMessage()
            ASCIIColors.success("OpenAI client requirements installed successfully.")
            ASCIIColors.info("----------------------")
            ASCIIColors.info("Attention:")
            ASCIIColors.info("----------------------")
            ASCIIColors.info("The OpenAI binding uses the OpenAI API, which is a paid service.")
            ASCIIColors.info("1. Create an account at https://platform.openai.com/")
            ASCIIColors.info("2. Generate an API key.")
            ASCIIColors.info("3. Provide the key in the binding settings or set the OPENAI_API_KEY environment variable.")
            ASCIIColors.info("4. For File Search, create a Vector Store and add its ID to the settings.")
        except ImportError:
            self.HideBlockingMessage()
            self.warning("pipmaster not found. Please install requirements manually: pip install openai tiktoken")
        except Exception as e:
            self.error(f"Installation failed: {e}")
            trace_exception(e)
            self.warning("Installation failed. Please ensure you have pip installed and internet access.", 20)
            self.HideBlockingMessage()


    def tokenize(self, prompt: str) -> List[int]:
        """
        Tokenizes the given prompt using tiktoken.

        Args:
            prompt: The text prompt to tokenize.

        Returns:
            A list of token IDs. Returns an empty list if tokenization fails.
        """
        if not prompt:
            return []
        try:
            # Use the currently selected model name if available for accuracy
            # Fallback to a common encoding if model not selected or unknown
            model_name = self.config.model_name if self.config.model_name else "gpt-4" # Default to gpt-4 encoding
            try:
                # tiktoken sometimes fails on specific model variants, try the base name
                base_model_name = model_name.split('-')[0] # e.g., 'gpt-4' from 'gpt-4-turbo-preview'
                if base_model_name in ["gpt", "o1", "o3", "o4"]: # Only use base if it's a known family
                     encoding = tiktoken.encoding_for_model(base_model_name)
                else:
                     encoding = tiktoken.encoding_for_model(model_name) # Try original name first
            except KeyError:
                try:
                    # Try the full name again if base failed or wasn't used
                    encoding = tiktoken.encoding_for_model(model_name)
                except KeyError:
                     ASCIIColors.warning(f"Tiktoken encoding not found for '{model_name}' or its base. Using 'cl100k_base'.")
                     encoding = tiktoken.get_encoding("cl100k_base") # Common for GPT-3.5/4

            return encoding.encode(prompt)
        except Exception as e:
            self.error(f"An unexpected error occurred during tokenization: {e}")
            trace_exception(e)
            return []


    def detokenize(self, tokens_list: List[int]) -> str:
        """
        Detokenizes the given list of tokens using tiktoken.

        Args:
            tokens_list: A list of token IDs.

        Returns:
            The detokenized text string. Returns an empty string if detokenization fails.
        """
        if not tokens_list:
            return ""
        try:
            # Use the currently selected model name if available
            model_name = self.config.model_name if self.config.model_name else "gpt-4"
            try:
                # Same logic as tokenize for finding the encoding
                base_model_name = model_name.split('-')[0]
                if base_model_name in ["gpt", "o1", "o3", "o4"]:
                     encoding = tiktoken.encoding_for_model(base_model_name)
                else:
                     encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                try:
                    encoding = tiktoken.encoding_for_model(model_name)
                except KeyError:
                    # Fallback is usually safe for decoding
                    encoding = tiktoken.get_encoding("cl100k_base")

            return encoding.decode(tokens_list)
        except Exception as e:
            self.error(f"An unexpected error occurred during detokenization: {e}")
            trace_exception(e)
            return ""


    def embed(self, text: Union[str, List[str]]) -> Optional[List[List[float]]]:
        """
        Computes text embeddings using the OpenAI API. Handles single string or list of strings.

        Args:
            text: The text (str) or list of texts (List[str]) to embed.

        Returns:
            A list of embedding lists (one for each input text), or None if an error occurs.
        """
        if not self.client:
            self.error("OpenAI client not initialized. Cannot compute embeddings.")
            return None

        # Recommended embedding model (check OpenAI docs for latest)
        embedding_model = "text-embedding-3-small" # Or text-embedding-3-large, etc.

        try:
            # Ensure input is a list for the API call
            input_texts = [text] if isinstance(text, str) else text
            if not isinstance(input_texts, list) or not all(isinstance(t, str) for t in input_texts):
                 self.error(f"Invalid input type for embedding: {type(text)}. Expected str or list[str].")
                 return None

            response = self.client.embeddings.create(
                input=input_texts,
                model=embedding_model
            )

            if response.data:
                 # Sort embeddings by index to match input order
                 embeddings_sorted = sorted(response.data, key=lambda e: e.index)
                 return [e.embedding for e in embeddings_sorted]
            else:
                 self.error("Embedding API returned no data.")
                 return None
        except openai.AuthenticationError:
             self.error("Authentication Error: Invalid OpenAI API key for embeddings.")
             return None
        except Exception as e:
            self.error(f"Failed to compute embedding: {e}")
            trace_exception(e)
            return None


    def generate_with_images(self,
                             prompt: str,
                             images: List[str],
                             n_predict: Optional[int] = None, # Changed default to None
                             callback: Optional[Callable[[str, int], bool]] = None,
                             verbose: bool = False,
                             **gpt_params) -> str:
        """
        Generates text using a prompt and optional images (for vision models).
        Uses the Chat Completions API. Web search and file search are NOT supported by this method.

        Args:
            prompt: The text prompt.
            images: A list of paths to image files.
            n_predict: Optional override for the maximum number of tokens to generate.
                       If None, uses the model's detected or configured max output tokens.
            callback: An optional callback function for streaming results.
            verbose: If True, prints more detailed information.
            **gpt_params: Additional parameters for the OpenAI API call.

        Returns:
            The generated text response.
        """
        if self.binding_config.config.get("enable_web_search", False) or self.binding_config.config.get("enable_file_search", False):
             self.warning("Web/File search is enabled but not supported by generate_with_images. Ignoring tools for this call.")
             # Fall through to normal image generation without tools

        if not self.client:
            self.error("OpenAI client not initialized.")
            if callback: callback("Error: OpenAI client not initialized.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return ""

        # Use effective max output tokens from config if n_predict is not provided
        effective_max_n_predict = self.config.max_n_predict
        if n_predict is not None:
             if 0 < n_predict <= self.config.max_n_predict:
                  effective_max_n_predict = n_predict
                  ASCIIColors.info(f"Using user-provided n_predict: {effective_max_n_predict}")
             elif n_predict > self.config.max_n_predict:
                  self.warning(f"Requested n_predict ({n_predict}) exceeds the effective model limit ({self.config.max_n_predict}). Capping at {self.config.max_n_predict}.")
                  effective_max_n_predict = self.config.max_n_predict
             # else: n_predict <= 0 or invalid, use config default

        if not (self.binding_type == BindingType.TEXT_IMAGE):
             self.error(f"Model '{self.config.model_name}' does not support image input according to its configuration.")
             if callback: callback(f"Error: Model '{self.config.model_name}' does not support image input.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""

        if not prompt and not images:
             self.error("Cannot generate response with empty prompt and no images.")
             if callback: callback("Error: Empty prompt and no images provided.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""

        model_name = self.config.model_name or ""
        if not model_name:
             self.error("Cannot generate response, no model selected.")
             if callback: callback("Error: No model selected.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""


        # --- Prepare API parameters ---
        default_params = {
            'temperature': self.config.temperature,
            # Add other lollms config defaults if needed (e.g., penalties)
        }
        final_params = {**default_params, **gpt_params}
        final_temperature = float(final_params.get("temperature", 1.0)) # Default to 1.0 if not provided

        seed = self.binding_config.config.get("seed", -1)
        api_params: Dict[str, Any] = { # Explicitly define type
            "model": model_name,
            "max_completion_tokens": effective_max_n_predict,
            "stream": True, # Always stream for vision? Or make optional? Keep streaming for now.
        }
        # Only add temp if they are not the default values OpenAI uses (usually 1.0)
        # to avoid sending unnecessary parameters. Check OpenAI defaults if unsure.
        if final_temperature != 1.0: api_params["temperature"] = final_temperature
        if seed is not None and seed != -1: api_params["seed"] = seed


        # --- Prepare message content (text + images) ---
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}] if prompt else []
        processed_image_count = 0
        invalid_image_paths = []
        # Rough token estimate per image (OpenAI uses complex formula, this is placeholder)
        # High detail can be ~1100 tokens, Low ~150. Using mid-range.
        # Let's use 0 for now, as cost calc only uses text tokens.
        image_token_estimate_per_image = 0
        total_image_token_estimate = 0

        max_image_width = self.binding_config.config.get("max_image_width", -1)

        for image_path_str in images:
             # Validate path and find the actual file
            image_path = Path(image_path_str)
            valid_image_path = find_first_available_file_path([image_path])

            if not valid_image_path or not is_file_path(valid_image_path) or not valid_image_path.exists():
                 self.warning(f"Image path not found or invalid: {image_path_str}. Skipping.")
                 invalid_image_paths.append(image_path_str)
                 if callback: callback(f"Warning: Image not found {image_path_str}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING)
                 continue

            try:
                encoded_image = encode_image(str(valid_image_path), max_image_width)
                if encoded_image:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}",
                            "detail": "auto" # Let OpenAI decide detail level (low, high, auto)
                        }
                    })
                    processed_image_count += 1
                    total_image_token_estimate += image_token_estimate_per_image
                    ASCIIColors.info(f"Successfully prepared image: {valid_image_path}")
                else:
                     self.warning(f"Could not encode image: {valid_image_path}")
                     invalid_image_paths.append(image_path_str)
                     if callback: callback(f"Warning: Failed to encode image {valid_image_path}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING)

            except Exception as img_ex:
                 self.error(f"Error processing image {valid_image_path}: {img_ex}")
                 invalid_image_paths.append(image_path_str)
                 trace_exception(img_ex)
                 if callback: callback(f"Error processing image {valid_image_path}: {img_ex}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)


        if not content or (not prompt and processed_image_count == 0):
            self.error("Failed to prepare any content (text or valid images) for the API request.")
            if callback: callback("Error: No valid text prompt or images provided.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return ""

        if invalid_image_paths:
             self.warning(f"Skipped {len(invalid_image_paths)} invalid or unloadable images.")

        api_params["messages"] = [{"role": "user", "content": content}]

        # --- Cost Estimation (Input - Text Tokens Only) ---
        prompt_tokens = 0
        if self.binding_config.config.turn_on_cost_estimation:
            if prompt:
                 prompt_tokens = len(self.tokenize(prompt))
            # Currently not including image token estimate in cost calculation
            input_tokens = prompt_tokens # + total_image_token_estimate
            self.binding_config.config["total_input_tokens"] += input_tokens
            # Use model name or 'default' for cost lookup
            input_cost_rate = self.input_costs_by_model.get(model_name, self.input_costs_by_model.get("default", 0))
            input_cost = input_tokens * input_cost_rate
            self.binding_config.config["total_input_cost"] += input_cost

        # --- Make API Call and Stream Response ---
        output = ""
        total_output_tokens = 0
        start_time = perf_counter()
        try:
            if verbose: ASCIIColors.verbose(f"Calling Chat Completions API with vision. Params: {api_params}")
            chat_completion = self.client.chat.completions.create(**api_params)

            stream_finished = False
            for chunk in chat_completion:
                chunk_text = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                finish_reason = chunk.choices[0].finish_reason

                if chunk_text:
                    output += chunk_text
                    if callback is not None:
                        # Only count tokens if needed for cost estimation later
                        # chunk_tokens = 0
                        # if self.binding_config.config.turn_on_cost_estimation:
                        #      chunk_tokens = len(self.tokenize(chunk_text))
                        # total_output_tokens += chunk_tokens

                        if not callback(chunk_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                            self.info("Generation stopped by callback.")
                            stream_finished = True # Mark as finished due to callback
                            # How to stop the stream? Breaking the loop is usually sufficient.
                            # chat_completion.close() # Check if stream object has a close method
                            break # Stop processing stream

                if finish_reason:
                    if verbose: ASCIIColors.verbose(f"Generation finished. Reason: {finish_reason}")
                    stream_finished = True
                    # Check for usage stats if available on the final chunk
                    if chunk.usage:
                         if verbose: ASCIIColors.verbose(f"Usage stats received: {chunk.usage}")
                         # Override token counts if provided by API (more accurate)
                         if self.binding_config.config.turn_on_cost_estimation:
                              # Subtract previously added prompt tokens, add API reported value
                              self.binding_config.config["total_input_tokens"] -= prompt_tokens
                              self.binding_config.config["total_input_tokens"] += chunk.usage.prompt_tokens
                              # Don't add completion tokens here, do it after loop from final output
                              # self.binding_config.config["total_output_tokens"] += chunk.usage.completion_tokens
                              total_output_tokens = chunk.usage.completion_tokens # Use API value

                    break # End loop once finished

            if not stream_finished:
                 self.info("Stream ended without explicit finish reason.")

        except openai.AuthenticationError as e:
            self.error("Authentication Error: Invalid OpenAI API key.")
            trace_exception(e)
            if callback: callback("Authentication Error: Invalid OpenAI API key.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return ""
        except openai.RateLimitError as e:
             self.error("OpenAI API rate limit exceeded.")
             trace_exception(e)
             if callback: callback("OpenAI API rate limit exceeded.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""
        except openai.BadRequestError as e:
             # Often related to image issues (format, size, safety) or unsupported params
             self.error(f'OpenAI API Bad Request Error: {e}. Check image validity, prompt, or parameters.')
             trace_exception(e)
             if callback: callback(f"OpenAI API Bad Request: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""
        except openai.APIError as api_ex:
             self.error(f'OpenAI API Error: {api_ex}')
             trace_exception(api_ex)
             if callback: callback(f"OpenAI API Error: {api_ex}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""
        except Exception as ex:
            self.error(f'Error during generation with images: {ex}')
            trace_exception(ex)
            if callback: callback(f"Error: {ex}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return ""
        finally:
             generation_time = perf_counter() - start_time
             ASCIIColors.info(f"Generation finished in {generation_time:.2f} seconds.")


        # --- Cost Estimation (Output - Text Tokens Only) ---
        if self.binding_config.config.turn_on_cost_estimation:
            # If API didn't provide completion tokens, tokenize the final output
            if total_output_tokens == 0 and output:
                total_output_tokens = len(self.tokenize(output))

            self.binding_config.config["total_output_tokens"] += total_output_tokens
            output_cost_rate = self.output_costs_by_model.get(model_name, self.output_costs_by_model.get("default", 0))
            output_cost = total_output_tokens * output_cost_rate
            self.binding_config.config["total_output_cost"] += output_cost
            self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]
            self.info(f'Accumulated cost (text tokens only): ${self.binding_config.config["total_cost"]:.6f}')
            self.binding_config.save() # Save updated costs

        return output


    def generate(self,
                 prompt: str,
                 n_predict: Optional[int] = None, # Changed default to None
                 callback: Optional[Callable[[str, int], bool]] = None,
                 verbose: bool = False,
                 **gpt_params) -> str:
        """
        Generates text using the OpenAI API, automatically handling tools or standard chat/legacy completion.

        - Uses Responses API (`responses.create`) if Web Search or File Search is enabled (disables streaming).
        - Uses Chat Completions API (`chat.completions.create`) for most modern models (supports streaming).
        - Uses Legacy Completions API (`completions.create`) for older instruct/legacy models (supports streaming).

        Args:
            prompt: The text prompt.
            n_predict: Optional override for the maximum number of tokens to generate.
                       If None, uses the model's detected or configured max output tokens.
            callback: An optional callback function for streaming results (only works when tools are disabled).
                      Signature: callback(token_or_full_response: str, message_type: int, metadata: dict) -> bool
            verbose: If True, prints more detailed information.
            **gpt_params: Additional parameters for the OpenAI API call (e.g., temperature).

        Returns:
            The generated text response, potentially including formatted citations if tools were used.
        """
        from time import perf_counter # Import here for performance measurement

        if not self.client:
            self.error("OpenAI client not initialized.")
            if callback: callback("Error: OpenAI client not initialized.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return ""

        model_name = self.config.model_name
        if not model_name:
             self.error("No model selected.")
             if callback: callback("Error: No model selected.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""

        # --- Parameter Preparation ---
        # Use effective max output tokens from config if n_predict is not provided
        effective_max_n_predict = self.config.max_n_predict
        if n_predict is not None:
             if 0 < n_predict <= self.config.max_n_predict:
                  effective_max_n_predict = n_predict
                  if verbose: ASCIIColors.verbose(f"Using user-provided n_predict: {effective_max_n_predict}")
             elif n_predict > self.config.max_n_predict:
                  self.warning(f"Requested n_predict ({n_predict}) exceeds the effective model limit ({self.config.max_n_predict}). Capping at {self.config.max_n_predict}.")
                  effective_max_n_predict = self.config.max_n_predict
             # else: n_predict <= 0 or invalid, use config default
        if verbose: ASCIIColors.verbose(f"Effective max_completion_tokens for this generation: {effective_max_n_predict}")


        default_params = {
            'temperature': self.config.temperature,
        }
        final_params = {**default_params, **gpt_params}

        # Clean and validate parameters
        final_temperature = float(final_params.get("temperature", 1.0))
        seed = self.binding_config.config.get("seed", -1)


        # Temperature Override Logic for specific families (Optional, adjust as needed)
        # Note: Tools usage might ignore temperature.
        temp_override_applied = False
        if model_name:
             if ('o4' in model_name or 'o3' in model_name) and final_temperature != 1.0:
                 self.warning(f"Model family '{model_name}' might work best with temp=1.0. Current: {final_temperature}.")
                 final_temperature = 1.0; 
                 temp_override_applied = True
                 final_freq_penalty = None
             elif 'search' in model_name: # Hypothetical search model family
                 if final_temperature != 1.0:
                     self.warning(f"Search models often ignore temperature. Setting to None for API call.")
                     final_temperature = None # Explicitly None might be better than 1.0
                     temp_override_applied = True

        # --- Cost Estimation (Input - Text Tokens Only) ---
        total_input_tokens = 0
        prompt_tokens = 0 # Keep track for potential usage stats correction
        if self.binding_config.config.turn_on_cost_estimation:
            prompt_tokens = len(self.tokenize(prompt))
            total_input_tokens = prompt_tokens
            self.binding_config.config["total_input_tokens"] += total_input_tokens
            input_cost_rate = self.input_costs_by_model.get(model_name, self.input_costs_by_model.get("default", 0))
            input_cost = total_input_tokens * input_cost_rate
            self.binding_config.config["total_input_cost"] += input_cost

        output = ""
        output_text = "" # For text part when using tools
        total_output_tokens = 0
        citation_text = ""
        metadata = {} # To store finish reason, usage, etc.

        # --- Determine API Path based on Tool Usage ---
        enable_web_search = self.binding_config.config.get("enable_web_search", False)
        enable_file_search = self.binding_config.config.get("enable_file_search", False)
        use_responses_api = enable_web_search or enable_file_search

        start_time = perf_counter()
        try:
            # ==================================================================
            # === BRANCH 1: Tools Enabled -> Use Responses API             ===
            # ==================================================================
            if use_responses_api:
                self.info(f"Tools enabled (WebSearch:{enable_web_search}, FileSearch:{enable_file_search}). Using Responses API. Streaming disabled.")
                if callback:
                    callback("INFO: Tool usage detected. Streaming is disabled. Full response will be provided at the end.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_INFO)

                # --- Prepare Tools List ---
                tools: List[Dict[str, Any]] = []
                tool_choice: Optional[Union[Dict[str, str], str]] = "auto" # Default: let model choose

                # Configure Web Search Tool
                if enable_web_search:
                    # Compatibility checks can be added here if needed
                    # if "nano" in model_name: self.warning(...)

                    web_search_tool: Dict[str, Any] = {"type": "web_search_preview"} # Current name
                    context_size = self.binding_config.config.get("web_search_context_size", "medium")
                    if context_size != "medium": web_search_tool["search_context_size"] = context_size

                    # Add user location if provided
                    user_location: Dict[str, str] = {}
                    loc_country = self.binding_config.config.get("web_search_user_location_country", "")
                    loc_city = self.binding_config.config.get("web_search_user_location_city", "")
                    loc_region = self.binding_config.config.get("web_search_user_location_region", "")
                    loc_tz = self.binding_config.config.get("web_search_user_location_timezone", "")
                    if loc_country: user_location["country"] = loc_country
                    if loc_city: user_location["city"] = loc_city
                    if loc_region: user_location["region"] = loc_region
                    if loc_tz: user_location["timezone"] = loc_tz
                    if user_location:
                        web_search_tool["user_location"] = {"type": "approximate", **user_location}

                    tools.append(web_search_tool)
                    if verbose: ASCIIColors.verbose(f"Configured web search tool: {web_search_tool}")

                    # Handle forcing the tool
                    if self.binding_config.config.get("web_search_force", False):
                        tool_choice = {"type": "web_search_preview"}
                        if verbose: ASCIIColors.verbose("Forcing web search tool.")


                # Configure File Search Tool
                if enable_file_search:
                    vector_store_id = self.binding_config.config.get("file_search_vector_store_id", "")
                    if not vector_store_id:
                        self.error("File Search enabled, but Vector Store ID is missing in configuration. Cannot use file search tool.")
                        if callback: callback("Error: File Search enabled, but Vector Store ID is missing.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
                        # Don't add the tool if ID is missing
                    else:
                        # Responses API uses 'file_search' tool type directly with vector_store_ids
                        file_search_tool: Dict[str, Any] = {"type": "file_search"}
                        file_search_tool["vector_store_ids"] = [vector_store_id]

                        max_results = self.binding_config.config.get("file_search_max_num_results", 20)
                        # Check if parameter exists or if it's controlled elsewhere (e.g., Vector Store settings)
                        # Assuming max_num_results is a valid param here, adjust if needed.
                        if max_results != 20: file_search_tool["max_num_results"] = max_results

                        tools.append(file_search_tool)
                        if verbose: ASCIIColors.verbose(f"Configured file search tool: {file_search_tool}")

                        # Handle forcing (only if web search wasn't already forced)
                        if self.binding_config.config.get("file_search_force", False) and not isinstance(tool_choice, dict):
                             tool_choice = {"type": "file_search"}
                             if verbose: ASCIIColors.verbose("Forcing file search tool.")

                if not tools:
                    self.error("Tools enabled in config, but no valid tools could be configured (e.g., missing Vector Store ID for file search). Aborting generation.")
                    return ""

                # If tool_choice is still 'auto' after individual forcing checks, set it to None for the API call (means auto)
                if tool_choice == "auto":
                    tool_choice = None
                    if verbose: ASCIIColors.verbose("Tool choice set to 'auto'.")

                # --- Prepare Responses API Call Parameters ---
                response_params: Dict[str, Any] = {
                    "model": model_name,
                    "input": prompt,
                    "tools": tools,
                    "max_completion_tokens": effective_max_n_predict, # Max tokens for the *model's response part*
                }
                if tool_choice: response_params["tool_choice"] = tool_choice
                # Add other params if they are not None/Default
                if final_temperature is not None: response_params["temperature"] = final_temperature
                if seed is not None and seed != -1: response_params["seed"] = seed

                # --- Make the API Call (Non-Streaming) ---
                if verbose: ASCIIColors.verbose(f"Calling Responses API. Params: {response_params}")
                api_response = self.client.responses.create(**response_params)
                if verbose: ASCIIColors.verbose(f"Responses API raw response: {api_response}") # Be careful logging full response

                # --- Process the Structured Response ---
                # The Responses API returns a list of message-like objects.
                # We need to parse it to find the assistant's message and annotations.
                web_citations = []
                file_citations = []

                if isinstance(api_response, list):
                     for item in api_response:
                          # Find the assistant's final message
                          if item.get("type") == "message" and item.get("role") == "assistant":
                               if item.get("content"):
                                    for content_item in item["content"]:
                                         # Extract the main text output
                                         if content_item.get("type") == "output_text":
                                              output_text = content_item.get("text", "")
                                              # Extract annotations/citations from this text block
                                              if content_item.get("annotations"):
                                                   for anno in content_item["annotations"]:
                                                        if anno.get("type") == "url_citation":
                                                             web_citations.append({
                                                                  "url": anno.get("url"),
                                                                  "title": anno.get("title", "N/A"),
                                                                  "text_content": anno.get("text_content", "") # Include cited text if available
                                                             })
                                                        elif anno.get("type") == "file_citation":
                                                             # Structure might vary, adapt based on actual API response
                                                             file_citations.append({
                                                                  "file_id": anno.get("file_id"),
                                                                  "quote": anno.get("quote", "N/A") # Original text snippet
                                                                  # Add other relevant fields if provided by API
                                                             })
                                              break # Found text content, process citations and exit inner loop
                               # Add usage data if available at this level
                               if item.get("usage"):
                                    metadata["usage"] = item["usage"]
                                    if verbose: ASCIIColors.verbose(f"Usage stats found in response: {item['usage']}")
                                    # Use API reported tokens if available
                                    if self.binding_config.config.turn_on_cost_estimation and item["usage"]:
                                         # Correct input tokens based on API report
                                         self.binding_config.config["total_input_tokens"] -= prompt_tokens # Subtract initial estimate
                                         self.binding_config.config["total_input_tokens"] += item["usage"].get("prompt_tokens", prompt_tokens) # Add API value or re-add estimate
                                         # Store output tokens for later cost calculation
                                         total_output_tokens = item["usage"].get("completion_tokens", 0)

                               break # Found the main assistant message, stop outer loop
                else:
                     self.warning(f"Unexpected response format from Responses API: {type(api_response)}")

                # Format citations for inclusion in the final output
                citations_parts = []
                if web_citations:
                    citations_parts.append("\n\n--- Web Citations ---")
                    for i, cit in enumerate(web_citations):
                        title = cit['title'] if cit['title'] != 'N/A' else cit['url']
                        cited_text_preview = f" \"{cit['text_content'][:100]}...\"" if cit.get('text_content') else ""
                        citations_parts.append(f"[{i+1}] {title} ({cit['url']}){cited_text_preview}")
                if file_citations:
                    citations_parts.append("\n\n--- File Citations ---")
                    for i, cit in enumerate(file_citations):
                         # Adjust formatting based on actual file citation structure
                         citations_parts.append(f"[F{i+1}] File ID: {cit.get('file_id', 'N/A')}, Quote: '{cit.get('quote', '...')}'")

                citation_text = "\n".join(citations_parts)
                output = output_text + citation_text # Combine main text and formatted citations

                # Send the full response via callback
                if callback:
                    # Combine all info into the metadata dict
                    metadata["citations"] = {"web": web_citations, "file": file_citations}
                    # Send the full processed output
                    callback(output, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_FULL_ANSWER, metadata)

            # ==================================================================
            # === BRANCH 2: Tools Disabled -> Use Chat/Legacy Completions  ===
            # ==================================================================
            else:
                # --- API Endpoint Selection (Legacy vs Chat) ---
                use_chat_completion = True
                legacy_indicators = ["instruct", "davinci", "curie", "babbage", "ada"]
                # Determine if it's a legacy model needing the completions API
                is_legacy = any(ind in model_name for ind in legacy_indicators) and "turbo" not in model_name and "gpt-3.5" not in model_name

                if is_legacy:
                    use_chat_completion = False
                    ASCIIColors.info(f"Using deprecated Completions API for legacy/instruct model '{model_name}'.")
                else:
                    ASCIIColors.info(f"Using Chat Completion API for model '{model_name}'. Streaming enabled.")

                # --- Prepare Streaming API Call Parameters ---
                stream_params: Dict[str, Any] = {
                    "model": model_name,
                    "max_completion_tokens": effective_max_n_predict,
                    "stream": True
                }
                # Add optional parameters only if they differ from defaults or are not None
                if final_temperature != 1.0: stream_params["temperature"] = final_temperature
                if seed is not None and seed != -1: stream_params["seed"] = seed

                # --- Execute Streaming Call ---
                stream_finished = False
                if use_chat_completion:
                    # Chat Completions API
                    messages = [{"role": "user", "content": prompt}]
                    stream_params["messages"] = messages
                    if verbose: ASCIIColors.verbose(f"Calling Chat Completions API. Params: {stream_params}")
                    completion_stream = self.client.chat.completions.create(**stream_params)

                    for chunk in completion_stream:
                        chunk_text = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                        finish_reason = chunk.choices[0].finish_reason
                        if chunk_text:
                            output += chunk_text
                            if callback and not callback(chunk_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                                self.info("Generation stopped by callback."); stream_finished=True; break
                        if finish_reason:
                             if verbose: ASCIIColors.verbose(f"Chat generation finished. Reason: {finish_reason}")
                             metadata["finish_reason"] = finish_reason
                             # Check for usage stats in the final chunk (sometimes present)
                             if chunk.usage:
                                  metadata["usage"] = chunk.usage
                                  if verbose: ASCIIColors.verbose(f"Usage stats received: {chunk.usage}")
                                  if self.binding_config.config.turn_on_cost_estimation:
                                       # Correct input tokens
                                       self.binding_config.config["total_input_tokens"] -= prompt_tokens
                                       self.binding_config.config["total_input_tokens"] += chunk.usage.prompt_tokens
                                       # Store output tokens
                                       total_output_tokens = chunk.usage.completion_tokens
                             stream_finished=True; break
                else:
                    # Legacy Completions API
                    stream_params["prompt"] = prompt
                    if verbose: ASCIIColors.verbose(f"Calling Legacy Completions API. Params: {stream_params}")
                    completion_stream = self.client.completions.create(**stream_params)

                    for chunk in completion_stream:
                        chunk_text = chunk.choices[0].text if chunk.choices else None
                        finish_reason = chunk.choices[0].finish_reason
                        if chunk_text:
                            output += chunk_text
                            if callback and not callback(chunk_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                                 self.info("Generation stopped by callback."); stream_finished=True; break
                        if finish_reason:
                             if verbose: ASCIIColors.verbose(f"Completion generation finished. Reason: {finish_reason}")
                             metadata["finish_reason"] = finish_reason
                             # Check for usage (less common in legacy stream, might be None)
                             if chunk.usage:
                                 metadata["usage"] = chunk.usage
                                 if verbose: ASCIIColors.verbose(f"Usage stats received: {chunk.usage}")
                                 if self.binding_config.config.turn_on_cost_estimation:
                                     self.binding_config.config["total_input_tokens"] -= prompt_tokens
                                     self.binding_config.config["total_input_tokens"] += chunk.usage.prompt_tokens
                                     total_output_tokens = chunk.usage.completion_tokens
                             stream_finished=True; break

                if not stream_finished: # Log only if streaming was expected and didn't explicitly finish
                      self.info("Stream ended without explicit finish reason.")
                # If streaming finished, send a final 'status' update via callback if desired
                if stream_finished and callback:
                    final_status_message = f"Generation finished: {metadata.get('finish_reason', 'Unknown')}"
                    ASCIIColors.info(final_status_message)


        # --- Exception Handling ---
        except openai.AuthenticationError as e: self.error(f"Authentication Error: {e}"); trace_exception(e); callback(f"Authentication Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION) if callback else None; return ""
        except openai.RateLimitError as e: self.error(f"Rate limit exceeded: {e}"); trace_exception(e); callback(f"Rate limit exceeded: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION) if callback else None; return ""
        except openai.BadRequestError as e: self.error(f"API Bad Request Error: {e}. Check model/params/tool compatibility/input format."); trace_exception(e); callback(f"API Bad Request: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION) if callback else None; return ""
        except openai.APIError as e: self.error(f'OpenAI API Error: {e}'); trace_exception(e); callback(f"API Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION) if callback else None; return ""
        except Exception as e: self.error(f'Error during generation: {e}'); trace_exception(e); callback(f"Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION) if callback else None; return ""
        finally:
            generation_time = perf_counter() - start_time
            ASCIIColors.info(f"Generation process finished in {generation_time:.2f} seconds.")


        # --- Cost Estimation (Output - Text Tokens Only) ---
        if self.binding_config.config.turn_on_cost_estimation:
            # Use the final text output (model response part, excluding citations)
            final_output_text = output_text if use_responses_api else output
            # If API didn't provide completion tokens (or tools weren't used), tokenize the final output
            if total_output_tokens == 0 and final_output_text:
                total_output_tokens = len(self.tokenize(final_output_text))
            elif total_output_tokens > 0:
                 if verbose: ASCIIColors.verbose(f"Using {total_output_tokens} output tokens reported by API/metadata.")
            else: # No output text and no API report
                 total_output_tokens = 0


            self.binding_config.config["total_output_tokens"] += total_output_tokens
            output_cost_rate = self.output_costs_by_model.get(model_name, self.output_costs_by_model.get("default", 0))
            output_cost = total_output_tokens * output_cost_rate
            self.binding_config.config["total_output_cost"] += output_cost
            # Recalculate total cost using potentially corrected input tokens and new output cost
            self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]

            cost_info = f'Accumulated cost (text tokens only, excludes tools): ${self.binding_config.config["total_cost"]:.6f}' if use_responses_api else f'Accumulated cost: ${self.binding_config.config["total_cost"]:.6f}'
            self.info(cost_info)
            self.binding_config.save() # Save updated costs

        return output # Return the full text including citations if tools were used


    def list_models(self) -> List[str]:
        """
        Lists the available models fetched from the API.
        Refreshes the list if it's empty and the client is available.

        Returns:
            A list of model ID strings. Returns an empty list on failure.
        """
        if not self.client:
             self.error("Cannot list models, OpenAI client not initialized (API Key likely missing or invalid).")
             return []

        if not self.available_models:
             ASCIIColors.warning("Model list was empty, attempting to fetch now...")
             self._update_available_models()
             if not self.available_models: # Check again after fetch attempt
                  self.error("Failed to fetch models. Check API key and network connection.")
                  return []

        return self.available_models


    def get_available_models(self, app: Optional[LoLLMsCom] = None) -> List[dict]:
        """
        Gets the available models list formatted for the LoLLMs UI, including estimated metadata.

        Args:
            app: The LoLLMsCom application instance (optional, fallback to self.lollmsCom).

        Returns:
            A list of dictionaries formatted for the LoLLMs models list.
        """
        lollms_com = self.lollmsCom or app # Use self.lollmsCom if available

        if not self.client:
             self.error("Cannot get available models for UI, OpenAI client not initialized.")
             if lollms_com: lollms_com.InfoMessage("OpenAI Error: Cannot fetch models. API Key missing or invalid.")
             return []

        if not self.available_models:
             self._update_available_models()
             if not self.available_models:
                  self.error("No models available to display after fetch attempt. Check API key/network.")
                  if lollms_com: lollms_com.InfoMessage("OpenAI Error: Failed to retrieve models from API.")
                  return []

        models_info: List[Dict[str, Any]] = []
        # Define icon paths relative to the binding's web folder
        # Ensure the binding name matches the folder structure if needed
        binding_folder = binding_folder_name if binding_folder_name else binding_name
        ui_path_prefix = f"/bindings/{binding_folder.lower()}/"
        default_icon = ui_path_prefix + "logo.png"

        # --- Model Metadata Enhancement ---
        # Add known licenses, creators etc. (Can be expanded)
        base_metadata = {
            "author": "OpenAI",
            "license": "Commercial API",
            "creation_date": None, # Can't easily get this
            "category": "API Model",
            "datasets": "Proprietary OpenAI Datasets",
            "commercial_use": True,
            "icon": default_icon,
            "model_creator": "OpenAI",
            "model_creator_link": "https://openai.com/",
            "provider": None,
            "type": "api",
        }

        for model_name in self.available_models:
            ctx_size, max_output = self.get_model_limits(model_name)

            # Determine Category (Vision/Multimodal vs Text)
            category = "text"
            if "vision" in model_name or "4o" in model_name:
                category = "multimodal"

            # Determine Rank and Size Proxy
            rank = 1.0 # Base rank
            if "o3" in model_name or "o4" in model_name or "o1-pro" in model_name: rank = 4.0 # Top reasoning
            elif "gpt-4.1" in model_name: rank = 3.5 # New powerful GPT
            elif "gpt-4o" == model_name: rank = 3.3 # Flagship omni
            elif "gpt-4" in model_name: rank = 3.0 # General GPT-4
            elif "gpt-4o-mini" == model_name: rank = 2.5 # Good balance
            elif "o1-mini" in model_name or "o3-mini" in model_name or "o4-mini" in model_name: rank = 2.3 # Small reasoning
            elif "gpt-3.5" in model_name: rank = 2.0 # Older but capable
            else: rank = 1.0 # Legacy/unknown

            size_proxy = -1
            if ctx_size is not None: size_proxy = ctx_size # Use context size as proxy

            # --- Build Lollms Model Dictionary ---
            model_entry = {
                **base_metadata,
                "name": model_name, # The actual ID used in configuration
                "display_name": model_name.replace("-preview", " Preview").replace("-", " ").title(), # Nicer name
                "category": category,
                "rank": rank,
                "description": f"OpenAI API model. Context: {ctx_size or 'Unknown'}. Max Output: {max_output or 'Unknown'}.",
                "ctx_size": ctx_size if ctx_size is not None else -1, # Use -1 for unknown
                # 'variants' structure for API models usually just lists the model itself
                "variants": [{"name": model_name, "size": size_proxy}]
            }
            models_info.append(model_entry)

        # Sort models: by rank (desc), then by name (asc)
        models_info.sort(key=lambda x: (-x['rank'], x['name']))

        ASCIIColors.success(f"Formatted {len(models_info)} OpenAI models for Lollms UI.")
        return models_info


# --- Main execution block for testing ---
if __name__ == "__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMsConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    from lollms.types import MSG_OPERATION_TYPE
    from time import perf_counter

    print("Initializing LoLLMs environment for testing...")
    lollms_paths = LollmsPaths.find_paths(force_local=True, tool_prefix="test_openai_")
    config = LOLLMsConfig.autoload(lollms_paths)
    # Minimal app for communication channel
    lollms_app = LollmsApplication("TestApp", config, lollms_paths, load_bindings=False, load_personalities=False, load_models=False)

    print("Creating OpenAIGPT binding instance...")
    # Use default install option for testing
    oai = OpenAIGPT(config, lollms_paths, installation_option=InstallOption.INSTALL_IF_NECESSARY, lollmsCom=lollms_app.com)

    # --- API Key Setup ---
    if not os.getenv('OPENAI_API_KEY') and not oai.binding_config.config.get("openai_key"):
         try:
             key_input = input("Enter OpenAI API Key for testing: ").strip()
             if key_input:
                 oai.binding_config.config["openai_key"] = key_input
                 # Don't save to permanent config during testing
                 # oai.binding_config.save()
                 print("API Key set for this session.")
             else:
                  print("No API key provided. Tests requiring API access will fail.")
         except EOFError:
             print("No API key input detected.")

    print("\nUpdating settings (initializes client and fetches models)...")
    oai.settings_updated() # This initializes the client and fetches models

    available_models = oai.list_models()
    if available_models:
        print(f"\nAvailable OpenAI Models ({len(available_models)} found):")
        # Print subset if list is long
        limit = 15
        if len(available_models) > limit:
            for m in available_models[:limit//2]: print(f"- {m}")
            print("  ...")
            for m in available_models[-limit//2:]: print(f"- {m}")
        else:
            for m in available_models: print(f"- {m}")

        # --- Test Setup ---
        # Select a common, capable model for testing
        test_model = "gpt-4o-mini" # Good balance of cost and capability
        if test_model not in available_models:
             print(f"\nWarning: Preferred test model '{test_model}' not found. Trying 'gpt-3.5-turbo'.")
             test_model = "gpt-3.5-turbo" # Common fallback
             if test_model not in available_models:
                  print(f"\nWarning: Model 'gpt-3.5-turbo' not found. Using first available model: {available_models[0] if available_models else 'None'}")
                  if available_models: test_model = available_models[0]
                  else: test_model = None

        if test_model:
             print(f"\nSelecting model for testing: {test_model}")
             oai.config.model_name = test_model
             oai.build_model() # Finalize setup for the selected model
             print(f"Effective limits for {test_model}: Context={oai.config.ctx_size}, Max Output={oai.config.max_n_predict}")

             # Define a callback function for testing
             def print_callback(chunk: str, msg_type: int, metadata: dict) -> bool:
                 """Prints output received from the generate method."""
                 type_str = MSG_OPERATION_TYPE(msg_type).name if msg_type in MSG_OPERATION_TYPE.__members__.values() else f"UNKNOWN({msg_type})"

                 if msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK:
                     print(chunk, end="", flush=True)
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_FULL_ANSWER:
                     print(f"\n--- [{type_str}] Full Response ---")
                     print(chunk)
                     if 'citations' in metadata: print(f"Citations: {metadata['citations']}")
                     print("--- End Full Response ---")
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION:
                     print(f"\n## [{type_str}] EXCEPTION: {chunk} ##")
                     return False # Stop on error
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_INFO or msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING:
                      print(f"\n## [{type_str}] INFO/WARN: {chunk} ##")
                 #else:
                 #    print(f"\n--- [{type_str}] ---")
                 #    print(chunk)
                 #    print(f"Metadata: {metadata}")
                 #    print(f"--- End [{type_str}] ---")

                 return True # Continue generation/processing

             # --- Test Case 1: Standard Chat (Streaming) ---
             print("\n" + "="*20 + " Test 1: Standard Chat (Streaming) " + "="*20)
             oai.binding_config.config["enable_web_search"] = False
             oai.binding_config.config["enable_file_search"] = False
             prompt1 = "Explain the concept of a Large Language Model (LLM) to a 5-year-old in one short paragraph."
             print(f"Prompt: {prompt1}")
             print("Response Stream:")
             try:
                 response1 = oai.generate(prompt1, n_predict=150, callback=print_callback, verbose=True) # Use verbose
                 print("\n-- Test 1 Complete --")
             except Exception as e: print(f"\nTest 1 failed: {e}"); trace_exception(e)

             # --- Test Case 2: Web Search Enabled (No Streaming) ---
             print("\n" + "="*20 + " Test 2: Web Search Enabled " + "="*20)
             oai.binding_config.config["enable_web_search"] = True
             oai.binding_config.config["enable_file_search"] = False
             # Example: Force web search
             # oai.binding_config.config["web_search_force"] = True
             prompt2 = "What notable AI developments happened in June 2024?" # Good prompt for search
             print(f"Prompt: {prompt2}")
             print("Response (Expecting Full Answer with potential citations):")
             try:
                 # Note: Callback will only receive FULL_ANSWER or EXCEPTION/INFO here
                 response2 = oai.generate(prompt2, n_predict=300, callback=print_callback, verbose=True)
                 print("\n-- Test 2 Complete --")
             except Exception as e: print(f"\nTest 2 failed: {e}"); trace_exception(e)
             finally: oai.binding_config.config["enable_web_search"] = False # Disable for next test

             # --- Test Case 3: File Search Enabled (No Streaming) ---
             # NOTE: This requires a VALID Vector Store ID!
             print("\n" + "="*20 + " Test 3: File Search Enabled (Requires Vector Store ID) " + "="*20)
             vector_store_id_test = oai.binding_config.config.get("file_search_vector_store_id", "")
             if not vector_store_id_test:
                 try:
                     vector_store_id_test = input("Enter Vector Store ID for File Search test (or press Enter to skip): ").strip()
                     if vector_store_id_test:
                         oai.binding_config.config["file_search_vector_store_id"] = vector_store_id_test
                     else:
                         print("-- Test 3 Skipped (No Vector Store ID provided) --")
                 except EOFError:
                      print("-- Test 3 Skipped (No input detected) --")

             if vector_store_id_test:
                 oai.binding_config.config["enable_file_search"] = True
                 # Example: Force file search
                 # oai.binding_config.config["file_search_force"] = True
                 prompt3 = "Summarize the document about project 'X'." # Adjust prompt based on your VS content
                 print(f"Prompt: {prompt3}")
                 print("Response (Expecting Full Answer with potential file info/citations):")
                 try:
                     response3 = oai.generate(prompt3, n_predict=300, callback=print_callback, verbose=True)
                     print("\n-- Test 3 Complete --")
                 except Exception as e: print(f"\nTest 3 failed: {e}"); trace_exception(e)
                 finally: oai.binding_config.config["enable_file_search"] = False # Disable after test
             else:
                 pass # Already printed skip message

             # --- Test Case 4: Vision Model (if applicable) ---
             print("\n" + "="*20 + " Test 4: Vision Input (Requires Vision Model & Image) " + "="*20)
             is_vision_model = "vision" in test_model or "4o" in test_model
             if is_vision_model:
                 # Create a dummy image file for testing
                 dummy_image_path = lollms_paths.personal_outputs_path/"test_image.png"
                 try:
                     from PIL import Image, ImageDraw
                     img = Image.new('RGB', (60, 30), color = 'red')
                     d = ImageDraw.Draw(img)
                     d.text((10,10), "Test", fill='white')
                     img.save(dummy_image_path)
                     print(f"Created dummy image: {dummy_image_path}")

                     prompt4 = "Describe this image."
                     images4 = [str(dummy_image_path)]
                     print(f"Prompt: {prompt4}")
                     print(f"Image: {images4[0]}")
                     print("Response Stream:")
                     try:
                         response4 = oai.generate_with_images(prompt4, images4, n_predict=100, callback=print_callback, verbose=True)
                         print("\n-- Test 4 Complete --")
                     except Exception as e: print(f"\nTest 4 failed: {e}"); trace_exception(e)
                     finally:
                         # Clean up dummy image
                         if dummy_image_path.exists(): dummy_image_path.unlink()

                 except ImportError:
                     print("PIL/Pillow not installed. Skipping image creation and vision test.")
                 except Exception as e:
                      print(f"Error during image creation/test: {e}")

             else:
                  print(f"-- Test 4 Skipped (Model '{test_model}' is not detected as a vision model) --")


             # --- Test Case 5: Cost Estimation ---
             print("\n" + "="*20 + " Test 5: Cost Estimation Check " + "="*20)
             oai.binding_config.config["turn_on_cost_estimation"] = True
             prompt5 = "Tell me a short story."
             print(f"Prompt: {prompt5}")
             print("Generating...")
             try:
                 # Use a non-streaming callback for simplicity here
                 response5 = oai.generate(prompt5, n_predict=50, callback=None)
                 print("Response Received (Cost info logged above/during generate).")
                 print(f"Final Accumulated Cost (text tokens only): ${oai.binding_config.config['total_cost']:.6f}")
                 print(f"Total Input Tokens: {oai.binding_config.config['total_input_tokens']}")
                 print(f"Total Output Tokens: {oai.binding_config.config['total_output_tokens']}")
                 print("\n-- Test 5 Complete --")
             except Exception as e: print(f"\nTest 5 failed: {e}"); trace_exception(e)
             finally: oai.binding_config.config["turn_on_cost_estimation"] = False # Disable after test


        else: print("\nSkipping generation tests as no suitable model could be selected/available.")
    else:
        print("\nCould not retrieve model list. Check API key and network connection.")
        print("Skipping tests.")

    print("\nScript finished.")
