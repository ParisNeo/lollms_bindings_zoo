######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying
# engine author : Inflection AI
# license       : Apache 2.0
# Description   :
# This binding provides an interface to the Inflection AI API for chat completions.
# It uses their proprietary models like inflection_3_pi and inflection_3_productivity.
# Update date   : 21/07/2024
######
from pathlib import Path
from typing import Callable, Any, Optional, List, Dict, Union, Tuple
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors, trace_exception
from lollms.types import MSG_OPERATION_TYPE

from lollms.com import LoLLMsCom
import subprocess
import yaml
import sys
import os
import json
from time import perf_counter

# Try to install necessary packages using pipmaster
try:
    import pipmaster as pm
    if not pm.is_installed("requests"):
        pm.install("requests")
except ImportError:
    print("Warning: pipmaster not found. Please install required packages manually: pip install requests")
    pass

# Import required libraries
try:
    import requests
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure 'requests' is installed (`pip install requests`)")
    raise e

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023-2024, ParisNeo"
__license__ = "Apache 2.0"

binding_name = "InflectionAI"
binding_folder_name = "inflection_ai"
binding_path = Path(__file__).parent

# ================= Known Models =================
# Source: Inflection API documentation provided
INFLECTION_MODELS: Dict[str, Dict[str, Any]] = {
    "inflection_3_pi": {"context_window": 8000, "max_output": 1024, "rank": 3.0, "category": "chat"}, # Context size is a guess
    "inflection_3_productivity": {"context_window": 8000, "max_output": 1024, "rank": 3.2, "category": "productivity_chat"}, # Context size is a guess
}

# --- Cost Estimation (Placeholder - Pricing not provided) ---
# Set turn_on_cost_estimation to False by default.
INPUT_COSTS_PLACEHOLDER: Dict[str, float] = {"default": 0.0 / 1_000_000}
OUTPUT_COSTS_PLACEHOLDER: Dict[str, float] = {"default": 0.0 / 1_000_000}

DEFAULT_CONFIG = {
    "inflection_key": "",
    "max_tokens": 256,
    "temperature": 1.0,
    "top_p": 0.95,
    "stop_tokens": "", # Comma-separated list
    "web_search": True,
    # Metadata fields
    "user_firstname": "",
    "user_timezone": "",
    "user_country": "",
    "user_region": "",
    "user_city": "",
    # Cost estimation (disabled)
    "turn_on_cost_estimation": False,
    "total_input_tokens": 0.0,
    "total_output_tokens": 0.0,
    "total_input_cost": 0.0,
    "total_output_cost": 0.0,
    "total_cost": 0.0,
    # API URL (fixed for now)
    "override_api_url": "https://layercake.pubwestus3.inf7ks8.com/external/api",
}


class InflectionAI(LLMBinding):
    """
    Binding class for interacting with the Inflection AI API.
    """
    def __init__(self,
                config: LOLLMSConfig,
                lollms_paths: LollmsPaths = None,
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY,
                lollmsCom:Optional[LoLLMsCom]=None) -> None:
        """
        Initialize the Binding.
        """
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        self.lollmsCom = lollmsCom

        # Configuration definition using ConfigTemplate
        binding_config_template = ConfigTemplate([
            # --- Credentials ---
            {"name":"inflection_key","type":"str","value":DEFAULT_CONFIG["inflection_key"],"help":"Your Inflection AI API key (token)."},

            # --- Generation Parameters ---
            {"name":"max_tokens","type":"int","value":DEFAULT_CONFIG["max_tokens"], "min":1, "max":1024, "help":"Maximum number of tokens to generate per response."},
            {"name":"temperature","type":"float","value":DEFAULT_CONFIG["temperature"],"min":0.0, "max":1.0, "help":"Controls randomness (0.0-1.0). Lower is more deterministic."},
            {"name":"top_p","type":"float","value":DEFAULT_CONFIG["top_p"],"min":0.0, "max":1.0, "help":"Nucleus sampling threshold (0.0-1.0)."},
            {"name":"stop_tokens","type":"str","value":DEFAULT_CONFIG["stop_tokens"],"help":"Comma-separated list of tokens to stop generation at."},
            {"name":"web_search","type":"bool","value":DEFAULT_CONFIG["web_search"],"help":"Allow the AI to use web search information."},

            # --- Optional User Metadata ---
            {"name":"user_firstname","type":"str","value":DEFAULT_CONFIG["user_firstname"],"help":"User's first name (optional metadata)."},
            {"name":"user_timezone","type":"str","value":DEFAULT_CONFIG["user_timezone"],"help":"User's timezone, e.g., America/Los_Angeles (optional metadata)."},
            {"name":"user_country","type":"str","value":DEFAULT_CONFIG["user_country"],"help":"User's country (optional metadata)."},
            {"name":"user_region","type":"str","value":DEFAULT_CONFIG["user_region"],"help":"User's region/state (optional metadata)."},
            {"name":"user_city","type":"str","value":DEFAULT_CONFIG["user_city"],"help":"User's city (optional metadata)."},

            # --- Cost Estimation (Placeholder - Disabled) ---
            {"name":"turn_on_cost_estimation","type":"bool", "value":DEFAULT_CONFIG["turn_on_cost_estimation"],"help":"Estimate costs (PLACEHOLDER - Not accurate for Inflection AI)."},
            {"name":"total_input_tokens","type":"float", "value":DEFAULT_CONFIG["total_input_tokens"],"help":"Accumulated input tokens (estimate)."},
            {"name":"total_output_tokens","type":"float", "value":DEFAULT_CONFIG["total_output_tokens"],"help":"Accumulated output tokens (estimate)."},
            {"name":"total_input_cost","type":"float", "value":DEFAULT_CONFIG["total_input_cost"],"help":"Accumulated input cost ($) (estimate)."},
            {"name":"total_output_cost","type":"float", "value":DEFAULT_CONFIG["total_output_cost"],"help":"Accumulated output cost ($) (estimate)."},
            {"name":"total_cost","type":"float", "value":DEFAULT_CONFIG["total_cost"],"help":"Total accumulated cost ($) (estimate)."},

            # --- Advanced ---
             {"name":"override_api_url","type":"str","value":DEFAULT_CONFIG["override_api_url"],"help":"Advanced: Override the default Inflection API base URL (use with caution)."},

        ])
        # Default values for the configuration
        binding_config_defaults = BaseConfig(config=DEFAULT_CONFIG)

        binding_config = TypedConfig(
            binding_config_template,
            binding_config_defaults
        )
        super().__init__(
                            binding_path,
                            lollms_paths,
                            config,
                            binding_config,
                            installation_option,
                            SAFE_STORE_SUPPORTED_FILE_EXTENSIONS=[], # Inflection API doesn't support file uploads
                            lollmsCom=lollmsCom
                        )
        # --- Configuration Sync ---
        # Inflection uses 'max_tokens' directly, map lollms' 'max_n_predict' to it
        # Let's use a fixed reasonable context size for LoLLMs internal display, as API doesn't expose it
        self.config.ctx_size = 8000 # Reasonable guess based on similar models
        self.config.max_n_predict = self.binding_config.config["max_tokens"] # Sync prediction limit

        self.inflection_key: Optional[str] = None
        self.session = requests.Session() # Use a session for potential connection reuse
        self.available_models: List[str] = list(INFLECTION_MODELS.keys()) # Hardcoded list

    def _get_api_key(self) -> Optional[str]:
        """Retrieves the API key from config or environment."""
        key = self.binding_config.config.get("inflection_key", "")
        if key and key.strip():
            return key
        key = os.getenv('INFLECTION_API_KEY')
        if key:
            return key
        return None

    def settings_updated(self) -> None:
        """Callback triggered when binding settings are updated in the UI."""
        ASCIIColors.info("Inflection AI settings updated.")
        key_updated = self._update_inflection_key()

        # Sync prediction limit from binding config to main config
        self.config.max_n_predict = self.binding_config.config["max_tokens"]

        if not key_updated:
            self.error("Inflection AI API Key is missing or invalid. Please configure it.")
            if self.lollmsCom:
                 self.lollmsCom.InfoMessage("Inflection AI Error: API Key is missing or invalid.")
        elif self.config.model_name:
             self.info(f"Settings updated. Current model: {self.config.model_name}. Max Output: {self.config.max_n_predict}")
        else:
             self.info("Settings updated. No model currently selected.")

    def build_model(self, model_name: Optional[str] = None) -> LLMBinding:
        """
        Sets up the binding for the selected model.
        """
        super().build_model(model_name) # Sets self.config.model_name

        if not self.inflection_key:
             if not self._update_inflection_key():
                 self.error("Model build failed: Inflection AI API key is missing or invalid.")
                 if self.lollmsCom:
                     self.lollmsCom.InfoMessage("Inflection AI Error: Cannot build model. API Key is missing or invalid.")
                 return self

        current_model_name = self.config.model_name or ""
        if not current_model_name:
            self.warning("No model name selected.")
            return self

        if current_model_name not in self.available_models:
             self.warning(f"Model '{current_model_name}' is not in the known list of Inflection AI models. It might not work.")

        # Sync prediction limit
        self.config.max_n_predict = self.binding_config.config["max_tokens"]

        # Determine Binding Type (Always Text)
        self.binding_type = BindingType.TEXT_ONLY
        self.SAFE_STORE_SUPPORTED_FILE_EXTENSIONS=[]

        ASCIIColors.success(f"Inflection AI binding built successfully. Model: {current_model_name}. API URL: {self.binding_config.config.get('override_api_url', DEFAULT_CONFIG['override_api_url'])}")
        return self

    def install(self) -> None:
        """Installs necessary Python packages."""
        super().install()
        self.ShowBlockingMessage("Installing Inflection AI requirements (requests)...")
        try:
            import pipmaster as pm
            requirements = ["requests"]
            for req in requirements:
                if not pm.is_installed(req):
                    self.info(f"Installing {req}...")
                    pm.install(req)
                else:
                    self.info(f"{req} already installed.")
            self.HideBlockingMessage()
            ASCIIColors.success("Inflection AI requirements installed successfully.")
            ASCIIColors.info("----------------------")
            ASCIIColors.info("Attention:")
            ASCIIColors.info("----------------------")
            ASCIIColors.info("The Inflection AI binding uses their private API.")
            ASCIIColors.info("1. Obtain an API key (token) from Inflection AI.")
            ASCIIColors.info("2. Provide the key in the binding settings or set the INFLECTION_API_KEY environment variable.")
            ASCIIColors.warning("3. Adhere to Inflection AI's API usage terms and restrictions.")
        except ImportError:
            self.HideBlockingMessage()
            self.warning("pipmaster not found. Please install requirements manually: pip install requests")
        except Exception as e:
            self.error(f"Installation failed: {e}")
            trace_exception(e)
            self.warning("Installation failed. Please ensure you have pip installed and internet access.", 20)
            self.HideBlockingMessage()

    def tokenize(self, prompt: str) -> List[int]:
        """Tokenization is not supported by the Inflection AI API."""
        ASCIIColors.warning("Inflection AI binding does not support client-side tokenization.")
        return []

    def detokenize(self, tokens_list: List[int]) -> str:
        """Detokenization is not supported by the Inflection AI API."""
        ASCIIColors.warning("Inflection AI binding does not support client-side detokenization.")
        return ""

    def embed(self, text: Union[str, List[str]]) -> Optional[List[List[float]]]:
        """Embedding is not supported by the Inflection AI API."""
        self.error("Inflection AI binding does not support embedding generation.")
        return None

    def generate_with_images(self, prompt: str, images: List[str], **kwargs) -> str:
        """Image input is not supported by the Inflection AI API."""
        self.error("Inflection AI binding does not support image inputs.")
        return "Error: Image input not supported by Inflection AI binding."

    def generate(self,
                 prompt: str,
                 n_predict: Optional[int] = None,
                 callback: Optional[Callable[[str, int], bool]] = None,
                 verbose: bool = False,
                 **generation_params) -> str:
        """
        Generates text using the Inflection AI Chat Completions API.

        Args:
            prompt: The text prompt (represents the latest user message).
            n_predict: Optional override for the maximum number of tokens to generate.
            callback: An optional callback function for streaming results.
            verbose: If True, prints more detailed information.
            **generation_params: Additional parameters (temperature, top_p, stop_tokens, web_search).

        Returns:
            The generated text response.
        """
        if not self.inflection_key:
            self.error("Inflection AI API key not set.")
            if callback: callback("Error: Inflection AI API key not configured.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return ""

        model_name = self.config.model_name
        if not model_name:
             self.error("No model selected.")
             if callback: callback("Error: No model selected.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""

        # --- Parameter Preparation ---
        effective_max_tokens = self.binding_config.config["max_tokens"]
        if n_predict is not None:
             if 0 < n_predict <= 1024: # API max limit is 1024
                 effective_max_tokens = n_predict
                 if verbose: ASCIIColors.verbose(f"Using user-provided max_tokens: {effective_max_tokens}")
             else:
                  self.warning(f"Requested n_predict ({n_predict}) is invalid or exceeds API limit (1024). Using configured value: {effective_max_tokens}.")
                  # Ensure effective_max_tokens is within valid range if config was > 1024
                  effective_max_tokens = min(effective_max_tokens, 1024)
        if verbose: ASCIIColors.verbose(f"Effective max_tokens for this generation: {effective_max_tokens}")

        # Combine default binding config with runtime params
        api_params: Dict[str, Any] = {
            "temperature": self.binding_config.config.get("temperature", DEFAULT_CONFIG["temperature"]),
            "top_p": self.binding_config.config.get("top_p", DEFAULT_CONFIG["top_p"]),
            "stop_tokens": self.binding_config.config.get("stop_tokens", DEFAULT_CONFIG["stop_tokens"]),
            "web_search": self.binding_config.config.get("web_search", DEFAULT_CONFIG["web_search"]),
        }
        api_params.update(generation_params) # Runtime params override config defaults

        # --- Prepare Context ---
        # TODO: Implement proper history management. For now, treat prompt as single user turn.
        context = [{"type": "Human", "text": prompt}]

        # --- Prepare Metadata ---
        metadata: Dict[str, str] = {}
        if self.binding_config.config.get("user_firstname"): metadata["user_firstname"] = self.binding_config.config["user_firstname"]
        if self.binding_config.config.get("user_timezone"): metadata["user_timezone"] = self.binding_config.config["user_timezone"]
        if self.binding_config.config.get("user_country"): metadata["user_country"] = self.binding_config.config["user_country"]
        if self.binding_config.config.get("user_region"): metadata["user_region"] = self.binding_config.config["user_region"]
        if self.binding_config.config.get("user_city"): metadata["user_city"] = self.binding_config.config["user_city"]

        # --- Prepare Stop Tokens ---
        stop_tokens_list = []
        stop_tokens_str = api_params["stop_tokens"]
        if isinstance(stop_tokens_str, str) and stop_tokens_str.strip():
             stop_tokens_list = [s.strip() for s in stop_tokens_str.split(',') if s.strip()]

        # --- Construct Payload ---
        payload: Dict[str, Any] = {
            "config": model_name,
            "context": context,
            "max_tokens": effective_max_tokens,
            "temperature": float(api_params["temperature"]),
            "top_p": float(api_params["top_p"]),
            "web_search": bool(api_params["web_search"]),
        }
        if stop_tokens_list: payload["stop_tokens"] = stop_tokens_list
        if metadata: payload["metadata"] = metadata

        # --- Determine API Endpoint and Headers ---
        is_streaming = callback is not None
        base_url = self.binding_config.config.get("override_api_url", DEFAULT_CONFIG["override_api_url"]).rstrip('/')
        api_url = f"{base_url}/inference/streaming" if is_streaming else f"{base_url}/inference"

        headers = {
            "Authorization": f"Bearer {self.inflection_key}",
            "Content-Type": "application/json",
            "Accept": "application/json", # API seems to always return JSON or JSON stream
        }

        output = ""
        callback_metadata = {} # For potentially passing finish reason etc.
        total_output_tokens = 0 # Placeholder
        total_input_tokens = 0 # Placeholder

        start_time = perf_counter()
        try:
            if verbose: ASCIIColors.verbose(f"Calling Inflection AI API ({'Streaming' if is_streaming else 'Non-streaming'}). URL: {api_url}. Payload: {json.dumps(payload, indent=2)}")

            response = self.session.post(
                api_url,
                headers=headers,
                json=payload,
                stream=is_streaming,
                timeout=300 # Add a reasonable timeout (e.g., 5 minutes)
            )
            response.raise_for_status() # Raise HTTPError for bad responses

            # --- Handle Streaming Response ---
            if is_streaming:
                stream_finished = False
                for line in response.iter_lines():
                    if line:
                        try:
                            decoded_line = line.decode('utf-8')
                            chunk_data = json.loads(decoded_line)
                            if verbose: ASCIIColors.verbose(f"Stream chunk received: {chunk_data}")

                            chunk_text = chunk_data.get("text")
                            if chunk_text:
                                output += chunk_text
                                if not callback(chunk_text, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                                    self.info("Generation stopped by callback.")
                                    stream_finished = True
                                    break # Stop processing stream
                            # NOTE: Inflection streaming doesn't seem to have a [DONE] or finish_reason marker per chunk in the docs.
                            # Assume stream ends when response closes.

                        except json.JSONDecodeError:
                            self.error(f"Failed to decode JSON stream line: {line}")
                            continue
                        except Exception as chunk_ex:
                             self.error(f"Error processing stream chunk: {chunk_ex}")
                             trace_exception(chunk_ex)
                             continue
                if not stream_finished:
                    self.info("Stream ended.")
                # Signal end of stream
                if callback:
                     callback("", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_STEP_END, {"finish_reason": "stop"}) # Assume normal stop


            # --- Handle Non-Streaming Response ---
            else:
                full_response_json = response.json()
                if verbose: ASCIIColors.verbose(f"Non-streaming response received: {json.dumps(full_response_json, indent=2)}")

                output = full_response_json.get("text", "")
                callback_metadata["created_timestamp"] = full_response_json.get("created")
                # No explicit finish reason or usage stats in non-streaming response docs

                if callback:
                     # Send the whole response as one chunk/message
                     callback(output, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_FULL_ANSWER, callback_metadata)

        # --- Exception Handling ---
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_message = f"Inflection AI API HTTP Error: {status_code}"
            try:
                error_details = e.response.json()
                # Try to extract more specific error if available
                error_message += f" - {error_details}"
            except json.JSONDecodeError:
                error_message += f" - {e.response.text}"
            self.error(error_message)
            trace_exception(e)
            if callback: callback(error_message, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return "" # Return empty on error
        except requests.exceptions.RequestException as e:
            error_message = f"Network error connecting to Inflection AI API: {e}"
            self.error(error_message)
            trace_exception(e)
            if callback: callback(error_message, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return ""
        except Exception as e:
            error_message = f"Error during Inflection AI generation: {e}"
            self.error(error_message)
            trace_exception(e)
            if callback: callback(error_message, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return ""
        finally:
             generation_time = perf_counter() - start_time
             ASCIIColors.info(f"Generation process finished in {generation_time:.2f} seconds.")

        # --- Cost Estimation (Placeholder) ---
        # API doesn't provide token counts, so cost estimation is not feasible.
        if self.binding_config.config.turn_on_cost_estimation:
             self.warning("Cost estimation is enabled but not supported/accurate for Inflection AI.")
             # You could estimate based on character counts if desired, but it's unreliable.

        return output

    def list_models(self) -> List[str]:
        """Lists the available models hardcoded for Inflection AI."""
        return self.available_models

    def get_available_models(self, app: Optional[LoLLMsCom] = None) -> List[dict]:
        """Gets the available models list formatted for the LoLLMs UI."""
        models_info: List[Dict[str, Any]] = []
        ui_path_prefix = f"/bindings/{binding_folder_name}/"
        default_icon = ui_path_prefix + "logo.png" # Assuming logo.png exists

        base_metadata = {
            "author": "Inflection AI",
            "license": "Proprietary API", # Assume commercial, follow terms
            "creation_date": None, # Unknown
            "category": "API Model", # Overridden below
            "datasets": "Proprietary Inflection AI Datasets",
            "commercial_use": False, # Check terms, assume restricted initially
            "icon": default_icon,
            "model_creator": "Inflection AI",
            "model_creator_link": "https://inflection.ai/",
            "provider": None,
            "type": "api",
            "binding_name": binding_name
        }

        for model_id, model_data in INFLECTION_MODELS.items():
            ctx = model_data.get("context_window", self.config.ctx_size) # Use known or default
            max_out = model_data.get("max_output", self.binding_config.config.max_tokens)
            rank = model_data.get("rank", 3.0)
            category = model_data.get("category", "chat")

            # --- Build Lollms Model Dictionary ---
            model_entry = {
                **base_metadata,
                "name": model_id,
                "display_name": model_id.replace("_", " ").title(),
                "category": category,
                "rank": rank,
                "description": f"Inflection AI API model. Context: {ctx}. Max Output: {max_out}.",
                "ctx_size": ctx,
                "max_n_predict": max_out,
                "variants": [{"name": model_id, "size": ctx}]
            }
            models_info.append(model_entry)

        models_info.sort(key=lambda x: (-x['rank'], x['name']))
        ASCIIColors.success(f"Formatted {len(models_info)} Inflection AI models for Lollms UI.")
        return models_info

# --- Main execution block for testing ---
if __name__ == "__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    from lollms.types import MSG_OPERATION_TYPE
    import sys

    print("Initializing LoLLMs environment for testing...")
    lollms_paths = LollmsPaths.find_paths(force_local=True, tool_prefix="test_inflection_")
    config = LOLLMSConfig.autoload(lollms_paths)
    try:
        lollms_app = LollmsApplication("TestApp", config, lollms_paths, load_bindings=False, load_personalities=False, load_models=False)
        lollms_com = lollms_app.com
    except Exception as e:
        print(f"Couldn't instantiate LollmsApplication (might be normal in CI): {e}")
        lollms_com = None

    print("Creating InflectionAI binding instance...")
    inf = InflectionAI(config, lollms_paths, installation_option=InstallOption.INSTALL_IF_NECESSARY, lollmsCom=lollms_com)

    # --- API Key Setup ---
    api_key_found = False
    if inf._get_api_key():
        print("API Key found in config or environment.")
        api_key_found = True
    else:
        try:
            if sys.stdin.isatty():
                key_input = input("Enter Inflection AI API Key (token) for testing: ").strip()
                if key_input:
                    inf.binding_config.config["inflection_key"] = key_input
                    print("API Key set for this session.")
                    api_key_found = True
                else:
                    print("No API key provided.")
            else:
                print("Running in non-interactive mode. Skipping API key input.")
                print("Ensure INFLECTION_API_KEY environment variable is set for tests.")
        except EOFError:
            print("No API key input detected.")

    if not api_key_found:
         print("API key not found. Skipping tests that require API access.")
         sys.exit(1)

    print("\nUpdating settings...")
    inf.settings_updated()

    available_models = inf.list_models()
    if available_models:
        print(f"\nAvailable Inflection AI Models (Hardcoded: {len(available_models)}):")
        for m in available_models: print(f"- {m}")

        # --- Test Setup ---
        test_model = "inflection_3_pi" # Default model
        if test_model not in available_models:
             print(f"\nWarning: Preferred test model '{test_model}' not found. Using first available: {available_models[0] if available_models else 'None'}")
             if available_models: test_model = available_models[0]
             else: test_model = None

        if test_model:
             print(f"\nSelecting model for testing: {test_model}")
             inf.config.model_name = test_model
             inf.build_model()
             print(f"Max Output set to: {inf.config.max_n_predict}")

             # Define a callback function for testing
             def print_callback(chunk: str, msg_type: int, metadata: dict) -> bool:
                 type_str = MSG_OPERATION_TYPE(msg_type).name if msg_type in MSG_OPERATION_TYPE.__members__.values() else f"UNKNOWN({msg_type})"
                 if msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK:
                     print(chunk, end="", flush=True)
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_STEP_END:
                     print(f"\n--- [{type_str}] Generation End ---")
                     print(f"Metadata: {metadata}")
                     print("--- End Step End ---")
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION:
                     print(f"\n## [{type_str}] EXCEPTION: {chunk} ##")
                     return False
                 return True

             # --- Test Case 1: Standard Chat (Streaming) ---
             print("\n" + "="*20 + " Test 1: Standard Chat (Streaming) " + "="*20)
             prompt1 = "Write a short poem about a robot learning to feel."
             print(f"Prompt: {prompt1}")
             print("Response Stream:")
             try:
                 # Use n_predict to override max_tokens for this call
                 response1 = inf.generate(prompt1, n_predict=100, callback=print_callback, verbose=True)
                 print("\n-- Test 1 Complete --")
             except Exception as e: print(f"\nTest 1 failed: {e}"); trace_exception(e)

             # --- Test Case 2: Web Search Disabled (Streaming) ---
             print("\n" + "="*20 + " Test 2: Web Search Disabled " + "="*20)
             inf.binding_config.config["web_search"] = False
             inf.settings_updated() # Apply change
             prompt2 = "What is the latest news about AI?" # Should not use web search
             print(f"Prompt: {prompt2}")
             print("Response Stream (Web search disabled):")
             try:
                 response2 = inf.generate(prompt2, n_predict=150, callback=print_callback, verbose=True)
                 print("\n-- Test 2 Complete --")
             except Exception as e: print(f"\nTest 2 failed: {e}"); trace_exception(e)
             finally:
                 inf.binding_config.config["web_search"] = True # Reset
                 inf.settings_updated()


             # --- Test Case 3: Non-Streaming Request ---
             print("\n" + "="*20 + " Test 3: Non-Streaming Request " + "="*20)
             prompt3 = "List three potential benefits of Artificial General Intelligence."
             print(f"Prompt: {prompt3}")
             print("Response (Full):")
             try:
                 response3 = inf.generate(prompt3, n_predict=200, callback=None, verbose=True) # No callback
                 print(response3)
                 print("\n-- Test 3 Complete --")
             except Exception as e: print(f"\nTest 3 failed: {e}"); trace_exception(e)

        else:
             print("\nSkipping generation tests as no suitable model could be selected.")
    else:
        print("\nCould not retrieve model list (hardcoded list empty?).")
        print("Skipping tests.")

    print("\nScript finished.")