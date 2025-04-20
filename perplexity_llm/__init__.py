######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying
# engine author : Perplexity AI
# license       : Apache 2.0
# Description   :
# This binding provides an interface to Perplexity AI's API for chat completions,
# leveraging their online models with built-in search capabilities.
# Update date   : 19/07/2024
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
    # Attempt direct import, assuming they might already be installed
    pass

# Import required libraries
try:
    import requests
except ImportError as e:
    print(f"Error importing required libraries: {e}")
    print("Please ensure 'requests' is installed (`pip install requests`)")
    raise e # Re-raise the exception to prevent the script from continuing without dependencies

# Exception Handling (Placeholder - Customize as needed)
class PerplexityAPIError(Exception):
    """Custom exception for Perplexity API errors."""
    pass

class HttpException(Exception):
    """Custom exception for HTTP errors."""
    pass


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023-2024, ParisNeo"
__license__ = "Apache 2.0"

binding_name = "Perplexity"
binding_folder_name = "perplexity_llm"
binding_path = Path(__file__).parent

# ================= Known Models =================
# Source: https://docs.perplexity.ai/docs/model-cards (check for updates)
# Note: Context sizes and max output are often estimations or common limits.
# Perplexity API might truncate responses differently.
# This dictionary serves as a fallback and source of metadata.
PERPLEXITY_MODELS: Dict[str, Dict[str, Any]] = {
    # Llama 3 Sonar Small (32k context)
    "llama-3-sonar-small-32k-chat": {"context_window": 32768, "max_output": 4096, "rank": 2.5, "category": "chat"},
    "llama-3-sonar-small-32k-online": {"context_window": 28000, "max_output": 4096, "rank": 2.8, "category": "online_chat"}, # Online models often have reduced effective input for search
    # Llama 3 Sonar Large (32k context)
    "llama-3-sonar-large-32k-chat": {"context_window": 32768, "max_output": 4096, "rank": 3.5, "category": "chat"},
    "llama-3-sonar-large-32k-online": {"context_window": 28000, "max_output": 4096, "rank": 3.8, "category": "online_chat"},
    # Mixtral 8x7b Instruct (16k context)
    "mixtral-8x7b-instruct": {"context_window": 16384, "max_output": 2048, "rank": 3.0, "category": "chat"},
    # Legacy/Hypothetical (keep commented or remove if unused)
    # "pplx-7b-chat": {"context_window": 8192, "max_output": 2048, "rank": 2.0, "category": "chat"},
    # "pplx-70b-chat": {"context_window": 4096, "max_output": 2048, "rank": 3.2, "category": "chat"},
    # "pplx-7b-online": {"context_window": 4096, "max_output": 1024, "rank": 2.2, "category": "online_chat"},
    # "pplx-70b-online": {"context_window": 4096, "max_output": 1024, "rank": 3.4, "category": "online_chat"},
}

# --- Cost Estimation (Placeholder - Perplexity pricing is complex: per request/chars) ---
# These are illustrative only and DO NOT reflect actual Perplexity costs.
# Set turn_on_cost_estimation to False by default.
INPUT_COSTS_PLACEHOLDER: Dict[str, float] = {"default": 0.0 / 1_000_000}
OUTPUT_COSTS_PLACEHOLDER: Dict[str, float] = {"default": 0.0 / 1_000_000}


class Perplexity(LLMBinding):
    """
    Binding class for interacting with the Perplexity AI API.
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

        # Configuration definition using ConfigTemplate
        binding_config_template = ConfigTemplate([
            # --- Credentials ---
            {"name":"perplexity_key","type":"str","value":"","help":"Your Perplexity AI API key. Found at https://docs.perplexity.ai/docs/getting-started"},

            # --- Model & Generation ---
            # Use manual limits as auto-detect is less reliable/documented for PPLX
            {"name":"ctx_size","type":"int","value":4096, "min":512, "help":"Model's maximum context size (input tokens). Adjust based on selected model and PPLX limits. Online models often have smaller effective input."},
            {"name":"max_n_predict","type":"int","value":1024, "min":1, "help":"Maximum number of tokens to generate per response. Perplexity API might have its own caps."},
            # Removed Seed as it's not listed in PPLX API docs

            # --- Perplexity Specific Parameters ---
            {"name":"temperature","type":"float","value":0.2,"min":0.0, "max":1.99, "help":"Amount of randomness (0.0-1.99). Lower values are more focused."},
            {"name":"top_p","type":"float","value":0.9,"min":0.0, "max":1.0, "help":"Nucleus sampling threshold (0.0-1.0)."},
            {"name":"top_k","type":"int","value":0,"min":0, "help":"Top-k filtering (0 to disable). Limits selection to k most likely tokens."},
            {"name":"presence_penalty","type":"float","value":0.0,"min":0.0, "max":2.0, "help":"Penalty for new topics (0.0-2.0). Higher values encourage new concepts."},
            {"name":"frequency_penalty","type":"float","value":1.0,"min":0.0, "max":2.0, "help":"Penalty for repetition (0.0-2.0). Higher values reduce repeating words."},
            {"name":"response_format_type", "type": "str", "value": "text", "options":["text", "json_object"], "help":"Set to 'json_object' to enable structured JSON output."},

            # --- Search Related Parameters (for Online models) ---
            {"name":"use_search_options", "type":"bool", "value":True, "help":"Enable configuration of search parameters for models that support them (e.g., 'online' models)."},
            {"name":"search_context_size", "type":"str", "value":"medium", "options":["low", "medium", "high"], "help":"(Web Search Option) Context size for web search (affects cost, quality, latency)."},
            {"name":"search_domain_filter", "type":"str", "value":"", "help":"Comma-separated list of domains to allow or deny (prefix with '-'). Max 3 domains (e.g., 'wikipedia.org,-badnews.com')."},
            {"name":"return_images", "type":"bool", "value":False, "help":"Request images in search results (if model supports). Note: Lollms UI may not display these images directly."},
            {"name":"return_related_questions", "type":"bool", "value":False, "help":"Request related questions (if model supports)."},
            {"name":"search_recency_filter", "type":"str", "value":"", "options":["", "day", "week", "month"], "help":"Filter search results by time."},

            # --- Cost Estimation (Placeholder - Disabled by default) ---
            {"name":"turn_on_cost_estimation","type":"bool", "value":False,"help":"Estimate costs (PLACEHOLDER - Not accurate for Perplexity's pricing model)."},
            {"name":"total_input_tokens","type":"float", "value":0,"help":"Accumulated input tokens (estimate)."},
            {"name":"total_output_tokens","type":"float", "value":0,"help":"Accumulated output tokens (estimate)."},
            {"name":"total_input_cost","type":"float", "value":0,"help":"Accumulated input cost ($) (estimate)."},
            {"name":"total_output_cost","type":"float", "value":0,"help":"Accumulated output cost ($) (estimate)."},
            {"name":"total_cost","type":"float", "value":0,"help":"Total accumulated cost ($) (estimate)."},

            # --- Advanced ---
             {"name":"override_api_url","type":"str","value":"https://api.perplexity.ai","help":"Advanced: Override the default Perplexity API URL."},

        ])
        # Default values for the configuration
        binding_config_defaults = BaseConfig(config={
            "perplexity_key": "",
            "ctx_size": 4096,
            "max_n_predict": 1024,
            "temperature": 0.2,
            "top_p": 0.9,
            "top_k": 0,
            "presence_penalty": 0.0,
            "frequency_penalty": 1.0,
            "response_format_type": "text",
            "use_search_options": True,
            "search_context_size": "medium",
            "search_domain_filter": "",
            "return_images": False,
            "return_related_questions": False,
            "search_recency_filter": "",
            "turn_on_cost_estimation": False, # Disabled by default
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_input_cost": 0,
            "total_output_cost": 0,
            "total_cost": 0,
            "override_api_url": "https://api.perplexity.ai",
        })

        binding_config = TypedConfig(
            binding_config_template,
            binding_config_defaults
        )
        super().__init__(
                            binding_path, # Use binding_path defined globally
                            lollms_paths,
                            config,
                            binding_config,
                            installation_option,
                            supported_file_extensions=[], # Perplexity Chat API doesn't support file uploads directly
                            lollmsCom=lollmsCom
                        )
        self.config.ctx_size = self.binding_config.config["ctx_size"]
        self.config.max_n_predict = self.binding_config.config["max_n_predict"]
        # Initialize available_models using list_models() which uses the static dict for now
        self.available_models: List[str] = self.list_models()
        self.perplexity_key: Optional[str] = None
        self.session = requests.Session() # Use a session for potential connection reuse

    def _update_perplexity_key(self) -> bool:
        """
        Sets the Perplexity API key from configuration or environment variables.

        Returns:
            True if an API key was found, False otherwise.
        """
        api_key = None
        source = None
        config_key = self.binding_config.config.get("perplexity_key", "")

        if config_key and config_key.strip():
            api_key = config_key
            source = "configuration"
        else:
            api_key_env = os.getenv('PERPLEXITY_API_KEY')
            if api_key_env:
                api_key = api_key_env
                source = "environment variable"

        if api_key:
            self.perplexity_key = api_key
            ASCIIColors.info(f"Using Perplexity API key from {source}.")
            # Simple validation: check if key is non-empty string
            if isinstance(api_key, str) and len(api_key) > 5: # Arbitrary length check
                return True
            else:
                 self.error(f"Invalid Perplexity API key format found from {source}.")
                 self.perplexity_key = None
                 return False
        else:
            self.warning("No Perplexity API key found in configuration or environment variables. Perplexity binding will not function.")
            self.perplexity_key = None
            return False

    # No _update_available_models needed as we use a hardcoded list via list_models()

    def settings_updated(self) -> None:
        """Callback triggered when binding settings are updated in the UI."""
        ASCIIColors.info("Perplexity settings updated.")
        key_updated = self._update_perplexity_key()

        # Update main Lollms config context/prediction from binding settings
        self.config.ctx_size = self.binding_config.config["ctx_size"]
        self.config.max_n_predict = self.binding_config.config["max_n_predict"]

        if not key_updated:
            self.error("Perplexity API Key is missing or invalid. Please configure it.")
            if self.lollmsCom:
                 self.lollmsCom.InfoMessage("Perplexity Error: API Key is missing or invalid.")
        elif self.config.model_name:
             self.info(f"Settings updated. Current model: {self.config.model_name}. Context: {self.config.ctx_size}, Max Output: {self.config.max_n_predict}")
        else:
             self.info("Settings updated. No model currently selected.")

    # No get_model_limits needed, relying on manual settings

    def build_model(self, model_name: Optional[str] = None) -> LLMBinding:
        """
        Sets up the binding for the selected model.

        Args:
            model_name: The name of the model to potentially load (used by parent).

        Returns:
            The instance of the binding.
        """
        super().build_model(model_name) # Sets self.config.model_name from argument or config

        if not self.perplexity_key:
             if not self._update_perplexity_key():
                 self.error("Model build failed: Perplexity API key is missing or invalid.")
                 if self.lollmsCom:
                     self.lollmsCom.InfoMessage("Perplexity Error: Cannot build model. API Key is missing or invalid.")
                 return self

        current_model_name = self.config.model_name or ""
        if not current_model_name:
            self.warning("No model name selected.")
            # Keep defaults from binding_config
            self.config.ctx_size = self.binding_config.config["ctx_size"]
            self.config.max_n_predict = self.binding_config.config["max_n_predict"]
            return self

        # Update available models list just in case (though it's static now)
        self.available_models = self.list_models()

        if current_model_name not in self.available_models:
             self.warning(f"Model '{current_model_name}' is not in the known list of Perplexity models. It might work, but parameters are based on defaults.")

        # Update main config context/prediction based on binding config (manual settings)
        self.config.ctx_size = self.binding_config.config["ctx_size"]
        self.config.max_n_predict = self.binding_config.config["max_n_predict"]

        ASCIIColors.info(f"Effective limits set: ctx_size={self.config.ctx_size}, max_n_predict={self.config.max_n_predict}")

        # Determine Binding Type (Always Text for Chat API)
        self.binding_type = BindingType.TEXT_ONLY
        self.supported_file_extensions=[] # No direct file uploads

        ASCIIColors.success(f"Perplexity binding built successfully. Model: {current_model_name}. API URL: {self.binding_config.config.get('override_api_url', 'https://api.perplexity.ai')}")
        return self

    def install(self) -> None:
        """Installs necessary Python packages using pipmaster."""
        super().install()
        self.ShowBlockingMessage("Installing Perplexity API client requirements...")
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
            ASCIIColors.success("Perplexity client requirements installed successfully.")
            ASCIIColors.info("----------------------")
            ASCIIColors.info("Attention:")
            ASCIIColors.info("----------------------")
            ASCIIColors.info("The Perplexity binding uses the Perplexity API.")
            ASCIIColors.info("1. Create an account and get an API key at https://docs.perplexity.ai/")
            ASCIIColors.info("2. Provide the key in the binding settings or set the PERPLEXITY_API_KEY environment variable.")
        except ImportError:
            self.HideBlockingMessage()
            self.warning("pipmaster not found. Please install requirements manually: pip install requests")
        except Exception as e:
            self.error(f"Installation failed: {e}")
            trace_exception(e)
            self.warning("Installation failed. Please ensure you have pip installed and internet access.", 20)
            self.HideBlockingMessage()

    def tokenize(self, prompt: str) -> List[int]:
        """
        Tokenizes the given prompt.
        NOTE: Perplexity API does not expose its tokenizer. This method is NOT accurate
              and should not be relied upon for precise token counting.
              It returns an empty list as a placeholder.

        Args:
            prompt: The text prompt to tokenize.

        Returns:
            An empty list.
        """
        ASCIIColors.warning("Perplexity binding does not support client-side tokenization. Token count estimations may be inaccurate.")
        # Option: return estimate using simple split? len(prompt.split())
        # Option: return character count? len(prompt)
        # Returning empty list to signal unavailability
        return []

    def detokenize(self, tokens_list: List[int]) -> str:
        """
        Detokenizes the given list of tokens.
        NOTE: As tokenization is not supported, detokenization is also not possible.

        Args:
            tokens_list: A list of token IDs.

        Returns:
            An empty string.
        """
        ASCIIColors.warning("Perplexity binding does not support client-side detokenization.")
        return ""

    def embed(self, text: Union[str, List[str]]) -> Optional[List[List[float]]]:
        """
        Computes text embeddings.
        NOTE: Perplexity API documentation does not currently list an embedding endpoint.

        Args:
            text: The text or list of texts to embed.

        Returns:
            None, as this feature is not supported.
        """
        self.error("Perplexity binding does not support embedding generation.")
        return None

    def generate_with_images(self, prompt: str, images: List[str], **kwargs) -> str:
        """
        Generates text using a prompt and images.
        NOTE: The Perplexity Chat Completions API does not support direct image input in the request.

        Args:
            prompt: The text prompt.
            images: A list of paths to image files.
            **kwargs: Additional parameters.

        Returns:
            An error message indicating lack of support.
        """
        self.error("Perplexity binding (Chat API) does not support image inputs.")
        return "Error: Image input not supported by Perplexity binding."


    def generate(self,
                 prompt: str,
                 n_predict: Optional[int] = None,
                 callback: Optional[Callable[[str, int, Optional[dict]], bool]] = None, # Adjusted callback signature
                 verbose: bool = False,
                 **generation_params) -> str:
        """
        Generates text using the Perplexity Chat Completions API.

        Args:
            prompt: The text prompt.
            n_predict: Optional override for the maximum number of tokens to generate.
                       If None, uses the binding's configured max_n_predict.
            callback: An optional callback function for streaming results.
                      Signature: callback(token_or_full_response: str, message_type: int, metadata: Optional[dict]) -> bool
            verbose: If True, prints more detailed information.
            **generation_params: Additional parameters for the API call (e.g., temperature, top_p, top_k).

        Returns:
            The generated text response.
        """
        if not self.perplexity_key:
            self.error("Perplexity API key not set.")
            if callback: callback("Error: Perplexity API key not configured.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION, None)
            raise PerplexityAPIError("API key not configured.")

        model_name = self.config.model_name
        if not model_name:
             self.error("No model selected.")
             if callback: callback("Error: No model selected.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION, None)
             raise PerplexityAPIError("No model selected.")

        # --- Parameter Preparation ---
        effective_max_n_predict = self.config.max_n_predict
        if n_predict is not None:
             if 0 < n_predict <= self.config.max_n_predict:
                  effective_max_n_predict = n_predict
                  if verbose: ASCIIColors.verbose(f"Using user-provided n_predict: {effective_max_n_predict}")
             elif n_predict > self.config.max_n_predict:
                  self.warning(f"Requested n_predict ({n_predict}) exceeds the configured limit ({self.config.max_n_predict}). Capping at {self.config.max_n_predict}.")
                  effective_max_n_predict = self.config.max_n_predict
        if verbose: ASCIIColors.verbose(f"Effective max_tokens for this generation: {effective_max_n_predict}")

        # Combine default binding config with runtime params
        api_params: Dict[str, Any] = {
            "temperature": self.binding_config.config.get("temperature", 0.2),
            "top_p": self.binding_config.config.get("top_p", 0.9),
            "top_k": self.binding_config.config.get("top_k", 0),
            "presence_penalty": self.binding_config.config.get("presence_penalty", 0.0),
            "frequency_penalty": self.binding_config.config.get("frequency_penalty", 1.0),
        }
        # Override with any generation_params passed at runtime
        api_params.update(generation_params)

        # Construct payload
        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}], # Simple user prompt
            # TODO: Consider adding system prompt from config if needed
            # messages = [{"role": "system", "content": self.config.system_prompt}, {"role": "user", "content": prompt}]
            "max_tokens": effective_max_n_predict,
            "temperature": float(api_params["temperature"]),
            "top_p": float(api_params["top_p"]),
            "top_k": int(api_params["top_k"]),
            "presence_penalty": float(api_params["presence_penalty"]),
            "frequency_penalty": float(api_params["frequency_penalty"]),
            "stream": bool(callback is not None), # Stream if callback is provided
        }

        # Add response format if not default 'text'
        response_format = self.binding_config.config.get("response_format_type", "text")
        if response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        # Add search-related options if enabled
        # PPLX API handles search based on model name implicitly ('online' models)
        # Some parameters might still be applicable if the API supports them
        if self.binding_config.config.get("use_search_options", True):
            # Note: Perplexity API reference doesn't explicitly list these under /chat/completions
            # They might be inferred or specific to certain models/endpoints. Keeping them conditional.
            # search_context = self.binding_config.config.get("search_context_size", "medium") # Not in Chat API docs?
            # if search_context != "medium": payload["search_context_size"] = search_context

            domain_filter_str = self.binding_config.config.get("search_domain_filter", "")
            if domain_filter_str:
                domains = [d.strip() for d in domain_filter_str.split(',') if d.strip()]
                if domains:
                    self.warning("search_domain_filter might not be supported by the Chat API endpoint.")
                    # payload["search_domain_filter"] = domains # Commented out unless confirmed

            if self.binding_config.config.get("return_images", False):
                 self.warning("return_images might not be supported by the Chat API endpoint.")
                 # payload["return_images"] = True # Commented out unless confirmed
            if self.binding_config.config.get("return_related_questions", False):
                 self.warning("return_related_questions might not be supported by the Chat API endpoint.")
                 # payload["return_related_questions"] = True # Commented out unless confirmed
            recency = self.binding_config.config.get("search_recency_filter", "")
            if recency:
                 self.warning("search_recency_filter might not be supported by the Chat API endpoint.")
                 # payload["search_recency_filter"] = recency # Commented out unless confirmed


        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.perplexity_key}",
            "Content-Type": "application/json",
            "Accept": "application/json" if not payload["stream"] else "text/event-stream"
        }

        api_url = self.binding_config.config.get("override_api_url", "https://api.perplexity.ai").rstrip('/') + "/chat/completions"

        output = ""
        metadata = {}
        total_output_tokens = 0 # Placeholder for cost estimation
        total_input_tokens = 0 # Placeholder

        start_time = perf_counter()
        try:
            if verbose: ASCIIColors.verbose(f"Calling Perplexity API ({'Streaming' if payload['stream'] else 'Non-streaming'}). URL: {api_url}. Payload: {json.dumps(payload, indent=2)}")

            response = self.session.post(
                api_url,
                headers=headers,
                json=payload,
                stream=payload["stream"]
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            # --- Handle Streaming Response ---
            if payload["stream"]:
                stream_finished = False
                client = self.sse_client(response) # Helper to parse SSE
                for event in client.events():
                    if event.event == "message":
                         if event.data.strip() == "[DONE]":
                             stream_finished = True
                             if verbose: ASCIIColors.verbose("Stream finished with [DONE].")
                             # Final usage might be in the last message or a separate event (check PPLX docs)
                             break # Exit loop on DONE

                         try:
                             chunk_data = json.loads(event.data)
                             if verbose: ASCIIColors.verbose(f"Stream chunk received: {chunk_data}")
                             chunk_metadata = {"chunk_id": chunk_data.get("id")} # Example metadata

                             # Extract text delta
                             delta_content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content")
                             if delta_content:
                                 output += delta_content
                                 if callback:
                                     if not callback(delta_content, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK, chunk_metadata):
                                         self.info("Generation stopped by callback.")
                                         stream_finished = True
                                         # Send cancellation step? Or just break?
                                         # callback("Operation cancelled", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_INFO, None) # Optional
                                         break # Stop processing stream

                             # Extract finish reason (usually on the last chunk)
                             finish_reason = chunk_data.get("choices", [{}])[0].get("finish_reason")
                             if finish_reason:
                                 metadata["finish_reason"] = finish_reason
                                 if verbose: ASCIIColors.verbose(f"Finish reason received: {finish_reason}")
                                 stream_finished = True # Mark finished if reason received

                             # Extract usage stats if present (PPLX API might send usage at the end or per chunk)
                             usage = chunk_data.get("usage")
                             if usage:
                                 metadata["usage"] = usage # Store the latest usage info
                                 if verbose: ASCIIColors.verbose(f"Usage stats received in chunk: {usage}")
                                 # Use these as current best estimates
                                 total_input_tokens = usage.get("prompt_tokens", total_input_tokens)
                                 total_output_tokens = usage.get("completion_tokens", total_output_tokens)

                         except json.JSONDecodeError:
                             self.error(f"Failed to decode JSON stream data: {event.data}")
                             continue # Skip malformed data
                         except Exception as chunk_ex:
                             self.error(f"Error processing stream chunk: {chunk_ex}")
                             trace_exception(chunk_ex)
                             continue

                    elif event.event == "error":
                         error_data = event.data
                         self.error(f"Perplexity API Stream Error: {error_data}")
                         try:
                            error_json = json.loads(error_data)
                            error_message = f"Stream Error: {error_json.get('detail', error_data)}"
                         except json.JSONDecodeError:
                            error_message = f"Stream Error: {error_data}"

                         if callback: callback(error_message, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION, None)
                         stream_finished = True # Assume stream ends on error
                         break
                    elif event.event == "ping":
                        if verbose: ASCIIColors.verbose("Received ping event.")
                        pass # Ignore pings
                    else: # Handle potential other event types if needed
                         if verbose: ASCIIColors.verbose(f"Received unexpected SSE event: {event.event} - Data: {event.data}")


                if not stream_finished:
                    self.info("Stream ended without explicit [DONE] message or finish reason.")

                # PPLX might send final usage *after* the [DONE] message or in a separate event/header.
                # Check documentation for specifics. Here, we assume usage might be in the last chunk's metadata.
                if "usage" in metadata:
                     final_usage = metadata["usage"]
                     total_input_tokens = final_usage.get("prompt_tokens", total_input_tokens)
                     total_output_tokens = final_usage.get("completion_tokens", total_output_tokens)
                     # Also check for total tokens if provided
                     total_output_tokens = final_usage.get("total_tokens", total_output_tokens) # PPLX might use total_tokens

                # Send final status update via callback if stream finished normally
                if stream_finished and "finish_reason" in metadata and callback:
                     final_status = f"Generation finished: {metadata.get('finish_reason', 'Unknown')}"
                     callback(final_status, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_STEP_END, metadata)


            # --- Handle Non-Streaming Response ---
            else:
                full_response_json = response.json()
                if verbose: ASCIIColors.verbose(f"Non-streaming response received: {json.dumps(full_response_json, indent=2)}")

                if full_response_json.get("choices"):
                     message = full_response_json["choices"][0].get("message")
                     if message:
                         output = message.get("content", "")
                     else:
                         output = "Error: No message found in response choice."
                     metadata["finish_reason"] = full_response_json["choices"][0].get("finish_reason")
                     metadata["response_id"] = full_response_json.get("id") # Store response ID
                else:
                     self.error("No 'choices' found in Perplexity API response.")
                     output = f"Error: Unexpected API response format. {full_response_json}"

                usage = full_response_json.get("usage")
                if usage:
                     metadata["usage"] = usage
                     total_input_tokens = usage.get("prompt_tokens", 0)
                     total_output_tokens = usage.get("completion_tokens", 0)
                     # Check for total_tokens as well
                     total_output_tokens = usage.get("total_tokens", total_output_tokens) # Use if completion_tokens is missing
                     if verbose: ASCIIColors.verbose(f"Usage stats received: {usage}")

                if callback:
                     # Send the whole response as one chunk/message
                     callback(output, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_FULL_ANSWER, metadata)


        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_message = f"Perplexity API HTTP Error: {status_code}"
            try:
                error_details = e.response.json()
                error_message += f" - {error_details.get('detail', e.response.text)}"
            except json.JSONDecodeError:
                error_message += f" - {e.response.text}" # Non-JSON error response

            self.error(error_message)
            trace_exception(e)
            if callback: callback(error_message, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION, None)
            raise HttpException(error_message) from e # Wrap in HttpException

        except requests.exceptions.RequestException as e:
            error_message = f"Network error connecting to Perplexity API: {e}"
            self.error(error_message)
            trace_exception(e)
            if callback: callback(error_message, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION, None)
            raise HttpException(error_message) from e # Wrap in HttpException

        except Exception as e:
            error_message = f"Error during Perplexity generation: {e}"
            self.error(error_message)
            trace_exception(e)
            if callback: callback(error_message, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION, None)
            raise # Re-raise other exceptions
        finally:
             generation_time = perf_counter() - start_time
             ASCIIColors.info(f"Generation process finished in {generation_time:.2f} seconds.")


        # --- Cost Estimation (Placeholder) ---
        if self.binding_config.config.turn_on_cost_estimation:
            # WARNING: These calculations are NOT accurate for Perplexity
            # Using placeholders based on token counts if available from API
            if total_input_tokens == 0 and prompt:
                # Rough estimate if API didn't provide prompt tokens
                # Simple split or char count might be better than non-functional tokenize
                total_input_tokens = len(prompt.split()) # Very rough
                ASCIIColors.warning("Using rough word count for input token estimate.")
            if total_output_tokens == 0 and output:
                # Rough estimate if API didn't provide completion tokens
                total_output_tokens = len(output.split()) # Very rough
                ASCIIColors.warning("Using rough word count for output token estimate.")


            self.binding_config.config["total_input_tokens"] += total_input_tokens
            self.binding_config.config["total_output_tokens"] += total_output_tokens
            input_cost_rate = INPUT_COSTS_PLACEHOLDER.get("default", 0)
            output_cost_rate = OUTPUT_COSTS_PLACEHOLDER.get("default", 0)
            input_cost = total_input_tokens * input_cost_rate
            output_cost = total_output_tokens * output_cost_rate
            self.binding_config.config["total_input_cost"] += input_cost
            self.binding_config.config["total_output_cost"] += output_cost
            self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]

            cost_info = f'Accumulated cost (PLACEHOLDER ESTIMATE): ${self.binding_config.config["total_cost"]:.6f} ({total_input_tokens} input, {total_output_tokens} output tokens this call)'
            self.info(cost_info)
            if callback:
                callback(cost_info, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_INFO, None)
            self.binding_config.save()

        return output

    def list_models(self) -> List[str]:
        """
        Lists the available models.
        Currently returns the hardcoded list from PERPLEXITY_MODELS.
        (Could be extended to call an API endpoint if Perplexity provides one).

        Returns:
            A list of model ID strings.
        """
        # For now, it relies on the static dictionary keys.
        # If Perplexity adds a model listing endpoint, this method could be updated.
        # Example structure if API call existed:
        # if not self.perplexity_key:
        #     self.warning("API key needed to fetch models from Perplexity API. Returning static list.")
        #     return list(PERPLEXITY_MODELS.keys())
        # try:
        #     headers = {"Authorization": f"Bearer {self.perplexity_key}"}
        #     # Replace with actual endpoint if available
        #     response = self.session.get("https://api.perplexity.ai/models", headers=headers)
        #     response.raise_for_status()
        #     models_data = response.json()
        #     # Assuming response format like OpenAI: {'data': [{'id': 'model1'}, ...]}
        #     return [model['id'] for model in models_data.get('data', [])]
        # except Exception as e:
        #     self.error(f"Failed to fetch models from Perplexity API: {e}. Returning static list.")
        #     return list(PERPLEXITY_MODELS.keys())

        # Return keys from the static dictionary
        return list(PERPLEXITY_MODELS.keys())


    def get_available_models(self, app: Optional[LoLLMsCom] = None) -> List[dict]:
        """
        Gets the available models list formatted for the LoLLMs UI.
        It uses self.list_models() as the primary source and falls back to
        the static PERPLEXITY_MODELS dictionary for metadata or if list_models fails.

        Args:
            app: The LoLLMsCom application instance (optional).

        Returns:
            A list of dictionaries formatted for the LoLLMs models list.
        """
        models_info: List[Dict[str, Any]] = []
        ui_path_prefix = f"/bindings/{binding_folder_name}/"
        default_icon = ui_path_prefix + "logo.png" # Assumes logo.png exists

        base_metadata = {
            "author": "Perplexity AI",
            "license": "Commercial API",
            "creation_date": None, # Unknown
            "category": "API Model", # Overridden below
            "datasets": "Proprietary Perplexity AI Datasets",
            "commercial_use": True,
            "icon": default_icon,
            "model_creator": "Perplexity AI",
            "model_creator_link": "https://perplexity.ai/",
            "quantizer": None,
            "type": "api",
            "binding_name": binding_name # Added for clarity
        }

        # --- Use list_models() as primary source ---
        model_ids_from_list = self.list_models() # Currently returns static list keys

        if model_ids_from_list:
            ASCIIColors.info(f"Found {len(model_ids_from_list)} models from list_models(). Fetching metadata...")
            for model_id in model_ids_from_list:
                # Fetch metadata from our static dictionary PERPLEXITY_MODELS
                # Use .get() for safe access, providing empty dict if model_id not found
                model_data = PERPLEXITY_MODELS.get(model_id, {})

                # Extract details using .get() with defaults for safety
                ctx = model_data.get("context_window", self.binding_config.config.ctx_size) # Use known or default binding ctx
                max_out = model_data.get("max_output", self.binding_config.config.max_n_predict) # Use known or default binding max_n
                rank = model_data.get("rank", 2.0) # Default rank if missing
                category = model_data.get("category", "chat") # Default category if missing
                is_online = "online" in category.lower()

                # --- Build Lollms Model Dictionary ---
                model_entry = {
                    **base_metadata,
                    "name": model_id,
                    "display_name": model_id.replace("-", " ").title(),
                    "category": category,
                    "rank": rank,
                    "description": f"Perplexity API model. Context: {ctx}. Max Output: {max_out}. {'Includes Web Search.' if is_online else ''}",
                    "ctx_size": ctx,
                    "max_n_predict": max_out,
                    "variants": [{"name": model_id, "size": ctx}] # Use context as size proxy
                }
                models_info.append(model_entry)
        else:
            # --- Fallback: Directly use PERPLEXITY_MODELS if list_models() was empty ---
            ASCIIColors.warning("list_models() returned empty. Falling back to static PERPLEXITY_MODELS dictionary.")
            for model_id, model_data in PERPLEXITY_MODELS.items():
                # Extract details (same logic as above)
                ctx = model_data.get("context_window", self.binding_config.config.ctx_size)
                max_out = model_data.get("max_output", self.binding_config.config.max_n_predict)
                rank = model_data.get("rank", 2.0)
                category = model_data.get("category", "chat")
                is_online = "online" in category.lower()

                # Build Lollms Model Dictionary
                model_entry = {
                    **base_metadata,
                    "name": model_id,
                    "display_name": model_id.replace("-", " ").title(),
                    "category": category,
                    "rank": rank,
                    "description": f"Perplexity API model. Context: {ctx}. Max Output: {max_out}. {'Includes Web Search.' if is_online else ''}",
                    "ctx_size": ctx,
                    "max_n_predict": max_out,
                    "variants": [{"name": model_id, "size": ctx}]
                }
                models_info.append(model_entry)

        # Sort models: by rank (desc), then by name (asc)
        models_info.sort(key=lambda x: (-x.get('rank', 0), x.get('name', ''))) # Safe sort

        if models_info:
            ASCIIColors.success(f"Formatted {len(models_info)} Perplexity models for Lollms UI.")
        else:
            ASCIIColors.warning("No Perplexity models found or formatted for the UI.")

        return models_info


    # Simple SSE Client (adapted from various sources)
    # Needs refinement for robustness, especially error handling and reconnection logic
    class sse_client:
        def __init__(self, response):
            self._response = response
            # self._chunk_decoder = json.JSONDecoder() # Not strictly needed if we parse each event data

        def _iter_sse_lines(self):
             """Iterates over SSE message blocks from the response."""
             buffer = ""
             for chunk in self._response.iter_content(chunk_size=1024, decode_unicode=True):
                 buffer += chunk
                 # SSE messages are separated by double newlines (\n\n)
                 while '\n\n' in buffer:
                     message_block, buffer = buffer.split('\n\n', 1)
                     if message_block: # Ensure it's not an empty block between messages
                         yield message_block.splitlines() # Yield list of lines in the message block

        def events(self):
            """Yields ServerSentEvent namedtuples."""
            from collections import namedtuple
            # Define the structure for a Server-Sent Event
            ServerSentEvent = namedtuple('ServerSentEvent', ['event', 'data', 'id', 'retry'])

            current_event_type = 'message' # Default event type
            current_data_lines = []
            current_id = None
            current_retry = None

            for lines in self._iter_sse_lines():
                # Reset for each message block (shouldn't be needed if block separation is correct)
                # event_type = 'message'
                # event_data = []
                # event_id = None
                # event_retry = None

                for line in lines:
                    if not line: continue # Skip empty lines within a block (shouldn't happen with splitlines?)
                    if line.startswith(':'): continue # Skip comments

                    # Split line into field and value
                    try:
                        field, value = line.split(':', 1)
                        value = value.lstrip() # Remove leading space if present
                    except ValueError:
                        # Handle lines without a colon (potentially invalid SSE)
                        field = line # Treat the whole line as the field (e.g., for simple data lines)
                        value = "" # No value part

                    # Process known SSE fields
                    if field == 'event': current_event_type = value
                    elif field == 'data': current_data_lines.append(value)
                    elif field == 'id': current_id = value
                    elif field == 'retry':
                        try:
                            current_retry = int(value)
                        except ValueError:
                            # Ignore invalid retry values
                            pass
                    # Ignore unknown fields as per SSE spec

                # After processing all lines in a block, construct and yield the event
                # Only yield if there's data associated with the event
                if current_data_lines:
                    # Join multi-line data with newlines as per SSE spec
                    full_data = '\n'.join(current_data_lines)
                    yield ServerSentEvent(event=current_event_type, data=full_data, id=current_id, retry=current_retry)

                # Reset for the next event message block
                current_event_type = 'message' # Reset to default
                current_data_lines = []
                # id and retry persist unless overwritten by the next message


# --- Main execution block for testing ---
if __name__ == "__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMsConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    from lollms.types import MSG_OPERATION_TYPE
    import sys

    print("Initializing LoLLMs environment for testing...")
    # Make sure paths are set correctly for testing
    lollms_paths = LollmsPaths.find_paths(tool_prefix="test_perplexity_")
    config = LOLLMsConfig.autoload(lollms_paths)
    # Minimal app for communication channel (optional, can be None)
    try:
        # You might not need the full app for binding tests
        lollms_app = None # LollmsApplication("TestApp", config, lollms_paths, load_bindings=False, load_personalities=False, load_models=False)
        lollms_com = None # lollms_app.com if lollms_app else None
        print("Running without full LollmsApplication context.")
    except Exception as e:
        print(f"Couldn't instantiate LollmsApplication (might be normal in CI/limited env): {e}")
        lollms_com = None # Fallback if app fails

    print("Creating Perplexity binding instance...")
    # Force install for testing if needed, or skip if requirements are met
    pplx = Perplexity(config, lollms_paths, installation_option=InstallOption.INSTALL_IF_NECESSARY, lollmsCom=lollms_com)

    # --- API Key Setup ---
    # Check environment variable first, then config, then prompt if interactive
    if not pplx.binding_config.config.get("perplexity_key") and not os.getenv('PERPLEXITY_API_KEY'):
         try:
             # Use input only if running interactively
             if sys.stdin.isatty():
                 key_input = input("Enter Perplexity API Key for testing: ").strip()
                 if key_input:
                     pplx.binding_config.config["perplexity_key"] = key_input
                     pplx.binding_config.save() # Save to config for this session if entered
                     print("API Key set and saved to binding config for this session.")
                 else:
                      print("No API key provided. Tests requiring API access will likely fail.")
             else:
                 print("Running in non-interactive mode. Skipping API key input.")
                 print("Ensure PERPLEXITY_API_KEY environment variable is set or key is in binding config for tests.")
         except EOFError:
             print("No API key input detected (EOF).")

    print("\nUpdating settings (initializes client and checks key)...")
    pplx.settings_updated() # This initializes the client and reads the key

    # --- List Models Test ---
    print("\n" + "="*20 + " Test: Listing Models " + "="*20)
    # Test the primary method used by UI
    formatted_models = pplx.get_available_models()
    if formatted_models:
        print(f"Formatted Models for UI ({len(formatted_models)}):")
        #for m in formatted_models: print(f"- {m['name']} (Rank: {m.get('rank', 'N/A')}, Ctx: {m.get('ctx_size', 'N/A')})")
        print(f"- {formatted_models[0]['name']} (Rank: {formatted_models[0].get('rank', 'N/A')}, Ctx: {formatted_models[0].get('ctx_size', 'N/A')})")
        print(f"... (showing first model only)")
    else:
        print("No models returned by get_available_models(). Check static list or potential API issues.")

    # Also show the raw list_models output
    raw_model_list = pplx.list_models()
    print(f"\nRaw Model List from list_models() ({len(raw_model_list)}):")
    if raw_model_list: print(f"- {raw_model_list[0]}, ...")
    else: print("list_models() returned empty.")


    # --- Generation Tests (Requires API Key) ---
    if pplx.perplexity_key:
        print("\nAPI Key found. Proceeding with generation tests...")

        # --- Test Setup ---
        # Select a common, capable model for testing
        test_model = "llama-3-sonar-small-32k-online" # Cheaper online model for testing search
        available_models = pplx.list_models() # Get current list
        if not available_models:
             print("\nError: No available models found. Cannot run generation tests.")
             test_model = None
        elif test_model not in available_models:
             print(f"\nWarning: Preferred test model '{test_model}' not found in list. Trying first available model: {available_models[0]}")
             test_model = available_models[0]

        if test_model:
             print(f"\nSelecting model for testing: {test_model}")
             pplx.config.model_name = test_model
             # Update binding config context/max based on potentially selected model (optional)
             # model_meta = PERPLEXITY_MODELS.get(test_model, {})
             # pplx.binding_config.config['ctx_size'] = model_meta.get('context_window', 4096)
             # pplx.binding_config.config['max_n_predict'] = model_meta.get('max_output', 1024)
             pplx.build_model() # Finalize setup for the selected model
             print(f"Effective limits for {test_model}: Context={pplx.config.ctx_size}, Max Output={pplx.config.max_n_predict}")

             # Define a callback function for testing
             def print_callback(chunk: str, msg_type: int, metadata: Optional[dict]) -> bool:
                 type_str = MSG_OPERATION_TYPE(msg_type).name if msg_type in MSG_OPERATION_TYPE._value2member_map_ else f"UNKNOWN({msg_type})"
                 meta_str = f" | Metadata: {metadata}" if metadata else ""

                 if msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK:
                     print(chunk, end="", flush=True)
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_FULL_ANSWER:
                     print(f"\n--- [{type_str}] Full Response ---")
                     print(chunk)
                     print(f"Metadata: {metadata}")
                     print("--- End Full Response ---")
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_STEP_END:
                      print(f"\n--- [{type_str}] Generation Step End --- {meta_str}")
                      print("--- End Step End ---")
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION:
                     print(f"\n## [{type_str}] EXCEPTION: {chunk} {meta_str} ##")
                     return False # Stop on error
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_INFO or msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING:
                      print(f"\n## [{type_str}]: {chunk} {meta_str} ##")
                 else: # Catch-all for other types if needed
                     print(f"\n--- [{type_str}] --- {meta_str}")
                     print(f"Chunk: {chunk}")
                     print(f"--- End [{type_str}] ---")
                 return True

             # --- Test Case 1: Standard Chat (Streaming) ---
             print("\n" + "="*20 + " Test 1: Standard Chat (Streaming) " + "="*20)
             prompt1 = "Explain the concept of a Large Language Model (LLM) in one concise paragraph."
             print(f"Prompt: {prompt1}")
             print("Response Stream:")
             try:
                 # Use moderate n_predict
                 response1 = pplx.generate(prompt1, n_predict=150, callback=print_callback, verbose=False) # Set verbose=True for detailed logs
                 print("\n-- Test 1 Complete --")
             except Exception as e: print(f"\nTest 1 failed: {e}"); trace_exception(e)

             # --- Test Case 2: Online Model with Search (Implicit) ---
             print("\n" + "="*20 + " Test 2: Online Model Search Test " + "="*20)
             if "online" in test_model:
                 # Search is implicit for 'online' models. No specific params needed unless testing advanced features (which might not work via Chat API).
                 # pplx.binding_config.config["use_search_options"] = True # Ensure it's enabled if logic depended on it
                 # pplx.binding_config.config["search_recency_filter"] = "week" # Test parameter (might generate warning)
                 # print(f"Testing with implicit search. Configured recency='{pplx.binding_config.config.get('search_recency_filter', '')}' (may not apply)")
                 prompt2 = "What's a recent interesting development in renewable energy?"
                 print(f"Prompt: {prompt2}")
                 print("Response Stream:")
                 try:
                     response2 = pplx.generate(prompt2, n_predict=300, callback=print_callback, verbose=False)
                     print("\n-- Test 2 Complete --")
                 except Exception as e: print(f"\nTest 2 failed: {e}"); trace_exception(e)
                 finally:
                     # Reset any tested params
                     # pplx.binding_config.config["search_recency_filter"] = ""
                     pass
             else:
                  print(f"-- Test 2 Skipped (Model '{test_model}' is not an 'online' model) --")

             # --- Test Case 3: Non-Streaming Request ---
             print("\n" + "="*20 + " Test 3: Non-Streaming Request " + "="*20)
             prompt3 = "Write a very short poem about a curious cat."
             print(f"Prompt: {prompt3}")
             print("Response (Full):")
             try:
                 # Pass callback=None to force non-streaming
                 response3 = pplx.generate(prompt3, n_predict=60, callback=None, verbose=False)
                 print(response3) # Print the returned full response
                 # You could inspect metadata if needed, e.g., via a custom wrapper or if generate returned it
                 print("\n-- Test 3 Complete --")
             except Exception as e: print(f"\nTest 3 failed: {e}"); trace_exception(e)

             # --- Test Case 4: JSON Output Request (if model supports/configured) ---
             print("\n" + "="*20 + " Test 4: JSON Output Request " + "="*20)
             pplx.binding_config.config["response_format_type"] = "json_object"
             print("Set response_format_type to json_object")
             prompt4 = "Provide a JSON object with two keys: 'name' (string, a fictional character) and 'age' (integer)."
             print(f"Prompt: {prompt4}")
             print("Response (Full JSON):")
             try:
                 response4 = pplx.generate(prompt4, n_predict=100, callback=None, verbose=False)
                 print(response4)
                 # Validate if it's JSON
                 try:
                     json.loads(response4)
                     print("Response appears to be valid JSON.")
                 except json.JSONDecodeError:
                     print("Warning: Response was not valid JSON!")
                 print("\n-- Test 4 Complete --")
             except Exception as e: print(f"\nTest 4 failed: {e}"); trace_exception(e)
             finally:
                 pplx.binding_config.config["response_format_type"] = "text" # Reset
                 print("Reset response_format_type to text")


        else: # test_model was None
             print("\nSkipping generation tests as no suitable model could be selected.")

    else: # No API Key
         print("\nSkipping generation tests as Perplexity API key is missing.")


    print("\nScript finished.")