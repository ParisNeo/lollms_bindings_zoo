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


__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023-2024, ParisNeo"
__license__ = "Apache 2.0"

binding_name = "Perplexity"
binding_folder_name = "perplexity"
binding_path = Path(__file__).parent

# ================= Known Models =================
# Source: https://docs.perplexity.ai/docs/model-cards (check for updates)
# Note: Context sizes and max output are often estimations or common limits.
# Perplexity API might truncate responses differently.
# We will rely on manual settings primarily, but keep these for UI display info.
PERPLEXITY_MODELS: Dict[str, Dict[str, Any]] = {
    # Llama 3 Sonar Small (32k context)
    "llama-3-sonar-small-32k-chat": {"context_window": 32768, "max_output": 4096, "rank": 2.5, "category": "chat"},
    "llama-3-sonar-small-32k-online": {"context_window": 28000, "max_output": 4096, "rank": 2.8, "category": "online_chat"}, # Online models often have reduced effective input for search
    # Llama 3 Sonar Large (32k context)
    "llama-3-sonar-large-32k-chat": {"context_window": 32768, "max_output": 4096, "rank": 3.5, "category": "chat"},
    "llama-3-sonar-large-32k-online": {"context_window": 28000, "max_output": 4096, "rank": 3.8, "category": "online_chat"},
    # Mixtral 8x7b Instruct (16k context)
    "mixtral-8x7b-instruct": {"context_window": 16384, "max_output": 2048, "rank": 3.0, "category": "chat"},
    # Perplexity 7B Chat (8k context) - Hypothetical/Legacy? Check docs if needed
    # "pplx-7b-chat": {"context_window": 8192, "max_output": 2048, "rank": 2.0, "category": "chat"},
    # Perplexity 70B Chat (4k context) - Hypothetical/Legacy? Check docs if needed
    # "pplx-70b-chat": {"context_window": 4096, "max_output": 2048, "rank": 3.2, "category": "chat"},
    # Perplexity 7B Online (Search focused) - Hypothetical/Legacy? Check docs if needed
    # "pplx-7b-online": {"context_window": 4096, "max_output": 1024, "rank": 2.2, "category": "online_chat"},
    # Perplexity 70B Online (Search focused) - Hypothetical/Legacy? Check docs if needed
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
        self.available_models: List[str] = list(PERPLEXITY_MODELS.keys()) # Hardcoded list
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

    # No _update_available_models needed as we use a hardcoded list

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
                 callback: Optional[Callable[[str, int], bool]] = None,
                 verbose: bool = False,
                 **generation_params) -> str:
        """
        Generates text using the Perplexity Chat Completions API.

        Args:
            prompt: The text prompt.
            n_predict: Optional override for the maximum number of tokens to generate.
                       If None, uses the binding's configured max_n_predict.
            callback: An optional callback function for streaming results.
                      Signature: callback(token_or_full_response: str, message_type: int, metadata: dict) -> bool
            verbose: If True, prints more detailed information.
            **generation_params: Additional parameters for the API call (e.g., temperature, top_p, top_k).

        Returns:
            The generated text response.
        """
        if not self.perplexity_key:
            self.error("Perplexity API key not set.")
            if callback: callback("Error: Perplexity API key not configured.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return ""

        model_name = self.config.model_name
        if not model_name:
             self.error("No model selected.")
             if callback: callback("Error: No model selected.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""

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

        # Add search-related options if enabled and applicable
        if self.binding_config.config.get("use_search_options", True): # and "online" in model_name? (Optional check)
            web_search_opts: Dict[str, Any] = {}
            search_context = self.binding_config.config.get("search_context_size", "medium")
            if search_context != "medium": # Add only if not default
                web_search_opts["search_context_size"] = search_context
            # No direct equivalent for web_search_options in PPLX docs provided, this might need adjustment
            # if web_search_opts: payload["web_search_options"] = web_search_opts

            domain_filter_str = self.binding_config.config.get("search_domain_filter", "")
            if domain_filter_str:
                domains = [d.strip() for d in domain_filter_str.split(',') if d.strip()]
                if domains: payload["search_domain_filter"] = domains

            if self.binding_config.config.get("return_images", False):
                payload["return_images"] = True
            if self.binding_config.config.get("return_related_questions", False):
                payload["return_related_questions"] = True
            recency = self.binding_config.config.get("search_recency_filter", "")
            if recency:
                payload["search_recency_filter"] = recency

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

                             # Extract text delta
                             delta_content = chunk_data.get("choices", [{}])[0].get("delta").get("content")
                             if delta_content:
                                 output += delta_content
                                 if callback:
                                     if not callback(delta_content, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                                         self.info("Generation stopped by callback.")
                                         stream_finished = True
                                         break # Stop processing stream

                             # Extract finish reason (usually on the last chunk)
                             finish_reason = chunk_data.get("choices", [{}])[0].get("finish_reason")
                             if finish_reason:
                                 metadata["finish_reason"] = finish_reason
                                 if verbose: ASCIIColors.verbose(f"Finish reason received: {finish_reason}")
                                 stream_finished = True # Mark finished if reason received

                             # Extract usage stats if present (might be per chunk or final)
                             usage = chunk_data.get("usage")
                             if usage:
                                 metadata["usage"] = usage # Store the latest usage info
                                 if verbose: ASCIIColors.verbose(f"Usage stats received: {usage}")
                                 # Update token counts if possible (use final values later if available)
                                 total_input_tokens = usage.get("prompt_tokens", 0)
                                 total_output_tokens = usage.get("completion_tokens", 0)

                         except json.JSONDecodeError:
                             self.error(f"Failed to decode JSON stream data: {event.data}")
                             continue # Skip malformed data
                         except Exception as chunk_ex:
                             self.error(f"Error processing stream chunk: {chunk_ex}")
                             trace_exception(chunk_ex)
                             continue

                    elif event.event == "error":
                         self.error(f"Perplexity API Stream Error: {event.data}")
                         if callback: callback(f"Stream Error: {event.data}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
                         stream_finished = True # Assume stream ends on error
                         break
                    elif event.event == "ping":
                        if verbose: ASCIIColors.verbose("Received ping event.")
                        pass # Ignore pings
                    else: # Handle potential other event types if needed
                         if verbose: ASCIIColors.verbose(f"Received unexpected SSE event: {event.event} - Data: {event.data}")


                if not stream_finished:
                    self.info("Stream ended without explicit [DONE] message or finish reason.")
                # Send final status update via callback if stream finished
                if stream_finished and callback:
                     final_status = f"Generation finished: {metadata.get('finish_reason', 'Unknown')}"
                     callback(final_status, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_STEP_END, metadata)


            # --- Handle Non-Streaming Response ---
            else:
                full_response_json = response.json()
                if verbose: ASCIIColors.verbose(f"Non-streaming response received: {json.dumps(full_response_json, indent=2)}")

                if full_response_json.get("choices"):
                     message = full_response_json["choices"][0].get("message")
                     output = message.get("content", "")
                     metadata["finish_reason"] = full_response_json["choices"][0].get("finish_reason")
                else:
                     self.error("No 'choices' found in Perplexity API response.")
                     output = f"Error: Unexpected API response format. {full_response_json}"

                usage = full_response_json.get("usage")
                if usage:
                     metadata["usage"] = usage
                     total_input_tokens = usage.get("prompt_tokens", 0)
                     total_output_tokens = usage.get("completion_tokens", 0)
                     if verbose: ASCIIColors.verbose(f"Usage stats received: {usage}")

                if callback:
                     # Send the whole response as one chunk/message
                     callback(output, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_FULL_ANSWER, metadata)


        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_message = f"Perplexity API HTTP Error: {status_code}"
            try:
                error_details = e.response.json()
            except json.JSONDecodeError:
                error_message += f" - {e.response.text}" # Non-JSON error response
                exc = HttpException(error_message)

            self.error(error_message)
            trace_exception(e)
            if callback: callback(error_message, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            raise exc # Re-raise the formatted exception

        except requests.exceptions.RequestException as e:
            error_message = f"Network error connecting to Perplexity API: {e}"
            self.error(error_message)
            trace_exception(e)
            if callback: callback(error_message, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            raise HttpException(error_message) from e # Wrap in HttpException

        except Exception as e:
            error_message = f"Error during Perplexity generation: {e}"
            self.error(error_message)
            trace_exception(e)
            if callback: callback(error_message, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            raise e # Re-raise other exceptions
        finally:
             generation_time = perf_counter() - start_time
             ASCIIColors.info(f"Generation process finished in {generation_time:.2f} seconds.")


        # --- Cost Estimation (Placeholder) ---
        if self.binding_config.config.turn_on_cost_estimation:
            # WARNING: These calculations are NOT accurate for Perplexity
            # Using placeholders based on token counts if available
            if total_input_tokens == 0: total_input_tokens = len(self.tokenize(prompt)) # Rough estimate if API didn't provide
            if total_output_tokens == 0: total_output_tokens = len(self.tokenize(output)) # Rough estimate

            self.binding_config.config["total_input_tokens"] += total_input_tokens
            self.binding_config.config["total_output_tokens"] += total_output_tokens
            input_cost_rate = INPUT_COSTS_PLACEHOLDER.get("default", 0)
            output_cost_rate = OUTPUT_COSTS_PLACEHOLDER.get("default", 0)
            input_cost = total_input_tokens * input_cost_rate
            output_cost = total_output_tokens * output_cost_rate
            self.binding_config.config["total_input_cost"] += input_cost
            self.binding_config.config["total_output_cost"] += output_cost
            self.binding_config.config["total_cost"] = self.binding_config.config["total_input_cost"] + self.binding_config.config["total_output_cost"]

            cost_info = f'Accumulated cost (PLACEHOLDER ESTIMATE): ${self.binding_config.config["total_cost"]:.6f}'
            self.info(cost_info)
            self.binding_config.save()

        return output

    def list_models(self) -> List[str]:
        """
        Lists the available models hardcoded for Perplexity.

        Returns:
            A list of model ID strings.
        """
        # Check if key is present, otherwise return empty list? Or allow listing anyway?
        # Let's allow listing even without a key for UI purposes.
        return self.available_models

    def get_available_models(self, app: Optional[LoLLMsCom] = None) -> List[dict]:
        """
        Gets the available models list formatted for the LoLLMs UI.

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

        for model_id, model_data in PERPLEXITY_MODELS.items():
            ctx = model_data.get("context_window", self.binding_config.config.ctx_size) # Use known or default
            max_out = model_data.get("max_output", self.binding_config.config.max_n_predict)
            rank = model_data.get("rank", 2.0) # Default rank
            category = model_data.get("category", "chat") # chat or online_chat

            # --- Build Lollms Model Dictionary ---
            model_entry = {
                **base_metadata,
                "name": model_id,
                "display_name": model_id.replace("-", " ").title(),
                "category": category,
                "rank": rank,
                "description": f"Perplexity API model. Context: {ctx}. Max Output: {max_out}. {'Includes Web Search.' if 'online' in category else ''}",
                "ctx_size": ctx,
                "max_n_predict": max_out, # Add max output info if available
                "variants": [{"name": model_id, "size": ctx}] # Use context as size proxy
            }
            models_info.append(model_entry)

        # Sort models: by rank (desc), then by name (asc)
        models_info.sort(key=lambda x: (-x['rank'], x['name']))

        ASCIIColors.success(f"Formatted {len(models_info)} Perplexity models for Lollms UI.")
        return models_info

    # Simple SSE Client (adapted from various sources)
    # Needs refinement for robustness, especially error handling and reconnection logic
    class sse_client:
        def __init__(self, response):
            self._response = response
            self._chunk_decoder = json.JSONDecoder() # For parsing potential JSON within data

        def _iter_sse_lines(self):
             """Iterates over SSE lines from the response."""
             buffer = ""
             for chunk in self._response.iter_content(chunk_size=1024, decode_unicode=True):
                 buffer += chunk
                 while '\n\n' in buffer:
                     message, buffer = buffer.split('\n\n', 1)
                     yield message.splitlines() # Yield list of lines in the message block

        def events(self):
            """Yields ServerSentEvent objects."""
            from collections import namedtuple
            ServerSentEvent = namedtuple('ServerSentEvent', ['event', 'data', 'id', 'retry'])

            for lines in self._iter_sse_lines():
                event_type = 'message'
                event_data = []
                event_id = None
                event_retry = None

                for line in lines:
                    if not line: continue # Skip empty lines within a block
                    if line.startswith(':'): continue # Skip comments

                    field, value = line.split(':', 1)
                    value = value.lstrip() # Remove leading space

                    if field == 'event': event_type = value
                    elif field == 'data': event_data.append(value)
                    elif field == 'id': event_id = value
                    elif field == 'retry': event_retry = int(value)
                    # Ignore unknown fields

                if not event_data: continue # Skip blocks without data

                yield ServerSentEvent(event=event_type, data='\n'.join(event_data), id=event_id, retry=event_retry)


# --- Main execution block for testing ---
if __name__ == "__main__":
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMsConfig
    from lollms.app import LollmsApplication
    from pathlib import Path
    from lollms.types import MSG_OPERATION_TYPE
    import sys

    print("Initializing LoLLMs environment for testing...")
    lollms_paths = LollmsPaths.find_paths(force_local=True, tool_prefix="test_perplexity_")
    config = LOLLMsConfig.autoload(lollms_paths)
    # Minimal app for communication channel
    try:
        lollms_app = LollmsApplication("TestApp", config, lollms_paths, load_bindings=False, load_personalities=False, load_models=False)
        lollms_com = lollms_app.com
    except Exception as e:
        print(f"Couldn't instantiate LollmsApplication (might be normal in CI): {e}")
        lollms_com = None # Fallback if app fails

    print("Creating Perplexity binding instance...")
    pplx = Perplexity(config, lollms_paths, installation_option=InstallOption.INSTALL_IF_NECESSARY, lollmsCom=lollms_com)

    # --- API Key Setup ---
    if not os.getenv('PERPLEXITY_API_KEY') and not pplx.binding_config.config.get("perplexity_key"):
         try:
             # Use input only if running interactively
             if sys.stdin.isatty():
                 key_input = input("Enter Perplexity API Key for testing: ").strip()
                 if key_input:
                     pplx.binding_config.config["perplexity_key"] = key_input
                     print("API Key set for this session.")
                 else:
                      print("No API key provided. Tests requiring API access will fail.")
             else:
                 print("Running in non-interactive mode. Skipping API key input.")
                 print("Ensure PERPLEXITY_API_KEY environment variable is set for tests.")
         except EOFError:
             print("No API key input detected.")

    print("\nUpdating settings (initializes client)...")
    pplx.settings_updated() # This initializes the client

    available_models = pplx.list_models()
    if available_models:
        print(f"\nAvailable Perplexity Models (Hardcoded: {len(available_models)}):")
        for m in available_models: print(f"- {m}")

        # --- Test Setup ---
        # Select a common, capable model for testing
        test_model = "llama-3-sonar-large-32k-online" # Good default with search
        if test_model not in available_models:
             print(f"\nWarning: Preferred test model '{test_model}' not found. Trying 'llama-3-sonar-small-32k-online'.")
             test_model = "llama-3-sonar-small-32k-online"
             if test_model not in available_models:
                  print(f"\nWarning: Model '{test_model}' not found. Using first available model: {available_models[0] if available_models else 'None'}")
                  if available_models: test_model = available_models[0]
                  else: test_model = None

        if test_model and pplx.perplexity_key: # Only run tests if model selected and key available
             print(f"\nSelecting model for testing: {test_model}")
             pplx.config.model_name = test_model
             pplx.build_model() # Finalize setup for the selected model
             print(f"Effective limits for {test_model}: Context={pplx.config.ctx_size}, Max Output={pplx.config.max_n_predict}")

             # Define a callback function for testing
             def print_callback(chunk: str, msg_type: int, metadata: dict) -> bool:
                 type_str = MSG_OPERATION_TYPE(msg_type).name if msg_type in MSG_OPERATION_TYPE.__members__.values() else f"UNKNOWN({msg_type})"

                 if msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK:
                     print(chunk, end="", flush=True)
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_FULL_ANSWER:
                     print(f"\n--- [{type_str}] Full Response ---")
                     print(chunk)
                     print(f"Metadata: {metadata}")
                     print("--- End Full Response ---")
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_STEP_END:
                      print(f"\n--- [{type_str}] Generation Step End ---")
                      print(f"Metadata: {metadata}")
                      print("--- End Step End ---")
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION:
                     print(f"\n## [{type_str}] EXCEPTION: {chunk} ##")
                     return False # Stop on error
                 elif msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_INFO or msg_type == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING:
                      print(f"\n## [{type_str}] INFO/WARN: {chunk} ##")
                 #else:
                 #    print(f"\n--- [{type_str}] ---")
                 #    print(f"Chunk: {chunk}")
                 #    print(f"Metadata: {metadata}")
                 #    print(f"--- End [{type_str}] ---")
                 return True

             # --- Test Case 1: Standard Chat (Streaming) ---
             print("\n" + "="*20 + " Test 1: Standard Chat (Streaming) " + "="*20)
             # Ensure search options are default or disabled if testing non-online model
             # pplx.binding_config.config["use_search_options"] = False # Example
             prompt1 = "Explain the concept of a Large Language Model (LLM) in one short paragraph."
             print(f"Prompt: {prompt1}")
             print("Response Stream:")
             try:
                 response1 = pplx.generate(prompt1, n_predict=150, callback=print_callback, verbose=True)
                 print("\n-- Test 1 Complete --")
             except Exception as e: print(f"\nTest 1 failed: {e}"); trace_exception(e)

             # --- Test Case 2: Test Search Parameters (Streaming, for online model) ---
             print("\n" + "="*20 + " Test 2: Online Model with Search Params " + "="*20)
             if "online" in test_model:
                 pplx.binding_config.config["use_search_options"] = True
                 pplx.binding_config.config["search_recency_filter"] = "week"
                 pplx.binding_config.config["return_related_questions"] = True
                 print(f"Configured search: Recency='{pplx.binding_config.config['search_recency_filter']}', RelatedQ={pplx.binding_config.config['return_related_questions']}")
                 prompt2 = "What notable AI developments happened recently?"
                 print(f"Prompt: {prompt2}")
                 print("Response Stream:")
                 try:
                     response2 = pplx.generate(prompt2, n_predict=300, callback=print_callback, verbose=True)
                     print("\n-- Test 2 Complete --")
                 except Exception as e: print(f"\nTest 2 failed: {e}"); trace_exception(e)
                 finally:
                     # Reset test params
                     pplx.binding_config.config["search_recency_filter"] = ""
                     pplx.binding_config.config["return_related_questions"] = False
             else:
                  print(f"-- Test 2 Skipped (Model '{test_model}' is not an 'online' model) --")

             # --- Test Case 3: Non-Streaming Request ---
             print("\n" + "="*20 + " Test 3: Non-Streaming Request " + "="*20)
             prompt3 = "Write a two-sentence horror story."
             print(f"Prompt: {prompt3}")
             print("Response (Full):")
             try:
                 # Pass callback=None to force non-streaming
                 response3 = pplx.generate(prompt3, n_predict=100, callback=None, verbose=True)
                 print(response3) # Print the returned full response
                 print("\n-- Test 3 Complete --")
             except Exception as e: print(f"\nTest 3 failed: {e}"); trace_exception(e)

        elif not pplx.perplexity_key:
             print("\nSkipping generation tests as Perplexity API key is missing.")
        else:
             print("\nSkipping generation tests as no suitable model could be selected.")
    else:
        print("\nCould not retrieve model list (hardcoded list empty?).")
        print("Skipping tests.")

    print("\nScript finished.")