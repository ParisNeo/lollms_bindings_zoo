######
# Project       : lollms
# File          : binding.py
# Author        : ParisNeo with the help of the community
# Underlying
# engine author : Google
# license       : Apache 2.0
# Description   :
# This is an interface class for lollms bindings.
# This binding is a wrapper to Google's Gemini API
######

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from lollms.binding import BindingType, LLMBinding, LOLLMSConfig
from lollms.com import LoLLMsCom
from lollms.config import BaseConfig, ConfigTemplate, InstallOption, TypedConfig
from lollms.helpers import ASCIIColors
from lollms.paths import LollmsPaths
from lollms.types import MSG_OPERATION_TYPE
from lollms.utilities import (
    find_first_available_file_path,
    is_file_path,
    trace_exception,
)

import pipmaster as pm
if not pm.is_installed("pillow"):
    pm.install("pillow")

from PIL import Image

# Ensure the required library is installed
if not pm.is_installed("google-generativeai"):
    pm.install("google-generativeai")

try:
    import google.generativeai as genai
    import google.api_core.exceptions
    from google.generativeai.types import GenerationConfig, ContentDict, PartDict
except ImportError:
    print("google-generativeai library not found.")
    print("Please install it using: pip install google-generativeai")
    genai = None  # Keep track that the import failed

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, ParisNeo"
__license__ = "Apache 2.0"

binding_name = "Gemini"
binding_folder_name = ""  # Keep empty if the folder name is the same as binding_name


class Gemini(LLMBinding):
    """
    A binding class for interacting with Google's Gemini API.

    This class handles configuration, model building, text generation,
    image-based generation, token counting, embedding, and model listing
    provided by the Google Generative AI SDK.
    """

    # --- Class Constants ---
    SAFETY_CATEGORIES = {
        "HARM_CATEGORY_HARASSMENT": "Harassment",
        "HARM_CATEGORY_HATE_SPEECH": "Hate Speech",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "Sexually Explicit",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "Dangerous Content",
    }
    SAFETY_OPTIONS = [
        "BLOCK_NONE",
        "BLOCK_ONLY_HIGH",
        "BLOCK_MEDIUM_AND_ABOVE",
        "BLOCK_LOW_AND_ABOVE",
    ]
    DEFAULT_SAFETY_SETTING = "BLOCK_MEDIUM_AND_ABOVE"
    EMBEDDING_MODEL_NAME = "models/text-embedding-004"

    def __init__(
        self,
        config: LOLLMSConfig,
        lollms_paths: Optional[LollmsPaths] = None,
        installation_option: InstallOption = InstallOption.INSTALL_IF_NECESSARY,
        lollmsCom: Optional[LoLLMsCom] = None,
    ) -> None:
        """
        Initializes the Gemini binding.

        Args:
            config: The main LOLLMS configuration object.
            lollms_paths: Paths configuration for LOLLMS.
            installation_option: Installation option (not directly used here).
            lollmsCom: Communication object with the LoLLMs application server.
        """
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        # Ensure genai is available before proceeding
        if genai is None:
            raise ImportError(
                "The 'google-generativeai' package is required but could not be imported. "
                "Please install it using: pip install google-generativeai"
            )

        # --- Configuration Template ---
        binding_config_template = ConfigTemplate([
            {"name":"google_api_key", "type":"str", "value":"", "help":"Your Google Generative AI API key. Get one from https://ai.google.dev/"},
            {"name": "auto_detect_limits", "type": "bool", "value": True, "help": "Auto-detect model context size and max output tokens. If false, uses manual values below.", "requires_restart": True},
            {"name":"ctx_size", "type":"int", "value":30720, "min":512, "help":"Context size (tokens). Auto-updated if 'auto_detect_limits' is on.", "read_only":True}, # Read-only controlled by auto_detect
            {"name":"max_output_tokens", "type":"int", "value":2048, "min":1, "help":"Max tokens per response. Auto-updated if 'auto_detect_limits' is on.", "read_only":True}, # Read-only controlled by auto_detect
            {"name":"seed", "type":"int", "value":-1, "help":"Random seed (-1 for random). Note: API support varies."},
            # Dynamically add safety settings
            *[
                {
                    "name": f"safety_setting_{cat.split('_')[-1].lower()}",
                    "type": "str",
                    "value": self.DEFAULT_SAFETY_SETTING,
                    "options": self.SAFETY_OPTIONS,
                    "help": f"Safety setting for {display_name} content.",
                }
                for cat, display_name in self.SAFETY_CATEGORIES.items()
            ],
        ])

        # --- Default Configuration ---
        binding_config_defaults = BaseConfig(config={
            "google_api_key": "",
            "auto_detect_limits": True,
            "ctx_size": 30720,        # Fallback default
            "max_output_tokens": 2048,# Fallback default
            "seed": -1,
            **{ # Dynamically generate default safety settings
                f"safety_setting_{cat.split('_')[-1].lower()}": self.DEFAULT_SAFETY_SETTING
                for cat in self.SAFETY_CATEGORIES
            }
        })

        # --- Initialize TypedConfig and Superclass ---
        binding_config = TypedConfig(binding_config_template, binding_config_defaults)
        super().__init__(
            Path(__file__).parent,
            lollms_paths,
            config,
            binding_config,
            installation_option,
            supported_file_extensions=['.png', '.jpg', '.jpeg', '.webp'], # Common vision formats
            lollmsCom=lollmsCom,
        )

        # --- Instance Attributes ---
        self.model: Optional[genai.GenerativeModel] = None
        self.genai = genai # Store the imported module
        self.safety_settings: Optional[dict[str, str]] = None
        self._stop_generation_requested = False
        self._is_generating = False

        # --- Initial Setup ---
        self.settings_updated()

    def settings_updated(self) -> None:
        """
        Called when binding settings are updated. Configures the API client,
        safety settings, and rebuilds the model.
        """
        if not self.binding_config.google_api_key:
            self.error("Google API key is missing. Please configure it in the binding settings.")
            if self.lollmsCom:
                self.lollmsCom.InfoMessage("Gemini Binding Error: Google API Key is missing.")
            self.model = None
            return

        try:
            self.genai.configure(api_key=self.binding_config.google_api_key)
            ASCIIColors.info("Google API key configured.")

            # Build safety settings dynamically from config
            self.safety_settings = {
                cat: getattr(self.binding_config, f"safety_setting_{cat.split('_')[-1].lower()}")
                for cat in self.SAFETY_CATEGORIES
            }
            ASCIIColors.debug(f"Safety settings configured: {self.safety_settings}")

            # Rebuild model, which also updates context/output limits based on detection setting
            self.build_model()

        except Exception as e:
            self.error(f"Failed to configure Google API or build model: {e}")
            trace_exception(e)
            if self.lollmsCom:
                self.lollmsCom.InfoMessage(f"Gemini Binding Error: Failed to configure API or build model.\nDetails: {e}")
            self.model = None
            # Reflect potential failure state in main config
            self.config.ctx_size = self.binding_config.ctx_size
            self.config.max_n_predict = self.binding_config.max_output_tokens

    def build_model(self, model_name: Optional[str] = None) -> 'Gemini':
        """
        Builds or rebuilds the generative model instance based on the configuration.

        Fetches model information to potentially set context size and max output tokens
        if 'auto_detect_limits' is enabled. Updates binding and main LOLLMS configs.

        Args:
            model_name: Optional model name to override the one in the config.

        Returns:
            The binding instance.
        """
        super().build_model(model_name) # Handles model name update in self.config

        if not self.binding_config.google_api_key:
            self.error("Cannot build model: Google API key not set.")
            self.model = None
            return self

        current_model_name = self.config.model_name
        if not current_model_name:
            self.warning("No model name selected in configuration. Cannot build model.")
            self.model = None
            return self

        ASCIIColors.info(f"Attempting to build Gemini model: {current_model_name}")

        try:
            # Ensure API is configured (redundant if called after settings_updated, but safe)
            self.genai.configure(api_key=self.binding_config.google_api_key)

            # --- Fetch Model Information ---
            actual_model_ctx_limit: Optional[int] = None
            actual_model_output_limit: Optional[int] = None
            model_supports_vision: bool = False

            try:
                model_info = self.genai.get_model(f"models/{current_model_name}")
                actual_model_ctx_limit = getattr(model_info, 'input_token_limit', None)
                actual_model_output_limit = getattr(model_info, 'output_token_limit', None)

                # Check vision support (simplified: check for vision in name/display or 1.5+)
                is_vision_candidate = ('vision' in current_model_name.lower() or
                                       'Vision' in getattr(model_info, 'display_name', '') or
                                       "1.5" in current_model_name.lower())
                # Vision models might also have 'embedContent' for multimodal input
                if is_vision_candidate and 'embedContent' in getattr(model_info, 'supported_generation_methods', []):
                    model_supports_vision = True

                if actual_model_ctx_limit and actual_model_output_limit:
                    ASCIIColors.success(f"Detected limits for {current_model_name}: "
                                        f"ctx={actual_model_ctx_limit}, max_out={actual_model_output_limit}")
                else:
                    self.warning(f"Could not automatically retrieve token limits for {current_model_name}.")

            except google.api_core.exceptions.NotFound:
                 self.error(f"Model 'models/{current_model_name}' not found via API.")
                 self.model = None
                 return self
            except Exception as info_err:
                self.warning(f"Could not retrieve detailed info for {current_model_name}. Using defaults/manual settings if applicable. Error: {info_err}")
                # Fallback vision check based on name if API info failed
                if ('vision' in current_model_name.lower() or "1.5" in current_model_name.lower()):
                     model_supports_vision = True

            # --- Determine Effective Limits ---
            effective_ctx_size = self.binding_config.ctx_size
            effective_max_output = self.binding_config.max_output_tokens
            update_config_read_only = False

            if self.binding_config.auto_detect_limits:
                ASCIIColors.info("Auto-detect limits enabled.")
                update_config_read_only = True
                if actual_model_ctx_limit and actual_model_output_limit:
                    effective_ctx_size = actual_model_ctx_limit
                    effective_max_output = actual_model_output_limit
                    ASCIIColors.info(f"Using auto-detected limits: ctx={effective_ctx_size}, max_out={effective_max_output}")
                else:
                    self.warning("Auto-detect enabled, but failed to fetch limits. Using fallback/previous values.")
            else:
                ASCIIColors.info("Auto-detect limits disabled. Using manually configured limits.")
                update_config_read_only = False
                # Warn if manual settings exceed detected limits
                if actual_model_ctx_limit and effective_ctx_size > actual_model_ctx_limit:
                    self.warning(f"Manual ctx_size ({effective_ctx_size}) exceeds detected limit ({actual_model_ctx_limit}).")
                if actual_model_output_limit and effective_max_output > actual_model_output_limit:
                     self.warning(f"Manual max_output ({effective_max_output}) exceeds detected limit ({actual_model_output_limit}).")

            # --- Update Configurations ---
            self.binding_config.config["ctx_size"] = effective_ctx_size
            self.binding_config.config["max_output_tokens"] = effective_max_output
            self.config.ctx_size = effective_ctx_size
            self.config.max_n_predict = effective_max_output # Keep max_n_predict aligned with max_output_tokens

            # Note: Updating read-only status in UI post-init might require LoLLMs core changes. Logging state for now.
            ASCIIColors.info(f"Context size and Max Output Tokens fields are {'read-only (auto-detected)' if update_config_read_only else 'editable (manual)'}.")

            # --- Update Binding Type ---
            self.binding_type = BindingType.TEXT_IMAGE if model_supports_vision else BindingType.TEXT_ONLY
            ASCIIColors.info(f"Binding type set to: {self.binding_type.name}")

            # --- Instantiate the Model ---
            ASCIIColors.info(f"Building Gemini model instance: {current_model_name}")
            self.model = self.genai.GenerativeModel(current_model_name)

            # --- Verify Model Operability ---
            try:
                self.model.count_tokens("hello") # Simple test
                ASCIIColors.success(f"Model '{current_model_name}' built and operational test passed.")
            except Exception as test_err:
                self.error(f"Model '{current_model_name}' failed operational test (token counting). Error: {test_err}")
                trace_exception(test_err)
                if self.lollmsCom:
                     self.lollmsCom.InfoMessage(f"Gemini Warning: Model '{current_model_name}' failed basic test. Check API key/permissions. Details: {test_err}")
                self.model = None

        except Exception as e:
            self.error(f"Failed to build Gemini model '{current_model_name}': {e}")
            trace_exception(e)
            if self.lollmsCom:
                self.lollmsCom.InfoMessage(f"Gemini Error: Failed to build model '{current_model_name}'. Details: {e}")
            self.model = None
            # Ensure main config reflects failure state or previous values
            self.config.ctx_size = self.binding_config.ctx_size
            self.config.max_n_predict = self.binding_config.max_output_tokens

        return self

    def count_tokens(self, prompt: str) -> int:
        """
        Counts the number of tokens in the prompt for the current model.

        Args:
            prompt: The text prompt to count tokens for.

        Returns:
            The total number of tokens, or -1 if counting fails.
        """
        if not self.model:
            self.error("Model not initialized. Cannot count tokens.")
            return -1
        if prompt=="":
            return 0
        try:
            token_count_response = self.model.count_tokens(prompt)
            return token_count_response.total_tokens
        except Exception as e:
            self.error(f"Failed to count tokens: {e}")
            trace_exception(e)
            return -1

    def embed(self, text: Union[str, List[str]], task_type: str = "retrieval_document") -> Optional[List[List[float]]]:
        """
        Computes text embeddings using a Google embedding model.

        Args:
            text: The text or list of texts to embed.
            task_type: The type of task (e.g., "retrieval_document").

        Returns:
            A list of embedding lists, or None if embedding fails.
        """
        if not self.genai:
            self.error("Gemini (genai) module not available for embedding.")
            return None
        if not self.binding_config.google_api_key:
             self.error("Google API key not configured for embedding.")
             return None

        try:
            # Ensure configuration is active
            self.genai.configure(api_key=self.binding_config.google_api_key)

            if isinstance(text, str):
                result = self.genai.embed_content(
                    model=self.EMBEDDING_MODEL_NAME, content=text, task_type=task_type
                )
                return [result['embedding']] if 'embedding' in result else None
            elif isinstance(text, list):
                # Prefer batch embedding if available and supported by the SDK version
                if hasattr(self.genai, 'batch_embed_contents'):
                    batch_result = self.genai.batch_embed_contents(
                        model=self.EMBEDDING_MODEL_NAME,
                        requests=[{'content': t, 'task_type': task_type} for t in text]
                    )
                    return [emb['values'] for emb in batch_result['embeddings']] if batch_result and 'embeddings' in batch_result else None
                else:
                    # Fallback to sequential embedding
                    self.warning("`batch_embed_contents` not found, using sequential embedding.")
                    embeddings = []
                    for item in text:
                        try:
                            single_result = self.genai.embed_content(model=self.EMBEDDING_MODEL_NAME, content=item, task_type=task_type)
                            embeddings.append(single_result['embedding'] if 'embedding' in single_result else None)
                        except Exception as item_e:
                            self.error(f"Error embedding item: {item[:50]}... Error: {item_e}")
                            embeddings.append(None)
                    return embeddings if any(e is not None for e in embeddings) else None
            else:
                 self.error(f"Invalid input type for embedding: {type(text)}. Expected str or list[str].")
                 return None

        except google.api_core.exceptions.PermissionDenied as e:
             self.error(f"Embedding failed: Permission denied. Check API key and Embedding API enablement. Error: {e}")
             return None
        except google.api_core.exceptions.InvalidArgument as e:
             self.error(f"Embedding failed: Invalid argument (check text or task type '{task_type}'). Error: {e}")
             return None
        except Exception as e:
            self.error(f"Failed to generate embeddings using {self.EMBEDDING_MODEL_NAME}: {e}")
            trace_exception(e)
            return None

    def _prepare_generation_config(self, n_predict: Optional[int], gpt_params: dict) -> Optional[GenerationConfig]:
        """
        Helper to build the GenerationConfig object, handling max tokens and other parameters.

        Args:
            n_predict: User-requested max tokens override.
            gpt_params: Dictionary of generation parameters (temperature, top_p, etc.).

        Returns:
            A GenerationConfig object or None if preparation fails.
        """
        if not self.model:
             self.error("Model not initialized, cannot prepare generation config.")
             return None

        # Determine max tokens for this call, respecting model limits
        model_max_tokens = self.binding_config.max_output_tokens
        generate_max_tokens = model_max_tokens

        if n_predict is not None:
            if 0 < n_predict <= model_max_tokens:
                generate_max_tokens = n_predict
                ASCIIColors.debug(f"Using user n_predict: {generate_max_tokens}")
            elif n_predict > model_max_tokens:
                 self.warning(f"Requested n_predict ({n_predict}) exceeds model limit ({model_max_tokens}). Capping.")
                 generate_max_tokens = model_max_tokens
            # else: Use default model_max_tokens if n_predict <= 0
        ASCIIColors.debug(f"Effective max_output_tokens for this call: {generate_max_tokens}")

        # Prepare generation parameters dictionary
        gen_config_params = {
            'candidate_count': 1,
            'max_output_tokens': generate_max_tokens,
            'temperature': float(gpt_params.get('temperature', 0.7)),
            'top_p': float(gpt_params.get('top_p', 0.95)),
            'top_k': int(gpt_params.get('top_k', 40)),
        }

        # Handle stop sequences carefully
        stop_sequences = gpt_params.get('stop_sequences') or gpt_params.get('stop')
        if stop_sequences:
            valid_stop_sequences = []
            if isinstance(stop_sequences, str):
                try: # Attempt to parse if it looks like a JSON list string
                    if stop_sequences.strip().startswith('[') and stop_sequences.strip().endswith(']'):
                         parsed = json.loads(stop_sequences)
                         if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
                             valid_stop_sequences = [s for s in parsed if s] # Filter empty strings
                         else:
                             self.warning("Parsed stop_sequences is not a list of strings. Using original string.")
                             if stop_sequences: valid_stop_sequences = [stop_sequences]
                    else: # Treat as a single sequence if not JSON list format
                        if stop_sequences: valid_stop_sequences = [stop_sequences]
                except json.JSONDecodeError:
                    self.warning(f"Could not parse stop_sequences string: '{stop_sequences}'. Using as single sequence.")
                    if stop_sequences: valid_stop_sequences = [stop_sequences]
            elif isinstance(stop_sequences, list) and all(isinstance(s, str) for s in stop_sequences):
                 valid_stop_sequences = [s for s in stop_sequences if s] # Filter empty
            else:
                 self.warning(f"Invalid format for stop_sequences: {stop_sequences}. Expected list of strings or JSON string list. Ignoring.")

            if valid_stop_sequences:
                 gen_config_params['stop_sequences'] = valid_stop_sequences
                 ASCIIColors.debug(f"Using stop sequences: {valid_stop_sequences}")

        # Seed handling (Note: Official SDK might not expose this in GenerationConfig)
        seed = gpt_params.get('seed', self.binding_config.seed)
        if seed is not None and seed != -1:
            # Note: Direct seed setting might not be supported via GenerationConfig.
            self.warning("Seed parameter provided, but direct support via SDK's GenerationConfig is uncertain/may not be available.")
            # gen_config_params['seed'] = seed # Uncomment if/when SDK supports it

        try:
            return GenerationConfig(**gen_config_params)
        except Exception as e:
            self.error(f"Failed to create GenerationConfig: {e}")
            trace_exception(e)
            return None

    def stop_generation(self) -> None:
        """
        Signals the binding to stop any ongoing generation stream.
        """
        if self._is_generating:
            self._stop_generation_requested = True
            ASCIIColors.info("Stop generation requested.")

    def _stream_generation(
        self,
        content: Union[str, List[Union[str, Image.Image]]],
        generation_config: GenerationConfig,
        callback: Optional[Callable[[str, int], bool]],
        verbose: bool = False,
    ) -> str:
        """
        Internal helper to handle the streaming logic for both text and vision generation.

        Args:
            content: The prompt string or list of content parts (text, images).
            generation_config: The prepared GenerationConfig object.
            callback: The streaming callback function.
            verbose: Whether to log verbose details.

        Returns:
            The full generated text, or an empty string if stopped or failed.
        """
        output = ""
        chunk_count = 0
        self._is_generating = True
        self._stop_generation_requested = False # Reset flag at start

        try:
            if verbose: ASCIIColors.info(f"Starting generation stream with config: {generation_config}")

            response_stream = self.model.generate_content(
                content,
                generation_config=generation_config,
                stream=True,
                safety_settings=self.safety_settings,
            )

            # --- Streaming Loop ---
            for chunk in response_stream:
                # --- Check for stop request ---
                if self._stop_generation_requested:
                    ASCIIColors.info("Generation stopped by external request.")
                    break

                # --- Check for API blocking reasons ---
                block_reason = None
                finish_reason_str = None

                # Prompt feedback (blocking before generation starts)
                if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                    block_reason = chunk.prompt_feedback.block_reason.name
                    self.error(f"Generation blocked by API (prompt feedback): {block_reason}")
                    if self.lollmsCom: self.lollmsCom.InfoMessage(f"Generation blocked by API. Reason: {block_reason}. Check prompt/safety settings.")
                    break

                # Candidate finish reason (blocking or completion during generation)
                if chunk.candidates:
                     candidate = chunk.candidates[0]
                     if candidate.finish_reason:
                         finish_reason = candidate.finish_reason
                         finish_reason_str = finish_reason.name
                         if finish_reason_str not in ["STOP", "MAX_TOKENS", "UNSPECIFIED"]: # Non-standard stops
                              self.warning(f"Generation potentially stopped by API. Finish Reason: {finish_reason_str}")
                              if finish_reason_str == "SAFETY": block_reason = finish_reason_str
                              if self.lollmsCom: self.lollmsCom.InfoMessage(f"Generation stopped/flagged by API. Reason: {finish_reason_str}")
                         if verbose: ASCIIColors.debug(f"Chunk finish reason: {finish_reason_str}")

                # --- Extract Text ---
                word = ""
                try:
                    if hasattr(chunk, 'text'): word = chunk.text
                except Exception as e:
                    self.error(f"Error accessing chunk text content: {e}. Chunk: {chunk}")

                # --- Process Text Chunk ---
                if word:
                    output += word
                    chunk_count += 1
                    if callback is not None:
                        if not callback(word, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                            ASCIIColors.info("Generation stopped by callback.")
                            self._stop_generation_requested = True # Ensure loop exit
                            break

                # If a blocking reason was found, stop after processing text in the current chunk
                if block_reason:
                    self.error(f"Stopping generation loop due to block reason: {block_reason}")
                    break

            # --- End Streaming Loop ---
            if verbose: ASCIIColors.success(f"Stream ended. Received {chunk_count} chunks.")

        # --- Exception Handling ---
        except google.api_core.exceptions.PermissionDenied as e:
             self.error(f"Permission Denied: {e}. Check API key/permissions.")
             if self.lollmsCom: self.lollmsCom.InfoMessage(f"Gemini API Error: Permission Denied.\nDetails: {e}")
             output = "" # Return empty on error
        except google.api_core.exceptions.ResourceExhausted as e:
             self.error(f"Resource Exhausted: {e}. Rate limits or quota likely hit.")
             if self.lollmsCom: self.lollmsCom.InfoMessage(f"Gemini API Error: Resource Exhausted (Quota/Rate Limit).\nDetails: {e}")
             output = ""
        except google.api_core.exceptions.InvalidArgument as e:
             self.error(f"Invalid Argument: {e}. Check prompt/image/parameters/safety settings.")
             if self.lollmsCom: self.lollmsCom.InfoMessage(f"Gemini API Error: Invalid Argument.\nDetails: {e}")
             output = ""
        except Exception as e:
            self.error(f"Gemini generation failed unexpectedly: {e}")
            trace_exception(e)
            if self.lollmsCom: self.lollmsCom.InfoMessage(f"Gemini Generation Error: Unexpected error.\nDetails: {e}")
            output = ""
        finally:
            self._is_generating = False
            self._stop_generation_requested = False # Reset flag

        return output

    def generate(
        self,
        prompt: str,
        n_predict: Optional[int] = None,
        callback: Optional[Callable[[str, int], bool]] = None,
        verbose: bool = False,
        **gpt_params,
    ) -> str:
        """
        Generates text based on a prompt using the configured Gemini model.

        Args:
            prompt: The text prompt for generation.
            n_predict: Overrides the effective max output tokens for this call.
            callback: Callback function for streaming tokens (token, type). Return False to stop.
            verbose: If true, logs more details.
            **gpt_params: Additional parameters (temperature, top_p, top_k, stop_sequences, seed).

        Returns:
            The generated text, or an empty string if generation fails or is stopped.
        """
        if not self.model:
            self.error("Model not initialized. Attempting recovery...")
            self.settings_updated()
            if not self.model:
                 self.error("Failed to recover model. Aborting generation.")
                 if self.lollmsCom: self.lollmsCom.InfoMessage("Gemini Error: Model not available.")
                 return ""

        # Prepare generation configuration
        generation_config = self._prepare_generation_config(n_predict, gpt_params)
        if generation_config is None:
            self.error("Failed to prepare generation config. Aborting.")
            if self.lollmsCom: self.lollmsCom.InfoMessage("Gemini Error: Could not create generation settings.")
            return ""

        # Use the internal streaming helper
        return self._stream_generation(prompt, generation_config, callback, verbose)


    def generate_with_images(
        self,
        prompt: str,
        images: List[str],
        n_predict: Optional[int] = None,
        callback: Optional[Callable[[str, int], bool]] = None,
        verbose: bool = False,
        **gpt_params,
    ) -> str:
        """
        Generates text based on a prompt and images using a Gemini vision model.

        Args:
            prompt: The text prompt.
            images: List of file paths to images.
            n_predict: Overrides effective max output tokens.
            callback: Streaming callback function.
            verbose: Verbose logging.
            **gpt_params: Generation parameters.

        Returns:
            Generated text or empty string on failure/stop.
        """
        if not self.model:
            self.error("Model not initialized. Attempting recovery...")
            self.settings_updated()
            if not self.model:
                 self.error("Failed to recover model. Aborting generation.")
                 if self.lollmsCom: self.lollmsCom.InfoMessage("Gemini Error: Model not available.")
                 return ""

        if self.binding_type != BindingType.TEXT_IMAGE:
            self.error(f"Model '{self.config.model_name}' doesn't support image input.")
            if self.lollmsCom:
                 self.lollmsCom.InfoMessage(f"Gemini Error: Model '{self.config.model_name}' is not vision-capable.")
            return ""

        if not images:
             self.warning("generate_with_images called with no images. Falling back to text generation.")
             return self.generate(prompt, n_predict, callback, verbose, **gpt_params)

        # --- Prepare generation configuration (consider vision defaults) ---
        vision_defaults = {'temperature': 0.4, 'top_k': 32} # Example vision defaults
        merged_params = {**vision_defaults, **gpt_params}
        generation_config = self._prepare_generation_config(n_predict, merged_params)
        if generation_config is None:
            self.error("Failed to prepare generation config for vision. Aborting.")
            if self.lollmsCom: self.lollmsCom.InfoMessage("Gemini Error: Could not create vision generation settings.")
            return ""

        # --- Prepare Content List [prompt, image1_PIL, image2_PIL, ...] ---
        content_parts: List[Union[str, Image.Image]] = [prompt]
        loaded_images: List[Image.Image] = [] # Keep track to close them later
        successful_image_loads = 0

        for image_path_str in images:
            img = None # Ensure img is None if loading fails
            try:
                # Validate path and find the actual file
                image_path = Path(image_path_str)
                valid_image_path = find_first_available_file_path([image_path])

                if not valid_image_path or not is_file_path(valid_image_path) or not valid_image_path.exists():
                     self.warning(f"Image path not found or invalid: {image_path_str}. Skipping.")
                     if callback: callback(f"\nWarning: Image not found {image_path_str}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING)
                     continue

                img = Image.open(valid_image_path)
                # img.verify() # Optional: Check image integrity
                # img = Image.open(valid_image_path) # Reopen after verify
                # Optional: Convert format if needed, though Gemini handles common ones
                # if img.mode not in ['RGB', 'RGBA']: img = img.convert('RGB')
                loaded_images.append(img) # Add to list for later cleanup
                content_parts.append(img) # Add PIL image directly
                successful_image_loads += 1
                ASCIIColors.info(f"Loaded image: {valid_image_path}")

            except FileNotFoundError: # Should be caught by exists(), but as fallback
                 self.warning(f"Image file not found (during open): {image_path_str}. Skipping.")
                 if callback: callback(f"\nWarning: Image not found {image_path_str}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING)
            except Exception as e:
                self.error(f"Failed to load/process image {image_path_str}: {e}")
                trace_exception(e)
                if callback: callback(f"\nWarning: Failed to load image {image_path_str}. Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING)
                if img: # Close if opened but failed later
                    try: img.close()
                    except Exception: pass # Ignore close error if already failed

        if successful_image_loads == 0:
             self.error("All provided images failed to load. Aborting vision generation.")
             if self.lollmsCom: self.lollmsCom.InfoMessage("Gemini Error: Failed to load any images.")
             # Ensure any partially loaded but failed images are closed
             for img_obj in loaded_images:
                 try: img_obj.close()
                 except Exception: pass
             return ""
        elif successful_image_loads < len(images):
             self.warning("Some images failed to load. Proceeding with successful ones.")

        # --- Use the internal streaming helper ---
        output = ""
        try:
            output = self._stream_generation(content_parts, generation_config, callback, verbose)
        finally:
            # --- Clean up loaded PIL Image objects ---
            ASCIIColors.info(f"Closing {len(loaded_images)} loaded image objects.")
            closed_count = 0
            for img_obj in loaded_images:
                try:
                    img_obj.close()
                    closed_count += 1
                except Exception as close_err:
                    self.warning(f"Could not close an image object: {close_err}")
            if closed_count != len(loaded_images):
                self.warning(f"Only closed {closed_count}/{len(loaded_images)} image objects.")

        return output

    def list_models(self) -> List[str]:
        """
        Lists the names (IDs) of available Gemini models supporting content generation.

        Returns:
            A list of model IDs, or an empty list if listing fails.
        """
        if not self.genai:
            self.error("Gemini (genai) module not available for listing models.")
            return []
        if not self.binding_config.google_api_key:
             self.error("Google API key not configured for listing models.")
             return []

        try:
            self.genai.configure(api_key=self.binding_config.google_api_key)
            ASCIIColors.info("Fetching list of available Gemini models from API...")
            api_models = self.genai.list_models()

            supported_models = [
                m.name.split('/')[-1] for m in api_models
                if 'generateContent' in getattr(m, 'supported_generation_methods', [])
            ]

            ASCIIColors.info(f"Found {len(supported_models)} models supporting 'generateContent'.")
            supported_models.sort()
            return supported_models

        except google.api_core.exceptions.PermissionDenied:
             self.error("Permission Denied listing models. Check API key/enablement.")
             return []
        except Exception as e:
            self.error(f"Failed to list Gemini models from API: {e}")
            trace_exception(e)
            return []

    def get_available_models(self, app: Optional[LoLLMsCom] = None) -> List[dict]:
        """
        Gets a list of available Gemini models formatted for the Lollms UI,
        including fetched metadata.

        Args:
            app: The LoLLMsCom object (optional, uses self.lollmsCom if available).

        Returns:
            A list of model dictionaries compatible with the Lollms UI format.
        """
        lollms_com = self.lollmsCom or app
        if not self.genai:
            self.error("Gemini (genai) module not available. Cannot get models.")
            return []
        if not self.binding_config.google_api_key:
             self.error("Google API key not configured. Cannot get models.")
             if lollms_com: lollms_com.InfoMessage("Gemini Error: Cannot fetch models. API Key missing.")
             return []

        lollms_models = []
        try:
            self.genai.configure(api_key=self.binding_config.google_api_key)
            ASCIIColors.info("Fetching detailed Gemini models list for Lollms UI...")
            api_models = self.genai.list_models()

            for model in api_models:
                # Only include models usable for generation
                if 'generateContent' not in getattr(model, 'supported_generation_methods', []):
                    continue

                model_id = model.name.split('/')[-1]
                display_name = getattr(model, 'display_name', model_id)
                description = getattr(model, 'description', 'N/A')
                input_limit = getattr(model, 'input_token_limit', 'N/A')
                output_limit = getattr(model, 'output_token_limit', 'N/A')

                # --- Determine Rank and Size Proxy ---
                rank = 1.0
                if "ultra" in model_id: rank = 3.0
                elif "pro" in model_id: rank = 2.0
                elif "flash" in model_id: rank = 1.5
                if "1.5" in model_id: rank += 0.5

                size_proxy = -1
                try:
                     if isinstance(input_limit, int) and isinstance(output_limit, int): size_proxy = input_limit + output_limit
                     elif isinstance(input_limit, int): size_proxy = input_limit
                except Exception: pass

                # --- Determine Category (Vision/Text) ---
                is_vision = (
                    ('vision' in model_id.lower() or 'Vision' in display_name or "1.5" in model_id.lower()) and
                    'embedContent' in getattr(model, 'supported_generation_methods', [])
                )
                category = "multimodal" if is_vision else "text"

                # --- Build Lollms Model Dictionary ---
                md = {
                    "category": category,
                    "datasets": "Proprietary Google",
                    "icon": '/bindings/gemini/logo.png',
                    "last_commit_time": datetime.now().timestamp(), # Placeholder
                    "license": "Commercial API",
                    "model_creator": "Google",
                    "model_creator_link": "https://ai.google.dev/",
                    "name": model_id,
                    "rank": rank,
                    "type": "api",
                    "variants": [{"name": model_id, "size": size_proxy}],
                    # Additional useful metadata
                    "description": description,
                    "display_name": display_name,
                    "input_token_limit": input_limit,
                    "output_token_limit": output_limit,
                    "supported_generation_methods": getattr(model, 'supported_generation_methods', []),
                }
                lollms_models.append(md)

            # Sort by rank (desc) then name (asc)
            lollms_models.sort(key=lambda x: (-x['rank'], x['name']))
            ASCIIColors.success(f"Formatted {len(lollms_models)} Gemini models for Lollms UI.")

        except google.api_core.exceptions.PermissionDenied:
             self.error("Permission Denied getting available models.")
             if lollms_com: lollms_com.InfoMessage("Gemini Error: Permission Denied fetching models.")
             return []
        except Exception as e:
            self.error(f"Failed to retrieve/process Gemini models for Lollms: {e}")
            trace_exception(e)
            if lollms_com: lollms_com.InfoMessage(f"Gemini Error: Failed to retrieve models.\nDetails: {e}")
            return []

        return lollms_models