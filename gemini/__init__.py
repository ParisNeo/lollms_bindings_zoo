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
import io
import json
import sys
import base64
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
    PackageManager,
    find_first_available_file_path,
    is_file_path,
    trace_exception,
)
from PIL import Image

# Ensure the required library is installed
if not PackageManager.check_package_installed("google-generativeai"):
    PackageManager.install_package("google-generativeai")

try:
    import google.generativeai as genai
    import google.api_core.exceptions
except ImportError:
    print("google-generativeai library not found.")
    print("Please install it using: pip install google-generativeai")
    # Optionally raise an error or sys.exit() if the binding cannot function without it.
    # raise ImportError("google-generativeai is required for the Gemini binding.") from None
    genai = None # Keep track that the import failed

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, ParisNeo"
__license__ = "Apache 2.0"

binding_name = "Gemini"
binding_folder_name = "" # Set to the folder name if different from binding_name

class Gemini(LLMBinding):

    # Class constants for safety categories and default settings
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

    def __init__(
        self,
        config: LOLLMSConfig,
        lollms_paths: Optional[LollmsPaths] = None,
        installation_option: InstallOption = InstallOption.INSTALL_IF_NECESSARY,
        lollmsCom: Optional[LoLLMsCom] = None,
    ) -> None:
        """
        Initialize the Binding.

        Args:
            config (LOLLMSConfig): The configuration object for LOLLMS.
            lollms_paths (LollmsPaths, optional): The paths object for LOLLMS. Defaults to LollmsPaths().
            installation_option (InstallOption, optional): The installation option for LOLLMS. (Not directly used here but part of standard signature).
            lollmsCom (LoLLMsCom, optional): Communication object with the LoLLMs application server.
        """
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        # Ensure genai is available before proceeding
        if genai is None:
            raise ImportError(
                "The 'google-generativeai' package is required but could not be imported. "
                "Please install it using: pip install google-generativeai"
            )

        self.lollmsCom = lollmsCom

        # Define the configuration template
        binding_config_template = ConfigTemplate([
            {"name":"google_api_key", "type":"str", "value":"", "help":"Your Google Generative AI API key. Get one from https://ai.google.dev/"},
            {"name": "auto_detect_limits", "type": "bool", "value": True, "help": "Automatically detect and use the selected model's context size and max output tokens. If false, uses the manually set values below (these fields will still display the detected limits for reference).", "requires_restart": True},
            {"name":"ctx_size", "type":"int", "value":30720, "min":512, "help":"Context size (in tokens). Automatically updated based on the selected model if 'auto_detect_limits' is enabled. Default shown is for Gemini 1.0 Pro.", "read_only":True}, # Read-only controlled by auto_detect
            {"name":"max_output_tokens", "type":"int", "value":2048, "min":1, "help":"Maximum number of tokens to generate per response. Automatically updated based on the selected model if 'auto_detect_limits' is enabled. Default shown is for Gemini 1.0 Pro.", "read_only":True}, # Read-only controlled by auto_detect
            {"name":"seed", "type":"int", "value":-1, "help":"Random seed for generation. -1 for random. Note: Seed support might vary or not be directly exposed in all API versions/SDK methods."},
            # Dynamic Safety Settings based on constants
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

        # Define the default configuration
        binding_config_defaults = BaseConfig(config={
            "google_api_key": "",
            "auto_detect_limits": True,
            "ctx_size": 30720,        # Fallback if detection fails or is off
            "max_output_tokens": 2048,# Fallback if detection fails or is off
            "seed": -1,
            **{ # Dynamically generate default safety settings
                f"safety_setting_{cat.split('_')[-1].lower()}": self.DEFAULT_SAFETY_SETTING
                for cat in self.SAFETY_CATEGORIES
            }
        })

        # Initialize TypedConfig
        binding_config = TypedConfig(
            binding_config_template,
            binding_config_defaults,
        )

        super().__init__(
            Path(__file__).parent,
            lollms_paths,
            config,
            binding_config,
            installation_option,
            supported_file_extensions=['.png', '.jpg', '.jpeg', '.webp'],
            lollmsCom=lollmsCom,
        )

        self.model: Optional[genai.GenerativeModel] = None
        self.genai = genai # Store the imported module
        self.safety_settings: Optional[dict] = None

        # Initial configuration and model build attempt
        self.settings_updated()


    def settings_updated(self):
        """Called when the settings are updated. Configures the API client and rebuilds the model."""
        if not self.binding_config.google_api_key:
            self.error("No Google API key is set! Please set it in the binding configuration.")
            # Potentially use InfoMessage here if lollmsCom is available and setup is complete
            if self.lollmsCom:
                self.lollmsCom.InfoMessage("Gemini Binding Error: Google API Key is missing. Please configure it in the binding settings.")
            self.model = None
            return # Stop further processing if key is missing

        try:
            self.genai.configure(api_key=self.binding_config.google_api_key)
            ASCIIColors.info("Google API key configured.")

            # Build safety settings dynamically from config
            self.safety_settings = {
                cat: getattr(self.binding_config, f"safety_setting_{cat.split('_')[-1].lower()}")
                for cat in self.SAFETY_CATEGORIES
            }
            ASCIIColors.info(f"Safety settings configured: {self.safety_settings}")

            # Rebuild model, which now considers auto_detect_limits and updates configs
            self.build_model() # This will also update self.config.ctx_size/max_n_predict

        except Exception as e:
            self.error(f"Failed to configure Google API or build model: {e}")
            trace_exception(e)
            if self.lollmsCom:
                self.lollmsCom.InfoMessage(f"Gemini Binding Error: Failed to configure API or build model.\nCheck your API key and network connection.\nDetails: {e}")
            self.model = None
            # Ensure main config reflects the potentially failed state or previous binding config values
            self.config.ctx_size = self.binding_config.ctx_size
            self.config.max_n_predict = self.binding_config.max_output_tokens


    def build_model(self, model_name: Optional[str] = None):
        """
        Builds the generative model based on the configuration.
        Fetches model information to set context size and max output tokens
        based on the 'auto_detect_limits' setting. Updates binding and lollms configs.
        """
        super().build_model(model_name) # Handles model name update in self.config

        if not self.binding_config.google_api_key:
            self.error("Cannot build model: Google API key not set.")
            self.model = None
            # No InfoMessage here as settings_updated handles the missing key message
            return self

        current_model_name = self.config.model_name
        if not current_model_name:
            self.warning("No model name selected in configuration. Cannot build model yet.")
            self.model = None
            return self

        try:
            # Ensure API is configured (redundant if called after settings_updated, but safe)
            self.genai.configure(api_key=self.binding_config.google_api_key)

            # --- Fetch Model Information ---
            fetched_ctx_size: Optional[int] = None
            fetched_max_output: Optional[int] = None
            model_supports_vision: bool = False
            actual_model_ctx_limit: Optional[int] = None
            actual_model_output_limit: Optional[int] = None

            try:
                ASCIIColors.info(f"Fetching details for model: models/{current_model_name}")
                model_info = self.genai.get_model(f"models/{current_model_name}")

                actual_model_ctx_limit = getattr(model_info, 'input_token_limit', None)
                actual_model_output_limit = getattr(model_info, 'output_token_limit', None)

                # Check for vision support based on methods or name patterns
                if 'embedContent' in model_info.supported_generation_methods and ('Vision' in model_info.display_name or 'vision' in current_model_name.lower() or "gemini-1.5" in current_model_name.lower()):
                     model_supports_vision = True

                if actual_model_ctx_limit and actual_model_output_limit:
                    ASCIIColors.success(f"Detected limits for {current_model_name}:")
                    ASCIIColors.info(f"  Input Token Limit (ctx_size): {actual_model_ctx_limit}")
                    ASCIIColors.info(f"  Output Token Limit (max_output_tokens): {actual_model_output_limit}")
                    fetched_ctx_size = actual_model_ctx_limit
                    fetched_max_output = actual_model_output_limit
                else:
                    self.warning(f"Could not automatically retrieve token limits for model {current_model_name}. API might not provide them.")

            except google.api_core.exceptions.NotFound:
                 self.error(f"Model 'models/{current_model_name}' not found via API. Please check the model name.")
                 self.model = None
                 return self # Cannot proceed without a valid model
            except Exception as info_err:
                self.warning(f"Could not retrieve detailed info for model {current_model_name}. Proceeding with defaults/manual settings if applicable. Error: {info_err}")
                # Fallback name check for vision if API info failed
                if "vision" in current_model_name.lower() or "gemini-1.5" in current_model_name.lower():
                     model_supports_vision = True

            # --- Determine Effective Limits ---
            # Start with values currently in binding_config (could be defaults or manually set if auto_detect was off)
            effective_ctx_size = self.binding_config.config["ctx_size"]
            effective_max_output = self.binding_config.config["max_output_tokens"]
            update_config_read_only = False

            if self.binding_config.auto_detect_limits:
                ASCIIColors.info("Auto-detect limits enabled.")
                update_config_read_only = True # Mark fields as read-only in UI
                if fetched_ctx_size and fetched_max_output:
                    effective_ctx_size = fetched_ctx_size
                    effective_max_output = fetched_max_output
                    ASCIIColors.info(f"Using auto-detected limits: ctx={effective_ctx_size}, max_output={effective_max_output}")
                else:
                    self.warning("Auto-detect enabled, but failed to fetch limits. Using fallback/previous values.")
                    # Keep the existing effective_ctx_size and effective_max_output from binding_config
            else:
                ASCIIColors.info("Auto-detect limits disabled. Using manually configured limits.")
                update_config_read_only = False # Allow manual editing
                # effective_ctx_size/effective_max_output are already loaded from binding_config
                # Warn if manual settings exceed detected limits (if detection worked)
                if actual_model_ctx_limit and effective_ctx_size > actual_model_ctx_limit:
                    self.warning(f"Manually set ctx_size ({effective_ctx_size}) exceeds the detected model limit ({actual_model_ctx_limit}). This may cause errors.")
                if actual_model_output_limit and effective_max_output > actual_model_output_limit:
                     self.warning(f"Manually set max_output_tokens ({effective_max_output}) exceeds the detected model limit ({actual_model_output_limit}). The API might cap the output.")

            # --- Update Configurations ---
            # Update binding config reflects the effective values used
            self.binding_config.config["ctx_size"] = effective_ctx_size
            self.binding_config.config["max_output_tokens"] = effective_max_output
            # Update main lollms config for global access and UI reflection
            self.config.ctx_size = effective_ctx_size
            self.config.max_n_predict = effective_max_output

            # Update read-only status in the config template for the UI (requires LoLLMsCom ideally)
            # This is tricky post-init, might need a mechanism in LoLLMs core to refresh field attributes
            # For now, we log the state. The UI might not immediately reflect read-only changes without a restart/refresh.
            ASCIIColors.info(f"Context size and Max Output Tokens fields are {'read-only (auto-detected)' if update_config_read_only else 'editable (manual)'}.")


            # --- Update Binding Type ---
            if model_supports_vision:
                self.binding_type = BindingType.TEXT_IMAGE
                ASCIIColors.info("Binding type set to TEXT_IMAGE (vision supported).")
            else:
                self.binding_type = BindingType.TEXT_ONLY
                ASCIIColors.info("Binding type set to TEXT_ONLY.")

            # --- Instantiate the Model ---
            ASCIIColors.info(f"Building Gemini model instance: {current_model_name}")
            self.model = self.genai.GenerativeModel(current_model_name)

            # --- Verify Model Operability ---
            try:
                # Use a simple, non-empty prompt for testing token counting
                test_prompt = "hello"
                self.model.count_tokens(test_prompt)
                ASCIIColors.success(f"Model '{current_model_name}' built and seems operational (token counting test passed).")
            except Exception as test_err:
                self.error(f"Model '{current_model_name}' instantiated but failed a basic operational test (token counting). Check API key, permissions, and model name. Error: {test_err}")
                trace_exception(test_err)
                if self.lollmsCom:
                     self.lollmsCom.InfoMessage(f"Gemini Binding Warning: Model '{current_model_name}' initialized but failed a basic test. It might not work correctly. Check API key/permissions.\nDetails: {test_err}")
                self.model = None # Mark model as non-operational
                return self # Return self, but model is None

        except Exception as e:
            self.error(f"Failed to build Gemini model '{current_model_name}': {e}")
            trace_exception(e)
            if self.lollmsCom:
                self.lollmsCom.InfoMessage(f"Gemini Binding Error: Failed to build model '{current_model_name}'.\nDetails: {e}")
            self.model = None
            # Ensure main config reflects failure state
            self.config.ctx_size = self.binding_config.ctx_size # Keep last known config values
            self.config.max_n_predict = self.binding_config.max_output_tokens
            return self

        return self


    def count_tokens(self, prompt: str) -> int:
        """
        Counts the number of tokens in the prompt for the current model.
        Returns -1 if counting fails.
        """
        if not self.model:
            self.error("Model not initialized. Cannot count tokens.")
            return -1
        try:
            token_count_response = self.model.count_tokens(prompt)
            return token_count_response.total_tokens
        except Exception as e:
            self.error(f"Failed to count tokens: {e}")
            trace_exception(e)
            # Don't use InfoMessage here, this is usually an internal check failure
            return -1

    def embed(self, text: Union[str, List[str]], task_type: str = "retrieval_document") -> Optional[List[List[float]]]:
        """
        Computes text embeddings using a Google embedding model.

        Args:
            text: The text or list of texts to embed.
            task_type: The type of task (e.g., "retrieval_document", "semantic_similarity").

        Returns:
            A list of embedding lists (one for each input text), or None if embedding fails.
        """
        if not self.genai:
            self.error("Gemini (genai) module not available. Cannot generate embeddings.")
            return None
        if not self.binding_config.google_api_key:
             self.error("Google API key not configured. Cannot generate embeddings.")
             # No InfoMessage here, as embedding is often background, error log sufficient
             return None

        embedding_model = "models/text-embedding-004" # Or choose dynamically if needed

        try:
            # Ensure configuration is active
            self.genai.configure(api_key=self.binding_config.google_api_key)

            results = []
            if isinstance(text, str):
                # Handle single text input
                result = self.genai.embed_content(
                    model=embedding_model,
                    content=text,
                    task_type=task_type,
                )
                if 'embedding' in result:
                    results = [result['embedding']] # Return as list of lists
                else:
                    self.error("Embedding failed or returned unexpected structure for single text.")
                    return None
            elif isinstance(text, list):
                # Handle list of texts using batch embedding if available
                if hasattr(self.genai, 'batch_embed_contents'):
                    batch_result = self.genai.batch_embed_contents(
                        model=embedding_model,
                        requests=[{'content': t, 'task_type': task_type} for t in text]
                    )
                    if batch_result and 'embeddings' in batch_result:
                        results = [emb['values'] for emb in batch_result['embeddings']]
                    else:
                        self.error("Batch embedding failed or returned unexpected structure.")
                        return None
                else:
                    # Fallback to sequential embedding if batch is not available (older SDK versions?)
                    self.warning("`batch_embed_contents` not found, attempting sequential embedding for list input.")
                    embeddings = []
                    for item in text:
                        try:
                            single_result = self.genai.embed_content(model=embedding_model, content=item, task_type=task_type)
                            if 'embedding' in single_result:
                                embeddings.append(single_result['embedding'])
                            else:
                                self.error(f"Failed to embed item: {item[:50]}...")
                                embeddings.append(None) # Keep list structure, mark failure
                        except Exception as item_e:
                            self.error(f"Error embedding item: {item[:50]}... Error: {item_e}")
                            embeddings.append(None)
                    # Only return if at least one embedding succeeded
                    if any(e is not None for e in embeddings):
                        results = embeddings # May contain None for failed items
                    else:
                        self.error("All items failed to embed sequentially.")
                        return None
            else:
                 self.error(f"Invalid input type for embedding: {type(text)}. Expected str or list[str].")
                 return None

            return results

        except google.api_core.exceptions.PermissionDenied as e:
             self.error(f"Embedding failed: Permission denied. Check API key and Embedding API enablement. Error: {e}")
             trace_exception(e)
             return None
        except google.api_core.exceptions.InvalidArgument as e:
             self.error(f"Embedding failed: Invalid argument. Check input text or task type ('{task_type}'). Error: {e}")
             trace_exception(e)
             return None
        except Exception as e:
            self.error(f"Failed to generate embeddings using {embedding_model}: {e}")
            trace_exception(e)
            return None


    def _prepare_generation_config(self, n_predict: Optional[int], gpt_params: dict) -> Optional[genai.types.GenerationConfig]:
        """Helper to build the GenerationConfig object and handle max tokens."""
        if not self.model: # Should be checked before calling, but double-check
             self.error("Model not initialized, cannot prepare generation config.")
             return None

        # Determine max tokens for this call
        model_max_tokens = self.binding_config.max_output_tokens # Effective limit from build_model
        generate_max_tokens = model_max_tokens

        if n_predict is not None:
            if 0 < n_predict <= model_max_tokens:
                generate_max_tokens = n_predict
                ASCIIColors.info(f"Using user-provided n_predict: {generate_max_tokens}")
            elif n_predict > model_max_tokens:
                 self.warning(f"Requested n_predict ({n_predict}) exceeds the effective model limit ({model_max_tokens}). Capping at {model_max_tokens}.")
                 generate_max_tokens = model_max_tokens
            else: # n_predict <= 0 or other invalid cases
                 self.warning(f"Invalid n_predict value ({n_predict}). Using effective model max output tokens: {model_max_tokens}")
                 # generate_max_tokens remains model_max_tokens
        else:
             ASCIIColors.info(f"Using effective max_output_tokens from configuration: {generate_max_tokens}")

        # Prepare generation parameters
        generation_config_params = {
            'candidate_count': 1, # Usually 1 for standard generation
            'max_output_tokens': generate_max_tokens,
            'temperature': float(gpt_params.get('temperature', 0.7)),
            'top_p': float(gpt_params.get('top_p', 0.95)), # Common default
            'top_k': int(gpt_params.get('top_k', 40)),     # Common default
        }

        # Handle stop sequences carefully
        stop_sequences = gpt_params.get('stop_sequences') or gpt_params.get('stop')
        if stop_sequences:
            # Ensure it's a list of strings
            if isinstance(stop_sequences, str):
                try:
                    # Try parsing if it looks like a JSON list
                    if stop_sequences.strip().startswith('[') and stop_sequences.strip().endswith(']'):
                         parsed_sequences = json.loads(stop_sequences)
                         if isinstance(parsed_sequences, list) and all(isinstance(s, str) for s in parsed_sequences):
                             stop_sequences = parsed_sequences
                         else:
                              self.warning(f"Parsed stop_sequences is not a list of strings: {parsed_sequences}. Using the original string as a single stop sequence.")
                              stop_sequences = [stop_sequences] # Treat original string as the sequence
                    else:
                         stop_sequences = [stop_sequences] # Treat as a single sequence if not JSON list format
                except json.JSONDecodeError:
                    self.warning(f"Could not parse stop_sequences string as JSON list: '{stop_sequences}'. Using it as a single stop sequence.")
                    stop_sequences = [stop_sequences]

            if isinstance(stop_sequences, list) and all(isinstance(s, str) for s in stop_sequences):
                 # Filter out empty strings which might cause issues
                 valid_stop_sequences = [s for s in stop_sequences if s]
                 if valid_stop_sequences:
                     generation_config_params['stop_sequences'] = valid_stop_sequences
                 else:
                     self.warning("Provided stop_sequences contained only empty strings. Ignoring.")
            else:
                 self.warning(f"Invalid format for stop_sequences: {stop_sequences}. Expected list of strings. Ignoring.")

        # Seed handling (Note: Direct seed setting is often not available in GenerationConfig)
        seed = gpt_params.get('seed', self.binding_config.seed) # Use binding config seed if not overridden
        if seed is not None and seed != -1:
            # Note: google-generativeai SDK as of early 2024 doesn't expose 'seed' in GenerationConfig
            # It might be implicitly used or controlled differently. We log a warning.
             self.warning("Seed parameter is provided but may not be directly supported or effective via the google-generativeai SDK's GenerationConfig.")
             # generation_config_params['seed'] = seed # Uncomment if SDK adds support

        try:
            return self.genai.types.GenerationConfig(**generation_config_params)
        except Exception as e:
            self.error(f"Failed to create GenerationConfig object: {e}")
            trace_exception(e)
            return None


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
            prompt (str): The text prompt for generation.
            n_predict (int, optional): Overrides the effective max output tokens for this call.
            callback (Callable[[str, int], bool], optional): Callback function for streaming tokens.
                                                                    Receives (token, type).
                                                                    Return False to stop generation.
            verbose (bool, optional): If true, logs more details.
            **gpt_params: Additional parameters (temperature, top_p, top_k, stop_sequences, seed).

        Returns:
            str: The generated text, or an empty string if generation fails or is stopped early.
        """
        if not self.model:
            self.error("Model not initialized. Attempting to rebuild...")
            self.settings_updated() # Attempt recovery
            if not self.model:
                 self.error("Failed to rebuild model. Aborting generation.")
                 if self.lollmsCom:
                      self.lollmsCom.InfoMessage("Gemini Generation Error: Model is not available. Please check configuration and logs.")
                 return ""

        # Prepare generation configuration
        generation_config = self._prepare_generation_config(n_predict, gpt_params)
        if generation_config is None:
            self.error("Failed to prepare generation configuration. Aborting.")
            if self.lollmsCom:
                 self.lollmsCom.InfoMessage("Gemini Generation Error: Could not create valid generation settings.")
            return ""

        output = ""
        try:
            if verbose: ASCIIColors.info(f"Generating text with effective config: {generation_config}")
            ASCIIColors.info("Starting generation stream...")

            response_stream = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True,
                safety_settings=self.safety_settings,
            )

            chunk_count = 0
            # --- Streaming Loop ---
            for chunk in response_stream:
                # Check for potential errors or blocks *before* accessing text
                block_reason = None
                finish_reason_str = None

                # Check for prompt feedback (blocking issues before generation starts)
                if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                    block_reason = chunk.prompt_feedback.block_reason.name
                    self.error(f"Generation stopped due to prompt feedback (safety/other): {block_reason}")
                    if self.lollmsCom:
                        self.lollmsCom.InfoMessage(f"Generation blocked by API before starting. Reason: {block_reason}. Check your prompt or safety settings.")
                    break # Stop processing chunks

                # Check for candidate finish reason (blocking or completion during generation)
                # NOTE: finish_reason might appear *with* the last text chunk or *in a separate final chunk*
                if chunk.candidates:
                     candidate = chunk.candidates[0] # Usually only one candidate
                     if candidate.finish_reason:
                         finish_reason = candidate.finish_reason # Get the enum value
                         finish_reason_str = finish_reason.name # Get the string name
                         if finish_reason_str not in ["STOP", "MAX_TOKENS", "UNSPECIFIED"]:
                              self.warning(f"Generation potentially stopped by API. Finish Reason: {finish_reason_str}")
                              if finish_reason_str == "SAFETY":
                                   block_reason = finish_reason_str # Treat safety finish as a block
                                   if self.lollmsCom:
                                        self.lollmsCom.InfoMessage("Generation stopped by API safety filter during response.")
                              elif finish_reason_str == "RECITATION":
                                   # Recitation is often less critical, maybe just warn
                                   self.warning("Generation flagged for potential recitation by API.")
                                   if self.lollmsCom:
                                        self.lollmsCom.InfoMessage("Warning: Generation flagged for potential recitation by the API.")
                              elif finish_reason_str == "OTHER":
                                   # Generic 'OTHER' block, needs investigation
                                   self.error(f"Generation stopped by API for 'OTHER' reason.")
                                   if self.lollmsCom:
                                        self.lollmsCom.InfoMessage("Generation stopped by API for an unspecified reason.")

                         if verbose: ASCIIColors.info(f"Chunk received with finish reason: {finish_reason_str}")


                # Extract text content if available
                word = ""
                try:
                    if hasattr(chunk, 'text'):
                        word = chunk.text
                    # Deprecated?: elif hasattr(chunk, 'parts') and chunk.parts: word = "".join(p.text for p in chunk.parts if hasattr(p,'text'))
                except AttributeError as e:
                    self.warning(f"Error accessing chunk text content: {e}. Chunk: {chunk}")
                    word = "" # Avoid adding potentially corrupt data
                except Exception as e: # Catch other potential errors during chunk processing
                    self.error(f"Error processing generation chunk content: {e}")
                    trace_exception(e)
                    word = ""


                # Process the extracted text chunk
                if word:
                    output += word
                    chunk_count += 1
                    if callback is not None:
                        if not callback(word, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                            ASCIIColors.info("Generation stopped by callback.")
                            # Attempt to cleanly stop? SDK might handle this implicitly on loop break.
                            # response_stream.close() # Or similar if available
                            break # Exit the loop

                # If a blocking reason was found, stop after processing any text in the current chunk
                if block_reason:
                    self.error(f"Stopping generation loop due to block reason: {block_reason}")
                    break

                # If a non-blocking but terminal finish reason (STOP, MAX_TOKENS) is present,
                # we can often break here as well, as no more text is expected.
                # However, sometimes metadata might follow, so letting the loop naturally end might be safer.
                # if finish_reason_str in ["STOP", "MAX_TOKENS"]:
                #    if verbose: ASCIIColors.info("Normal finish reason detected, ending stream processing.")
                #    break # Optimization: Exit loop if normally finished


            # --- End Streaming Loop ---
            ASCIIColors.success(f"Generation process completed. Received {chunk_count} chunks.")

        # --- Exception Handling ---
        except google.api_core.exceptions.PermissionDenied as e:
             self.error(f"Permission Denied: {e}. Check API key/permissions.")
             trace_exception(e)
             if self.lollmsCom:
                 self.lollmsCom.InfoMessage(f"Gemini API Error: Permission Denied. Please check your API key and ensure the Generative Language API is enabled for your project.\nDetails: {e}")
             return "" # Return empty on error
        except google.api_core.exceptions.ResourceExhausted as e:
             self.error(f"Resource Exhausted: {e}. Rate limits or quota likely hit.")
             trace_exception(e)
             if self.lollmsCom:
                 self.lollmsCom.InfoMessage(f"Gemini API Error: Resource Exhausted. You may have exceeded your usage quota or hit a rate limit. Please check your Google Cloud console or try again later.\nDetails: {e}")
             return ""
        except google.api_core.exceptions.InvalidArgument as e:
             # This can be complex: prompt too long, invalid safety, bad parameters etc.
             self.error(f"Invalid Argument: {e}. Check prompt length (ctx: {self.binding_config.ctx_size}), parameters ({generation_config}), or safety settings.")
             trace_exception(e)
             if self.lollmsCom:
                 self.lollmsCom.InfoMessage(f"Gemini API Error: Invalid Argument. This could be due to the prompt exceeding the context limit, invalid generation parameters, or conflicting safety settings. Please review your input and configuration.\nDetails: {e}")
             return ""
        except Exception as e: # Catch-all for other unexpected API or streaming errors
            self.error(f"Gemini generation failed with an unexpected error: {e}")
            trace_exception(e)
            if self.lollmsCom:
                 self.lollmsCom.InfoMessage(f"Gemini Generation Error: An unexpected error occurred during generation.\nPlease check the Lollms logs for details.\nError: {e}")
            return ""

        return output


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
            prompt (str): The text prompt.
            images (List[str]): List of file paths to images.
            n_predict (int, optional): Overrides effective max output tokens for this call.
            callback (Callable): Streaming callback (same signature as generate).
            verbose (bool): Verbose logging.
            **gpt_params: Generation parameters.

        Returns:
            str: Generated text or empty string on failure.
        """
        if not self.model:
            self.error("Model not initialized. Attempting to rebuild...")
            self.settings_updated() # Attempt recovery
            if not self.model:
                 self.error("Failed to rebuild model. Aborting generation.")
                 if self.lollmsCom:
                      self.lollmsCom.InfoMessage("Gemini Generation Error: Model is not available. Please check configuration and logs.")
                 return ""

        if self.binding_type != BindingType.TEXT_IMAGE:
            self.error(f"The selected model '{self.config.model_name}' does not support image input according to its configuration.")
            if self.lollmsCom:
                 self.lollmsCom.InfoMessage(f"Gemini Generation Error: The current model ({self.config.model_name}) is not configured for vision input. Please select a vision-capable model (e.g., gemini-pro-vision, gemini-1.5-pro).")
            return ""

        if not images:
             self.warning("generate_with_images called but no images were provided. Falling back to text-only generation.")
             # Use the same parameters for the text-only fallback
             return self.generate(prompt, n_predict, callback, verbose, **gpt_params)

        # --- Prepare generation configuration (uses the same helper) ---
        # Vision models might have different optimal defaults, adjust if needed
        # Default vision temp/top_k often lower, handled if not in gpt_params
        vision_defaults = {'temperature': 0.4, 'top_k': 32}
        merged_params = {**vision_defaults, **gpt_params} # User params override defaults
        generation_config = self._prepare_generation_config(n_predict, merged_params)
        if generation_config is None:
            self.error("Failed to prepare generation configuration for vision task. Aborting.")
            if self.lollmsCom:
                 self.lollmsCom.InfoMessage("Gemini Generation Error: Could not create valid generation settings for vision task.")
            return ""


        # --- Prepare Content List [prompt, image1_PIL, image2_PIL, ...] ---
        content_parts: List[Union[str, Image.Image]] = [prompt]
        loaded_images: List[Image.Image] = [] # Keep track to close them later
        successful_image_loads = 0

        for image_path_str in images:
            try:
                # Validate path and find the actual file
                image_path = Path(image_path_str)
                valid_image_path = find_first_available_file_path([image_path]) # Handles relative/absolute paths

                if not valid_image_path or not is_file_path(valid_image_path) or not valid_image_path.exists():
                     self.warning(f"Image path not found or invalid: {image_path_str}. Skipping.")
                     if callback: callback(f"\nWarning: Image not found {image_path_str}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING)
                     continue

                # Load image using PIL
                img = Image.open(valid_image_path)
                # Optional: Convert to RGB if needed (Gemini usually handles common formats)
                # if img.mode != 'RGB':
                #     img = img.convert('RGB')
                loaded_images.append(img) # Add to list for later cleanup
                content_parts.append(img) # Add PIL image directly to content list
                successful_image_loads += 1
                ASCIIColors.info(f"Successfully loaded image for processing: {valid_image_path}")

            except FileNotFoundError: # Should be caught by exists() check, but as fallback
                 self.error(f"Image file not found: {image_path_str}. Skipping.")
                 if callback: callback(f"\nWarning: Image not found {image_path_str}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING)
            except Exception as e:
                self.error(f"Failed to load or process image {image_path_str}: {e}")
                trace_exception(e)
                if callback: callback(f"\nWarning: Failed to load image {image_path_str}. Error: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING)

        if successful_image_loads == 0:
             self.error("All provided images failed to load. Aborting vision generation.")
             # No images loaded, close any potentially opened (though unlikely)
             for img in loaded_images: img.close()
             if self.lollmsCom:
                  self.lollmsCom.InfoMessage("Gemini Generation Error: Failed to load any of the provided images. Cannot proceed with vision task.")
             return ""
        elif successful_image_loads < len(images):
             self.warning("Some images failed to load. Proceeding with the successfully loaded ones.")
             # Optionally inform user via callback or InfoMessage


        output = ""
        try:
            if verbose: ASCIIColors.info(f"Generating text with images. Effective config: {generation_config}")
            ASCIIColors.info(f"Starting generation stream with {successful_image_loads} images...")

            response_stream = self.model.generate_content(
                content_parts, # Pass the list [prompt, img1, img2, ...]
                generation_config=generation_config,
                stream=True,
                safety_settings=self.safety_settings,
            )

            chunk_count = 0
            # --- Streaming Loop (Identical logic to text generation for handling chunks/errors) ---
            for chunk in response_stream:
                block_reason = None
                finish_reason_str = None

                if hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                    block_reason = chunk.prompt_feedback.block_reason.name
                    self.error(f"Generation stopped due to prompt/image feedback (safety/other): {block_reason}")
                    if self.lollmsCom:
                        self.lollmsCom.InfoMessage(f"Generation blocked by API before starting. Reason: {block_reason}. Check your prompt, images, or safety settings.")
                    break

                if chunk.candidates:
                     candidate = chunk.candidates[0]
                     if candidate.finish_reason:
                         finish_reason = candidate.finish_reason
                         finish_reason_str = finish_reason.name
                         if finish_reason_str not in ["STOP", "MAX_TOKENS", "UNSPECIFIED"]:
                             self.warning(f"Generation potentially stopped by API. Finish Reason: {finish_reason_str}")
                             if finish_reason_str == "SAFETY":
                                  block_reason = finish_reason_str
                                  if self.lollmsCom: self.lollmsCom.InfoMessage("Generation stopped by API safety filter during response.")
                             elif finish_reason_str == "RECITATION":
                                  self.warning("Generation flagged for potential recitation by API.")
                                  if self.lollmsCom: self.lollmsCom.InfoMessage("Warning: Generation flagged for potential recitation by the API.")
                             elif finish_reason_str == "OTHER":
                                  self.error("Generation stopped by API for 'OTHER' reason.")
                                  if self.lollmsCom: self.lollmsCom.InfoMessage("Generation stopped by API for an unspecified reason.")
                         if verbose: ASCIIColors.info(f"Chunk received with finish reason: {finish_reason_str}")

                word = ""
                try:
                    if hasattr(chunk, 'text'): word = chunk.text
                except AttributeError as e:
                    self.warning(f"Error accessing chunk text content: {e}. Chunk: {chunk}")
                    word = ""
                except Exception as e:
                    self.error(f"Error processing generation chunk content: {e}")
                    trace_exception(e)
                    word = ""

                if word:
                    output += word
                    chunk_count += 1
                    if callback is not None:
                        if not callback(word, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                            ASCIIColors.info("Generation stopped by callback.")
                            break

                if block_reason:
                    self.error(f"Stopping generation loop due to block reason: {block_reason}")
                    break

            # --- End Streaming Loop ---
            ASCIIColors.success(f"Generation with images process completed. Received {chunk_count} chunks.")

        # --- Exception Handling (Use InfoMessage for user-facing API errors) ---
        except google.api_core.exceptions.PermissionDenied as e:
             self.error(f"Permission Denied: {e}. Check API key/permissions.")
             trace_exception(e)
             if self.lollmsCom: self.lollmsCom.InfoMessage(f"Gemini API Error: Permission Denied. Check API key and API enablement.\nDetails: {e}")
             return ""
        except google.api_core.exceptions.ResourceExhausted as e:
             self.error(f"Resource Exhausted: {e}. Rate limits or quota hit.")
             trace_exception(e)
             if self.lollmsCom: self.lollmsCom.InfoMessage(f"Gemini API Error: Resource Exhausted (Quota/Rate Limit). Check Google Cloud console or wait.\nDetails: {e}")
             return ""
        except google.api_core.exceptions.InvalidArgument as e:
             self.error(f"Invalid Argument: {e}. Check image formats/sizes/content, prompt length (ctx: {self.binding_config.ctx_size}), parameters ({generation_config}), or safety settings.")
             trace_exception(e)
             if self.lollmsCom: self.lollmsCom.InfoMessage(f"Gemini API Error: Invalid Argument. Check images, prompt, parameters, or safety settings.\nDetails: {e}")
             return ""
        except Exception as e:
            self.error(f"Gemini generation with images failed with an unexpected error: {e}")
            trace_exception(e)
            if self.lollmsCom: self.lollmsCom.InfoMessage(f"Gemini Generation Error: An unexpected error occurred during vision generation.\nCheck logs.\nError: {e}")
            return ""
        finally:
             # --- Clean up loaded PIL Image objects ---
             ASCIIColors.info(f"Closing {len(loaded_images)} loaded image objects.")
             closed_count = 0
             for img in loaded_images:
                 try:
                     img.close()
                     closed_count += 1
                 except Exception as close_err:
                     # Log warning but don't stop everything
                     self.warning(f"Could not close an image object: {close_err}")
             if closed_count != len(loaded_images):
                 self.warning(f"Only closed {closed_count} out of {len(loaded_images)} image objects.")


        return output


    def list_models(self) -> List[str]:
        """
        Lists the names (IDs) of available Gemini models that support content generation.
        Returns an empty list if listing fails.
        """
        if not self.genai:
            self.error("Gemini (genai) module not available. Cannot list models.")
            return []
        if not self.binding_config.google_api_key:
             self.error("Google API key not configured. Cannot list models.")
             # No InfoMessage needed here, error log is sufficient for this background task
             return []

        try:
            # Ensure API is configured
            self.genai.configure(api_key=self.binding_config.google_api_key)
            ASCIIColors.info("Fetching list of available Gemini models from API...")
            api_models = self.genai.list_models()

            supported_models = []
            for m in api_models:
                # Check if the model supports the 'generateContent' method
                if 'generateContent' in m.supported_generation_methods:
                    # Extract the model ID (usually the part after 'models/')
                    model_id = m.name.split('/')[-1]
                    supported_models.append(model_id)

            ASCIIColors.info(f"Found {len(supported_models)} models supporting 'generateContent'.")
            supported_models.sort() # Sort alphabetically for consistency
            return supported_models

        except google.api_core.exceptions.PermissionDenied:
             self.error("Permission Denied listing models. Check your API key and API enablement.")
             trace_exception(None) # Log the context
             # No InfoMessage here
             return []
        except Exception as e:
            self.error(f"Failed to list Gemini models from API: {e}")
            trace_exception(e)
            # No InfoMessage here
            return []


    def get_available_models(self, app: Optional[LoLLMsCom] = None) -> List[dict]:
        """
        Gets a list of available Gemini models formatted for the Lollms UI.
        Includes fetched metadata like token limits and potential capabilities.

        Args:
            app (LoLLMsCom, optional): The LoLLMsCom object, passed by LoLLMs.

        Returns:
            List[dict]: A list of model dictionaries compatible with the LoLLMs models list format.
        """
        # Use self.lollmsCom if available, otherwise use the passed app instance
        lollms_com = self.lollmsCom or app
        # Check prerequisites first
        if not self.genai:
            self.error("Gemini (genai) module not available. Cannot get available models.")
            return []
        if not self.binding_config.google_api_key:
             self.error("Google API key not configured. Cannot get available models.")
             if lollms_com:
                 lollms_com.InfoMessage("Gemini Binding Error: Cannot fetch models. Google API Key is missing.")
             return []


        lollms_models = []
        try:
            # Ensure API is configured
            self.genai.configure(api_key=self.binding_config.google_api_key)
            ASCIIColors.info("Fetching detailed list of available Gemini models from API for Lollms UI...")
            api_models = self.genai.list_models()

            for model in api_models:
                # Only include models usable for generation
                if 'generateContent' not in model.supported_generation_methods:
                    continue

                model_id = model.name.split('/')[-1] # e.g., 'gemini-1.5-pro-latest'
                display_name = getattr(model, 'display_name', model_id)
                description = getattr(model, 'description', 'N/A')
                input_limit = getattr(model, 'input_token_limit', 'N/A')
                output_limit = getattr(model, 'output_token_limit', 'N/A')

                # --- Determine Rank and Size Proxy ---
                rank = 1.0 # Base rank
                if "ultra" in model_id: rank = 3.0
                elif "pro" in model_id: rank = 2.0
                elif "flash" in model_id: rank = 1.5 # Flash is faster/cheaper than Pro

                if "1.5" in model_id: rank += 0.5 # Newer versions get a boost
                elif "1.0" in model_id: rank += 0.0 # Baseline
                # Add more version checks if needed (e.g., "2.0")

                size_proxy = -1 # Default/unknown size
                try:
                     # Use sum of limits as a rough proxy for 'size' or capability
                     if isinstance(input_limit, int) and isinstance(output_limit, int):
                          size_proxy = input_limit + output_limit
                     elif isinstance(input_limit, int): # Use input limit if output is unknown
                          size_proxy = input_limit
                except Exception:
                     pass # Keep -1 if conversion fails

                # --- Determine Category (Vision/Multimodal vs Text) ---
                category = "text" # Default category
                # Check for vision based on methods, name, or display name
                is_vision = (
                    'embedContent' in model.supported_generation_methods and # Often needed for multimodal
                    ('Vision' in model.display_name or 'vision' in model_id.lower() or "1.5" in model_id.lower()) # Name/version patterns
                )
                if is_vision:
                     category = "multimodal"


                # --- Build Lollms Model Dictionary ---
                md = {
                    "category": category,
                    "datasets": "Proprietary Google datasets", # General assumption
                    "icon": '/bindings/gemini/logo.png', # Path relative to Lollms web root
                    "last_commit_time": datetime.now().timestamp(), # Placeholder
                    "license": "Commercial API",
                    "model_creator": "Google",
                    "model_creator_link": "https://ai.google.dev/",
                    "name": model_id, # The actual ID used in configuration
                    "quantizer": None, # Not applicable to APIs
                    "rank": rank,
                    "type": "api", # Indicates it's an API-based model
                    "variants": [{"name": model_id, "size": size_proxy}], # API models usually have one variant per ID
                    # Additional metadata from API (optional but helpful)
                    "description": description,
                    "display_name": display_name, # User-friendly name
                    "input_token_limit": input_limit,
                    "output_token_limit": output_limit,
                    "supported_generation_methods": getattr(model, 'supported_generation_methods', []),
                    # Default generation params (informational)
                    "temperature": getattr(model, 'temperature', 'N/A'),
                    "top_p": getattr(model, 'top_p', 'N/A'),
                    "top_k": getattr(model, 'top_k', 'N/A'),
                }
                lollms_models.append(md)

            # Sort models by rank (descending) then name (ascending) for better UI presentation
            lollms_models.sort(key=lambda x: (-x['rank'], x['name']))
            ASCIIColors.success(f"Formatted {len(lollms_models)} Gemini models for Lollms UI.")

        except google.api_core.exceptions.PermissionDenied:
             self.error("Permission Denied getting available models. Check your API key and API enablement.")
             trace_exception(None)
             if lollms_com:
                 lollms_com.InfoMessage("Gemini Binding Error: Permission Denied when fetching models. Check API key.")
             return []
        except Exception as e:
            self.error(f"Failed to retrieve or process Gemini models for Lollms: {e}")
            trace_exception(e)
            if lollms_com:
                 lollms_com.InfoMessage(f"Gemini Binding Error: Failed to retrieve models from API.\nCheck logs.\nError: {e}")
            return []

        return lollms_models