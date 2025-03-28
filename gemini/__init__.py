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
from pathlib import Path
from typing import Callable, Any, List, Union
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors
from lollms.types import MSG_OPERATION_TYPE
from lollms.utilities import (
    detect_antiprompt,
    remove_text_from_string,
    trace_exception,
    PackageManager,
    find_first_available_file_path,
    is_file_path
)
from lollms.com import LoLLMsCom
import subprocess
import yaml
import sys
import json
from datetime import datetime
from PIL import Image
import base64
import io
import pipmaster as pm

pm.install_if_missing("google-generativeai")
import google.generativeai as genai
import google.api_core.exceptions

__author__ = "parisneo"
__github__ = "https://github.com/ParisNeo/lollms_bindings_zoo"
__copyright__ = "Copyright 2023, ParisNeo"
__license__ = "Apache 2.0"

binding_name = "Gemini"
binding_folder_name = "" # Set to the folder name if different from binding_name

# Kept for potential use cases, though the SDK prefers PIL objects directly
def encode_image(image_path, max_image_width=-1):
    # (encode_image function remains unchanged)
    try:
        image = Image.open(image_path)
        width, height = image.size

        if max_image_width != -1 and width > max_image_width:
            ratio = max_image_width / width
            new_width = max_image_width
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height))

        # Convert to RGB if necessary (common requirement for models)
        if image.mode != 'RGB':
             image = image.convert('RGB')

        # Gemini prefers PNG, JPEG, WEBP. Let's use PNG for consistency here.
        byte_arr = io.BytesIO()
        image.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()

        return base64.b64encode(byte_arr).decode('utf-8')
    except Exception as e:
        trace_exception(e)
        return None


class Gemini(LLMBinding):

    def __init__(self,
                config: LOLLMSConfig,
                lollms_paths: LollmsPaths = None,
                installation_option:InstallOption=InstallOption.INSTALL_IF_NECESSARY, # This option is less relevant for API bindings
                lollmsCom=None) -> None:
        """
        Initialize the Binding.

        Args:
            config (LOLLMSConfig): The configuration object for LOLLMS.
            lollms_paths (LollmsPaths, optional): The paths object for LOLLMS. Defaults to LollmsPaths().
            installation_option (InstallOption, optional): The installation option for LOLLMS. Defaults to InstallOption.INSTALL_IF_NECESSARY.
        """
        if lollms_paths is None:
            lollms_paths = LollmsPaths()
        # installation_option is handled by the PackageManager check above for the library
        self.installation_option = installation_option # Store it, though less critical here

        # Define the configuration template
        # Use placeholders for ctx_size and max_output_tokens initially, they will be updated by build_model
        binding_config_template = ConfigTemplate([
                {"name":"google_api_key","type":"str","value":"", "help":"Your Google Generative AI API key. Get one from https://ai.google.dev/"},
                # Context Size - Will be updated from model info
                {"name":"ctx_size","type":"int","value":30720, "min":512, "help":"Context size (in tokens). Automatically updated based on the selected model. Default shown is for Gemini 1.0 Pro.", "read_only":True},
                # Max Output Tokens - Will be updated from model info
                {"name":"max_output_tokens","type":"int","value":2048, "min":1, "help":"Maximum number of tokens to generate per response. Automatically updated based on the selected model. Default shown is for Gemini 1.0 Pro.", "read_only":True},
                {"name":"seed","type":"int","value":-1,"help":"Random seed for generation. -1 for random. Note: Seed support might vary or not be directly exposed in all API versions/SDK methods."},
                # Safety Settings
                {"name": "safety_setting_harassment", "type": "str", "value": "BLOCK_MEDIUM_AND_ABOVE", "options":["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"], "help":"Safety setting for harassment content."},
                {"name": "safety_setting_hate_speech", "type": "str", "value": "BLOCK_MEDIUM_AND_ABOVE", "options":["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"], "help":"Safety setting for hate speech content."},
                {"name": "safety_setting_sexually_explicit", "type": "str", "value": "BLOCK_MEDIUM_AND_ABOVE", "options":["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"], "help":"Safety setting for sexually explicit content."},
                {"name": "safety_setting_dangerous_content", "type": "str", "value": "BLOCK_MEDIUM_AND_ABOVE", "options":["BLOCK_NONE", "BLOCK_ONLY_HIGH", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_LOW_AND_ABOVE"], "help":"Safety setting for dangerous content."}
            ])

        # Define the default configuration
        binding_config_defaults = BaseConfig(config={
                "google_api_key": "",
                "safety_setting_harassment": "BLOCK_MEDIUM_AND_ABOVE",
                "safety_setting_hate_speech": "BLOCK_MEDIUM_AND_ABOVE",
                "safety_setting_sexually_explicit": "BLOCK_MEDIUM_AND_ABOVE",
                "safety_setting_dangerous_content": "BLOCK_MEDIUM_AND_ABOVE",
                # ctx_size and max_output_tokens will be populated by build_model
                "ctx_size": 30720, # Default fallback
                "max_output_tokens": 2048, # Default fallback
                "seed": -1
            })

        # Initialize TypedConfig
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
                            supported_file_extensions=['.png', '.jpg', '.jpeg', '.webp'], # Common image formats supported by Gemini
                            lollmsCom=lollmsCom
                        )

        self.model = None
        self.genai = genai # Store the imported module
        self.safety_settings = None # To be built in settings_updated

        # Trigger initial configuration and model build
        self.settings_updated()


    def settings_updated(self):
        """Called when the settings are updated."""
        if not self.binding_config.google_api_key:
            self.error("No Google API key is set! Please set it in the binding configuration.")
            self.model = None
            # Do not nullify self.genai, the library is still loaded
        else:
            try:
                # Configure the API key globally for the genai library
                if self.genai:
                    self.genai.configure(api_key=self.binding_config.google_api_key)
                    ASCIIColors.info("Google API key configured.")
                else:
                    self.error("google-generativeai library not available.")
                    return

                # Update safety settings
                self.safety_settings = {
                    'HARM_CATEGORY_HARASSMENT': self.binding_config.safety_setting_harassment,
                    'HARM_CATEGORY_HATE_SPEECH': self.binding_config.safety_setting_hate_speech,
                    'HARM_CATEGORY_SEXUALLY_EXPLICIT': self.binding_config.safety_setting_sexually_explicit,
                    'HARM_CATEGORY_DANGEROUS_CONTENT': self.binding_config.safety_setting_dangerous_content,
                }

                # Rebuild the model. This will also update context sizes.
                # self.config.model_name might be None initially, build_model handles this
                self.build_model() # Call without args to use self.config.model_name

            except Exception as e:
                self.error(f"Failed to configure Google API or build model: {e}")
                trace_exception(e)
                self.model = None # Ensure model is None if config fails

        # Sync lollms config with binding config *after* potential updates in build_model
        # These values in self.config are primarily informational for the lollms UI.
        self.config.ctx_size = self.binding_config.ctx_size
        self.config.max_n_predict = self.binding_config.max_output_tokens


    def build_model(self, model_name=None):
        """
        Builds the generative model based on the configuration.
        Fetches model information to set context size and max output tokens.
        """
        # If a model_name is provided, update self.config first
        super().build_model(model_name) # Handles model name update in self.config

        if not self.binding_config.google_api_key:
            self.error("Cannot build model: Google API key not set.")
            self.model = None
            return self # Return self, unusable state

        current_model_name = self.config.model_name
        if not current_model_name:
            self.warning("No model name selected in configuration. Cannot build model yet.")
            # Attempt to list models and select the first one? Or just wait for user?
            # For now, just warn and exit.
            self.model = None
            return self

        try:
            # Ensure API key is configured before building
            self.genai.configure(api_key=self.binding_config.google_api_key)

            # --- Fetch Model Information ---
            fetched_ctx_size = None
            fetched_max_output = None
            model_supports_vision = False
            try:
                ASCIIColors.info(f"Fetching details for model: models/{current_model_name}")
                model_info = self.genai.get_model(f"models/{current_model_name}") # API expects 'models/' prefix

                fetched_ctx_size = getattr(model_info, 'input_token_limit', None)
                fetched_max_output = getattr(model_info, 'output_token_limit', None)

                # Crude check for vision support based on name (API doesn't seem to have a direct flag)
                if "vision" in current_model_name.lower() or "gemini-1.5" in current_model_name.lower(): # Gemini 1.5 models are multimodal
                    model_supports_vision = True

                if fetched_ctx_size and fetched_max_output:
                    ASCIIColors.success(f"Successfully fetched limits for {current_model_name}:")
                    ASCIIColors.info(f"  Input Token Limit (ctx_size): {fetched_ctx_size}")
                    ASCIIColors.info(f"  Output Token Limit (max_output_tokens): {fetched_max_output}")

                    # --- Update Configurations ---
                    # Update binding config (read_only fields)
                    self.binding_config.config["ctx_size"] = fetched_ctx_size
                    self.binding_config.config["max_output_tokens"] = fetched_max_output
                    # Update main lollms config
                    self.config.ctx_size = fetched_ctx_size
                    self.config.max_n_predict = fetched_max_output
                else:
                    self.warning(f"Could not retrieve token limits for model {current_model_name}. Using defaults.")
                    # Keep existing/default values from binding_config
                    self.config.ctx_size = self.binding_config.ctx_size
                    self.config.max_n_predict = self.binding_config.max_output_tokens

                # Update binding type based on check
                if model_supports_vision:
                    self.binding_type = BindingType.TEXT_IMAGE
                    ASCIIColors.info("Binding type set to TEXT_IMAGE")
                else:
                    self.binding_type = BindingType.TEXT_ONLY
                    ASCIIColors.info("Binding type set to TEXT")


            except Exception as info_err:
                self.warning(f"Could not retrieve detailed info for model {current_model_name}. Using configured defaults/fallbacks. Error: {info_err}")
                # Fallback to name check for vision if get_model fails
                if "vision" in current_model_name.lower() or "gemini-1.5" in current_model_name.lower() or "gemini-2.0" in current_model_name.lower() or "gemini-2.5" in current_model_name.lower():
                    self.binding_type = BindingType.TEXT_IMAGE
                else:
                    self.binding_type = BindingType.TEXT_ONLY
                # Ensure main config reflects binding config defaults/previous values
                self.config.ctx_size = self.binding_config.ctx_size
                self.config.max_n_predict = self.binding_config.max_output_tokens

            # --- Instantiate the Model ---
            ASCIIColors.info(f"Building Gemini model instance: {current_model_name}")
            self.model = self.genai.GenerativeModel(current_model_name)

            # --- Verify Model Operability ---
            try:
                # Simple test: count tokens
                self.model.count_tokens("test")
                ASCIIColors.success(f"Model {current_model_name} built and seems operational.")
            except Exception as test_err:
                self.error(f"Model {current_model_name} instantiated but failed a basic test. Check API key and model name. Error: {test_err}")
                self.model = None # Mark as non-operational
                # Keep potentially fetched context sizes, but model is unusable
                return self

        except Exception as e:
            self.error(f"Failed to build Gemini model '{current_model_name}': {e}")
            trace_exception(e)
            self.model = None # Ensure model is None on failure
            # Reset context sizes to defaults if build fails entirely? Or keep last known good/defaults?
            # Let's keep the last values set in binding_config for now.
            self.config.ctx_size = self.binding_config.ctx_size
            self.config.max_n_predict = self.binding_config.max_output_tokens
            return self

        return self # Return self, hopefully ready


    def count_tokens(self, prompt: str):
        """
        Counts the number of tokens in the prompt for the current model.

        Args:
            prompt (str): The text prompt to count tokens for.

        Returns:
            int: The total number of tokens, or -1 if an error occurs or model is not available.
                 Returning -1 signals unavailability or error, differentiating from an empty prompt (0 tokens).
        """
        if not self.model:
            self.error("Model not initialized. Cannot count tokens.")
            return -1 # Indicate error/unavailability
        try:
            # This counts tokens based on the model's internal tokenizer
            token_count_response = self.model.count_tokens(prompt)
            return token_count_response.total_tokens
        except Exception as e:
            self.error(f"Failed to count tokens: {e}")
            trace_exception(e)
            return -1 # Indicate error

    # Removed tokenize/detokenize as they were using tiktoken (incorrect for Gemini)
    # and the SDK doesn't provide a direct public equivalent. Rely on count_tokens.

    def embed(self, text: Union[str, List[str]], task_type: str = "retrieval_document"):
        """
        Computes text embeddings using a Google embedding model.

        Args:
            text (Union[str, List[str]]): The text or list of texts to embed.
            task_type (str): The intended task for the embedding (e.g., "retrieval_document", "semantic_similarity").
                           Check Google API documentation for valid task types. Defaults to "retrieval_document".

        Returns:
            List[List[float]]: A list of embeddings (list of floats), one for each input text.
                               Returns None if embedding fails.
        """
        if not self.genai:
            self.error("Gemini not configured (genai module not loaded). Cannot generate embeddings.")
            return None
        if not self.binding_config.google_api_key:
             self.error("Google API key not configured. Cannot generate embeddings.")
             return None

        # Consider making the embedding model configurable? For now, hardcode a common one.
        embedding_model = "models/text-embedding-004" # Newer model as of early 2024
        # embedding_model = "models/embedding-001" # Older model

        try:
            # Ensure API key is configured
            self.genai.configure(api_key=self.binding_config.google_api_key)

            # The API handles both single string and list of strings for batching
            result = self.genai.embed_content(
                model=embedding_model,
                content=text,
                task_type=task_type,
                # title="Optional title if task_type requires it" # Add if needed
            )

            # The result structure is {'embedding': [float,...]} for single input
            # and {'embedding': [float,...]} even for batch input (iterating through list internally)
            # The SDK documentation seems inconsistent here. Let's test and adapt.
            # Testing reveals `embed_content` with a list returns a dict with 'embedding' key,
            # containing the embedding for the *first* item only if not iterated.
            # Need to call it per item or use batch_embed_content if available.
            # `batch_embed_content` is the correct method for lists.

            if isinstance(text, list):
                # Use batch embedding for lists
                batch_result = self.genai.batch_embed_contents(
                    model=embedding_model,
                    requests=[{'content': t, 'task_type': task_type} for t in text]
                )
                if batch_result and 'embeddings' in batch_result:
                     # Note: The actual key might be different, check API response structure
                     # Assuming `batch_embed_contents` returns a list of embeddings directly
                     # Let's refine based on actual SDK behavior if different.
                     # According to docs, it returns a list of ContentEmbedding objects.
                     # Update: The structure seems to be `BatchEmbedContentsResponse` containing `embeddings` list.
                     return [emb['values'] for emb in batch_result['embeddings']] # Extract numerical values
                else:
                     self.error("Batch embedding failed or returned unexpected structure.")
                     return None
            else:
                 # Single text case, use embed_content
                 if 'embedding' in result:
                     return [result['embedding']] # Return as list of embeddings for consistency
                 else:
                     self.error("Expected 'embedding' key in single text result, but not found.")
                     return None

        except AttributeError:
             # Handle cases where `batch_embed_contents` might not be available in older SDK versions
             self.warning("`batch_embed_contents` not found, attempting sequential embedding for list input.")
             embeddings = []
             if isinstance(text, list):
                 for item in text:
                     try:
                         single_result = self.genai.embed_content(model=embedding_model, content=item, task_type=task_type)
                         if 'embedding' in single_result:
                             embeddings.append(single_result['embedding'])
                         else:
                             self.error(f"Failed to embed item: {item[:50]}...")
                             embeddings.append(None) # Add None to maintain list structure size
                     except Exception as item_e:
                         self.error(f"Error embedding item: {item[:50]}... Error: {item_e}")
                         embeddings.append(None)
                 # Check if all items failed
                 if all(e is None for e in embeddings):
                     return None
                 # Return list, potentially containing Nones for failed items
                 return embeddings
             else:
                 # This path shouldn't be reached if AttributeError occurred on batch call, but handle defensively
                 self.error("Internal error during embedding.")
                 return None

        except Exception as e:
            self.error(f"Failed to generate embeddings using {embedding_model}: {e}")
            trace_exception(e)
            if isinstance(e, google.api_core.exceptions.PermissionDenied):
                 self.error("Permission denied. Check your API key and ensure the Embedding API is enabled.")
            elif isinstance(e, google.api_core.exceptions.InvalidArgument):
                 self.error(f"Invalid argument. Check the input text and task type ('{task_type}').")
            return None


    def generate(self,
                 prompt: str,
                 n_predict: int = None, # User override for max tokens
                 callback: Callable[[str, int, dict], bool] = None,
                 verbose: bool = False,
                 **gpt_params) -> str:
        """
        Generates text based on a prompt using the configured Gemini model.

        Args:
            prompt (str): The text prompt for generation.
            n_predict (int, optional): Overrides the maximum number of tokens to generate for this specific call.
                                      If None, uses the model's default max output tokens.
            callback (Callable[[str, int, dict], bool], optional): Callback function for streaming tokens.
            verbose (bool, optional): If true, logs more details.
            **gpt_params: Additional parameters (temperature, top_p, top_k, stop_sequences).

        Returns:
            str: The generated text, or an empty string if generation fails.
        """
        if not self.model:
            self.error("Model not initialized.")
            self.settings_updated() # Attempt recovery
            if not self.model:
                 self.error("Failed to rebuild model. Aborting generation.")
                 return ""

        # Determine max tokens for this specific generation call:
        # 1. Use n_predict if provided by the user and it's less than the model's max.
        # 2. Otherwise, use the model's max output tokens stored in binding_config.
        model_max_tokens = self.binding_config.max_output_tokens
        generate_max_tokens = model_max_tokens
        if n_predict is not None:
            if n_predict > 0 and n_predict <= model_max_tokens:
                generate_max_tokens = n_predict
                ASCIIColors.info(f"Using user-provided n_predict: {generate_max_tokens}")
            elif n_predict > model_max_tokens:
                 self.warning(f"Requested n_predict ({n_predict}) exceeds model's limit ({model_max_tokens}). Using model limit.")
                 # generate_max_tokens remains model_max_tokens
            else: # n_predict is <= 0 or invalid
                 self.warning(f"Invalid n_predict value ({n_predict}). Using model's max output tokens: {model_max_tokens}")
                 # generate_max_tokens remains model_max_tokens
        else:
             ASCIIColors.info(f"Using model's default max_output_tokens: {generate_max_tokens}")


        # Prepare generation configuration
        generation_config_params = {
            'candidate_count': 1, # Usually 1 for chat/generation
            'max_output_tokens': generate_max_tokens, # Use calculated limit
            'temperature': float(gpt_params.get('temperature', 0.7)),
            'top_p': float(gpt_params.get('top_p', 0.95)),
            'top_k': int(gpt_params.get('top_k', 40)),
        }
        # Add stop sequences if provided
        stop_sequences = gpt_params.get('stop_sequences', None) or gpt_params.get('stop', None) # Allow 'stop' alias
        if stop_sequences:
            if isinstance(stop_sequences, str):
                # Handle potential JSON string list from some frontends
                try:
                     stop_sequences = json.loads(stop_sequences)
                except json.JSONDecodeError:
                     stop_sequences = [stop_sequences] # Treat as single sequence
            if isinstance(stop_sequences, list) and all(isinstance(s, str) for s in stop_sequences):
                 generation_config_params['stop_sequences'] = stop_sequences
            else:
                 self.warning(f"Invalid stop_sequences format: {stop_sequences}. Must be a list of strings. Ignoring.")


        # Handle seed warning (as before)
        if gpt_params.get('seed', -1) != -1:
             self.warning("Seed parameter is provided but not directly supported in google-generativeai's GenerationConfig. Generation might not be deterministic.")

        # Warn about unsupported parameters (as before)
        # unsupported_params = ['repeat_penalty', 'presence_penalty', 'frequency_penalty', 'logit_bias'] # Add others if known
        # for param in unsupported_params:
        #     if param in gpt_params:
        #         self.warning(f"{param} is not directly supported by the Gemini API via this SDK and will be ignored.")

        output = ""
        try:
            generation_config = self.genai.types.GenerationConfig(**generation_config_params)

            if verbose: ASCIIColors.info(f"Generating text with effective config: {generation_config_params}")
            ASCIIColors.info("Starting generation stream...")

            # Start generation stream
            response_stream = self.model.generate_content(
                prompt,
                generation_config=generation_config,
                stream=True,
                safety_settings=self.safety_settings
            )

            chunk_count = 0
            # --- Streaming Loop ---
            try:
                for chunk in response_stream:
                    # (Streaming loop logic remains the same as before)
                    try:
                        if hasattr(chunk, 'text'):
                            word = chunk.text
                        elif hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                            reason = chunk.prompt_feedback.block_reason.name # Use name for readability
                            self.error(f"Generation stopped due to prompt feedback: {reason}")
                            if callback: callback(f"\nGeneration stopped by safety filter: {reason}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
                            break
                        elif chunk.candidates and chunk.candidates[0].finish_reason:
                            finish_reason = chunk.candidates[0].finish_reason.name # Use name
                            if finish_reason not in ["STOP", "MAX_TOKENS", "UNSPECIFIED"]: # Log warnings for others
                                self.warning(f"Generation finish reason: {finish_reason}")
                            if verbose: ASCIIColors.info(f"Generation finished with reason: {finish_reason}")
                            word=""
                        else:
                            if verbose: ASCIIColors.debug(f"Received non-text chunk: {chunk}")
                            word = ""
                    except AttributeError as e:
                        self.warning(f"Error accessing chunk content: {e}. Chunk: {chunk}")
                        word = ""
                    except StopIteration:
                        ASCIIColors.info("Generation stream finished.")
                        break
                    except Exception as e:
                        self.error(f"Error processing generation chunk: {e}")
                        trace_exception(e)
                        word = ""

                    if word:
                        output += word
                        chunk_count += 1
                        if callback is not None:
                            if not callback(word, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                                ASCIIColors.info("Generation stopped by callback.")
                                # Consider trying to close the stream explicitly if SDK supports it
                                # response_stream.close() # Fictional method
                                break
            except Exception as ex:
                trace_exception(ex)
            # --- End Streaming Loop ---

            ASCIIColors.success("Generation process completed.")

        # --- Exception Handling ---
        # (Exception handling remains the same)
        except google.api_core.exceptions.PermissionDenied as e:
             self.error(f"Permission Denied: {e}. Check your API key and ensure the Generative Language API is enabled.")
             if callback: callback(f"\nError: Permission Denied - {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""
        except google.api_core.exceptions.ResourceExhausted as e:
             self.error(f"Resource Exhausted: {e}. You may have hit API rate limits or quota.")
             if callback: callback(f"\nError: API Quota Exceeded or Rate Limited - {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""
        except google.api_core.exceptions.InvalidArgument as e:
             # Provide more context if possible
             self.error(f"Invalid Argument: {e}. Check prompt length against context window ({self.binding_config.ctx_size}), parameters ({generation_config_params}), or safety settings.")
             if callback: callback(f"\nError: Invalid Argument - {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""
        except Exception as e:
            self.error(f"Gemini generation failed: {e}")
            trace_exception(e)
            if callback: callback(f"\nError: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return ""

        return output


    def generate_with_images(self,
                prompt:str,
                images:list=[],
                n_predict: int = None, # User override
                callback: Callable[[str, int, dict], bool] = None,
                verbose: bool = False,
                **gpt_params ):
        """
        Generates text based on a prompt and images using a Gemini vision model.

        Args:
            prompt (str): The text prompt.
            images (list): List of file paths to images.
            n_predict (int, optional): Overrides max tokens for this call.
            callback (Callable): Streaming callback.
            verbose (bool): Verbose logging.
            **gpt_params: Generation parameters.

        Returns:
            str: Generated text or empty string on failure.
        """
        if not self.model:
            self.error("Model not initialized.")
            self.settings_updated() # Attempt recovery
            if not self.model:
                 self.error("Failed to rebuild model. Aborting generation.")
                 return ""

        if self.binding_type != BindingType.TEXT_IMAGE:
            self.error(f"The selected model '{self.config.model_name}' does not support image input.")
            if callback: callback(f"\nError: Model {self.config.model_name} does not support images.", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return ""

        if not images:
             self.warning("generate_with_images called but no images were provided. Falling back to text-only generation.")
             return self.generate(prompt, n_predict, callback, verbose, **gpt_params)

        # --- Determine max tokens (same logic as text generation) ---
        model_max_tokens = self.binding_config.max_output_tokens
        generate_max_tokens = model_max_tokens
        if n_predict is not None:
             if n_predict > 0 and n_predict <= model_max_tokens:
                 generate_max_tokens = n_predict
                 ASCIIColors.info(f"Using user-provided n_predict: {generate_max_tokens}")
             elif n_predict > model_max_tokens:
                 self.warning(f"Requested n_predict ({n_predict}) exceeds model's limit ({model_max_tokens}). Using model limit.")
             else:
                 self.warning(f"Invalid n_predict value ({n_predict}). Using model's max output tokens: {model_max_tokens}")
        else:
             ASCIIColors.info(f"Using model's default max_output_tokens: {generate_max_tokens}")

        # --- Prepare generation configuration (same logic as text generation) ---
        # Vision models might benefit from different defaults, but let's keep consistency for now
        generation_config_params = {
            'candidate_count': 1,
            'max_output_tokens': generate_max_tokens,
            'temperature': float(gpt_params.get('temperature', 0.4)), # Slightly lower default for vision?
            'top_p': float(gpt_params.get('top_p', 0.95)),
            'top_k': int(gpt_params.get('top_k', 32)), # Common default for vision
        }
        stop_sequences = gpt_params.get('stop_sequences', None) or gpt_params.get('stop', None)
        if stop_sequences:
            # (Same stop sequence handling as in generate)
            if isinstance(stop_sequences, str):
                try: stop_sequences = json.loads(stop_sequences)
                except json.JSONDecodeError: stop_sequences = [stop_sequences]
            if isinstance(stop_sequences, list) and all(isinstance(s, str) for s in stop_sequences):
                 generation_config_params['stop_sequences'] = stop_sequences
            else:
                 self.warning(f"Invalid stop_sequences format: {stop_sequences}. Ignoring.")

        if gpt_params.get('seed', -1) != -1:
             self.warning("Seed parameter is provided but not directly supported in google-generativeai's GenerationConfig.")
        # (Unsupported param warnings remain the same)

        # --- Prepare Content List [prompt, image1_PIL, image2_PIL, ...] ---
        content_parts = [prompt]
        loaded_images = []
        for image_path in images:
            try:
                # (Image loading logic remains the same, using PIL)
                valid_image_path = find_first_available_file_path([image_path])
                if not valid_image_path or not is_file_path(valid_image_path):
                     self.warning(f"Image path not found or invalid: {image_path}. Skipping.")
                     continue
                img = Image.open(valid_image_path)
                # It's good practice to ensure format compatibility if needed, but Gemini usually handles common ones.
                # img = img.convert("RGB") # Optional: Ensure RGB
                loaded_images.append(img) # Keep track for closing later
                content_parts.append(img) # Append PIL object
                ASCIIColors.info(f"Added image: {valid_image_path}")
            except FileNotFoundError:
                 self.error(f"Image file not found: {image_path}. Skipping.")
                 if callback: callback(f"\nWarning: Image not found {image_path}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING)
            except Exception as e:
                self.error(f"Failed to load or process image {image_path}: {e}")
                trace_exception(e)
                if callback: callback(f"\nWarning: Failed to load image {image_path}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_WARNING)
                # Decide whether to continue or fail - let's continue if at least one image loads

        if len(content_parts) == 1: # Only prompt left
             self.error("All provided images failed to load. Aborting vision generation.")
             # Close any PIL images that *might* have loaded before a later failure (though unlikely in current loop)
             for img in loaded_images: img.close()
             return ""
        elif len(loaded_images) < len(images):
             self.warning("Some images failed to load. Proceeding with the successfully loaded ones.")

        output = ""
        try:
            generation_config = self.genai.types.GenerationConfig(**generation_config_params)
            if verbose: ASCIIColors.info(f"Generating text with images. Effective config: {generation_config_params}")
            ASCIIColors.info("Starting generation stream (with images)...")

            # Start generation stream
            response_stream = self.model.generate_content(
                content_parts, # Pass the list [prompt, PIL_Image1, ...]
                generation_config=generation_config,
                stream=True,
                safety_settings=self.safety_settings
            )

            # --- Streaming Loop (Identical to text generation) ---
            for chunk in response_stream:
                # (Streaming loop logic remains the same)
                try:
                    if hasattr(chunk, 'text'):
                         word = chunk.text
                    elif hasattr(chunk, 'prompt_feedback') and chunk.prompt_feedback.block_reason:
                         reason = chunk.prompt_feedback.block_reason.name
                         self.error(f"Generation stopped due to prompt feedback: {reason}")
                         if callback: callback(f"\nGeneration stopped by safety filter: {reason}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
                         break
                    elif chunk.candidates and chunk.candidates[0].finish_reason:
                         finish_reason = chunk.candidates[0].finish_reason.name
                         if finish_reason not in ["STOP", "MAX_TOKENS", "UNSPECIFIED"]: self.warning(f"Generation finish reason: {finish_reason}")
                         if verbose: ASCIIColors.info(f"Generation finished with reason: {finish_reason}")
                         word = ""
                    else:
                         if verbose: ASCIIColors.debug(f"Received non-text chunk: {chunk}")
                         word = ""
                except AttributeError as e:
                    self.warning(f"Error accessing chunk content: {e}. Chunk: {chunk}")
                    word = ""
                except StopIteration:
                     ASCIIColors.info("Generation stream finished.")
                     break
                except Exception as e:
                     self.error(f"Error processing generation chunk: {e}")
                     trace_exception(e)
                     word = ""

                if word:
                    output += word
                    if callback is not None:
                        if not callback(word, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK):
                            ASCIIColors.info("Generation stopped by callback.")
                            break
            # --- End Streaming Loop ---

            ASCIIColors.success("Generation with images process completed.")

        # --- Exception Handling (Mostly same as text generation) ---
        except google.api_core.exceptions.PermissionDenied as e:
             # (Same handling)
             self.error(f"Permission Denied: {e}. Check API key/permissions.")
             if callback: callback(f"\nError: Permission Denied - {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""
        except google.api_core.exceptions.ResourceExhausted as e:
             # (Same handling)
             self.error(f"Resource Exhausted: {e}. Rate limits or quota hit.")
             if callback: callback(f"\nError: API Quota Exceeded or Rate Limited - {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""
        except google.api_core.exceptions.InvalidArgument as e:
             # More likely related to images here
             self.error(f"Invalid Argument: {e}. Check image formats/sizes/content, prompt length, parameters ({generation_config_params}), or safety settings.")
             if callback: callback(f"\nError: Invalid Argument - {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
             return ""
        except Exception as e:
            # (Same handling)
            self.error(f"Gemini generation with images failed: {e}")
            trace_exception(e)
            if callback: callback(f"\nError: {e}", MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION)
            return ""
        finally:
             # --- Clean up loaded PIL Image objects ---
             ASCIIColors.info(f"Closing {len(loaded_images)} loaded image objects.")
             for img in loaded_images:
                 try:
                     img.close()
                 except Exception as close_err:
                     # Log but don't fail the whole process
                     self.warning(f"Could not close image object: {close_err}")

        return output


    def list_models(self):
        """
        Lists the names of available Gemini models that support content generation.

        Returns:
            list: A list of model name strings (e.g., "gemini-pro", "gemini-1.5-pro-latest").
                  Returns empty list on failure.
        """
        # (list_models logic remains the same)
        if not self.genai:
            self.error("Gemini (genai) client not configured. Cannot list models.")
            if self.binding_config.google_api_key:
                 try: self.genai.configure(api_key=self.binding_config.google_api_key)
                 except Exception as e:
                    self.error(f"Failed to configure genai for listing models: {e}")
                    return []
            else: return []

        try:
            self.genai.configure(api_key=self.binding_config.google_api_key)
            ASCIIColors.info("Fetching list of available Gemini models from API...")
            api_models = self.genai.list_models()

            supported_models = []
            for m in api_models:
                if 'generateContent' in m.supported_generation_methods:
                    model_id = m.name.split('/')[-1]
                    supported_models.append(model_id)

            ASCIIColors.info(f"Found {len(supported_models)} models supporting content generation.")
            supported_models.sort()
            return supported_models

        except google.api_core.exceptions.PermissionDenied:
             self.error("Permission Denied listing models. Check your API key.")
             return []
        except Exception as e:
            self.error(f"Failed to list Gemini models from API: {e}")
            trace_exception(e)
            return []


    def get_available_models(self, app: LoLLMsCom = None):
        """
        Gets a list of available Gemini models formatted for the Lollms UI.
        Includes fetched token limits in the metadata.

        Args:
            app (LoLLMsCom, optional): The Lollms communication object (unused here).

        Returns:
            list: List of model dictionaries for Lollms, or empty list on failure.
        """
        # (get_available_models logic remains largely the same, but ensures limits are included)
        if not self.genai:
            self.error("Gemini (genai) client not configured. Cannot get available models.")
            # (API key check/config retry remains the same)
            if self.binding_config.google_api_key:
                 try: self.genai.configure(api_key=self.binding_config.google_api_key)
                 except Exception as e:
                    self.error(f"Failed to configure genai for getting models: {e}")
                    return []
            else: return []

        lollms_models = []
        try:
            self.genai.configure(api_key=self.binding_config.google_api_key)
            ASCIIColors.info("Fetching detailed list of available Gemini models from API for Lollms UI...")
            api_models = self.genai.list_models()

            for model in api_models:
                if 'generateContent' not in model.supported_generation_methods:
                    continue

                model_id = model.name.split('/')[-1]
                display_name = getattr(model, 'display_name', model_id)
                description = getattr(model, 'description', 'N/A')
                # Fetch limits directly here for the list
                input_limit = getattr(model, 'input_token_limit', 'N/A')
                output_limit = getattr(model, 'output_token_limit', 'N/A')

                # Rank heuristic
                rank = 1.0
                if "ultra" in model_id: rank = 3.0
                elif "pro" in model_id: rank = 2.0
                elif "flash" in model_id: rank = 1.5
                if "1.5" in model_id: rank += 0.5 # Boost 1.5 models

                # Size proxy using limits
                size_proxy = -1
                try:
                     if isinstance(input_limit, int) and isinstance(output_limit, int):
                          size_proxy = input_limit + output_limit # Simple sum as proxy
                     elif isinstance(input_limit, int):
                          size_proxy = input_limit
                except: pass # Ignore errors calculating proxy

                # Check for vision support for category
                category = "generic"
                if "vision" in model_id.lower() or "gemini-1.5" in model_id.lower() or "gemini-2.0" in model_id.lower() or "gemini-2.5" in model_id.lower():
                    category="multimodal"


                md = {
                    "category": category,
                    "datasets": "proprietary", # More accurate than unknown
                    "icon": '/bindings/gemini/logo.png',
                    "last_commit_time": datetime.now().timestamp(), # Placeholder
                    "license": "commercial",
                    "model_creator": "google",
                    "model_creator_link": "https://ai.google.dev/",
                    "name": model_id,
                    "quantizer": None,
                    "rank": rank,
                    "type": "api",
                    "variants": [
                        {
                            "name": model_id,
                            "size": size_proxy # Use proxy or -1
                        }
                    ],
                    # Add useful metadata from API
                    "description": description,
                    "display_name": display_name,
                    "input_token_limit": input_limit,
                    "output_token_limit": output_limit,
                    "temperature": getattr(model, 'temperature', 'N/A'), # Default generation params if available
                    "top_p": getattr(model, 'top_p', 'N/A'),
                    "top_k": getattr(model, 'top_k', 'N/A'),
                }
                lollms_models.append(md)

            lollms_models.sort(key=lambda x: (-x['rank'], x['name']))
            ASCIIColors.success(f"Formatted {len(lollms_models)} models for Lollms.")

        except google.api_core.exceptions.PermissionDenied:
             self.error("Permission Denied getting available models. Check your API key.")
             return []
        except Exception as e:
            self.error(f"Failed to retrieve or process Gemini models for Lollms: {e}")
            trace_exception(e)
            return []

        return lollms_models