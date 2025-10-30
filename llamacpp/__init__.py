# bindings/llamacpp_server/binding.py
import json
import os
import pprint
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict, Any, Set
import base64
import requests # For HTTP client

# LOLPMS imports
from lollms.config import BaseConfig, TypedConfig, ConfigTemplate, InstallOption
from lollms.paths import LollmsPaths
from lollms.binding import LLMBinding, LOLLMSConfig, BindingType
from lollms.helpers import ASCIIColors, trace_exception
from lollms.com import NotificationType, LoLLMsCom
from lollms.types import MSG_OPERATION_TYPE, MSG_TYPE
from lollms.utilities import discussion_path_to_url, AdvancedGarbageCollector


import pipmaster as pm
import platform

binding_name = "LlamaCpp_Server" 

# --- llama-cpp-binaries installation function ---
def install_llama_cpp_binaries_pkg():
    """Installs the llama-cpp-binaries package from oobabooga's releases."""
    ASCIIColors.info("Attempting to install llama-cpp-binaries...")
    system = platform.system()
    
    version_tag = "v0.56.0" 
    cuda_suffix = "+cu124" 

    if system == "Windows":
        url = f"https://github.com/oobabooga/llama-cpp-binaries/releases/download/{version_tag}/llama_cpp_binaries-{version_tag.lstrip('v')}{cuda_suffix}-py3-none-win_amd64.whl"
    elif system == "Linux":
        url = f"https://github.com/oobabooga/llama-cpp-binaries/releases/download/{version_tag}/llama_cpp_binaries-{version_tag.lstrip('v')}{cuda_suffix}-py3-none-linux_x86_64.whl"
    else:
        ASCIIColors.error(f"Unsupported OS for precompiled llama-cpp-binaries: {system}. "
                          "You might need to set 'llama_server_binary_path' in the binding config "
                          "to point to a manually compiled llama.cpp server binary.")
        return False

    ASCIIColors.info(f"Downloading and installing from: {url}")
    if pm.install(url): 
        ASCIIColors.green("llama-cpp-binaries installed successfully via pipmaster.")
        try:
            import llama_cpp_binaries 
            ASCIIColors.info(f"llama_cpp_binaries version: {getattr(llama_cpp_binaries, '__version__', 'unknown')}")
            return True
        except ImportError:
            ASCIIColors.error("Failed to import llama-cpp-binaries after installation.")
            return False
    else:
        ASCIIColors.error("Failed to install llama-cpp-binaries using pipmaster.")
        return False

llama_cpp_binaries_module = None

def _lazy_load_llama_cpp_binaries():
    global llama_cpp_binaries_module
    if llama_cpp_binaries_module is None:
        try:
            import llama_cpp_binaries
            llama_cpp_binaries_module = llama_cpp_binaries
        except ImportError:
            ASCIIColors.debug("llama-cpp-binaries package not yet loaded or installed.")
    return llama_cpp_binaries_module

_QUANT_COMPONENTS_SET: Set[str] = {
    "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q2_K_S", "Q3_K_S", "Q4_K_S", "Q5_K_S",
    "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q3_K_L", "Q2_K_XS", "Q3_K_XS", "Q4_K_XS", "Q5_K_XS", "Q6_K_XS",
    "Q2_K_XXS", "Q3_K_XXS", "Q4_K_XXS", "Q5_K_XXS", "Q6_K_XXS", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
    "F16", "FP16", "F32", "FP32", "BF16", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
    "IQ3_XXS", "IQ3_S", "IQ3_M", "IQ4_NL", "IQ4_XS", "IQ3_M_K", "IQ3_S_K", "IQ4_XS_K", "IQ4_NL_K",
    "I8", "I16", "I32", "ALL_F32", "MOSTLY_F16", "MOSTLY_Q4_0", "MOSTLY_Q4_1", "MOSTLY_Q5_0",
    "MOSTLY_Q5_1", "MOSTLY_Q8_0", "MOSTLY_Q2_K", "MOSTLY_Q3_K_S", "MOSTLY_Q3_K_M", "MOSTLY_Q3_K_L",
    "MOSTLY_Q4_K_S", "MOSTLY_Q4_K_M", "MOSTLY_Q5_K_S", "MOSTLY_Q5_K_M", "MOSTLY_Q6_K",
    "MOSTLY_IQ1_S", "MOSTLY_IQ1_M", "MOSTLY_IQ2_XXS", "MOSTLY_IQ2_XS", "MOSTLY_IQ2_S", "MOSTLY_IQ2_M",
    "MOSTLY_IQ3_XXS", "MOSTLY_IQ3_S", "MOSTLY_IQ3_M", "MOSTLY_IQ4_NL", "MOSTLY_IQ4_XS"
}
_MODEL_NAME_SUFFIX_COMPONENTS_SET: Set[str] = {
    "instruct", "chat", "GGUF", "HF", "ggml", "pytorch", "AWQ", "GPTQ", "EXL2",
    "base", "cont", "continue", "ft", "v0.1", "v0.2", "v1.0", "v1.1", "v1.5", "v1.6", "v2.0",
}
_ALL_REMOVABLE_COMPONENTS: List[str] = sorted(
    list(_QUANT_COMPONENTS_SET.union(_MODEL_NAME_SUFFIX_COMPONENTS_SET)), key=len, reverse=True
)

def get_gguf_model_base_name(file_path_or_name: Union[str, Path]) -> str:
    if isinstance(file_path_or_name, str): p = Path(file_path_or_name)
    elif isinstance(file_path_or_name, Path): p = file_path_or_name
    else: raise TypeError(f"Input must be a string or Path. Got: {type(file_path_or_name)}")
    name_part = p.name
    if name_part.lower().endswith(".gguf"): name_part = name_part[:-5]
    while True:
        original_name_part_len = len(name_part)
        stripped_in_this_iteration = False
        for component in _ALL_REMOVABLE_COMPONENTS:
            component_lower = component.lower()
            for separator in [".", "-", "_"]:
                pattern_to_check = f"{separator}{component_lower}"
                if name_part.lower().endswith(pattern_to_check):
                    name_part = name_part[:-(len(pattern_to_check))]
                    stripped_in_this_iteration = True; break
            if stripped_in_this_iteration: break
        if not stripped_in_this_iteration or not name_part: break
    while name_part and name_part[-1] in ['.', '-', '_']: name_part = name_part[:-1]
    return name_part

DEFAULT_LLAMACPP_SERVER_HOST = "127.0.0.1" 

class LlamaCppServerProcess:
    def __init__(self, 
                 model_path: str,
                 lollms_com: LoLLMsCom,
                 server_binary_path: str, 
                 port: int,               
                 server_args: Dict[str, Any],
                 clip_model_path: Optional[str] = None,
                 ):
        self.model_path = Path(model_path)
        self.lollms_com = lollms_com 
        self.clip_model_path = Path(clip_model_path) if clip_model_path else None
        
        self.server_binary_path = Path(server_binary_path)
        self.server_args = server_args if server_args is not None else {} 
        self.port = port 

        self.process: Optional[subprocess.Popen] = None
        self.session = requests.Session() 
        self.host = self.server_args.get("host",DEFAULT_LLAMACPP_SERVER_HOST) 
        self.base_url = f"http://{self.host}:{self.port}"
        self.is_healthy = False
        self._stderr_lines: List[str] = [] 
        self._stdout_lines: List[str] = []
        self._stderr_thread: Optional[threading.Thread] = None
        self._stdout_thread: Optional[threading.Thread] = None

        if not self.model_path.exists(): raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not self.server_binary_path or not self.server_binary_path.exists(): 
            raise FileNotFoundError(f"Llama.cpp server binary not found: {self.server_binary_path}")

    def _log_output_line(self, line: str, source: str, buffer: List[str]):
        line_strip = line.strip()
        buffer.append(line_strip)
        if len(buffer) > 100: buffer.pop(0) 
        
        log_prefix = f"[LLAMA_SERVER_{source.upper()}]"
        if "llama_model_loaded" in line_strip or "error" in line_strip.lower() or "failed" in line_strip.lower():
            self.lollms_com.debug(f"{log_prefix} {line_strip}") 
        elif ("running on port" in line_strip and "http" in line_strip) or ("HTTP server listening" in line_strip):
            self.lollms_com.info(f"{log_prefix} {line_strip}")

    def _stream_output_pipe(self, pipe, source_name:str, buffer:List[str]):
        try:
            for line_bytes in iter(pipe.readline, b''):
                if line_bytes:
                    line_str = line_bytes.decode('utf-8', errors='replace')
                    self._log_output_line(line_str, source_name, buffer)
                else: break 
        except ValueError: self.lollms_com.debug(f"Pipe {source_name} closed for LlamaCppServerProcess.")
        except Exception as e: self.lollms_com.warning(f"Exception in {source_name} stream thread: {e}")
        finally: pipe.close()

    def start(self):
        cmd = [
            str(self.server_binary_path),
            "--model", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
        ]
        arg_map = {
            "n_ctx": "--ctx-size", "n_gpu_layers": "--gpu-layers", "main_gpu": "--main-gpu",
            "tensor_split": "--tensor-split", 
            "use_mmap": (lambda v: ["--no-mmap"] if not v else []),
            "use_mlock": (lambda v: ["--mlock"] if v else []), 
            "seed": "--seed", "n_batch": "--batch-size",
            "n_threads": "--threads", "n_threads_batch": "--threads-batch",
            #"rope_scaling_type": "--rope-scaling", "rope_freq_base": "--rope-freq-base",
            #"rope_freq_scale": "--rope-freq-scale",
            "embedding_mode": (lambda v: ["--embedding"] if v else []), 
            "verbose_server": (lambda v: ["--verbose"] if v else []),
            "chat_template": "--chat-template",
            "parallel": "--parallel",
            "cont_batching": (lambda v: ["--cont-batching"] if v else []),
        }
        
        if self.clip_model_path and self.clip_model_path.exists():
            cmd.extend(["--mmproj", str(self.clip_model_path)])
            if "llava" in self.model_path.name.lower() and not self.server_args.get("chat_template"):
                 self.lollms_com.info("LLaVA model detected. Consider setting a specific 'chat_template' (e.g., 'llava-1.5') if issues arise.")

        for key, cli_arg_or_fn in arg_map.items():
            val = self.server_args.get(key)
            if val is not None:
                if callable(cli_arg_or_fn): cmd.extend(cli_arg_or_fn(val))
                else: cmd.extend([cli_arg_or_fn, str(val)])
        
        extra_cli_flags_str = self.server_args.get("extra_cli_flags", "")
        if isinstance(extra_cli_flags_str, str) and extra_cli_flags_str.strip():
            cmd.extend(extra_cli_flags_str.split())

        self.lollms_com.info(f"Starting Llama.cpp server: {' '.join(cmd)}")

        env = os.environ.copy()
        if os.name == 'posix' and self.server_binary_path.parent != Path('.'): 
            lib_path_str = str(self.server_binary_path.parent.resolve())
            env['LD_LIBRARY_PATH'] = f"{lib_path_str}:{env.get('LD_LIBRARY_PATH', '')}".strip(':')

        try:
            self.process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=False, bufsize=0, env=env)
        except Exception as e:
            self.lollms_com.error(f"Failed to start llama.cpp server process: {e}"); trace_exception(e); raise

        self._stderr_thread = threading.Thread(target=self._stream_output_pipe, args=(self.process.stderr, "stderr", self._stderr_lines), daemon=True)
        self._stdout_thread = threading.Thread(target=self._stream_output_pipe, args=(self.process.stdout, "stdout", self._stdout_lines), daemon=True)
        self._stderr_thread.start(); self._stdout_thread.start()

        health_url = f"{self.base_url}/health"
        max_wait_time = self.server_args.get("server_startup_timeout", 120) 
        start_time = time.time(); logged_waiting = False
        
        while time.time() - start_time < max_wait_time:
            if self.process.poll() is not None:
                msg = (f"Server process terminated unexpectedly (code {self.process.poll()}) during startup.\n"
                       f"Stderr:\n{''.join(self._stderr_lines[-10:])}\nStdout:\n{''.join(self._stdout_lines[-10:])}")
                self.lollms_com.error(msg); raise RuntimeError(msg)
            try:
                response = self.session.get(health_url, timeout=2)
                if response.status_code == 200:
                    try:
                        health_json = response.json()
                        status = health_json.get("status")
                        if status == "ok":
                            self.is_healthy = True; self.lollms_com.success(f"Server started on port {self.port} for {self.model_path.name}."); return
                        elif status == "loading":
                            if not logged_waiting: self.lollms_com.info("Server loading model..."); logged_waiting=True
                            time.sleep(2); continue
                        else: self.lollms_com.warning(f"Health check OK but status is '{status}'.")
                    except json.JSONDecodeError: 
                         self.is_healthy = True; self.lollms_com.success(f"Server (non-JSON health) started on port {self.port}."); return
            except requests.exceptions.ConnectionError: time.sleep(1) 
            except Exception as e: self.lollms_com.debug(f"Health check attempt failed: {e}"); time.sleep(1)
        
        self.is_healthy = False; self.stop() 
        timeout_msg = (f"Server failed to become healthy on port {self.port} within {max_wait_time}s.\n"
                       f"Stderr:\n{''.join(self._stderr_lines[-10:])}\nStdout:\n{''.join(self._stdout_lines[-10:])}")
        self.lollms_com.error(timeout_msg); raise TimeoutError(timeout_msg)

    def stop(self):
        self.is_healthy = False
        if self.process:
            pid = self.process.pid
            self.lollms_com.info(f"Stopping Llama.cpp server (PID: {pid})...")
            try:
                self.process.terminate(); self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.lollms_com.warning(f"Server (PID: {pid}) kill timeout. Forcing kill..."); self.process.kill()
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired: self.lollms_com.error(f"Failed to kill server (PID: {pid}).")
            except Exception as e: self.lollms_com.error(f"Error stopping server (PID: {pid}): {e}")
            finally:
                self.process = None
                for thread in [self._stderr_thread, self._stdout_thread]:
                    if thread and thread.is_alive(): thread.join(timeout=1)
                self.lollms_com.info(f"Llama.cpp server (was PID: {pid}) stopped.")


class LlamaCpp_Server(LLMBinding):
    binding_name = "LlamaCpp_Server" 
    
    DEFAULT_SERVER_ARGS_TEMPLATE = {
        "host": DEFAULT_LLAMACPP_SERVER_HOST, 
        "preferred_port":9641, 
        "n_gpu_layers": -1, "main_gpu": 0, "n_ctx": 4096, "n_batch": 512, "embedding_mode": False, 
        "verbose_server": False, "server_startup_timeout": 120, "use_mmap": True, "use_mlock": False,
        "seed": -1, "n_threads": 0, "n_threads_batch": 0, 
        "rope_scaling_type": "", "rope_freq_base": 0.0, "rope_freq_scale": 0.0, 
        "chat_template": "", "llama_server_binary_path": "", "extra_cli_flags": "",
        "clip_model_name_hint": "", "parallel":1, "cont_batching": False,
        "temperature": 0.7, "top_k": 40, "top_p": 0.9, "repeat_penalty": 1.1, "repeat_last_n": 64,
        "mirostat_mode": 0, "mirostat_tau": 5.0, "mirostat_eta": 0.1, "grammar_string": "",
        "generation_timeout": 300,
    }

    def __init__(self, 
                config: LOLLMSConfig, 
                lollms_paths: Optional[LollmsPaths] = None, # Changed: Optional with default None
                installation_option: InstallOption = InstallOption.INSTALL_IF_NECESSARY,
                lollmsCom: Optional[LoLLMsCom] = None      # Kept Optional[LoLLMsCom] = None
                ) -> None:
        
        if lollms_paths is None: # Added: Instantiate if None
            lollms_paths = LollmsPaths()
            
        bc_template_list = []
        for k, v_default in self.DEFAULT_SERVER_ARGS_TEMPLATE.items():
            entry = {"name":k, "value":v_default}
            if isinstance(v_default, bool): entry["type"] = "bool"
            elif isinstance(v_default, int): entry["type"] = "int"
            elif isinstance(v_default, float): entry["type"] = "float"
            else: entry["type"] = "str" 
            bc_template_list.append(entry)

        binding_config_template = ConfigTemplate(bc_template_list)
        binding_config_vals = BaseConfig.from_template(binding_config_template)
        
        # Pass lollmsCom (which can be None) directly to super().
        # LLMBinding's __init__ will handle creating a default LoLLMsCom if it's None.
        super().__init__(
            Path(__file__).parent, 
            lollms_paths = lollms_paths, 
            config = config, 
            binding_config = TypedConfig(binding_config_template, binding_config_vals), 
            installation_option = installation_option, 
            SAFE_STORE_SUPPORTED_FILE_EXTENSIONS=['.gguf'], 
            models_dir_names=["gguf", f"gguf_{self.binding_name.lower()}"],
            lollmsCom=lollmsCom # Pass lollmsCom directly
        )
        self.config.ctx_size = self.binding_config.n_ctx 

        self.server_process: Optional[LlamaCppServerProcess] = None
        self.port: Optional[int] = None 
        self.current_model_path: Optional[Path] = None
        self.current_clip_model_path: Optional[Path] = None
        self.server_binary_actual_path: Optional[Path] = None

        _lazy_load_llama_cpp_binaries() 

    def settings_updated(self):
        self.config.ctx_size = self.binding_config.n_ctx
        # Note: Unlike the example, we don't set self.config.max_n_predict here,
        # as max_n_predict for the server is a per-request generation parameter,
        # not a persistent server configuration reflected in LOLLMSConfig.
        if self.server_process and self.server_process.is_healthy:
            self.InfoMessage("Binding settings updated. If server parameters (GPU, context, paths, etc.) or generation defaults changed, "
                             "you may need to reload the model for some changes to take effect. "
                             "Generation parameter changes (e.g. temperature) usually apply to the next generation.")


    def _get_server_binary_path(self) -> Path:
        custom_path_str = self.binding_config.llama_server_binary_path
        if custom_path_str:
            custom_path = Path(custom_path_str)
            if custom_path.exists() and custom_path.is_file():
                self.InfoMessage(f"Using custom server binary: {custom_path}"); return custom_path.resolve()
            else: self.WarningMessage(f"Custom binary path '{custom_path_str}' invalid. Falling back.")

        binaries_mod = _lazy_load_llama_cpp_binaries()
        if not binaries_mod: raise FileNotFoundError("llama-cpp-binaries not installed/loaded and no custom path set.")
        
        try:
            server_bin_path_str = binaries_mod.get_binary_path()
            if server_bin_path_str:
                server_bin_path = Path(server_bin_path_str)
                if server_bin_path.exists() and server_bin_path.is_file():
                    self.InfoMessage(f"Using server binary from llama-cpp-binaries: {server_bin_path}"); return server_bin_path.resolve()
            raise FileNotFoundError("Could not locate valid server binary via llama-cpp-binaries.")
        except Exception as e: self.error(f"Error getting server binary path: {e}"); trace_exception(e); raise

    def _find_available_port(self) -> int:
        host_to_check = self.binding_config.host 
        current_port = self.binding_config.preferred_port
        for _ in range(100): 
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try: 
                    s.bind((host_to_check, current_port))
                    return s.getsockname()[1] 
                except socket.error as e:
                    if e.errno == socket.errno.EADDRINUSE: current_port += 1
                    else: raise
        raise RuntimeError(f"Could not find an available port starting from {self.binding_config.preferred_port} on host {host_to_check}.")

    def searchModelPath(self, model_name:str):
        model_path=self.searchModelFolder(model_name)
        mp:Path = None
        for f in model_path.iterdir():
            for ff in f.iterdir():
                a =  Path(model_name).stem.lower()
                b = ff.name.lower()
                if a in b :
                    mp = ff
                    return mp
        return None

    def build_model(self, model_name=None): 
        super().build_model(model_name) 
        
        if not _lazy_load_llama_cpp_binaries() and not self.binding_config.llama_server_binary_path:
            self.error("llama-cpp-binaries not installed or custom path not set. Please install binding or set path."); return None
        
        model_path = self.get_model_path()
        if not model_path or not model_path.exists(): self.error(f"Model not found: {model_path}"); return None

        # Check if server is already running with the same model and settings that affect server startup
        # For simplicity, we'll just check model_path. A more robust check would compare critical server args.
        if self.server_process and self.server_process.is_healthy and self.current_model_path == model_path:
            self.InfoMessage(f"Model '{model_path.name}' already loaded on port {self.port}."); return self

        if self.server_process: self.unload_model()
        self.current_model_path = model_path
        
        try:
            self.port = self._find_available_port()
            self.server_binary_actual_path = self._get_server_binary_path()
        except Exception as e: self.error(f"Failed to prep server start: {e}"); trace_exception(e); self.current_model_path = None; return None
        
        self.InfoMessage(f"Starting server for {self.current_model_path.name} on port {self.port}")
        current_server_args = self.binding_config.config

        self.current_clip_model_path = None; base_name_no_ext = get_gguf_model_base_name(self.current_model_path.stem)
        clip_hint = self.binding_config.clip_model_name_hint
        search_paths_for_clip = [self.current_model_path.parent]
        # Use self.models_dir_names to access "gguf_llamacpp_server" correctly
        binding_specific_models_path_segment = self.models_dir_names[1] if len(self.models_dir_names) > 1 else self.binding_folder_name

        for sub_dir in ["projectors", "clip", "mmproj"]: 
            search_paths_for_clip.append(self.lollms_paths.personal_models_path / binding_specific_models_path_segment / sub_dir)
        
        potential_clip_filenames = []
        if clip_hint: potential_clip_filenames.append(clip_hint)
        potential_clip_filenames.extend([
            f"{base_name_no_ext}.mmproj", f"mmproj-{base_name_no_ext}.gguf", self.current_model_path.with_suffix(".mmproj").name
        ])

        for search_dir in search_paths_for_clip:
            if search_dir.exists() and search_dir.is_dir():
                for fname in potential_clip_filenames:
                    p_clip = search_dir / fname
                    if p_clip.exists(): self.current_clip_model_path = p_clip; break
            if self.current_clip_model_path: break
        
        if self.current_clip_model_path:
            self.InfoMessage(f"LLaVA clip model found: {self.current_clip_model_path}"); self.binding_type = BindingType.TEXT_IMAGE
        else:
            self.binding_type = BindingType.TEXT_ONLY
            if any(kw in self.current_model_path.name.lower() for kw in ["llava", "bakllava", "vision"]):
                self.WarningMessage("Vision model name, but no .mmproj found. Vision may not work.")
        
        try:
            self.server_process = LlamaCppServerProcess(
                model_path=str(self.current_model_path), 
                lollms_com=self.lollmsCom, # Use self.lollmsCom, which is set by super().__init__
                server_binary_path=str(self.server_binary_actual_path), 
                port=self.port, 
                server_args=current_server_args, 
                clip_model_path=str(self.current_clip_model_path) if self.current_clip_model_path else None
            )
            self.server_process.start() 
            if self.server_process.is_healthy: self.model = self.server_process; return self 
            else: self.unload_model(); return None 
        except Exception as e: self.error(f"Failed to start server for {self.current_model_path.name}: {e}"); trace_exception(e); self.unload_model(); return None

    def unload_model(self):
        if self.server_process: self.server_process.stop(); self.server_process = None
        self.current_model_path = self.current_clip_model_path = self.port = self.model = None
        self.InfoMessage("Server and model unloaded."); AdvancedGarbageCollector.collect()
    
    def __del__(self): 
        self.unload_model()

    def _get_request_url(self, endpoint: str) -> str:
        if not (self.server_process and self.server_process.is_healthy):
            if self.current_model_path : 
                self.WarningMessage("Server unhealthy. Attempting restart..."); 
                if not self.build_model(self.config.model_name): 
                    raise ConnectionError("Server restart failed.")
            else: raise ConnectionError("Server not running/healthy. Load model first.")
        return f"{self.server_process.base_url}{endpoint}"

    def _prepare_generation_payload(self, messages:list, n_predict:Optional[int]=None,
                                   images:Optional[List[str]]=None, use_chat_format:bool=True, stream:bool=False,
                                   gen_params:Dict[str,Any]={}) -> Dict:
        payload = {k: self.binding_config.config.get(k,None) for k in [ # Use get_safe for TypedConfig
            "temperature", "top_k", "top_p", "repeat_penalty", "repeat_last_n", 
            "mirostat_tau", "mirostat_eta", "grammar_string" 
        ]}
        payload["seed"] = self.binding_config.seed if self.binding_config.seed != -1 else None 
        payload["mirostat"] = self.binding_config.mirostat_mode 

        payload.update(gen_params) 
        if n_predict is not None: payload['n_predict'] = n_predict
        
        payload = {k: v for k, v in payload.items() if v is not None} 
        if payload.get("grammar_string") == "": payload.pop("grammar_string",None)

        if use_chat_format:
            if images and self.binding_type == BindingType.TEXT_IMAGE:
                image_parts = []
                for img_path_str in images:
                    try:
                        img_path = Path(img_path_str) 
                        with open(img_path, "rb") as f: encoded_img = base64.b64encode(f.read()).decode("utf-8")
                        img_type = img_path.suffix[1:].lower() or "png"
                        if img_type == "jpg": img_type = "jpeg"
                        image_parts.append({"type": "image_url", "image_url": {"url": f"data:image/{img_type};base64,{encoded_img}"}})
                    except Exception as ex: self.error(f"Error processing image {img_path_str}: {ex}"); trace_exception(ex)
                if image_parts: messages.append(image_parts) # type: ignore
            final_payload = {"messages": messages, "stream": stream, **payload}
            if 'n_predict' in final_payload: final_payload['max_tokens'] = final_payload.pop('n_predict') 
            return final_payload
        else: 
            full_prompt = "\n".join([f"{self.lollmsCom.user_full_header if m['role']=='user' else self.lollmsCom.ai_full_header if m['role']=='assistant' else self.lollmsCom.system_full_header}:{m['content']}" for m in messages])+self.lollmsCom.ai_full_header
            final_payload = {"prompt": full_prompt, "stream": stream, **payload}
            if images and self.binding_type == BindingType.TEXT_IMAGE:
                img_data_list = []
                for i, img_path_str in enumerate(images):
                    try:
                        img_path = Path(img_path_str)
                        with open(img_path, "rb") as f: encoded_img = base64.b64encode(f.read()).decode("utf-8")
                        img_data_list.append({"data": encoded_img, "id": i + 10}) 
                    except Exception as e: self.error(f"Error encoding image {img_path_str} for /completion: {e}")
                if img_data_list:
                    final_payload["image_data"] = img_data_list
                    if not any(f"<image {i+1}>" in final_payload["prompt"] for i in range(len(img_data_list))):
                        self.WarningMessage("Images for /completion, but prompt may lack <image N> placeholders.")
            return final_payload

    def generate(self, prompt:str, n_predict:int=None, callback:Optional[Callable[[str,int,dict],bool]]=None,
                 verbose:bool=False, **gpt_params) -> str:
        use_chat_format = gpt_params.pop("use_chat_completions_format", True)
        images = gpt_params.pop("images", None)
        messages=self.lollmsCom.parse_to_openai(prompt)
        payload = self._prepare_generation_payload(messages, n_predict, images, use_chat_format, callback is not None, gpt_params)
        endpoint = "/v1/chat/completions" if use_chat_format else "/completion"
        
        req_url = "" 
        try: 
            req_url = self._get_request_url(endpoint)
        except ConnectionError as e: 
            self.error(f"Generation failed: {e}")
            if callback: callback("",MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION) # Pass enum value
            return ""

        full_response_txt = ""
        try:
            timeout_val = self.binding_config.generation_timeout
            response = self.server_process.session.post(req_url, json=payload, stream=(callback is not None), timeout=timeout_val)
            response.raise_for_status()

            if callback: 
                for line_bytes in response.iter_lines():
                    if not line_bytes: continue
                    line_str = line_bytes.decode('utf-8').strip()
                    if line_str.startswith('data: '): line_str = line_str[len('data: '):]
                    if line_str == '[DONE]': break
                    try:
                        chunk_data = json.loads(line_str)
                        chunk_content = (chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '') if use_chat_format 
                                         else chunk_data.get('content', ''))
                        if chunk_content:
                            full_response_txt += chunk_content
                            if not callback(chunk_content, MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_ADD_CHUNK): # Pass enum value
                                self.InfoMessage("Streaming stopped by callback."); response.close(); break
                        if (not use_chat_format and any(chunk_data.get(k,False) for k in ['stop','stopped_eos','stopped_limit'])) or \
                           (use_chat_format and chunk_data.get('choices', [{}])[0].get('finish_reason') is not None): break
                    except json.JSONDecodeError: self.WarningMessage(f"Failed to decode stream chunk: {line_str}")
                return full_response_txt
            else: 
                resp_data = response.json()
                return (resp_data.get('choices', [{}])[0].get('message', {}).get('content', '') if use_chat_format
                        else resp_data.get('content', ''))
        except requests.exceptions.RequestException as e:
            details = e.response.text[:200] if e.response else "No response details"
            self.error(f"Server request error for {req_url}: {e} - Details: {details}"); 
            if callback: callback(e,MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION) # Pass enum value
            return ""
        except Exception as ex: 
            self.error(f"Generation error with {req_url}: {ex}"); trace_exception(ex)
            if callback: callback(e,MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION) # Pass enum value
            return ""
    
    def tokenize(self, text: str) -> List[int]:
        try:
            resp = self.server_process.session.post(self._get_request_url("/tokenize"), json={"content": text})
            resp.raise_for_status(); return resp.json().get("tokens", [])
        except Exception as e: self.error(f"Tokenization error: {e}"); trace_exception(e); return []

    def detokenize(self, tokens: List[int]) -> str:
        try:
            resp = self.server_process.session.post(self._get_request_url("/detokenize"), json={"tokens": tokens})
            resp.raise_for_status(); return resp.json().get("content", "")
        except Exception as e: self.error(f"Detokenization error: {e}"); trace_exception(e); return ""

    def embed(self, text_or_texts: Union[str, List[str]], **kwargs) -> Union[List[float], List[List[float]]]:
        if not self.binding_config.embedding_mode: 
            self.error("Embedding mode is not enabled in the server configuration.")
            raise Exception("Embedding not enabled in server config.")
            
        is_list = isinstance(text_or_texts, list); inputs = text_or_texts if is_list else [text_or_texts]
        all_embs = []
        try:
            payload = {"input": inputs[0] if len(inputs) == 1 and not is_list else inputs}
            if "model" in kwargs: payload["model"] = kwargs["model"] 
            
            url_v1 = self._get_request_url("/v1/embeddings")
            resp_v1 = self.server_process.session.post(url_v1, json=payload)

            if resp_v1.status_code == 200:
                data = resp_v1.json()
                if "data" in data and isinstance(data["data"], list): 
                    all_embs = [item["embedding"] for item in sorted(data["data"], key=lambda x: x.get("index",0))]
                else: raise ValueError(f"Unexpected /v1/embeddings format: {data}")
            elif resp_v1.status_code == 404: 
                self.InfoMessage("/v1/embeddings not found, trying /embedding (single text mode). This may be less efficient for multiple texts.")
                if len(inputs)>1:
                    self.WarningMessage("Using /embedding fallback for multiple texts. This will make multiple API calls.")
                for txt in inputs:
                    resp_emb = self.server_process.session.post(self._get_request_url("/embedding"), json={"content": txt})
                    resp_emb.raise_for_status(); data_emb = resp_emb.json()
                    if "embedding" in data_emb: all_embs.append(data_emb["embedding"])
                    else: raise ValueError(f"Unexpected /embedding format: {data_emb}")
            else: resp_v1.raise_for_status()
            return all_embs if is_list else (all_embs[0] if all_embs else []) 
        except requests.exceptions.RequestException as e: 
            err_details = e.response.text[:100] if e.response else "No response details"
            err=f"Embedding request error: {e} - {err_details}"
            self.error(err); raise Exception(err) from e
        except Exception as ex: 
            err=f"Embedding failed: {ex}"
            self.error(err); trace_exception(ex); raise Exception(err) from ex
        
    def list_models(self) -> List[Dict[str, str]]:
        models_found = []
        scan_paths = set()
        for models_dir_name in self.models_dir_names: 
            if self.lollms_paths.personal_models_path.exists():
                 scan_paths.add(self.lollms_paths.personal_models_path / models_dir_name)
            if self.lollms_paths.models_zoo_path.exists():
                 scan_paths.add(self.lollms_paths.models_zoo_path / models_dir_name)
        
        if self.lollms_paths.personal_models_path.exists():
            scan_paths.add(self.lollms_paths.personal_models_path / "gguf") # Common shared GGUF

        unique_files = set()
        for spath in scan_paths:
            if spath.exists() and spath.is_dir():
                for model_file in spath.rglob(f"*{self.SAFE_STORE_SUPPORTED_FILE_EXTENSIONS[0]}"): 
                    if model_file.is_file() and model_file not in unique_files:
                        try: size_gb_str = f"{model_file.stat().st_size / (1024**3):.2f} GB"
                        except: size_gb_str = "Unknown size"
                        models_found.append({'name': model_file.name, 'size': size_gb_str, 'path': str(model_file)})
                        unique_files.add(model_file)
        return [m["name"] for m in sorted(models_found, key=lambda m: m['name'])]

    def install(self):
        self.ShowBlockingMessage("Installing LlamaCpp_Server binding requirements...")
        self.InfoMessage("Ensuring base packages (requests, pillow)...")
        try: 
            pm.ensure_packages(["requests","Pillow"]) 
            self.success("Base packages ensured.")
        except Exception as e: 
            self.error(f"Failed ensuring base packages: {e}")
            self.HideBlockingMessage()
            return

        global llama_cpp_binaries_module
        _lazy_load_llama_cpp_binaries() 
        if not llama_cpp_binaries_module and not self.binding_config.llama_server_binary_path:
            self.InfoMessage("llama-cpp-binaries not found or custom path not set. Attempting installation...")
            if install_llama_cpp_binaries_pkg(): 
                self.InfoMessage("llama-cpp-binaries installed successfully.")
                _lazy_load_llama_cpp_binaries() 
            else: 
                self.error("Failed to install llama-cpp-binaries. "
                                "Please set 'llama_server_binary_path' in the binding configuration to a precompiled server binary, "
                                "or try installing this binding again. Check the console for error messages from pipmaster.")
        elif llama_cpp_binaries_module: 
            self.InfoMessage(f"llama-cpp-binaries (version: {getattr(llama_cpp_binaries_module, '__version__', 'unknown')}) already available.")
        else: 
            self.InfoMessage(f"Using custom server binary path: '{self.binding_config.llama_server_binary_path}'. Skipping llama-cpp-binaries installation check.")
        
        self.HideBlockingMessage()
        self.InfoMessage("Installation process for LlamaCpp_Server binding finished.")


    def uninstall(self):
        super().uninstall() 
        if self.server_process: self.unload_model()
        self.InfoMessage("LlamaCpp_Server binding uninstalled.")
        self.InfoMessage("If 'llama-cpp-binaries' was installed by this binding and is no longer needed by other tools, "
                         f"you can uninstall it manually using: `{sys.executable} -m pip uninstall llama-cpp-binaries`")

if __name__ == '__main__':
    from lollms.paths import LollmsPaths
    from lollms.main_config import LOLLMSConfig

    class TestCom(LoLLMsCom):
        def __init__(self,binding_instance_or_name): super().__init__(binding_instance_or_name) 
        def InfoMessage(self,m): ASCIIColors.info(f"INFO:{m}")
        def ErrorMessage(self,m): ASCIIColors.error(f"ERR:{m}")
        def WarningMessage(self,m): ASCIIColors.warning(f"WARN:{m}")
        def SuccessMessage(self,m): ASCIIColors.green(f"OK:{m}")
        def DebugMessage(self,m): ASCIIColors.blue(f"DEBUG:{m}")
        def ShowBlockingMessage(self,m): print(f"SHOW_BLOCK:{m}")
        def HideBlockingMessage(self): print(f"HIDE_BLOCK")

    ASCIIColors.yellow("--- LlamaCpp_Server Binding Test ---")
    # Use a unique prefix for test paths to avoid conflicts
    lollms_paths_instance = LollmsPaths.find_paths(force_local=True, tool_prefix="lcpp_srv_test_") 
    global_config = LOLLMSConfig.autoload(lollms_paths_instance) 

    active_binding = None
    try:
        # For testing, explicitly pass lollms_paths and a TestCom instance
        active_binding = LlamaCpp_Server(
            config=global_config, 
            lollms_paths=lollms_paths_instance, 
            lollmsCom=TestCom("LlamaCpp_Server_Test_Instance") # Pass instance or name
        )
        
        # --- Installation Test (Uncomment to run) ---
        # print("Running install test...")
        # active_binding.install() 
        # print("Install finished. Please ensure a GGUF model is available and configured.")
        # print("If llama-cpp-binaries was installed, you might need to restart for path updates if it was the first time.")
        # exit() 

        models = active_binding.list_models()
        if not models: 
            print(f"No GGUF models found. Searched in paths like: {lollms_paths_instance.personal_models_path / 'gguf'}, etc.")
            print("Please place GGUF files in appropriate model paths and ensure 'model_name' in config is correct.")
            exit()
        print(f"Found models (first 3): {[m['name'] for m in models[:3]]}...")
        
        # !!! IMPORTANT: Set model_name in global_config for build_model() to pick it up !!!
        # global_config.model_name = models[0]['name'] # Example: use the first model found
        # Or specify a known model file name that exists in your model paths:
        # global_config.model_name = "your-test-model.gguf" 
        
        # Check if model_name is set in global_config, if not, try to use the first one from list
        if not global_config.model_name and models:
            print(f"Warning: global_config.model_name is not set. Attempting to use first model: {models[0]['name']}")
            global_config.model_name = models[0]['name']
        elif not global_config.model_name and not models:
            print("Error: No models found and global_config.model_name is not set. Cannot proceed with build_model test.")
            exit()


        print(f"Attempting to load model: {global_config.model_name}")
        active_binding.binding_config.config.n_gpu_layers = 0 
        active_binding.binding_config.config.embedding_mode = True
        active_binding.binding_config.config.verbose_server = False # Keep false for cleaner test output unless debugging server

        if not active_binding.build_model(): # build_model uses self.config.model_name
            raise RuntimeError(f"Model build failed for {global_config.model_name}. Check logs and model path.")
        
        print("\n--- Streaming Generation Test ---")
        def test_cb(chunk, msg_type_val, meta): 
            if msg_type_val == MSG_OPERATION_TYPE.MSG_OPERATION_TYPE_EXCEPTION:
                 print(f"\nCALLBACK_ERROR: {chunk} (Meta: {meta})")
                 return False 
            print(chunk, end="", flush=True)
            return True

        active_binding.generate("What is the capital of France? Respond concisely.", callback=test_cb, n_predict=50)
        
        print("\n\n--- Embedding Test ---")
        try:
            emb = active_binding.embed("Test embedding sentence.")
            print(f"Embedding (first 3 dims): {emb[:3] if emb else 'None'}... (Total dims: {len(emb) if emb else 0})")
            
            embs_list = active_binding.embed(["Sentence 1 for embedding.", "Sentence 2 for batch embedding."])
            if embs_list and len(embs_list) > 0:
                print(f"List Embeddings (2 sentences, first 3 dims of first sentence): {embs_list[0][:3]}... (Total sentences: {len(embs_list)})")
            else:
                print("List Embeddings: Received empty or no embeddings.")

        except Exception as e:
            print(f"Embedding test failed: {e}")
            if "Embedding not enabled" in str(e):
                print("Note: 'embedding_mode' might be false in server config (binding_config).")

    except Exception as e: 
        print(f"\n--- TEST FAILED ---")
        trace_exception(e)
    finally:
        if active_binding: 
            print("\nUnloading model...")
            active_binding.unload_model()
    print("\n--- Test Finished ---")