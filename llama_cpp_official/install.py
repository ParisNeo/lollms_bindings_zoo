import subprocess
from pathlib import Path
from lollms.binding import BindingConfig, BindingInstaller
import yaml

class Install(BindingInstaller):
    def __init__(self, config:BindingConfig=None):
        # Build parent
        super().__init__(config)
        # Get the current directory
        current_dir = Path(__file__).resolve().parent
        install_file = current_dir / ".installed"

        if not install_file.exists():
            print("-------------- llama_cpp_official binding -------------------------------")
            print("This is the first time you are using this binding.")
            # Step 2: Install dependencies using pip from requirements.txt
            requirements_file = current_dir / "requirements.txt"
            subprocess.run(["pip", "install", "--no-cache-dir", "-r", str(requirements_file)])

            # Define the environment variables
            env = {"CMAKE_ARGS":"-DLLAMA_CUBLAS=on", "FORCE_CMAKE":"1"}
            subprocess.run(["pip", "install", "--no-cache-dir", "-r", str(requirements_file)], env=env)

            # Create ther models folder
            models_folder = Path(config.models_path/"llama_cpp_official")
            models_folder.mkdir(exist_ok=True, parents=True)

            # Create the configuration file
            self.create_config_file()
            
            #Create the install file 
            with open(install_file,"w") as f:
                f.write("ok")
            print("Installed successfully")

    def create_config_file(self):
        """
        Create a config_local.yaml file with predefined data.

        The function creates a config_local.yaml file with the specified data. The file is saved in the parent directory
        of the current file.

        Args:
            None

        Returns:
            None
        """
        data = {
            "n_gpu_layers": 20,     # number of layers to put in gpu
        }
        path = Path(__file__).parent / 'config_local.yaml'
        with open(path, 'w') as file:
            yaml.dump(data, file)

    def reinstall_pytorch_with_cuda(self):
        subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio", "--no-cache-dir", "--index-url", "https://download.pytorch.org/whl/cu117"])
        