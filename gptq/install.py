import subprocess
from pathlib import Path
from lollms.binding import LOLLMSConfig, BindingInstaller
from lollms.paths import  LollmsPaths
import yaml
import os

class Install(BindingInstaller):
    def __init__(self, config:LOLLMSConfig=None):
        # Build parent
        super().__init__(config)
        # Get the current directory
        current_dir = Path(__file__).resolve().parent
        install_file = current_dir / ".installed"

        if not install_file.exists():
            print("-------------- GPTQ binding -------------------------------")
            print("This is the first time you are using this binding.")
            print("Installing ...")
            # Example of installing py torche
            try:
                print("Checking pytorch")
                import torch
                if torch.cuda.is_available():
                    print("CUDA is supported.")
                else:
                    print("CUDA is not supported. Reinstalling PyTorch with CUDA support.")
                    self.reinstall_pytorch_with_cuda()
            except Exception as ex:
                self.reinstall_pytorch_with_cuda()

            # Step 2: Install dependencies using pip from requirements.txt
            env = os.environ.copy()
            requirements_file = current_dir / "requirements.txt"
            subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)], env=env)

            # Create the models folder
            models_folder = config.lollms_paths.personal_models_path/f"{Path(__file__).parent.stem}"
            models_folder.mkdir(exist_ok=True, parents=True)

            # Create configuration file
            self.create_config_file()
            # Create the install file (a file that is used to insure the installation was done correctly)
            with open(install_file,"w") as f:
                f.write("ok")
            print("Installed successfully")
        
    def create_config_file(self):
        """
        Create a local_config.yaml file with predefined data.

        The function creates a local_config.yaml file with the specified data. The file is saved in the parent directory
        of the current file.

        Args:
            None

        Returns:
            None
        """
        data = {
            "device": "cuda:0",     # good
        }
        path = self.config.lollms_paths.personal_configuration_path / 'binding_gptq_config.yaml'
        with open(path, 'w') as file:
            yaml.dump(data, file)
        
    def reinstall_pytorch_with_cuda(self):
        """Installs pytorch with cuda (if you have a gpu) 
        """
        subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio", "--upgrade", "--no-cache-dir", "--index-url", "https://download.pytorch.org/whl/cu117"])
        