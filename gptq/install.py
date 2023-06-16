import subprocess
from pathlib import Path
from lollms.binding import LOLLMSConfig, BindingInstaller
from lollms.helpers import ASCIIColors
import yaml
import os

class Install(BindingInstaller):
    def __init__(self, config:LOLLMSConfig=None, force_reinstall=False):
        # Build parent
        super().__init__(config)
        # Get the current directory
        current_dir = Path(__file__).resolve().parent
        install_file = current_dir / ".installed"

        if not install_file.exists() or force_reinstall:
            ASCIIColors.info("-------------- GPTQ binding -------------------------------")
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

            # Define the environment variables
            requirements_file = current_dir / "requirements.txt"
            env = os.environ.copy()
            result = subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "auto_gptq-0.2.0+cu118-cp310-cp310-linux_x86_64.whl"], env=env)

            if result.returncode != 0:
                print("Couldn't find Cuda build tools on your PC. Reverting to CPU. ")
                subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "auto-gptq"])
            subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])

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
        ASCIIColors.error("----------------------")
        ASCIIColors.error("Attention please")
        ASCIIColors.error("----------------------")
        print()
        sel = None
        device = ""
        while sel is None:
            ASCIIColors.error("Select the device to use (if you choose cuda please make sure you do have a cuda compatible GPU)")
            ASCIIColors.success("1) cpu")
            ASCIIColors.success("2) cuda:0")
            sel = input("?:")
            if sel=="1":
                device = "cpu"
            elif sel=="2":
                device = "cuda:0"
            else:
                sel=None

        data = {
            "device": device,     # cpu
        }
        path = self.config.lollms_paths.personal_configuration_path / 'binding_gptq_config.yaml'
        with open(path, 'w') as file:
            yaml.dump(data, file)
        
