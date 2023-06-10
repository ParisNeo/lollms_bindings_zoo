import subprocess
from pathlib import Path
from lollms.binding import LOLLMSConfig, BindingInstaller
import yaml

class Install(BindingInstaller):
    def __init__(self, config:LOLLMSConfig=None, force_reinstall=False):
        # Build parent
        super().__init__(config)
        # Get the current directory
        current_dir = Path(__file__).resolve().parent
        install_file = current_dir / ".installed"

        if not install_file.exists() or force_reinstall:
            print("-------------- cTransformers binding -------------------------------")
            print("This is the first time you are using this binding.")
            print("Installing ...")
            """
            try:
                print("Checking pytorch")
                import torch
                import torchvision
                if torch.cuda.is_available():
                    print("CUDA is supported.")
                else:
                    print("CUDA is not supported. Reinstalling PyTorch with CUDA support.")
                    self.reinstall_pytorch_with_cuda()
            except Exception as ex:
                self.reinstall_pytorch_with_cuda()
            """

            # Step 2: Install dependencies using pip from requirements.txt
            requirements_file = current_dir / "requirements.txt"
            subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])

            # Create ther models folder
            models_folder = config.lollms_paths.personal_models_path/f"{Path(__file__).parent.stem}"
            models_folder.mkdir(exist_ok=True, parents=True)

            # Create the configuration file
            self.create_config_file(config.lollms_paths.personal_configuration_path / "c_transformers_config.yaml")

            #Create the install file 
            with open(install_file,"w") as f:
                f.write("ok")
            print("Installed successfully")

    def create_config_file(self, path):
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
            "use_avx2": True,     # use avx2
        }
        with open(path, 'w') as file:
            yaml.dump(data, file)

    def reinstall_pytorch_with_cuda(self):
        subprocess.run(["pip", "install", "--upgrade", "torch", "torchvision", "torchaudio", "--no-cache-dir", "--index-url", "https://download.pytorch.org/whl/cu117"])
        