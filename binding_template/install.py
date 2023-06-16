import subprocess
from pathlib import Path
from lollms.binding import LOLLMSConfig, BindingInstaller
import yaml
from lollms.helpers import ASCIIColors

class Install(BindingInstaller):
    def __init__(self, config:LOLLMSConfig=None, force_reinstall=False):
        # Build parent
        super().__init__(config)
        # Get the current directory
        current_dir = Path(__file__).resolve().parent
        install_file = current_dir / ".installed"

        if not install_file.exists() or force_reinstall:
            ASCIIColors.info("-------------- Template binding -------------------------------")
            print("This is the first time you are using this binding.")
            print("Installing ...")
            # Example of installing py torche
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

            # Create the models folder
            models_folder = config.lollms_paths.personal_models_path/f"{Path(__file__).parent.stem}"
            models_folder.mkdir(exist_ok=True, parents=True)

            # The local config can be used to store personal information that shouldn't be shared like chatgpt Key 
            # or other personal information
            # This file is never commited to the repository as it is ignored by .gitignore
            # You can remove this if you don't need custom local configurations
            self.create_config_file(config.lollms_paths.personal_configuration_path/'binding_template_config.yaml')
            
            #Create the install file (a file that is used to insure the installation was done correctly)
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
            "parameter": "value",     # good
        }
        with open(path, 'w') as file:
            yaml.dump(data, file)
                