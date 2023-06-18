import subprocess
from pathlib import Path
from lollms.binding import LOLLMSConfig, BindingInstaller
from lollms.helpers import ASCIIColors
import yaml

class Install(BindingInstaller):
    def __init__(self, config:LOLLMSConfig=None, force_reinstall=False):
        # Build parent
        super().__init__(config)
        # Get the current directory
        current_dir = Path(__file__).resolve().parent
        install_file = current_dir / ".installed"

        if not install_file.exists() or force_reinstall:
            ASCIIColors.info("-------------- Remote LoLLMs Binding -------------------------------")
            print("This is the first time you are using this binding.")
            print("Installing ...")
            # Step 2: Install dependencies using pip from requirements.txt
            requirements_file = current_dir / "requirements.txt"
            subprocess.run(["pip", "install", "--upgrade", "--no-cache-dir", "-r", str(requirements_file)])

            # Create the models folder
            models_folder = config.lollms_paths.personal_models_path/f"{Path(__file__).parent.stem}"
            models_folder.mkdir(exist_ok=True, parents=True)

            # Create the configuration file
            binding_config_path = config.lollms_paths.personal_configuration_path / "binding_remote_lollms_config.yaml"
            self.create_config_file(binding_config_path)

            #Create the install file (a file that is used to insure the installation was done correctly)
            with open(install_file,"w") as f:
                f.write("ok")
            print("Installed successfully")
            print(f"Don't forget to add your servers list here: {binding_config_path}")
            

    def create_config_file(self, binding_config_path):
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
            "servers_addresses":  [], # List of paths to text generation lollms servers
            "keep_only_active_servers": True # Keeps only active servers
        }
        with open(binding_config_path, 'w') as file:
            yaml.dump(data, file)