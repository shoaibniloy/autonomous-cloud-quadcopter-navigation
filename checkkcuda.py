import sys
import platform
import torch

def show_env_details():
    env_details = {
        "Python Version": sys.version,
        "Python Compiler": sys.version_info,
        "Platform": platform.system(),
        "Platform Version": platform.version(),
        "Architecture": platform.architecture(),
        "Processor": platform.processor(),
        "PyTorch Version": torch.__version__,
        "CUDA Available": torch.cuda.is_available(),
        "CUDA Version": torch.version.cuda if torch.cuda.is_available() else "Not available"
    }

    for key, value in env_details.items():
        print(f"{key}: {value}")

# Call the function to display the environment details
show_env_details()
