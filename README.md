# Pipeline Setup and Usage Guide

This README provides step-by-step instructions to set up and run the pipeline for integrating Depth Anything V2, SmolVLM, and LLM fine-tuning in a single environment. Follow these steps to ensure proper setup and execution.

## Prerequisites
- This pipeline has been tested on **Ubuntu 20.04**. Ensure your system is compatible or make necessary adjustments for other operating systems.
- All components and dependencies must be installed within a **single folder**, which will serve as your working environment for this pipeline.

## Setup Instructions

1. **Create a Project Folder**
   - Create a single folder (e.g., `pipeline_env`) to house all components, dependencies, and the virtual environment.
   - Example: `mkdir pipeline_env && cd pipeline_env`
   - This folder will be your working environment for the entire pipeline.

2. **Set Up a Virtual Environment**
   - Create and activate a Python virtual environment inside the project folder to manage dependencies:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - Ensure all subsequent installations are performed within this activated virtual environment.

3. **Install Requirements**
   - Create a `requirements.txt` file in the project folder with the necessary dependencies for your pipeline (e.g., `torch`, `opencv-python`, etc.).
   - Install the requirements:
     ```bash
     pip install -r requirements.txt
     ```
   - Note: Ensure `requirements.txt` includes dependencies compatible with Depth Anything V2 and other components. Refer to the respective repositories for specific requirements.

4. **Install Depth Anything V2**
   - Clone the Depth Anything V2 repository into the project folder:
     ```bash
     git clone https://github.com/DepthAnything/Depth-Anything-V2
     ```
   - Navigate to the cloned repository and install its dependencies:
     ```bash
     cd Depth-Anything-V2
     pip install -r requirements.txt
     cd ..
     ```
   - Download the pre-trained model checkpoints from the [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2) and place them in the `Depth-Anything-V2/checkpoints` directory.

5. **Install llama.cpp**
   - Install the `llama-cpp-python` package within the virtual environment:
     ```bash
     pip install llama-cpp-python
     ```
   - This package provides the necessary bindings to run the llama.cpp server for SmolVLM integration.

6. **Install SmolVLM Real-Time Webcam Demo**
   - Clone the SmolVLM repository into the project folder:
     ```bash
     git clone https://github.com/ngxson/smolvlm-realtime-webcam
     ```
   - Follow the instructions in the [SmolVLM repository](https://github.com/ngxson/smolvlm-realtime-webcam) to set up the llama.cpp server and run the demo:
     ```bash
     cd smolvlm-realtime-webcam
     ```
     - Start the llama.cpp server with the SmolVLM model:
       ```bash
       llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF
       ```
     - Optionally, add `-ngl 99` to enable GPU support if using an NVIDIA/AMD/Intel GPU.
     - Open `index.html` in a browser, adjust instructions if needed (e.g., for JSON output), and click "Start" to run the real-time webcam demo.
     - Return to the project folder: `cd ..`

7. **Install Ollama**
   - Install Ollama in the project folder by following the official instructions for Ubuntu:
     ```bash
     curl -fsSL https://ollama.com/install.sh | sh
     ```
   - Verify the installation by running:
     ```bash
     ollama --version
     ```
   - Ensure Ollama models and configurations are stored within the project folder (e.g., by setting appropriate environment variables or configurations).

8. **Download Models**
   - Download the necessary models for Depth Anything V2 and SmolVLM:
     - For **Depth Anything V2**, download the desired model checkpoints (e.g., Small, Base, Large) from the links provided in the [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2) and place them in `Depth-Anything-V2/checkpoints`.
     - For **SmolVLM**, ensure the `SmolVLM-500M-Instruct-GGUF` model is downloaded via the llama.cpp server as specified in step 6.
     - For **Ollama**, pull the required LLMs using:
       ```bash
       ollama pull <model_name>
       ```
       Replace `<model_name>` with the desired model (e.g., `llama3`, `mistral`). Store these models within the project folder.

9. **Fine-Tuning LLMs**
   - For fine-tuning large language models, clone the One-Click LLM Fine-Tuning repository into the project folder:
     ```bash
     git clone https://github.com/shoaibniloy/One-Click-LLM-Fine-Tuning-in-Google-Colab
     ```
   - Follow the instructions in the [One-Click LLM Fine-Tuning repository](https://github.com/shoaibniloy/One-Click-LLM-Fine-Tuning-in-Google-Colab) to set up and run fine-tuning in Google Colab.
   - Note: Fine-tuning may require uploading datasets or configurations to Google Colab. Ensure any fine-tuned models are downloaded and stored in the project folder for local use.

10. **Using LLMs Directly**
    - You can use LLMs directly (e.g., via Ollama or llama.cpp) without fine-tuning if desired.
    - Ensure all model files, configurations, and scripts remain within the project folder for consistency.
    - Example: Run an Ollama model locally:
      ```bash
      ollama run <model_name>
      ```

## Notes
- **Single Folder Requirement**: Ensure all repositories, models, and the virtual environment are contained within the same project folder (`pipeline_env`) to maintain a cohesive environment.
- **Compatibility**: Verify compatibility between dependencies (e.g., Python versions, PyTorch for Depth Anything V2, and llama.cpp for SmolVLM). Use the same Python version across all components (e.g., Python 3.8 or higher).
- **GPU Support**: For GPU acceleration, ensure CUDA or appropriate drivers are installed for Depth Anything V2 and llama.cpp. Use the `-ngl 99` flag for llama.cpp if applicable.
- **Storage**: Ensure sufficient disk space for model checkpoints, especially for larger models like Depth Anything V2 Large (335.3M parameters) or Ollama LLMs.

## Running the Pipeline
- Activate the virtual environment:
  ```bash
  source venv/bin/activate
  ```
- Run Depth Anything V2 for depth estimation:
  ```bash
  cd Depth-Anything-V2
  python run.py --encoder vitl --img-path assets/examples --outdir depth_vis
  ```
- Run SmolVLM for real-time object detection:
  ```bash
  cd smolvlm-realtime-webcam
  llama-server -hf ggml-org/SmolVLM-500M-Instruct-GGUF
  ```
  Then open `index.html` in a browser.
- Run Ollama for LLM tasks:
  ```bash
  ollama run <model_name>
  ```
- For fine-tuning, follow the Google Colab instructions in the `One-Click-LLM-Fine-Tuning-in-Google-Colab` repository.

## Troubleshooting
- If you encounter dependency conflicts, ensure all packages are installed in the virtual environment and check for version compatibility.
- For issues with Depth Anything V2, refer to the [official repository](https://github.com/DepthAnything/Depth-Anything-V2) for detailed setup instructions.
- For SmolVLM issues, check the [SmolVLM repository](https://github.com/ngxson/smolvlm-realtime-webcam) for server or model configuration details.
- For Ollama or fine-tuning issues, consult the respective repositories or official documentation.

## License
- Refer to the individual repositories for their respective licenses:
  - Depth Anything V2: Apache-2.0 (Small model), CC-BY-NC-4.0 (Base/Large/Giant models)
  - SmolVLM and One-Click LLM Fine-Tuning: Check the respective repositories for licensing details.
- Ensure compliance with all licenses when using or distributing models and code.
