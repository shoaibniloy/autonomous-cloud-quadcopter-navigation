# Install system dependencies (run once in terminal if not already installed)
# sudo apt-get update && sudo apt-get install -y build-essential cmake
import os
import torch
import torchvision
import subprocess
import torchvision.ops
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
# Define paths
lora_path = "/home/shoaib/Drone/YOLO+LLM"  # Path to LoRA adapters folder
output_dir = "/home/shoaib/ollama_models"
gguf_output = os.path.join(output_dir, "model-q4_0.gguf")
merged_model_path = os.path.join(output_dir, "merged_model")
llama_cpp_path = "/home/shoaib/Drone/YOLO+LLM/llama.cpp"

# Verify LoRA adapters folder
if not os.path.exists(lora_path):
    raise FileNotFoundError(f"LoRA adapters folder not found at {lora_path}")
required_files = ["adapter_config.json", "adapter_model.safetensors"]
if not all(os.path.exists(os.path.join(lora_path, f)) for f in required_files):
    print(f"Warning: Expected files {required_files} not found in {lora_path}. Checking for .bin...")
    required_files = ["adapter_config.json", "adapter_model.bin"]
    if not all(os.path.exists(os.path.join(lora_path, f)) for f in required_files):
        raise FileNotFoundError(f"Missing required files in {lora_path}: {required_files}")

# Verify llama.cpp installation
if not os.path.exists(llama_cpp_path):
    raise FileNotFoundError(f"llama.cpp not found at {llama_cpp_path}. Please install it.")

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Build llama.cpp with CMake (only if llama-quantize is missing)
quantize_bin = os.path.join(llama_cpp_path, "/home/shoaib/Drone/YOLO+LLM/llama.cpp/build/bin/llama-quantize")
if not os.path.exists(quantize_bin):
    print("Building llama.cpp with CMake...")
    os.makedirs(os.path.join(llama_cpp_path, "build"), exist_ok=True)
    result = subprocess.run(f"cd {llama_cpp_path}/build && cmake .. && cmake --build . --config Release", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to build llama.cpp: {result.stderr}")
    print("llama.cpp build complete.")
else:
    print("llama-quantize found, skipping build.")

# Load base model and tokenizer
print("Loading base model and tokenizer...")
model_name = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Base model and tokenizer loaded.")

# Load and merge LoRA adapters
print("Loading and merging LoRA adapters...")
model = PeftModel.from_pretrained(base_model, lora_path)
merged_model = model.merge_and_unload()
print("LoRA adapters merged.")

# Save merged model
print("Saving merged model...")
merged_model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print(f"Merged model saved to {merged_model_path}.")

# Convert to GGUF with f16
print("Converting to GGUF (f16)...")
result = subprocess.run(f"python {os.path.join(llama_cpp_path, 'convert_hf_to_gguf.py')} {merged_model_path} --outfile {os.path.join(output_dir, 'model-f16.gguf')} --outtype f16", shell=True, capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError(f"GGUF conversion failed: {result.stderr}")
print("GGUF (f16) conversion complete.")

# Quantize to Q4_0
print("Quantizing to Q4_0...")
result = subprocess.run(f"{quantize_bin} {os.path.join(output_dir, 'model-f16.gguf')} {gguf_output} Q4_0", shell=True, capture_output=True, text=True)
if result.returncode != 0:
    raise RuntimeError(f"Quantization failed: {result.stderr}")
print("Q4_0 quantization complete.")

# Verify GGUF file
if not os.path.exists(gguf_output):
    raise FileNotFoundError(f"GGUF file not created at {gguf_output}")
print(f"GGUF file created at {gguf_output}. Proceed to set up Ollama with the bash script.")