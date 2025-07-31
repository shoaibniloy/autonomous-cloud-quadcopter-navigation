from flask import Flask, jsonify
import psutil
import subprocess
import platform
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

def get_cpu_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    return cpu_usage

def get_ram_usage():
    mem = psutil.virtual_memory()
    return mem.percent

def get_gpu_usage():
    if platform.system() == 'Linux':
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            gpu_usage = result.stdout.decode('utf-8').strip()
            if gpu_usage:
                return gpu_usage
            else:
                return "GPU not in use"
        except FileNotFoundError:
            return "No NVIDIA GPU or nvidia-smi not installed"
    else:
        return "Not a Linux system or no GPU"

@app.route('/system-usage', methods=['GET'])
def system_usage():
    cpu_usage = get_cpu_usage()
    ram_usage = get_ram_usage()
    gpu_usage = get_gpu_usage()

    return jsonify({
        'cpu_usage': cpu_usage,
        'ram_usage': ram_usage,
        'gpu_usage': gpu_usage
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
