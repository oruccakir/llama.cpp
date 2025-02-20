import argparse
import subprocess

GREEN = "\033[92m"
RESET = "\033[0m" 


parser = argparse.ArgumentParser(description="build llama.cpp")

parser.add_argument('-d', '--device_type', type=str, required=True, help="device type to build llama.cpp")

# Parse the arguments
args = parser.parse_args()

device_type = args.device_type

config_command = None
build_command  = None

if device_type == "gpu":
    config_command = "cmake -B build -DGGML_CUDA=ON"
elif device_type == "cpu":
    config_command = "cmake -B build -DGGML_CUDA=OFF"

build_command = "cmake --build build --config Release -j $(nproc)"

config_command_res = subprocess.run(config_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(config_command_res.stdout.decode())
print(f"{GREEN}Configuration completed successfully{RESET}")
build_command_res = subprocess.run(build_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
print(build_command_res.stdout.decode())
print(f"{GREEN}Build completed successfully{RESET}")



