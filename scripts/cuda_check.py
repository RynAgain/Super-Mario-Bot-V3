import torch
print("PyTorch CUDA build:", torch.version.cuda)           # e.g., '12.1'
print("CUDA available?   ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:        ", torch.cuda.get_device_name(0))
    print("Capability:    ", torch.cuda.get_device_capability(0))  # e.g., (8, 6)
    print("cuDNN version: ", torch.backends.cudnn.version())
