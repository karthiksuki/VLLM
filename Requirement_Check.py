import torch

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA not available. Still using CPU.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")