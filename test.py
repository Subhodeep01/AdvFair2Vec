import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version used by PyTorch: {torch.version.cuda}")
print(f"Is CUDA available? {torch.cuda.is_available()}")