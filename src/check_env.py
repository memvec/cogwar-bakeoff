import platform

import torch

print(f"Python version: {platform.python_version()}")
print(f"torch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")
