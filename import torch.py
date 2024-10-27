import torch

# Create a dummy tensor
dummy_tensor = torch.zeros((1, 3, 480, 640), dtype=torch.float32)

# Attempt to copy a None value to the dummy tensor
try:
    dummy_tensor.copy_(None)
except TypeError as e:
    print(f"Error: {e}")
