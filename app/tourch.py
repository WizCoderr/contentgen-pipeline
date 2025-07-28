import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show your GPU name
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")