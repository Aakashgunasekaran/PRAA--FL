from medmnist import INFO, PathMNIST
from torchvision import transforms

# Dataset information
info = INFO["pathmnist"]
DataClass = PathMNIST

# Image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load datasets 
train_dataset = DataClass(
    split="train",
    transform=transform,
    download=True
)

val_dataset = DataClass(
    split="val",
    transform=transform,
    download=True
)

test_dataset = DataClass(
    split="test",
    transform=transform,
    download=True
)

# Print dataset details
print("=" * 60)
print("PathMNIST Dataset Information")
print("=" * 60)

print(f"Training Images   : {len(train_dataset)}")
print(f"Validation Images : {len(val_dataset)}")
print(f"Testing Images    : {len(test_dataset)}")

print("\nDataset Loaded Successfully!")