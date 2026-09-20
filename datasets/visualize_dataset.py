import matplotlib.pyplot as plt
from medmnist import INFO, PathMNIST
from torchvision import transforms

# Dataset information
info = INFO["pathmnist"]
DataClass = PathMNIST

# Transform
transform = transforms.ToTensor()

# Load training dataset
train_dataset = DataClass(
    split="train",
    transform=transform,
    download=True
)

# Class names
labels = list(info["label"].values())

# Plot first 9 images
plt.figure(figsize=(8, 8))

for i in range(9):
    image, label = train_dataset[i]

    plt.subplot(3, 3, i + 1)
    plt.imshow(image.permute(1, 2, 0))
    plt.title(labels[label.item()])
    plt.axis("off")

plt.tight_layout()
plt.show()