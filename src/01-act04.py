import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np

# Load the FashionMNIST dataset
train_dataset = datasets.FashionMNIST(root='data', train=True, download=False)
y_train = train_dataset.targets.numpy()

# Define class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

# Dictionary to store 10 sample indices per class
classes = {i: [] for i in range(10)}

# Collect 10 images per class efficiently
indices = np.argsort(y_train)  # Sort indices by class labels
y_sorted = y_train[indices]

start = 0
for label in range(10):
    end = np.searchsorted(y_sorted, label + 1, side='left')
    classes[label] = indices[start:end][:10]  # Take the first 10 indices of each class
    start = end

# Create the plot grid (10 rows, 11 columns)
fig, axes = plt.subplots(10, 11, figsize=(20, 15))

for row, label in enumerate(range(10)):
    class_name = class_names[label]
    
    # First column: Class name
    axes[row, 0].text(0.5, 0.5, class_name, fontsize=12, ha="center", va="center")
    axes[row, 0].axis("off")  # Hide frame
    
    # Next 10 columns: Images
    for col, idx in enumerate(classes[label]):
        ax = axes[row, col + 1]  # Start from column 1
        image = train_dataset.data[idx]
        ax.imshow(image.numpy(), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

plt.tight_layout()
plt.show()
