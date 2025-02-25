import torch
import matplotlib.pyplot as plt
from skimage import data, color
import numpy as np

def convolution3d(image, kernel):
    """Performs 3D convolution across all RGB channels."""
    output_shape = (image.shape[0] - kernel.shape[0] + 1, 
                    image.shape[1] - kernel.shape[1] + 1, 
                    image.shape[2])
    
    output = torch.zeros(output_shape, dtype=torch.float32)
    
    # Apply kernel to each channel separately
    for c in range(image.shape[2]):  # Iterate over RGB channels
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                output[i, j, c] = torch.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1], c] * kernel)
    
    return output

# Load astronaut image and convert it to a tensor (normalize to [0,1])
image = data.astronaut() / 255.0  # Normalize
image_tensor = torch.tensor(image, dtype=torch.float32)  # Shape: (H, W, 3)

# Define 3D sharpening kernel (same for all channels)
sharpen_kernel = torch.tensor([
    [ 1/9, 1/9, 1/9],
    [ 1/9, 1/9, 1/9],
    [ 1/9, 1/9, 1/9]
], dtype=torch.float32)

# Apply convolution
sharpened_image = convolution3d(image_tensor, sharpen_kernel)

# Convert tensors to NumPy arrays for visualization
original_np = image_tensor.numpy()
sharpened_np = sharpened_image.numpy()

# Ensure values are in [0,1] range
sharpened_np = np.clip(sharpened_np, 0, 1)

# Display images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_np)
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[1].imshow(sharpened_np)
axes[1].set_title("Blurred Image")
axes[1].axis("off")
plt.show()
