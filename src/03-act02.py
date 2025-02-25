import torch
import matplotlib.pyplot as plt
from skimage import data, color
import numpy as np

def convolution2d(image, kernel):
    output_shape = (image.shape[0] - kernel.shape[0] + 1, image.shape[1] - kernel.shape[1] + 1)
    output = torch.zeros(output_shape)
        
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            output[i, j] = torch.sum(image[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return output

# Load astronaut image and convert to grayscale
image = data.astronaut()
gray_image = color.rgb2gray(image)
image_tensor = torch.tensor(gray_image, dtype=torch.float32)
# Define sharpening kernel
sharpen_kernel = torch.tensor([[ 1/9, 1/9, 1/9],
[1/9, 1/9, 1/9],
[ 1/9, 1/9, 1/9]],
dtype=torch.float32)
# Apply convolution
sharpened_image = convolution2d(image_tensor, sharpen_kernel)

# Convert tensors to NumPy arrays for visualization
original_np = image_tensor.numpy()
sharpened_np = sharpened_image.numpy()
# Display images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_np, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")
axes[1].imshow(sharpened_np, cmap='gray')
axes[1].set_title("Sharpened Image")
axes[1].axis("off")
plt.show()
