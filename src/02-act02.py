import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.images = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                self.images.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations (Resize added)
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(60),
    transforms.RandomGrayscale(p=0.1),
    transforms.Pad(1, fill=0, padding_mode='constant'),
    transforms.ToTensor()
])

# Create dataset and DataLoader
custom_data = CustomDataset('../data/custom', transform=transform)
data_loader = DataLoader(dataset=custom_data, batch_size=32, shuffle=True)

# Function to visualize a batch
def show_batch(loader):
    images, labels = next(iter(loader)) 
    grid = torchvision.utils.make_grid(images, nrow=8, padding=2, normalize=True)
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Batch of Images from Custom Dataset")
    plt.axis("off")
    plt.show()

# Show a batch of images
show_batch(data_loader)
