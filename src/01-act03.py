import torch    

# Generate tensor
a = torch.tensor([[[1, 2, 3, 4, 5, 6, 7, 8], [10, 11, 12 , 13, 14, 15, 16, 17]], [[71, 72, 73, 74, 75, 76, 77, 78], [81, 82, 83, 84, 85, 86, 87, 88]]])

# Extract specified tensors
print(f"First exercise is {a[0, 0, 0:7]}")
print(f"Second exercise is {a[0, :, 0:3]}")
