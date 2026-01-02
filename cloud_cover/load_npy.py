import numpy as np

file_path = "/home/echofusion/Hemanth/Cloud_Cover/data/RFCD_pixel_training_data/y_TOA.npy"

data = np.load(file_path)

# print(data[data==0].shape)
print("Data shape:", data.shape)

print("Data:", data[0:9])