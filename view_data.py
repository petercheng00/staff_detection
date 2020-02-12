import os
import matplotlib.pyplot as plt
from staff_image_dataset import StaffImageDataset

bw_dir = 'Training_BW/Training_BW'
gt_dir = 'Training_GT/Training_GT'
bw_files = [os.path.join(bw_dir, f) for f in os.listdir(bw_dir)]
gt_files = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir)]
dataset = StaffImageDataset(bw_files, gt_files)

print(f'length = {len(dataset)}')

bw, nostaff = dataset[0]

plt.imshow(bw)
plt.figure()
plt.imshow(nostaff)
plt.show()
