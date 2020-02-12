import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from staff_image_dataset import StaffImageDataset
from unet_mini import UNetMini
from torchvision import transforms

bw_dir = 'Training_BW/Training_BW'
gt_dir = 'Training_GT/Training_GT'
bw_files = [os.path.join(bw_dir, f) for f in sorted(os.listdir(bw_dir))]
gt_files = [os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir))]

bw_train, bw_test, gt_train, gt_test = train_test_split(bw_files, gt_files, test_size=0.2, random_state=0)


train_dataset = StaffImageDataset(bw_train, gt_train)
test_dataset = StaffImageDataset(bw_test, gt_test)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = UNetMini(2).to(device)

checkpoint = torch.load('checkpoint9.ckpt')
net.load_state_dict(checkpoint['model_state_dict'])

net.eval()

for bw, gt in train_dataset:

    t = transforms.ToPILImage()
    plt.imshow(t(bw))
    plt.figure()
    plt.imshow(t(gt))

    bw_tensor = bw.unsqueeze(0).to(device)
    result = net(bw_tensor).cpu()
    criterion = torch.nn.CrossEntropyLoss()
    gt_long = gt.type(torch.LongTensor)
    print(criterion(result, gt_long))

    softmaxer = torch.nn.LogSoftmax(0)
    result_softmaxed = softmaxer(result[0])
    result_label = result_softmaxed.max(0)[1].float()
    result_im = t(result_label)

    plt.figure()
    plt.imshow(result_im)

    plt.show()
