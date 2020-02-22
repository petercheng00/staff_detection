import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from staff_image_dataset import StaffImageDataset
from unet.unet import UNet
from torchvision import transforms

in_prefix = 'Training_GRAY_part_1/Training_GRAY_part_1/GR_'
gt_prefix = 'Training_GT/Training_GT/GT_'

file_suffixes = ["%04d.png" % x for x in range(1, 1001)]
in_files = [in_prefix + fs for fs in file_suffixes]
gt_files = [gt_prefix + fs for fs in file_suffixes]

in_train, in_test, gt_train, gt_test = train_test_split(in_files, gt_files, test_size=0.2, random_state=0)

# train_dataset = StaffImageDataset(in_train, gt_train)
test_dataset = StaffImageDataset(in_test, gt_test)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# net = UNet().to(device)
net = UNet(depth=3, wf=5, padding=True, up_mode='upsample').to(device)
checkpoint = torch.load('checkpoint0.ckpt')
net.load_state_dict(checkpoint['model_state_dict'])

net.eval()

for in_image, gt in test_dataset:
    t = transforms.ToPILImage()

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(t(in_image), cmap='gray')
    fig.add_subplot(1, 3, 2)
    plt.imshow(t(gt))
    fig.add_subplot(1, 3, 3)

    in_tensor = in_image.unsqueeze(0).to(device)
    result = net(in_tensor).cpu()
    criterion = torch.nn.CrossEntropyLoss()
    gt_long = gt.type(torch.LongTensor)
    print(criterion(result, gt_long))

    softmaxer = torch.nn.LogSoftmax(0)
    result_softmaxed = softmaxer(result[0])
    result_label = result_softmaxed.max(0)[1].float()
    result_im = t(result_label)

    plt.imshow(result_im)

    plt.show()
