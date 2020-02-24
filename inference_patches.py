import matplotlib.pyplot as plt
import torch
from staff_image_dataset import train_dataset, test_dataset
from unet import UNet
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = UNet(depth=3, num_initial_channels=32, conv_padding=1).to(device)
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
