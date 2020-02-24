import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from unet import UNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = UNet(depth=3, num_initial_channels=32, conv_padding=1).to(device)
checkpoint = torch.load('checkpoint4.ckpt')
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

patch_size = 512
softmaxer = torch.nn.LogSoftmax(0)

in_prefix = 'Test_GRAY/GR_'
out_prefix = 'Test_Pred_GRAY/GR_'
for index in range(1, 2001):
    if index % 100 == 0:
        print(f'{index} / 2000')
    filename = in_prefix + ('%04d.png' % index)
    in_image = Image.open(filename)
    width, height = in_image.size
    output_image = Image.new('L', (width, height))
    ys = list(range(0, height, patch_size))
    xs = list(range(0, width, patch_size))
    ys.append(height-patch_size)
    xs.append(width-patch_size)
    for y in ys:
        for x in xs:
            in_image_patch = TF.crop(in_image, y, x, patch_size, patch_size)
            in_tensor_patch_gpu = TF.to_tensor(in_image_patch).unsqueeze(0).to(device)
            result = net(in_tensor_patch_gpu).cpu()
            result_softmaxed = softmaxer(result[0])
            result_label = result_softmaxed.max(0)[1].float()
            result_im = TF.to_pil_image(result_label)
            output_image.paste(result_im, (x, y))


    output_image.save(out_prefix + ('%04d.png' % index))
    # plt.imshow(in_image)
    # plt.figure()
    # plt.imshow(output_image)
    # plt.show()
