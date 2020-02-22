from apex import amp
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from staff_image_dataset import StaffImageDataset
from unet.unet import UNet
# from unet_mini import UNetMini

batch_size=4
epochs=1
learning_rate=0.01
momentum=0.9

in_prefix = 'Training_GRAY_part_1/Training_GRAY_part_1/GR_'
gt_prefix = 'Training_GT/Training_GT/GT_'

file_suffixes = ["%04d.png" % x for x in range(1, 1001)]
in_files = [in_prefix + fs for fs in file_suffixes]
gt_files = [gt_prefix + fs for fs in file_suffixes]

in_train, in_test, gt_train, gt_test = train_test_split(in_files, gt_files, test_size=0.2, random_state=0)

train_dataset = StaffImageDataset(in_train, gt_train)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = UNet(depth=3, wf=5, padding=True, up_mode='upsample').to(device)
# net = UNetMini(2).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

# The training loop
total_steps = len(train_data_loader)
print(f"{epochs} epochs, {total_steps} total_steps per epoch")

for epoch in range(epochs):
    for i, (in_images, gt_images) in enumerate(train_data_loader, 1):
        in_images = in_images.to(device)
        gt_images = gt_images.type(torch.LongTensor) #cross entropy requires longtensor targets
        gt_images = gt_images.reshape(gt_images.shape[0], gt_images.shape[2], gt_images.shape[3]) # drop the channels dimension
        gt_images = gt_images.to(device)


        # Forward pass
        outputs = net(in_images)

        # compute loss
        loss = criterion(outputs, gt_images)

        # zero all gradients
        optimizer.zero_grad()

        # back propagate loss
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        # update optimizer
        optimizer.step()

        if (i) % 10 == 0:
            print (f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}")

    torch.save({'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
    }, 'checkpoint' + str(epoch) + '.ckpt')
