import os
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from staff_image_dataset import StaffImageDataset
from unet_mini import UNetMini

batch_size=32
epochs=10
learning_rate=0.01
momentum=0.9

bw_dir = 'Training_BW/Training_BW'
gt_dir = 'Training_GT/Training_GT'
bw_files = [os.path.join(bw_dir, f) for f in sorted(os.listdir(bw_dir))]
gt_files = [os.path.join(gt_dir, f) for f in sorted(os.listdir(gt_dir))]

bw_train, bw_test, gt_train, gt_test = train_test_split(bw_files, gt_files, test_size=0.2, random_state=0)

train_dataset = StaffImageDataset(bw_train, gt_train)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = UNetMini(2).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

# The training loop
total_steps = len(train_data_loader)
print(f"{epochs} epochs, {total_steps} total_steps per epoch")

for epoch in range(epochs):
    for i, (bw_images, gt_images) in enumerate(train_data_loader, 1):
        bw_images = bw_images.to(device)
        gt_images = gt_images.type(torch.LongTensor) #cross entropy requires longtensor targets
        gt_images = gt_images.reshape(gt_images.shape[0], gt_images.shape[2], gt_images.shape[3]) # drop the channels dimension
        gt_images = gt_images.to(device)


        # Forward pass
        outputs = net(bw_images)

        # compute loss
        loss = criterion(outputs, gt_images)

        # zero all gradients
        optimizer.zero_grad()

        # back propagate loss
        loss.backward()

        # update optimizer
        optimizer.step()

        if (i) % 10 == 0:
            print (f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}")

    torch.save({'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
    }, 'checkpoint' + str(epoch) + '.ckpt')
