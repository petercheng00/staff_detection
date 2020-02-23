from apex import amp
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from staff_image_dataset import train_data_loader, test_data_loader
from unet import UNet

# Hyperparameters
batch_size=8
data_loader_parallel=4
epochs=1
learning_rate=0.01

train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=data_loader_parallel)
test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=data_loader_parallel)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = UNet(depth=3, num_initial_channels=32, conv_padding=1).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

net, optimizer = amp.initialize(net, optimizer, opt_level="O1")

# The training loop
total_steps = len(train_data_loader)
print(f"{epochs} epochs, {total_steps} total_steps per epoch")

for epoch in range(epochs):
    net.train()
    for i, (in_images, gt_images) in enumerate(train_data_loader, 1):
        preds = net(in_images.to(device))
        gt_images = gt_images.squeeze(1).type(torch.LongTensor).to(device)
        loss = criterion(preds, gt_images)

        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        if (i) % 10 == 0:
            print (f"Epoch [{epoch + 1}/{epochs}], Step [{i}/{total_steps}], Loss: {loss.item():4f}")

    # Save after each epoch
    torch.save({'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
    }, 'checkpoint' + str(epoch) + '.ckpt')

    # Evaluate validation after each epoch
    net.eval()
    with torch.no_grad():
        sum_loss = 0
        for i, (in_images, gt_images) in enumerate(test_data_loader, 1):
            preds = net(in_images.to(device))
            gt_images = gt_images.squeeze(1).type(torch.LongTensor).to(device)
            sum_loss += criterion(preds, gt_images)
        print(f'validation loss: {(sum_loss / len(test_data_loader)):4f}')
