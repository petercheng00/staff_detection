from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class StaffImageDataset(Dataset):
    def __init__(self, in_files, gt_files, transforms=None):
        self.in_files = in_files
        self.gt_files = gt_files
        self.transforms = transforms

    def process(self, image, size):
        t2 = transforms.CenterCrop(size)
        image = t2(image)
        t = transforms.ToTensor()
        image = t(image)
        if self.transforms:
            image = self.transforms(image)
        return image

    def __getitem__(self, index):
        in_image = Image.open(self.in_files[index])
        nostaff_image = Image.open(self.gt_files[index])

        return self.process(in_image, 512), self.process(nostaff_image, 512)

    def __len__(self):
        return len(self.in_files)
