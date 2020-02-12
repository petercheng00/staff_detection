from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class StaffImageDataset(Dataset):
    def __init__(self, bw_files, nostaff_files, transforms=None):
        self.bw_files = bw_files
        self.nostaff_files = nostaff_files
        self.transforms = transforms

    def process(self, image):
        t2 = transforms.CenterCrop(1024)
        image = t2(image)
        t = transforms.ToTensor()
        image = t(image)
        if self.transforms:
            image = self.transforms(image)
        return image

    def __getitem__(self, index):
        bw_image = Image.open(self.bw_files[index])
        nostaff_image = Image.open(self.nostaff_files[index])

        return self.process(bw_image), self.process(nostaff_image)

    def __len__(self):
        return len(self.bw_files)
