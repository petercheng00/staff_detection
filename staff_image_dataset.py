from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms

class StaffImageDataset(Dataset):
    def __init__(self, in_files, gt_files, size=(512, 512)):
        self.in_files = in_files
        self.gt_files = gt_files
        self.size = size

    def __getitem__(self, index):
        in_image = Image.open(self.in_files[index])
        gt_image = Image.open(self.gt_files[index])

        y, x, h, w = transforms.RandomCrop.get_params(in_image, output_size=self.size)

        in_image = TF.crop(in_image, y, x, h, w)
        gt_image = TF.crop(gt_image, y, x, h, w)
        return (TF.to_tensor(in_image), TF.to_tensor(gt_image))

    def __len__(self):
        return len(self.in_files)

# Data setup
in_prefix = 'Training_GRAY/GR_'
gt_prefix = 'Training_GT/Training_GT/GT_'
file_suffixes = ["%04d.png" % x for x in range(1, 4001)]
in_files = [in_prefix + fs for fs in file_suffixes]
gt_files = [gt_prefix + fs for fs in file_suffixes]
in_train, in_test, gt_train, gt_test = train_test_split(in_files, gt_files, test_size=0.1, random_state=0)

train_dataset = StaffImageDataset(in_train, gt_train)
test_dataset = StaffImageDataset(in_test, gt_test)
