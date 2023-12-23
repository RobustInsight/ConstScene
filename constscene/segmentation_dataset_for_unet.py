import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class SegmentationDataset(Dataset):
    def __init__(self, input_dir, transform=None):
        self.root_dir = input_dir

        self.transform = transform
        image_file_names = [f for f in os.listdir(self.root_dir) if '.jpg' in f]
        mask_file_names = [f for f in os.listdir(self.root_dir) if '.png' in f]

        self.images = sorted(image_file_names)
        self.masks = sorted(mask_file_names)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root_dir, self.images[index]))
        mask = Image.open(os.path.join(self.root_dir, self.masks[index]))

        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask) * 255
        return img, mask


def get_loaders(train_dir, val_dir, test_dir, batch_size, train_transform, val_transform, test_transform):
    train_ds = SegmentationDataset(input_dir=train_dir, transform=train_transform)

    train_loader_ = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = SegmentationDataset(input_dir=val_dir, transform=val_transform)

    val_loader_ = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    test_ds = SegmentationDataset(input_dir=test_dir, transform=test_transform)

    test_loader_ = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_loader_, val_loader_, test_loader_
