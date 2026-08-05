import torch, os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

NUM_WORKERS = min(4, os.cpu_count() or 1)

train_transform = transforms.Compose([
    transforms.RandomCrop(64, padding=4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                          std=[0.2764, 0.2689, 0.2816]),
    transforms.RandomErasing(p=0.25),
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                          std=[0.2764, 0.2689, 0.2816]),
])

class TinyImageNetDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        label = sample["label"]

        if self.transform is not None:
            image = self.transform(image).clone()

        return image, label

def create_dataloaders(batch_size: int=64, num_workers: int=NUM_WORKERS):
    train_dataset = load_dataset('Maysee/tiny-imagenet', split='train')
    valid_dataset = load_dataset('Maysee/tiny-imagenet', split='valid')

    class_names = train_dataset.features["label"].names

    train_dataset = TinyImageNetDataset(train_dataset, train_transform)
    valid_dataset = TinyImageNetDataset(valid_dataset, valid_transform)


    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, valid_loader, class_names

def main():
    train_loader, valid_loader, class_names = create_dataloaders()

if __name__ == "__main__":
    main()