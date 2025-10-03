from torch.utils.data import Dataset

class CryoGEMDataset(Dataset):
    def __init__(self, data_dir, image_size=128):
        self.data_dir = data_dir
        self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize between -1 and 1
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("L")  # Convert to grayscale
        return self.transform(img)
