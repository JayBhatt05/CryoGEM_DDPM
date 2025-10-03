import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqd
from PIL import Image
import os
from pytorch_fid import fid_score
import numpy as np
import tempfile

# Dataset class for CryoGEM
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
    
    
def calculate_fid(real_loader, gen_images, device):
    """
    Calculate FID score between real images from dataloader and generated images.
    
    Args:
        real_loader: DataLoader containing real images
        gen_images: Tensor of generated images (normalized to [-1, 1])
        device: Device to run computation on
    
    Returns:
        float: FID score
    """
    # Initialize FID metric
    fid = FrechetInceptionDistance(normalize=True).to(device)
    
    # Process generated images
    # Convert from [-1, 1] to [0, 255]
    gen_images = ((gen_images + 1) * 127.5).byte()
    
    # If images are grayscale, convert to RGB by repeating channels
    if gen_images.shape[1] == 1:
        gen_images = gen_images.repeat(1, 3, 1, 1)
    
    # Update FID with generated images
    fid.update(gen_images, real=False)
    
    # Process real images
    for batch in tqdm(real_loader, desc="Computing FID"):
        # Convert to [0, 255] range
        real_batch = ((batch + 1) * 127.5).byte()
        
        # If images are grayscale, convert to RGB
        if real_batch.shape[1] == 1:
            real_batch = real_batch.repeat(1, 3, 1, 1)
        
        real_batch = real_batch.to(device)
        fid.update(real_batch, real=True)
    
    # Compute and return the FID score
    return fid.compute().item()


def evaluate_model_fid(model, diffusion, data_loader, device, n_samples=250, batch_size=16):
    """
    Generate samples from the model and calculate FID score against real data using pytorch-fid.
    
    Args:
        model: DDPM model
        diffusion: Diffusion process
        data_loader: DataLoader with real images
        device: Device to run computation on
        n_samples: Number of samples to generate
        batch_size: Batch size for generation
        
    Returns:
        float: FID score
    """
    model.eval()
    
    # Create temporary directories for real and generated images
    with tempfile.TemporaryDirectory() as real_dir, tempfile.TemporaryDirectory() as gen_dir:
        # Save real images to disk
        print("Saving real images...")
        for i, images in enumerate(tqdm(data_loader)):
            for j, img in enumerate(images):
                # Convert from [-1, 1] to [0, 1]
                img = (img.clamp(-1, 1) + 1) / 2
                # Convert to PIL Image and save
                img_pil = Image.fromarray(
                    (img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                )
                img_pil.save(os.path.join(real_dir, f"real_{i}_{j}.png"))
                
                # Stop if we have enough real images
                if i * data_loader.batch_size + j >= n_samples:
                    break
            if i * data_loader.batch_size + j >= n_samples:
                break
        
        # Generate and save fake images
        print("Generating fake images...")
        img_count = 0
        for i in range(0, n_samples, batch_size):
            current_batch_size = min(batch_size, n_samples - i)
            batch = diffusion.sample(model, n=current_batch_size)
            
            # Save generated images
            for j, img in enumerate(batch):
                img_pil = Image.fromarray(
                    (img.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                )
                img_pil.save(os.path.join(gen_dir, f"gen_{img_count}.png"))
                img_count += 1
                
        # Calculate FID score using pytorch-fid
        print("Calculating FID score...")
        fid_value = fid_score.calculate_fid_given_paths(
            [real_dir, gen_dir],
            batch_size=batch_size,
            device=device,
            dims=2048
        )
    
    model.train()
    return fid_value