import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob
from tqdm import tqdm
import numpy as np
from torchvision.utils import save_image
from DDPM_model import CryoETDDPM
from utils import CryoGEMDataset, evaluate_model_fid


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=128, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        # Define noise schedule
        self.betas = torch.linspace(beta_start, beta_end, noise_steps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def noise_images(self, x, t):
        """Add noise to images at timestep t"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise

    def sample_timesteps(self, n):
        """Sample random timesteps for training"""
        return torch.randint(low=1, high=self.noise_steps, size=(n,))
    
    @torch.no_grad()
    def sample(self, model, n=8, save_path=None):
        """Generate samples using the diffusion model"""
        model.eval()
        
        # Start with random noise
        x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
        
        # Progressively denoise
        for i in reversed(range(1, self.noise_steps)):
            t = torch.ones(n, dtype=torch.long).to(self.device) * i
            predicted_noise = model(x, t)
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise
        
        # Normalize to [0, 1] for saving
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        
        if save_path:
            # Save the generated images
            for idx, img in enumerate(x):
                save_image(img, os.path.join(save_path, f"sample_{idx}.png"))
        
        model.train()
        return x


# Dataset class for CryoGEM
# class CryoGEMDataset(Dataset):
#     def __init__(self, data_dir, image_size=128):
#         self.data_dir = data_dir
#         self.image_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png') or f.endswith('.jpg')]
#         self.transform = transforms.Compose([
#             transforms.Resize((image_size, image_size)),
#             transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))  # Normalize between -1 and 1
#         ])

#     def __len__(self):
#         return len(self.image_files)

#     def __getitem__(self, idx):
#         img = Image.open(self.image_files[idx]).convert("L")  # Convert to grayscale
#         return self.transform(img)


def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    """Save model checkpoint"""
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    """Load model checkpoint"""
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_file, map_location="cuda")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    
    # If we want to modify learning rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    print(f"Checkpoint loaded from {checkpoint_file}")


def train(args):
    global counter
    counter = 0
    
    # Set device
    device = args.device if hasattr(args, 'device') else (
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    # Set paths for data
    train_data_paths = "/home/22ucs095/CryoET/real_data"
    val_data_paths = "/home/22ucs095/CryoET/real_data"
    
    # Create directory for saving results
    os.makedirs(args.results_folder, exist_ok=True)
    os.makedirs(args.checkpoints, exist_ok=True)
    
    # Create dataset and dataloader
    dataset = CryoGEMDataset(data_dir=train_data_paths, image_size=args.image_size)
    val_dataset = CryoGEMDataset(data_dir=val_data_paths, image_size=args.image_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model and optimizer
    ddpm = CryoETDDPM(
        time_dim=args.emb_dim, 
        img_channels=1,  # Grayscale
        dim=args.dim
    )
    
    if torch.cuda.device_count() > 1:
        ddpm = torch.nn.DataParallel(ddpm)
    ddpm = ddpm.to(device) 
    
    optimizer = optim.AdamW(
        ddpm.parameters(), 
        lr=args.lr, 
        weight_decay=0.01
    )
    
    # Load checkpoint if specified
    if args.load_model:
        load_checkpoint(
            os.path.join(args.checkpoints, args.checkpoint_name),
            ddpm, optimizer, args.lr,
        )
    
    # Loss functions
    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    
    # Initialize minimum average loss for RTT and best_fid
    min_avg_loss = float("inf")
    best_fid = float("inf")
    
    # Create diffusion process
    diffusion = Diffusion(
        noise_steps=args.noise_steps,
        img_size=args.image_size,
        device=device
    )
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        pbar = tqdm(dataloader)
        avg_loss = 0
        count1 = 0  # Counter for first retry
        count2 = 0  # Counter for successful RTT
        
        for i, images in enumerate(pbar):
            # For CryoET dataset, we only have images (no labels)
            images = images.to(device)
            
            # Sample timesteps and add noise
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            # Repetitive Training Technique (RTT)
            for rep in range(3):  # Maximum 3 retries
                if rep == 1:
                    count1 += 1
                
                # Predict noise
                predicted_noise = ddpm(x_t, t)
                
                # Calculate loss
                loss = mse(noise, predicted_noise) + l1(noise, predicted_noise)
                
                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                
                # Break if loss is better than min_avg_loss
                if loss < min_avg_loss:
                    if rep == 2:
                        count2 += 1
                    break
            
            # Update average loss
            avg_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix(
                epoch=epoch, 
                AVG_MSE=avg_loss / (i+1), 
                count1=count1, 
                count2=count2, 
                MIN_MSE=min_avg_loss
            )
            
            # Generate and save samples periodically
            if i % ((len(dataloader)-1)//2) == 0 and i != 0:
                os.makedirs(f"{args.results_folder}/during_training/epoch_{epoch}_step_{i}", exist_ok=True)
                samples = diffusion.sample(
                    ddpm, 
                    n=10, 
                    save_path=os.path.join(args.results_folder, "during_training", f"epoch_{epoch}_step_{i}")
                )
                counter += 1
        
        # If current epoch's average loss is better than minimum, save checkpoint
        current_avg_loss = avg_loss / len(dataloader)
        if min_avg_loss > current_avg_loss:
            min_avg_loss = current_avg_loss
            # save_checkpoint(
            #     ddpm, 
            #     optimizer, 
            #     filename=os.path.join(args.checkpoints, f"ddpm_epoch_{epoch}.pth.tar")
            # )
            
            # Also save the best model separately
            save_checkpoint(
                ddpm, 
                optimizer, 
                filename=os.path.join(args.checkpoints, "ddpm_best.pth.tar")
            )
        
        print(f"Epoch {epoch} completed. Average loss: {current_avg_loss:.6f}")
        
        if epoch % 10 == 0:
            print(f"Calculating FID score for epoch {epoch}...")
            fid_score = evaluate_model_fid(
                ddpm, 
                diffusion, 
                eval_dataloader,
                device, 
                n_samples=250,
                batch_size=args.batch_size
            )
            print(f"FID Score: {fid_score:.4f}")
            
            # Log FID score
            with open(os.path.join(args.results_folder, "fid_scores.txt"), "a") as f:
                f.write(f"{epoch},{fid_score:.4f},{current_avg_loss:.6f}\n")
            
            # Save model if FID is better
            if fid_score < best_fid:
                best_fid = fid_score
                save_checkpoint(
                    ddpm, 
                    optimizer, 
                    filename=os.path.join(args.checkpoints, "ddpm_best_fid.pth.tar")
                )
                print(f"New best FID: {best_fid:.4f}, saved model")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--noise_steps", type=int, default=1000)
    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--checkpoint_name", type=str, default="ddpm_best.pth.tar")
    parser.add_argument("--checkpoints", type=str, default="checkpoints")
    parser.add_argument("--results_folder", type=str, default="/home/22ucs095/CryoET/fake_data")
    parser.add_argument("--data_pattern", type=str, default="datasets/cryoET/*.tif")
    
    args = parser.parse_args()
    
    train(args)