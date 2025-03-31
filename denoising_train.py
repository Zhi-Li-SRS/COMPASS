import argparse
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataload import create_dataloaders, load_data
from loss import Metrics, Denoise_Loss, msle_loss
from denoising_model import RamanDenoise
from utils import *
from vis_utils import *

import random
import matplotlib.pyplot as plt

class Denoise_Trainer:
    def __init__(self, args):
        self.args = args

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = self._prepare_model()
        self.criterion =  Denoise_Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-6)
        
        self.train_loader, self.val_loader, self.class_names = self._prepare_data()

        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.patience = 0
        self.early_stop = False


    def _prepare_model(self):
        model = RamanDenoise(
            input_channels=self.args.input_channels,
            base_channels=self.args.base_channels,
        )
        model = model.to(self.device)

        return model
    
    def _prepare_data(self) -> Tuple[DataLoader, DataLoader, list]:
        return create_dataloaders(
                train_path=self.args.train_path,
                val_path=self.args.val_path,
                target_path=self.args.target_path,
                batch_size=self.args.batch_size,
        )
    
    def _save_checkpoint(self, epoch, is_best=True):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        if epoch % self.args.save_freq == 0:
            periodic_path = os.path.join(self.args.checkpoint_dir, f"checkpoint_{epoch}.pth")
            torch.save(checkpoint, periodic_path)
            print(f"Saved checkpoint at {periodic_path}")

        if is_best:
            best_model_path = os.path.join(self.args.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            print(f"Saved best model at {best_model_path} with accuracy {self.best_val_acc:.4f}%")

    def _check_early_stop(self, val_loss: float) -> bool:
        """Check if training should be stopped early based on validation loss"""
        if val_loss < self.best_val_loss * (1 - self.args.min_improve):
            self.best_val_loss = val_loss
            self.patience = 0
            return False
        else:
            self.patience += 1
            if self.patience >= self.args.patience:
                print(f"Validation loss did not improve for {self.args.patience} epochs. Early stopping.")
                return True
            return False

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore training state"""
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]

        self.best_val_loss = checkpoint["best_val_loss"]

        start_epoch = checkpoint["epoch"] + 1

        print(f"Restored checkpoint from epoch {checkpoint['epoch']}")

        return start_epoch
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Denoising - Training")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            # Add channel dimension if needed
            if len(data.shape) == 2:
                data = data.unsqueeze(1)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Denoising - avg_loss": f"{total_loss/(batch_idx+1):.4f}"})

        return total_loss / len(self.train_loader)

    def validate(self):
        val_loss = 0.0
        self.model.eval()

        data_iterator = iter(self.val_loader)  # Create an iterator over the dataloader
        data, target = next(data_iterator)  # Fetch one batch

        data, target = data.to(self.device), target.to(self.device)
        batch_size = data.shape[0]
        random_indices = random.sample(range(batch_size), 10)

        data_samples = data[random_indices]
        target_samples = target[random_indices]

        with torch.no_grad():
            denoised_data_samples = self.model(data_samples.unsqueeze(1)).cpu().numpy()
            for data, target in tqdm(self.val_loader, desc="Denoising - Validation"):
                data, target = data.to(self.device), target.to(self.device)
                if len(data.shape) == 2:
                    data = data.unsqueeze(1)

                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)

        # Visualize Denoising Results
        plt.figure(figsize=(16,4))
        for i in range(5):
            plt.subplot(1, 5, i + 1)
            plt.plot(data_samples[i].cpu().numpy(), label="Noisy")
            plt.plot(target_samples[i].cpu().numpy(), label="Original", linestyle="dashed")
            plt.plot(denoised_data_samples[i], label="Denoised", color="black")
            plt.legend()
            plt.title(f"Sample {i+1}")


        plt.show()

        return avg_val_loss

    def train(self):
        if self.args.resume_checkpoint:
            start_epoch = self.load_checkpoint(self.args.resume_checkpoint)
        else:
            start_epoch = 1
            print("Starting fresh training")
        print(f"Training from epoch {start_epoch} to {self.args.epochs}")
        for epoch in range(1, self.args.epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.epochs}")

            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            val_loss = self.validate()
            self.val_losses.append(val_loss)

            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")

            self.scheduler.step()

            # Save checkpoints
            if epoch % self.args.save_freq == 0:
                self._save_checkpoint(epoch)

            if epoch > self.args.warmup_epochs:
                if self._check_early_stop(val_loss):
                    print(f"Early stopping at epoch {epoch}")
                    break
        print(f"Training completed.")

def main():
    print("Starting denoising training.")
    parser = argparse.ArgumentParser(description="Train RamanNet model on lipid dataset")

    # Model paramaters 
    parser.add_argument("--input_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--base_channels", type=int, default=16, help="Number of base channels in the model")
    parser.add_argument("--num_classes", type=int, default=25, help="Number of classes in the dataset")


    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="training batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data to total data")
    parser.add_argument("--sim_weight", type=float, default=0.5, help="Weight for cosine similarity loss term")

    parser.add_argument(
    "--resume_checkpoint", type=str, default=None, help="Path to checkpoint to resume training from"
    )

    # Early stopping parameters
    parser.add_argument(
        "--patience", type=int, default=20, help="Number of epochs to wait before early stopping"
    )
    parser.add_argument(
        "--min_improve", type=float, default=1e-4, help="Minimum improvement for early stopping"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=10, help="Number of warmup epochs before early stopping"
    )

    # Data parameters
    parser.add_argument(
        "--train_path", type=str, default="dataset/denoising_train_data.csv", help="Path to the denoising dataset"
    )
    parser.add_argument(
        "--val_path", type=str, default="dataset/denoising_val_data.csv", help="Path to the denoising dataset"
    )
    parser.add_argument(
        "--target_path", type=str, default="dataset/denoising_target_data.csv", help="Path to the target dataset"
    )

    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints"
    )
    parser.add_argument("--save_freq", type=int, default=50, help="Frequency of saving checkpoints (epochs)")

    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    trainer = Denoise_Trainer(args)
    trainer.train()

if __name__ == "__main__":
    main()