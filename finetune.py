import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from dataload import create_dataloaders
from loss import Loss
from model import LipidNet
from utils import set_seed
from vis_utils import plot_train_history


class FineTuner:
    def __init__(self, args):
        self.args = args
        set_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load pretrained_model
        self.model = self._load_pretrained_model()

        # Prepare data loaders with background class
        self.train_loader, self.val_loader, self.class_names = create_dataloaders(
            train_path=args.train_data_path, val_path=args.val_data_path, batch_size=args.batch_size
        )

        self.criterion = Loss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-6)
        # Training metrics
        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []
        self.patience = 0

        # Create output directory
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    def _load_pretrained_model(self):
        """Load the pretrained LipidNet model and modify the output layer for fine-tuning."""
        model = LipidNet(
            input_channels=self.args.input_channels,
            base_channels=self.args.base_channels,
            num_classes=self.args.original_num_classes,
        )

        checkpoint = torch.load(self.args.pretrained_model_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])

        new_num_classes = self.args.original_num_classes + 1  # Add background class

        in_features = model.classifier[0].in_features
        hidden_features = model.classifier[0].out_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(hidden_features, new_num_classes),
        )
        return model.to(self.device)

    def save_checkpoint(self, epoch, is_best=True):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_accuracies": self.val_accuracies,
            "val_losses": self.val_losses,
            "class_names": self.class_names,
        }

        if is_best:
            best_path = os.path.join(self.args.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation accuracy: {self.best_val_acc:.2f}%")

        if epoch % self.args.save_freq == 0:
            epoch_path = os.path.join(self.args.checkpoint_dir, f"epoch_{epoch}.pth")
            torch.save(checkpoint, epoch_path)
            print(f"Saved checkpoint at epoch {epoch}")

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

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training")

        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            if len(data.shape) == 2:
                data = data.unsqueeze(1)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"avg_loss": f"{total_loss/(batch_idx+1):.4f}"})

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)

                if len(data.shape) == 2:
                    data = data.unsqueeze(1)

                output = self.model(data)
                loss = self.criterion(output, target)
                val_loss += loss.item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        accuracy = 100.0 * correct / total
        avg_val_loss = val_loss / len(self.val_loader)
        return accuracy, avg_val_loss

    def finetune(self):
        for epoch in range(1, self.args.epochs + 1):
            print(f"\nEpoch {epoch}/{self.args.epochs}")
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            val_accuracy, val_loss = self.validate()
            self.val_accuracies.append(val_accuracy)
            self.val_losses.append(val_loss)

            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.2f}%")

            self.scheduler.step()

            # Save best model
            is_best = val_accuracy > self.best_val_acc
            if is_best:
                self.best_val_acc = val_accuracy

            if is_best or epoch % self.args.save_freq == 0:
                self.save_checkpoint(epoch, is_best)

            # Early stopping
            if epoch > self.args.warmup_epochs:
                if self._check_early_stop(val_loss):
                    print(f"Early stopping at epoch {epoch}")
                    break

        print(f"Fine-tuning completed. Best validation accuracy: {self.best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LipidNet model with background class")

    # Model parameters
    parser.add_argument("--input_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--base_channels", type=int, default=8, help="Number of base channels")
    parser.add_argument(
        "--original_num_classes", type=int, default=18, help="Number of classes in pretrained model"
    )

    # Fine-tuning parameters
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="checkpoints_18sub/best_model.pth",
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="Raman_dataset/train_data_with_bg.csv",
        help="Path to training data with background class",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="Raman_dataset/val_data_with_bg.csv",
        help="Path to validation data with background class",
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints_with_bg", help="Directory to save fine-tuned model"
    )

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (lower than initial training)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--save_freq", type=int, default=50, help="Save frequency (epochs)")

    # Early stopping
    parser.add_argument("--patience", type=int, default=20, help="Patience for early stopping")
    parser.add_argument(
        "--min_improve", type=float, default=1e-4, help="Minimum improvement for early stopping"
    )
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Warmup epochs before early stopping")

    args = parser.parse_args()
    fine_tuner = FineTuner(args)
    fine_tuner.finetune()


if __name__ == "__main__":
    main()
