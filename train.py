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
from loss import Loss, Metrics
from model import LipidNet
from utils import *
from vis_utils import *


class Trainer:
    def __init__(self, args):
        self.args = args
        set_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = Metrics()

        self.model = self._prepare_model()
        self.criterion = Loss(use_corr=args.use_corr, corr_weight=args.corr_weight)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs, eta_min=1e-6)
        # Updated data loading
        self.train_loader, self.val_loader, self.class_names = self._prepare_data()

        self.best_val_acc = 0.0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_accuracies = []
        self.val_losses = []
        self.patience = 0
        self.early_stop = False

        self.vis_dir = os.path.join(args.checkpoint_dir, "train_figures")
        os.makedirs(self.vis_dir, exist_ok=True)

    def _prepare_model(self):
        model = LipidNet(
            input_channels=self.args.input_channels,
            base_channels=self.args.base_channels,
            num_classes=self.args.num_classes,
        )
        return model.to(self.device)

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        return create_dataloaders(
            train_path=self.args.train_data_path,
            val_path=self.args.val_data_path,
            batch_size=self.args.batch_size,
        )

    def _save_checkpoint(self, epoch, is_best=True):
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
        }

        if epoch % self.args.save_freq == 0:
            periodic_path = os.path.join(self.args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save(checkpoint, periodic_path)
            print(f"Saved periodic checkpoint at epoch {epoch}")

        if is_best:
            best_model_path = os.path.join(self.args.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_model_path)
            print(f"Saved new best model with validation accuracy: {self.best_val_acc:.2f}%")

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
        self.val_accuracies = checkpoint["val_accuracies"]

        self.best_val_acc = checkpoint["best_val_acc"]
        self.best_val_loss = checkpoint["best_val_loss"]

        start_epoch = checkpoint["epoch"] + 1

        print(f"Restored checkpoint from epoch {checkpoint['epoch']}")
        print(f"Best validation accuracy so far: {self.best_val_acc:.2f}%")

        return start_epoch

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)

            # Add channel dimension if needed
            if len(data.shape) == 2:
                data = data.unsqueeze(1)

            self.optimizer.zero_grad()
            output = self.model(data)

            if self.args.use_corr:
                loss = self.criterion(output, target, data, data)
            else:
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
                if self.args.use_corr:
                    loss = self.criterion(output, target, data, data)
                else:
                    loss = self.criterion(output, target)
                val_loss += loss.item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        accuracy = 100.0 * correct / total
        avg_val_loss = val_loss / len(self.val_loader)
        return accuracy, avg_val_loss

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

            val_accuracy, val_loss = self.validate()
            self.val_accuracies.append(val_accuracy)
            self.val_losses.append(val_loss)

            print(f"Training Loss: {train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.2f}%")

            self.scheduler.step()

            # Save checkpoints
            is_best = val_accuracy > self.best_val_acc
            if is_best:
                self.best_val_acc = val_accuracy

            if epoch % self.args.save_freq == 0 or is_best:
                self._save_checkpoint(epoch, is_best)

            if epoch > self.args.warmup_epochs:
                if self._check_early_stop(val_loss):
                    print(f"Early stopping at epoch {epoch}")
                    break
        print(f"Training completed. Best validation accuracy: {self.best_val_acc:.2f}%")

        self._create_train_figures()

    def _create_train_figures(self):
        """Create training and evaluation figures"""
        self.model.eval()
        y_true = []
        y_pred = []
        y_pred_proba = []

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Final Evaluation"):
                data, target = data.to(self.device), target.to(self.device)
                if len(data.shape) == 2:
                    data = data.unsqueeze(1)

                output = self.model(data)
                probs = torch.softmax(output, dim=1)

                y_true.extend(target.cpu().numpy())
                y_pred.extend(output.argmax(dim=1, keepdim=True).cpu().numpy())
                y_pred_proba.extend(probs.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)

        precision, recall, f1 = self.metrics.compute_precision_recall_f1(y_true, y_pred)

        class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        class_metrics = {"precision": class_precision, "recall": class_recall, "f1": class_f1}

        plot_train_history(
            save_dir=self.vis_dir,
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            val_accuracies=self.val_accuracies,
            y_true=y_true,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            class_names=self.class_names,
            precision=precision,
            recall=recall,
            f1=f1,
            class_metrics=class_metrics,
        )

        print("\nFinal Evaluation Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"\nVisualization and metrics saved to: {self.vis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train LipidNet model on lipid dataset")

    # Model parameters
    parser.add_argument("--input_channels", type=int, default=1, help="Number of input channels")
    parser.add_argument("--base_channels", type=int, default=8, help="Number of base channels in the model")
    parser.add_argument("--num_classes", type=int, default=18, help="Number of classes in the dataset")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data to total data")
    parser.add_argument(
        "--use_corr", type=bool, default=True, help="Use correlation loss in addition to CrossEntropy"
    )
    parser.add_argument("--corr_weight", type=float, default=0.2, help="Weight for correlation loss term")
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
        "--train_data_path", type=str, default="dataset/train_data.csv", help="Path to the dataset"
    )
    parser.add_argument(
        "--val_data_path", type=str, default="dataset/val_data.csv", help="Path to the dataset"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints"
    )
    parser.add_argument("--save_freq", type=int, default=50, help="Frequency of saving checkpoints (epochs)")

    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
