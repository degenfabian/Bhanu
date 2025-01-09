import torch
import torch.nn as nn
import os
from tqdm import tqdm
from model import Bhanu
from preprocess_data import get_dataloaders, PPGDataset
from metrics import BinaryClassificationMetrics


class Config:
    """
    Configuration class holding all hyperparameters and settings for the model.

    Attributes:
        save_interval (int): Number of epochs between checkpoints
        epochs (int): Total number of training epochs
        prediction_threshold (float): Threshold for binary classification
        early_stopping_patience (int): Number of epochs without improvement before stopping
        batch_size (int): Number of samples per batch
        num_workers (int): Number of DataLoader workers
        pin_memory (bool): Whether to pin memory in DataLoader
        persistent_workers (bool): Whether to keep DataLoader workers alive
        dropout (float): Dropout rate
        learning_rate (float): Learning rate for optimizer
        weight_decay (float): Weight decay for optimizer
        gradient_clipping_threshold (float): Maximum gradient norm
        device (str): Device to use for training ('cuda', 'mps', or 'cpu')
        loss_function (nn.Module): Loss function for training
        model_path (str): Path to save/load model weights
        data_path (str): Path to data directory
    """

    save_interval = 1
    epochs = 90
    prediction_threshold = 0.5
    early_stopping_patience = 10
    batch_size = 128
    num_workers = 8
    pin_memory = True
    persistent_workers = True
    dropout = 0.5
    learning_rate = 1e-5
    weight_decay = 0.1
    gradient_clipping_threshold = 1.0
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.325]).to(device))
    model_path = "model_weights/"
    data_path = "data/"


def evaluate(cfg, model, data_loader, metrics):
    """
    Evaluates model performance on a given dataset.

    Args:
        cfg (Config): Configuration object containing model settings
        model (nn.Module): The model to evaluate
        data_loader (DataLoader): DataLoader containing the evaluation data
        metrics (BinaryClassificationMetrics): Metrics tracker object

    Note:
        Updates the metrics object in-place and prints the evaluation results.
    """

    model = model.to(cfg.device)
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for ppg, label in tqdm(data_loader, desc="Evaluating", position=0, leave=True):
            ppg = ppg.to(cfg.device)
            label = label.to(cfg.device)

            output = model(ppg)
            loss = cfg.loss_function(output, label)
            total_loss += loss.item()

            # Apply sigmoid to get probabilities
            raw_prediction = nn.functional.sigmoid(output)
            metrics.track_batch_results(
                cfg, raw_prediction, label
            )  # Update internal metric counters

    avg_loss = total_loss / len(data_loader)

    # Calculate and print metrics
    metrics.calculate_and_print_metrics(avg_loss, 0)


def save_checkpoint(
    cfg,
    model,
    optimizer,
    epoch,
    scheduler,
    train_metrics,
    val_metrics,
    checkpoint_name,
):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        cfg (Config): Configuration object containing model settings and paths
        model (nn.Module): The model to save
        optimizer (Optimizer): Optimizer for updating model parameters
        epoch (int): Current epoch
        scheduler (lr_scheduler): Learning rate scheduler instance
        train_metrics (BinaryClassificationMetrics): Training metrics history
        val_metrics (BinaryClassificationMetrics): Validation metrics history
        checkpoint_name (str): Name of the checkpoint file to save (e.g., "checkpoint_best.pt")
    """

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "scheduler": scheduler.state_dict(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    os.makedirs(cfg.model_path, exist_ok=True)
    torch.save(checkpoint, os.path.join(cfg.model_path, checkpoint_name))


def train(
    cfg,
    model,
    train_loader,
    val_loader=None,
    checkpoint=None,
):
    """
    Trains the model using the specified configuration and data.

    Args:
        cfg (Config): Configuration object containing training settings
        model (nn.Module): The model to train
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader, optional): DataLoader for validation data
        checkpoint (dict, optional): Checkpoint to resume training from

    Notes:
        - Implements early stopping based on AUC-ROC score
        - Saves periodic checkpoints and best model based on validation AUC-ROC
        - Implements gradient clipping for training stability
        - Implements linear learning rate warmup and decay as described in the original ViT paper
        - Maintains comprehensive metrics history for both training and validation
        - Can resume training from a checkpoint including optimizer, scheduler,
          and metrics state
    """

    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # Linear learning rate warmup and decay as described in the original ViT paper
    steps_per_epoch = len(train_loader)
    linear_warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1, total_iters=3 * steps_per_epoch
    )
    linear_decay = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.001,
        total_iters=(cfg.epochs - 3) * steps_per_epoch,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[linear_warmup, linear_decay],
        milestones=[3 * steps_per_epoch],
    )

    best_auc_roc = 0
    early_stopping = 0
    start_epoch = 0

    # Metric history
    train_metrics = BinaryClassificationMetrics()
    val_metrics = BinaryClassificationMetrics()

    # Resume from checkpoint if provided
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        train_metrics = checkpoint["train_metrics"]
        val_metrics = checkpoint["val_metrics"]

        print(f"Resuming training from epoch {start_epoch}")

    # Main training loop
    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Training", position=0):
        model.train()
        training_loss = 0.0

        # Batch training loop
        for ppg, label in tqdm(
            train_loader, desc=f"Epoch {epoch}", position=0, leave=True
        ):
            ppg = ppg.to(cfg.device)
            label = label.to(cfg.device)

            optimizer.zero_grad()

            output = model(ppg)
            loss = cfg.loss_function(output, label)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                model.parameters(), cfg.gradient_clipping_threshold
            )

            optimizer.step()
            scheduler.step()

            training_loss += loss.item()

            # Apply sigmoid before thresholding for predictions
            raw_prediction = nn.functional.sigmoid(output)
            train_metrics.track_batch_results(
                cfg, raw_prediction, label
            )  # Update internal metric counters

        avg_loss = training_loss / len(train_loader)

        print(f"Training metrics for epoch {epoch}: \n")
        train_metrics.calculate_and_print_metrics(
            avg_loss, scheduler.get_last_lr()[0]
        )  # Calculate and print training metrics

        # Validation phase
        if val_loader is not None:
            # Print validation metrics
            print(f"Validation metrics for epoch {epoch}: \n")
            evaluate(cfg, model, val_loader, val_metrics)

            current_auc_roc = val_metrics.get_current_auc_roc()

            # Save best model based on AUC-ROC
            if current_auc_roc > best_auc_roc:
                best_auc_roc = current_auc_roc
                save_checkpoint(
                    cfg,
                    model,
                    optimizer,
                    epoch,
                    scheduler,
                    train_metrics,
                    val_metrics,
                    "checkpoint_best.pt",
                )
                print(f"Best model saved at epoch {epoch}")
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping >= cfg.early_stopping_patience:
                    print(f"Stopping training at epoch {epoch} due to early stopping")
                    break

        # Save checkpoint every cfg.save_interval epochs
        if epoch % cfg.save_interval == cfg.save_interval - 1:
            save_checkpoint(
                cfg,
                model,
                optimizer,
                epoch,
                scheduler,
                train_metrics,
                val_metrics,
                f"checkpoint_epoch_{epoch}.pt",
            )
            print(f"Checkpoint saved for epoch {epoch}")


def test(cfg, model, test_loader):
    """
    Tests the model using the best checkpoint from training.

    Args:
        cfg (Config): Configuration object containing model settings
        model (nn.Module): The model to test
        test_loader (DataLoader): DataLoader containing test data
    """

    checkpoint = torch.load(os.path.join(cfg.model_path, "checkpoint_best.pt"))
    model.load_state_dict(checkpoint["model"])
    test_metrics = BinaryClassificationMetrics()

    print(f"Testing model trained for {checkpoint['epoch']} epochs\n")
    print("Testing metrics: \n")
    evaluate(cfg, model, test_loader, test_metrics)


def main():
    """
    Main function that sets up the training process:
    1. Initializes the model with the configuration
    2. Trains the model
    3. Saves the best model based on validation AUC-ROC
    4. Tests the performance on test set
    """

    cfg = Config()
    model = Bhanu(cfg)

    # Use cuDNN benchmark for faster training if using CUDA
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {cfg.device}")

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    train(
        cfg,
        model,
        train_loader,
        val_loader,
    )
    torch.save(model.state_dict(), os.path.join(cfg.model_path, "Bhanu.pt"))

    test(cfg, model, test_loader)


if __name__ == "__main__":
    main()
