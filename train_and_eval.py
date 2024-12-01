import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
from preprocess_data import get_dataloaders, PPGDataset
from model import Heart_GPT_FineTune, ClassificationHead
from metrics import BinaryClassificationMetrics

"""
The values for the Config class and large parts of the main function were taken from: 
https://github.com/harryjdavies/HeartGPT/blob/main/HeartGPT_finetuning.py

Check out the paper by Davies et al. (2024) "Interpretable Pre-Trained Transformers for Heart Time-Series Data" 
available at https://arxiv.org/abs/2407.20775 to find out more about their amazing work!
"""


class Config:
    """
    Configuration class holding all hyperparameters and settings for the model.

    Attributes:
        save_interval (int): Number of epochs between checkpoints
        epochs (int): Total number of training epochs
        prediction_threshold (float): Threshold for binary classification
        early_stopping_threshold (int): Number of epochs without improvement before stopping
        batch_size (int): Number of samples per batch
        num_workers (int): Number of DataLoader workers
        pin_memory (bool): Whether to pin memory in DataLoader
        persistent_workers (bool): Whether to keep DataLoader workers alive
        use_amp (bool): Whether to use automatic mixed precision
        block_size (int): Size of transformer blocks
        n_embd (int): Embedding dimension
        n_head (int): Number of attention heads
        n_layer (int): Number of transformer layers
        dropout (float): Dropout rate
        learning_rate (float): Learning rate for optimizer
        blocks_to_unfreeze (int): Number of transformer blocks to unfreeze for fine-tuning
        vocab_size (int): Size of the vocabulary
        device (str): Device to use for training ('cuda', 'mps', or 'cpu')
        model_path (str): Path to save/load model weights
        data_path (str): Path to data directory
        loss_function (nn.Module): Loss function
    """

    save_interval = 1
    epochs = 100
    prediction_threshold = 0.5
    early_stopping_threshold = 15
    batch_size = 384
    num_workers = 4
    pin_memory = True
    persistent_workers = True
    use_amp = True
    block_size = 500
    n_embd = 64
    n_head = 8
    n_layer = 8
    dropout = 0.2
    learning_rate = 3e-04
    blocks_to_unfreeze = 5
    vocab_size = 102
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model_path = "model_weights/"
    data_path = "data/"
    loss_function = nn.BCEWithLogitsLoss()


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
            label = label.to(torch.float32)
            loss = cfg.loss_function(output, label)
            total_loss += loss.item()

            # Apply sigmoid before thresholding for predictions
            prediction = (
                nn.functional.sigmoid(output) > cfg.prediction_threshold
            ).float()
            metrics.update_metrics(prediction, label)

    print(metrics.__str__())

    avg_loss = total_loss / len(data_loader)

    return avg_loss


def save_checkpoint(
    cfg, model, optimizer, epoch, scaler, metric_history, checkpoint_name
):
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        cfg (Config): Configuration object containing model settings and paths
        model (nn.Module): The model to save
        optimizer (Optimizer): Optimizer for updating model parameters
        epoch (int): Current epoch
        scaler (GradScaler): AMP GradScaler for mixed precision training
        metric_history (dict): Dictionary containing training and validation metric history
        checkpoint_name (str): Name of the checkpoint file to save (e.g., "checkpoint_best.pt")
    """

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metric_history": metric_history,
    }

    if cfg.use_amp:
        checkpoint["scaler"] = scaler.state_dict()

    os.makedirs(cfg.model_path, exist_ok=True)
    torch.save(checkpoint, os.path.join(cfg.model_path, checkpoint_name))


def train(
    cfg,
    model,
    optimizer,
    train_loader,
    val_loader=None,
    checkpoint=None,
):
    """
    Trains the model using the specified configuration and data.

    Args:
        cfg (Config): Configuration object containing training settings
        model (nn.Module): The model to train
        optimizer (Optimizer): Optimizer for updating model parameters
        train_loader (DataLoader): DataLoader for training data
        val_loader (DataLoader, optional): DataLoader for validation data
        checkpoint (dict, optional): Checkpoint to resume training from

    Notes:
        - Implements early stopping based on validation accuracy
        - Saves checkpoints periodically and when best performance is achieved
        - Uses mixed precision training when cfg.use_amp is True
    """

    model = model.to(cfg.device)
    scaler = torch.amp.GradScaler()
    best_val_acc = 0
    early_stopping = 0
    start_epoch = 0

    # Metric history
    metrics_history = {
        "train_loss": [],
        "train_acc": [],
        "train_sensitivity": [],
        "train_specificity": [],
        "train_f1": [],
        "val_loss": [],
        "val_acc": [],
        "val_sensitivity": [],
        "val_specificity": [],
        "val_f1": [],
    }

    # Resume from checkpoint if provided
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        metrics_history = checkpoint["metric_history"]

        if cfg.use_amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
            print("Restored AMP scaler state from checkpoint")

        print(f"Resuming training from epoch {start_epoch}")

    # Main training loop
    for epoch in tqdm(range(start_epoch, cfg.epochs), desc="Training", position=0):
        model.train()
        training_loss = 0.0
        train_metrics = BinaryClassificationMetrics(0, 0, 0, 0)

        # Batch training loop
        for ppg, label in tqdm(
            train_loader, desc=f"Epoch {epoch}", position=0, leave=True
        ):
            ppg = ppg.to(cfg.device)
            label = label.to(cfg.device)

            optimizer.zero_grad()
            with torch.autocast(
                device_type=cfg.device, dtype=torch.float16, enabled=cfg.use_amp
            ):
                output = model(ppg)
                label = label.to(torch.float32)
                loss = cfg.loss_function(output, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            training_loss += loss.item()
            # Apply sigmoid before thresholding for predictions
            prediction = (
                nn.functional.sigmoid(output) > cfg.prediction_threshold
            ).float()
            train_metrics.update_metrics(prediction, label)

        avg_loss = training_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        # Print training metrics
        print(f"Training metrics for epoch {epoch}: \n")
        print(f"Loss: {avg_loss}")
        print(train_metrics.__str__())

        # Calculate and store all training metrics
        metrics_history["train_loss"].append(avg_loss)
        metrics_history["train_acc"].append(train_metrics.accuracy())
        metrics_history["train_sensitivity"].append(train_metrics.sensitivity())
        metrics_history["train_specificity"].append(train_metrics.specificity())
        metrics_history["train_f1"].append(train_metrics.f1_score())

        # Validation phase
        if val_loader is not None:
            val_metrics = BinaryClassificationMetrics(0, 0, 0, 0)

            # Print validation metrics
            print("Validation metrics: \n")
            val_loss = evaluate(cfg, model, val_loader, val_metrics)

            # Calculate and store all validation metrics
            metrics_history["val_loss"].append(val_loss)
            metrics_history["val_acc"].append(val_metrics.accuracy())
            metrics_history["val_sensitivity"].append(val_metrics.sensitivity())
            metrics_history["val_specificity"].append(val_metrics.specificity())
            metrics_history["val_f1"].append(val_metrics.f1_score())

            # Save best model
            if val_metrics.accuracy() > best_val_acc:
                best_val_acc = val_metrics.accuracy()
                save_checkpoint(
                    cfg,
                    model,
                    optimizer,
                    epoch,
                    scaler,
                    metrics_history,
                    "checkpoint_best.pt",
                )
                print(f"Best model saved at epoch {epoch}")
                early_stopping = 0
            else:
                early_stopping += 1
                if early_stopping >= cfg.early_stopping_threshold:
                    print(f"Stopping training at epoch {epoch} due to early stopping")
                    break

        # Save checkpoint every cfg.save_interval epochs
        if epoch % cfg.save_interval == cfg.save_interval - 1:
            save_checkpoint(
                cfg,
                model,
                optimizer,
                epoch,
                scaler,
                metrics_history,
                f"checkpoint_epoch_{epoch}.pt",
            )
            print(f"Checkpoint saved for epoch {epoch}")

    return metrics_history


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
    test_metrics = BinaryClassificationMetrics(0, 0, 0, 0)

    # Print test metrics
    print("Testing metrics: \n")
    evaluate(cfg, model, test_loader, test_metrics)


def main():
    """
    Main function that handles the model fine-tuning process:
    1. Loads pretrained model
    2. Freezes all layers except the last blocks_to_unfreeze blocks and the classification head
    3. Trains the model
    4. Saves the fine-tuned model
    5. Tests the performance of fine-tuned model
    """

    cfg = Config()
    model = Heart_GPT_FineTune(cfg).to(cfg.device)

    # Use cuDNN benchmark for faster training if using CUDA
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"Using device: {cfg.device}")

    model.load_state_dict(
        torch.load(
            os.path.join(cfg.model_path, "PPGPT_500k_iters.pth"),
            map_location=cfg.device,
            weights_only=True,
        )
    )

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    model.lm_head = ClassificationHead(cfg)

    # Replace and unfreeze head
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # Unfreeze layer normalization
    for param in model.ln_f.parameters():
        param.requires_grad = True

    last_blocks = model.blocks[-cfg.blocks_to_unfreeze :]

    # Unfreeze last four transformer blocks
    for block in last_blocks:
        for param in block.parameters():
            param.requires_grad = True

    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    train(
        cfg,
        model,
        optimizer,
        train_loader,
        val_loader,
    )
    torch.save(model.state_dict(), os.path.join(cfg.model_path, "PPGPT_finetuned.pth"))

    test(cfg, model, test_loader)


if __name__ == "__main__":
    main()
