import torch
from sklearn.metrics import roc_auc_score


class BinaryClassificationMetrics:
    """
    A comprehensive metrics tracker for binary classification models that calculates
    and maintains common performance metrics across training epochs.

    This class provides functionality to:
    - Track raw predictions and ground truth labels
    - Calculate threshold-based metrics using confusion matrix elements
    - Compute common classification metrics like accuracy, AUC-ROC, sensitivity, etc.
    - Maintain historical values of metrics across epochs
    - Reset counts between epochs or evaluation phases

    Attributes:
        Confusion Matrix Components:
            tp (int): True positives count
            fp (int): False positives count
            tn (int): True negatives count
            fn (int): False negatives count

        Raw Data:
            raw_predictions (list): Model predictions before thresholding
            raw_labels (list): Corresponding ground truth labels

        Metric History:
            loss (list): Loss per epoch
            auc_roc (list): Area Under ROC Curve scores per epoch
            accuracy (list): Classification accuracy per epoch
            sensitivity (list): True Positive Rate (recall) per epoch
            specificity (list): True Negative Rate (precision) per epoch
            f1_score (list): Harmonic mean of recall and precision per epoch

    Notes:
        - Between epochs, everything except metric history is reset.
    """

    def __init__(self):
        # These counts and raw_data are used to calculate metrics at the end of each epoch and get reset between epochs
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.tn = 0  # True Negatives
        self.fn = 0  # False Negatives
        self.raw_predictions = []  # Raw predictions before thresholding
        self.raw_labels = []  # Ground truth labels

        # These lists store metric values across epochs for plotting after training is completed
        self.loss = []
        self.learning_rate = []
        self.auc_roc = []
        self.accuracy = []
        self.sensitivity = []
        self.specificity = []
        self.f1_score = []

    def reset(self):
        """
        Resets all counters and raw data for the next epoch.
        Maintains historical metrics.
        """

        self.tp = self.fp = self.tn = self.fn = 0
        self.raw_predictions = []
        self.raw_labels = []

    def track_batch_results(self, cfg, raw_prediction, label):
        """
        Updates metrics with predictions from the current batch.

        Args:
            cfg : Config
                Configuration object containing prediction_threshold
            raw_prediction : torch.Tensor
                Model predictions (probabilities)
        label : torch.Tensor
            Ground truth labels (0 or 1)
        """

        # Store raw values for calculating AUC-ROC
        self.raw_predictions.extend(raw_prediction.detach().cpu().numpy().flatten())
        self.raw_labels.extend(label.detach().cpu().numpy().flatten())

        # Apply threshold to predictions and convert to boolean
        thresholded_prediction = (raw_prediction > cfg.prediction_threshold).bool()
        label = label.bool()

        # Update confusion matrix counts
        self.tp += torch.sum((thresholded_prediction & label)).item()
        self.fp += torch.sum((thresholded_prediction & ~label)).item()
        self.fn += torch.sum((~thresholded_prediction & label)).item()
        self.tn += torch.sum((~thresholded_prediction & ~label)).item()

    def calculate_and_print_metrics(self, loss, learning_rate):
        """
        Computes and stores all metrics for the current epoch.

        Args:
            loss: Average loss value for the epoch
            learning_rate: Learning rate for the current epoch

        Returns:
            None
            (Updates internal metric history lists)
        """

        # Calculate metrics for current epoch
        metrics = {
            "loss": loss,
            "learning_rate": learning_rate,
            "auc_roc": self.calculate_auc_roc(),
            "accuracy": self.calculate_accuracy(),
            "sensitivity": self.calculate_sensitivity(),
            "specificity": self.calculate_specificity(),
            "f1_score": self.calculate_f1_score(),
        }

        # Update tracking lists
        for metric_name, value in metrics.items():
            getattr(self, metric_name).append(value)

        # Reset counts and raw data for next epoch
        self.reset()

        # Print metrics for the epoch
        print(self.__str__())

    def get_current_auc_roc(self):
        return self.auc_roc[-1]

    def calculate_auc_roc(self):
        return roc_auc_score(self.raw_labels, self.raw_predictions)

    def calculate_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    def calculate_sensitivity(self):
        return self.tp / (self.tp + self.fn)

    def calculate_specificity(self):
        return self.tn / (self.tn + self.fp)

    def calculate_f1_score(self):
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)

    def __str__(self):
        return (
            f"Loss: {self.loss[-1]}\n"
            f"Learning Rate: {self.learning_rate[-1]}\n"
            f"AUC-ROC: {self.auc_roc[-1]}\n"
            f"Accuracy: {self.accuracy[-1]}\n"
            f"Sensitivity: {self.sensitivity[-1]}\n"
            f"Specificity: {self.specificity[-1]}\n"
            f"F1 Score: {self.f1_score[-1]}\n"
        )
