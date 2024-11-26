import torch


class BinaryClassificationMetrics:
    """
    A class to track and calculate binary classification metrics.

    This class maintains counts of true positives (tp), false positives (fp),
    true negatives (tn), and false negatives (fn) to compute various performance metrics
    for binary classification tasks.

    Attributes:
        tp (int): Count of true positives
        fp (int): Count of false positives
        tn (int): Count of true negatives
        fn (int): Count of false negatives
    """

    def __init__(self, tp, fp, tn, fn):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

    def update_metrics(self, prediction, label):
        """
        Update metrics based on new predictions and labels.

        Args:
            prediction (torch.Tensor): Model predictions (0 or 1)
            label (torch.Tensor): True labels (0 or 1)
        """

        # Convert predictions and labels to booleans for comparison
        pred = prediction.bool()
        lab = label.bool()

        self.tp += torch.sum((pred & lab)).item()
        self.fp += torch.sum((pred & ~lab)).item()
        self.fn += torch.sum((~pred & lab)).item()
        self.tn += torch.sum((~pred & ~lab)).item()

    def accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.fp + self.tn + self.fn)

    def sensitivity(self):
        return self.tp / (self.tp + self.fn)

    def specificity(self):
        return self.tn / (self.tn + self.fp)

    def f1_score(self):
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)

    def confusion_matrix(self):
        return [[self.tp, self.fp], [self.fn, self.tn]]

    def __str__(self):
        return (
            f"Accuracy: {self.accuracy()}\n"
            f"Sensitivity: {self.sensitivity()}\n"
            f"Specificity: {self.specificity()}\n"
            f"F1 score: {self.f1_score()}\n"
            f"Confusion matrix: {self.confusion_matrix()}\n"
        )
