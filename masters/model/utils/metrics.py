from torchmetrics import MetricCollection
from torchmetrics.classification import AUROC, Accuracy, F1Score, Precision, Recall


def threshold_metrics(
    num_classes: int, ignore_index: int | None = None
) -> MetricCollection:
    task = "binary" if num_classes == 2 else "multiclass"

    metrics = [
        Accuracy(task=task, num_classes=num_classes, ignore_index=ignore_index),
        Precision(task=task, num_classes=num_classes, ignore_index=ignore_index),
        Recall(task=task, num_classes=num_classes, ignore_index=ignore_index),
        F1Score(task=task, num_classes=num_classes, ignore_index=ignore_index),
    ]

    return MetricCollection(metrics)


def continuous_metrics(
    num_classes: int, ignore_index: int | None = None
) -> MetricCollection:
    task = "binary" if num_classes == 2 else "multiclass"

    return MetricCollection(
        [AUROC(task=task, num_classes=num_classes, ignore_index=ignore_index)]
    )
