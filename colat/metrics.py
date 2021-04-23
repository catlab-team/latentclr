import torch


class LossMetric:
    """Keeps track of the loss over an epoch"""

    def __init__(self) -> None:
        self.running_loss = 0
        self.count = 0

    def update(self, loss: float, batch_size: int) -> None:
        self.running_loss += loss * batch_size
        self.count += batch_size

    def compute(self) -> float:
        return self.running_loss / self.count

    def reset(self) -> None:
        self.running_loss = 0
        self.count = 0


class AccuracyMetric:
    """Keeps track of the top-k accuracy over an epoch

    Args:
        k (int): Value of k for top-k accuracy
    """

    def __init__(self, k: int = 1) -> None:
        self.correct = 0
        self.total = 0
        self.k = k

    def update(self, out: torch.Tensor, target: torch.Tensor) -> None:
        # Computes top-k accuracy
        _, indices = torch.topk(out, self.k, dim=-1)
        target_in_top_k = torch.eq(indices, target[:, None]).bool().any(-1)
        total_correct = torch.sum(target_in_top_k, dtype=torch.int).item()
        total_samples = target.shape[0]

        self.correct += total_correct
        self.total += total_samples

    def compute(self) -> float:
        return self.correct / self.total

    def reset(self) -> None:
        self.correct = 0
        self.total = 0
