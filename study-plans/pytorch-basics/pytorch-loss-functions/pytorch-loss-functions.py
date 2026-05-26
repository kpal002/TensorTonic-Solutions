import torch
import torch.nn.functional as F

def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: scalar loss
    """

    pred = torch.tensor(pred, dtype=torch.float32)

    if method in ["mse", "mae", "huber"]:
        target = torch.tensor(target, dtype=torch.float32)
        error = pred - target

        if method == "mse":
            loss = error ** 2

        elif method == "mae":
            loss = torch.abs(error)

        elif method == "huber":
            abs_error = torch.abs(error)
            loss = torch.where(
                abs_error <= delta,
                0.5 * error ** 2,
                delta * (abs_error - 0.5 * delta)
            )

        return loss.mean()   # IMPORTANT

    elif method == "cross_entropy":
        # pred = logits [batch, num_classes]
        # target = class indices [batch]

        target = torch.tensor(target, dtype=torch.long)
        loss = F.cross_entropy(pred, target)

        return loss

    else:
        raise ValueError(f"Unknown method: {method}")