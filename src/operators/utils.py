import torch


def resolve_device(device_str: str = "auto", local_rank: int | None = None) -> torch.device:
    device_str = str(device_str).lower()

    if device_str == "cpu":
        return torch.device("cpu")

    if device_str.startswith("cuda"):
        if local_rank is not None and torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device(device_str)

    if device_str == "auto":
        if torch.cuda.is_available():
            if local_rank is not None:
                return torch.device(f"cuda:{local_rank}")
            return torch.device("cuda")
        return torch.device("cpu")

    raise ValueError(f"Unknown device string: {device_str}")
