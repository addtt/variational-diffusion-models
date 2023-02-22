import dataclasses
import json
import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchinfo
import yaml
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm


@dataclass
class TrainConfig:
    embedding_dim: float
    n_blocks: int
    n_attention_heads: int
    dropout_prob: float
    norm_groups: int
    input_channels: int
    use_fourier_features: bool
    attention_everywhere: bool
    batch_size: int
    noise_schedule: str
    gamma_min: float
    gamma_max: float
    antithetic_time_sampling: bool
    lr: float
    weight_decay: float
    clip_grad_norm: bool


def print_model_summary(model, *, batch_size, shape, depth=4, batch_size_torchinfo=1):
    summary = torchinfo.summary(
        model,
        [(batch_size_torchinfo, *shape), (batch_size_torchinfo,)],
        depth=depth,
        col_names=["input_size", "output_size", "num_params"],
        verbose=0,  # quiet
    )
    log(summary)
    if batch_size is None or batch_size == batch_size_torchinfo:
        return
    output_bytes_large = summary.total_output_bytes / batch_size_torchinfo * batch_size
    total_bytes = summary.total_input + output_bytes_large + summary.total_param_bytes
    log(
        f"\n--- With batch size {batch_size} ---\n"
        f"Forward/backward pass size: {output_bytes_large / 1e9:0.2f} GB\n"
        f"Estimated Total Size: {total_bytes / 1e9:0.2f} GB\n"
        + "=" * len(str(summary).splitlines()[-1])
        + "\n"
    )


def cycle(dl):
    # We don't use itertools.cycle because it caches the entire iterator.
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def sample_batched(model, num_samples, batch_size, n_sample_steps, clip_samples):
    samples = []
    for i in range(0, num_samples, batch_size):
        corrected_batch_size = min(batch_size, num_samples - i)
        samples.append(model.sample(corrected_batch_size, n_sample_steps, clip_samples))
    return torch.cat(samples, dim=0)


def get_date_str():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


class DeviceAwareDataLoader(DataLoader):
    """A DataLoader that moves batches to a device. If device is None, it is equivalent to a standard DataLoader."""

    def __init__(self, *args, device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device

    def __iter__(self):
        for batch in super().__iter__():
            yield self.move_to_device(batch)

    def move_to_device(self, batch):
        if self.device is None:
            return batch
        if isinstance(batch, (tuple, list)):
            return [self.move_to_device(x) for x in batch]
        elif isinstance(batch, dict):
            return {k: self.move_to_device(v) for k, v in batch.items()}
        elif isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        else:
            return batch


def evaluate_model(model, dataloader):
    all_metrics = defaultdict(list)
    for batch in tqdm(dataloader, desc="evaluation"):
        loss, metrics = model(batch)
        for k, v in metrics.items():
            try:
                v = v.item()
            except AttributeError:
                pass
            all_metrics[k].append(v)
    return {k: sum(v) / len(v) for k, v in all_metrics.items()}  # average over dataset


def log_and_save_metrics(avg_metrics, dataset_split, step, filename):
    log(f"\n{dataset_split} metrics:")
    for k, v in avg_metrics.items():
        log(f"    {k}: {v}")

    avg_metrics = {"step": step, "set": dataset_split, **avg_metrics}
    with open(filename, "a") as f:
        json.dump(avg_metrics, f)
        f.write("\n")


def dict_stats(dictionaries: list[dict]) -> dict:
    """Computes the average and standard deviation of metrics in a list of dictionaries.

    Args:
        dictionaries: A list of dictionaries, where each dictionary contains the same keys,
            and the values are numbers.

    Returns:
        A dictionary of the same keys as the input dictionaries, with the average and
        standard deviation of the values. If the list has length 1, the original dictionary
        is returned instead.
    """
    if len(dictionaries) == 1:
        return dictionaries[0]

    # Convert the list of dictionaries to a dictionary of lists.
    lists = defaultdict(list)
    for d in dictionaries:
        for k, v in d.items():
            lists[k].append(v)

    # Compute the average and standard deviation of each list.
    stats = {}
    for k, v in lists.items():
        stats[f"{k}_avg"] = np.mean(v)
        stats[f"{k}_std"] = np.std(v)
    return stats


def evaluate_model_and_log(model, dataloader, filename, split, step=None, n=1):
    # Call evaluate_model multiple times. Each call returns a dictionary of metrics, and
    # we then compute their average and standard deviation.
    if n > 1:
        log(f"\nRunning {n} evaluations to compute average metrics")
    metrics = dict_stats([evaluate_model(model, dataloader) for _ in range(n)])
    log_and_save_metrics(metrics, split, step, filename)


@torch.no_grad()
def zero_init(module: nn.Module) -> nn.Module:
    """Sets to zero all the parameters of a module, and returns the module."""
    for p in module.parameters():
        nn.init.zeros_(p.data)
    return module


def maybe_unpack_batch(batch):
    if isinstance(batch, (tuple, list)) and len(batch) == 2:
        return batch
    else:
        return batch, None


def make_cifar(*, train, download):
    return CIFAR10(
        root="data",
        download=download,
        train=train,
        transform=transforms.Compose([transforms.ToTensor()]),
    )


def handle_results_path(res_path: str, default_root: str = "./results") -> Path:
    """Sets results path if it doesn't exist yet."""
    if res_path is None:
        results_path = Path(default_root) / get_date_str()
    else:
        results_path = Path(res_path)
    log(f"Results will be saved to '{results_path}'")
    return results_path


def unsqueeze_right(x, num_dims=1):
    """Unsqueezes the last `num_dims` dimensions of `x`."""
    return x.view(x.shape + (1,) * num_dims)


def init_config_from_args(cls, args):
    """Initializes a dataclass from a Namespace, ignoring unknown fields."""
    return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})


def check_config_matches_checkpoint(config, checkpoint_path):
    with open(checkpoint_path / "config.yaml", "r") as f:
        ckpt_config = yaml.safe_load(f)
    if dataclasses.asdict(config) != ckpt_config:
        raise ValueError(
            f"Config mismatch:\n"
            f" > Config: {dataclasses.asdict(config)}\n"
            f" > Checkpoint: {ckpt_config}"
        )


_accelerator: Optional[Accelerator] = None


def init_logger(accelerator: Accelerator):
    global _accelerator
    if _accelerator is not None:
        raise ValueError("Accelerator already set")
    _accelerator = accelerator


def log(message):
    global _accelerator
    if _accelerator is None:
        warnings.warn("Accelerator not set, using print instead.")
        print_fn = print
    else:
        print_fn = _accelerator.print
    print_fn(message)
