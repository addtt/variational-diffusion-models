import argparse
import math
from pathlib import Path

import torch
import yaml
from accelerate.utils import set_seed
from ema_pytorch import EMA
from torch.utils.data import Subset
from torchvision.utils import save_image

from utils import (
    DeviceAwareDataLoader,
    TrainConfig,
    evaluate_model_and_log,
    get_date_str,
    has_int_squareroot,
    log,
    make_cifar,
    print_model_summary,
    sample_batched,
)
from vdm import VDM
from vdm_unet import UNetVDM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--results-path", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-sample-steps", type=int, default=250)
    parser.add_argument("--clip-samples", type=bool, default=True)
    parser.add_argument("--n-samples-for-eval", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)

    # Load config from YAML.
    with open(Path(args.results_path) / "config.yaml", "r") as f:
        cfg = TrainConfig(**yaml.safe_load(f))

    model = UNetVDM(cfg)
    print_model_summary(model, batch_size=None, shape=(3, 32, 32))
    train_set = make_cifar(train=True, download=True)
    validation_set = make_cifar(train=False, download=False)
    diffusion = VDM(model, cfg, image_shape=train_set[0][0].shape)
    Evaluator(
        diffusion,
        train_set,
        validation_set,
        config=cfg,
        eval_batch_size=args.batch_size,
        results_path=Path(args.results_path),
        num_dataloader_workers=args.num_workers,
        device=args.device,
        n_sample_steps=args.n_sample_steps,
        clip_samples=args.clip_samples,
        n_samples_for_eval=args.n_samples_for_eval,
    ).eval()


class Evaluator:
    def __init__(
        self,
        diffusion_model,
        train_set,
        validation_set,
        config,
        *,
        eval_batch_size,
        device,
        results_path,
        num_samples=64,
        num_dataloader_workers=1,
        n_sample_steps=250,
        clip_samples=True,
        n_samples_for_eval=4,
    ):
        assert has_int_squareroot(num_samples), "num_samples must have an integer sqrt"
        self.num_samples = num_samples
        self.cfg = config
        self.n_sample_steps = n_sample_steps
        self.clip_samples = clip_samples
        self.device = device
        self.eval_batch_size = eval_batch_size
        self.n_samples_for_eval = n_samples_for_eval

        def make_dataloader(dataset, limit_size=None):
            # If limit_size is not None, only use a subset of the dataset
            if limit_size is not None:
                dataset = Subset(dataset, range(limit_size))
            return DeviceAwareDataLoader(
                dataset,
                eval_batch_size,
                device=device,
                shuffle=False,
                pin_memory=True,
                num_workers=num_dataloader_workers,
                drop_last=True,
            )

        self.validation_dataloader = make_dataloader(validation_set)
        self.train_eval_dataloader = make_dataloader(train_set, len(validation_set))
        self.diffusion_model = diffusion_model.eval().to(self.device)
        # No need to set EMA parameters since we only use it for eval from checkpoint.
        self.ema = EMA(self.diffusion_model).to(self.device)
        self.ema.ema_model.eval()
        self.path = results_path
        self.eval_path = self.path / f"eval_{get_date_str()}"
        self.eval_path.mkdir()
        self.checkpoint_file = self.path / f"model.pt"
        with open(self.eval_path / "eval_config.yaml", "w") as f:
            eval_conf = {
                "n_sample_steps": n_sample_steps,
                "clip_samples": clip_samples,
                "n_samples_for_eval": n_samples_for_eval,
            }
            yaml.dump(eval_conf, f)
        self.load_checkpoint()

    def load_checkpoint(self):
        data = torch.load(self.checkpoint_file, map_location=self.device)
        log(f"Loading checkpoint '{self.checkpoint_file}'")
        self.diffusion_model.load_state_dict(data["model"])
        self.ema.load_state_dict(data["ema"])

    @torch.no_grad()
    def eval(self):
        self.eval_model(self.diffusion_model, is_ema=False)
        self.eval_model(self.ema.ema_model, is_ema=True)

    def eval_model(self, model, *, is_ema):
        log(f"\n *** Evaluating {'EMA' if is_ema else 'online'} model\n")
        self.sample_images(model, is_ema=is_ema)
        for validation in [True, False]:
            evaluate_model_and_log(
                model,
                self.validation_dataloader
                if validation
                else self.train_eval_dataloader,
                self.eval_path / ("ema-metrics.jsonl" if is_ema else "metrics.jsonl"),
                "validation" if validation else "train",
                n=self.n_samples_for_eval,
            )

    def sample_images(self, model, *, is_ema):
        samples = sample_batched(
            model,
            self.num_samples,
            self.eval_batch_size,
            self.n_sample_steps,
            self.clip_samples,
        )
        path = self.eval_path / f"sample{'-ema' if is_ema else ''}.png"
        save_image(samples, str(path), nrow=int(math.sqrt(self.num_samples)))


if __name__ == "__main__":
    main()
