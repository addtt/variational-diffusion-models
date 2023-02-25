import argparse
import dataclasses
import math
from argparse import BooleanOptionalAction

import torch
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed
from ema_pytorch import EMA
from torch.utils.data import Subset
from torchvision.utils import save_image
from tqdm.auto import tqdm

from utils import (
    DeviceAwareDataLoader,
    TrainConfig,
    check_config_matches_checkpoint,
    cycle,
    evaluate_model_and_log,
    get_date_str,
    handle_results_path,
    has_int_squareroot,
    init_config_from_args,
    init_logger,
    log,
    make_cifar,
    print_model_summary,
    sample_batched,
)
from vdm import VDM
from vdm_unet import UNetVDM


def main():
    parser = argparse.ArgumentParser()

    # Architecture
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--n-blocks", type=int, default=32)
    parser.add_argument("--n-attention-heads", type=int, default=1)
    parser.add_argument("--dropout-prob", type=float, default=0.1)
    parser.add_argument("--norm-groups", type=int, default=32)
    parser.add_argument("--input-channels", type=int, default=3)
    parser.add_argument("--use-fourier-features", type=bool, default=True)
    parser.add_argument("--attention-everywhere", type=bool, default=False)

    # Training
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-schedule", type=str, default="fixed_linear")
    parser.add_argument("--gamma-min", type=float, default=-13.3)
    parser.add_argument("--gamma-max", type=float, default=5.0)
    parser.add_argument("--antithetic-time-sampling", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", action=BooleanOptionalAction, default=True)

    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--results-path", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed(args.seed)
    accelerator = Accelerator(split_batches=True)
    init_logger(accelerator)
    cfg = init_config_from_args(TrainConfig, args)

    model = UNetVDM(cfg)
    print_model_summary(model, batch_size=cfg.batch_size, shape=(3, 32, 32))
    with accelerator.local_main_process_first():
        train_set = make_cifar(train=True, download=accelerator.is_local_main_process)
    validation_set = make_cifar(train=False, download=False)
    diffusion = VDM(model, cfg, image_shape=train_set[0][0].shape)
    Trainer(
        diffusion,
        train_set,
        validation_set,
        accelerator,
        make_opt=lambda params: torch.optim.AdamW(
            params, cfg.lr, betas=(0.9, 0.99), weight_decay=cfg.weight_decay, eps=1e-8
        ),
        config=cfg,
        save_and_eval_every=args.eval_every,
        results_path=handle_results_path(args.results_path),
        resume=args.resume,
        num_dataloader_workers=args.num_workers,
    ).train()


class Trainer:
    def __init__(
        self,
        diffusion_model,
        train_set,
        validation_set,
        accelerator,
        make_opt,
        config,
        *,
        train_num_steps=10_000_000,
        ema_decay=0.9999,
        ema_update_every=1,
        ema_power=3 / 4,  # 0.999 at 10k, 0.9997 at 50k, 0.9999 at 200k
        save_and_eval_every=1000,
        num_samples=64,
        results_path=None,
        resume=False,
        num_dataloader_workers=1,
        n_sample_steps=250,
        clip_samples=True,
    ):
        super().__init__()
        assert has_int_squareroot(num_samples), "num_samples must have an integer sqrt"
        self.num_samples = num_samples
        self.save_and_eval_every = save_and_eval_every
        self.cfg = config
        self.train_num_steps = train_num_steps
        self.n_sample_steps = n_sample_steps
        self.clip_samples = clip_samples
        self.accelerator = accelerator
        self.step = 0

        def make_dataloader(dataset, limit_size=None, *, train=False):
            if limit_size is not None:
                dataset = Subset(dataset, range(limit_size))
            dataloader = DeviceAwareDataLoader(
                dataset,
                config.batch_size,
                shuffle=train,
                pin_memory=True,
                num_workers=num_dataloader_workers,
                drop_last=True,
                device=accelerator.device if not train else None,  # None -> standard DL
            )
            if train:
                dataloader = accelerator.prepare(dataloader)
            return dataloader

        self.train_dataloader = cycle(make_dataloader(train_set, train=True))
        self.validation_dataloader = make_dataloader(validation_set)
        self.train_eval_dataloader = make_dataloader(train_set, len(validation_set))

        self.path = results_path
        self.checkpoint_file = self.path / f"model.pt"
        if accelerator.is_main_process:
            self.ema = EMA(
                diffusion_model.to(accelerator.device),
                beta=ema_decay,
                update_every=ema_update_every,
                power=ema_power,
            )
            self.ema.ema_model.eval()
            self.path.mkdir(exist_ok=True, parents=True)
        self.diffusion_model = accelerator.prepare(diffusion_model)
        self.opt = accelerator.prepare(make_opt(self.diffusion_model.parameters()))
        if resume:
            self.load_checkpoint()
        else:
            if len(list(self.path.glob("*.pt"))) > 0:
                raise ValueError(f"'{self.path}' contains checkpoints but resume=False")
            if accelerator.is_main_process:
                with open(self.path / "config.yaml", "w") as f:
                    yaml.dump(dataclasses.asdict(config), f)

    def save_checkpoint(self):
        tmp_file = self.checkpoint_file.with_suffix(f".tmp.{get_date_str()}.pt")
        if self.checkpoint_file.exists():
            self.checkpoint_file.rename(tmp_file)  # Rename old checkpoint to temp file
        checkpoint = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.diffusion_model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_file)
        tmp_file.unlink(missing_ok=True)  # Delete temp file

    def load_checkpoint(self):
        check_config_matches_checkpoint(self.cfg, self.path)
        data = torch.load(self.checkpoint_file, map_location=self.accelerator.device)
        self.step = data["step"]
        log(f"Resuming from checkpoint '{self.checkpoint_file}' (step {self.step})")
        model = self.accelerator.unwrap_model(self.diffusion_model)
        model.load_state_dict(data["model"])
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

    def train(self):
        with tqdm(
            initial=self.step,
            total=self.train_num_steps,
            disable=not self.accelerator.is_main_process,
        ) as pbar:
            while self.step < self.train_num_steps:
                data = next(self.train_dataloader)
                self.opt.zero_grad()
                loss, _ = self.diffusion_model(data)
                self.accelerator.backward(loss)
                if self.cfg.clip_grad_norm:
                    self.accelerator.clip_grad_norm_(
                        self.diffusion_model.parameters(), 1.0
                    )
                self.opt.step()
                pbar.set_description(f"loss: {loss.item():.4f}")
                self.step += 1
                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.ema.update()
                    if self.step % self.save_and_eval_every == 0:
                        self.eval()
                pbar.update()

    @torch.no_grad()
    def eval(self):
        self.save_checkpoint()
        self.sample_images(self.ema.ema_model, is_ema=True)
        self.sample_images(self.diffusion_model, is_ema=False)
        self.evaluate_ema_model_and_log(validation=True)
        self.evaluate_ema_model_and_log(validation=False)

    def evaluate_ema_model_and_log(self, *, validation):
        evaluate_model_and_log(
            self.ema.ema_model,
            self.validation_dataloader if validation else self.train_eval_dataloader,
            self.path / "metrics_log.jsonl",
            "validation" if validation else "train",
            self.step,
        )

    def sample_images(self, model, *, is_ema):
        train_state = model.training
        model.eval()
        samples = sample_batched(
            self.accelerator.unwrap_model(model),
            self.num_samples,
            self.cfg.batch_size,
            self.n_sample_steps,
            self.clip_samples,
        )
        path = self.path / f"sample-{'ema-' if is_ema else ''}{self.step}.png"
        save_image(samples, str(path), nrow=int(math.sqrt(self.num_samples)))
        model.train(train_state)


if __name__ == "__main__":
    main()
