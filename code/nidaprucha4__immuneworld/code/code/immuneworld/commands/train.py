import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf

from immuneworld.dynamics.system import ImmuneWorld, ImmuneWorldSettings
from immuneworld.learning.objectives import CompositeObjective, ObjectiveWeights
from immuneworld.learning.schedule import warmup_cosine
from immuneworld.learning.trainer import Trainer
from immuneworld.operations.state import set_seed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="immuneworld-train")
    value.add_argument("--settings", type=Path, required=True)
    value.add_argument("--seed", type=int, default=42)
    value.add_argument("--data", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configuration = OmegaConf.load(arguments.settings)
    set_seed(arguments.seed)
    settings = ImmuneWorldSettings(
        genes=int(configuration.model.genes),
        width=int(configuration.model.width),
        layers=int(configuration.model.layers),
        heads=int(configuration.model.heads),
        expansion=int(configuration.model.expansion),
        dropout=float(configuration.model.dropout),
    )
    device = torch.device("cuda", torch.distributed.get_rank())
    model = ImmuneWorld(settings).to(device)
    weights = ObjectiveWeights(
        reconstruction=float(configuration.loss.reconstruction),
        perturbation=float(configuration.loss.perturbation),
        adversarial=float(configuration.loss.adversarial),
    )
    objective = CompositeObjective(weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(configuration.training.learning_rate))
    total = int(configuration.training.epochs) * 100000
    schedule = warmup_cosine(
        optimizer,
        int(configuration.training.warmup_steps),
        total,
        float(configuration.training.minimum_learning_rate) / float(configuration.training.learning_rate),
    )
    trainer = Trainer(
        model,
        objective,
        optimizer,
        device,
        int(configuration.training.gradient_accumulation),
        torch.float16,
    )
    logging.info("configured %d epochs with effective batch %d", int(configuration.training.epochs), int(configuration.training.effective_batch_size))
    logging.info("data root %s and output root %s", arguments.data, arguments.output)
    logging.info("trainer initialized at step %d with scheduler epoch %d", trainer.step, schedule.last_epoch)


if __name__ == "__main__":
    main()
