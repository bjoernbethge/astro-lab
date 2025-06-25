import argparse
import logging
import sys

from astro_lab.data.datamodule import AstroDataModule
from astro_lab.models.config import ModelConfig
from astro_lab.training.config import TrainingConfig
from astro_lab.training.trainer import AstroTrainer
from astro_lab.utils.config.loader import ConfigLoader

logger = logging.getLogger("astro_lab.cli.train")


def main():
    parser = argparse.ArgumentParser(description="AstroLab Training CLI")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--dataset", help="Dataset for quick training")
    parser.add_argument("--model", help="Model for quick training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max-samples", type=int, default=1000, help="Maximum number of samples"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--devices", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument(
        "--strategy", default="auto", help="Training strategy (auto/ddp/fsdp)"
    )
    parser.add_argument(
        "--precision",
        default="16-mixed",
        help="Training precision (32/16-mixed/bf16-mixed)",
    )
    parser.add_argument(
        "--accumulate", type=int, default=1, help="Gradient accumulation steps"
    )
    args = parser.parse_args()

    try:
        if args.config:
            config_loader = ConfigLoader(args.config)
            config_loader.load_config()
            training_dict = config_loader.get_training_config()
            model_dict = config_loader.get_model_config()
            model_config = ModelConfig(**model_dict)
            training_config = TrainingConfig(
                name=training_dict.get("name", "config_training"),
                model=model_config,
                **{
                    k: v
                    for k, v in training_dict.items()
                    if k != "name" and k != "model"
                },
            )
            survey = model_dict.get("name", "gaia")
            datamodule = AstroDataModule(
                survey=survey,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
            )
            datamodule.setup()
            if datamodule.num_classes:
                model_config.output_dim = datamodule.num_classes
                logger.info(f"Detected {datamodule.num_classes} classes from data")
            trainer = AstroTrainer(training_config=training_config)
            trainer.fit(datamodule=datamodule)
        elif args.dataset and args.model:
            datamodule = AstroDataModule(
                survey=args.dataset,
                batch_size=args.batch_size,
                max_samples=args.max_samples,
            )
            datamodule.setup()
            if not datamodule.num_classes:
                logger.error(
                    f"Could not detect number of classes from {args.dataset} dataset. Please check your data."
                )
                sys.exit(2)
            logger.info(
                f"Detected {datamodule.num_classes} classes from {args.dataset} data"
            )
            if args.model in [
                "gaia_classifier",
                "lsst_transient",
                "lightcurve_classifier",
            ]:
                model_config = ModelConfig(
                    name=args.model,
                    output_dim=datamodule.num_classes,
                    task="classification",
                )
            elif args.model in ["sdss_galaxy", "galaxy_modeler"]:
                model_config = ModelConfig(
                    name=args.model,
                    task="regression",
                )
            else:
                model_config = ModelConfig(
                    name=args.model,
                    output_dim=datamodule.num_classes,
                )
            training_config = TrainingConfig(
                name="quick_training",
                model=model_config,
                scheduler={"max_epochs": args.epochs},
                hardware={"devices": args.devices, "precision": args.precision},
            )
            trainer = AstroTrainer(training_config=training_config)
            logger.info(
                f"Starting training with model={args.model}, num_classes={datamodule.num_classes}"
            )
            trainer.fit(datamodule=datamodule)
        else:
            logger.error(
                "Error: Either --config or both --dataset and --model must be specified"
            )
            sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
