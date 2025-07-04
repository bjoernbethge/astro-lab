"""Simple training script for AstroLab models."""

import logging
import platform
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from astro_lab.config import get_combined_config, get_data_paths
from astro_lab.data import AstroLabDataModule, AstroLabInMemoryDataset, get_sampler
from astro_lab.models import AstroModel, create_cosmic_web_model, create_stellar_model
from astro_lab.training.trainer import AstroTrainer

logger = logging.getLogger(__name__)


def prepare_datamodule(
    survey: str,
    task: str,
    config: dict,
) -> Tuple[AstroLabDataModule, dict]:
    batch_size = int(config.get("batch_size", 32))
    k_neighbors = int(config.get("k_neighbors", 8))
    num_workers = int(config.get("num_workers", 8))  # Default to 8 workers
    sampling_strategy = config.get("sampling_strategy", "knn")
    sampler_kwargs = {"k": k_neighbors} if k_neighbors else {}

    try:
        dataset = AstroLabInMemoryDataset(
            survey_name=survey,
            sampling_strategy=sampling_strategy,
            sampler_kwargs=sampler_kwargs,
            task=task,  # Pass task explicitly
        )

        # Ensure dataset is processed
        processed_path = Path(dataset.processed_paths[0])
        if not processed_path.exists():
            logger.info(f"Processing dataset for {survey}...")
            dataset.process()

        sampler = get_sampler(sampling_strategy)
        datamodule = AstroLabDataModule(
            dataset=dataset,
            sampler=sampler,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        datamodule.setup()
        info = dataset.get_info()

        # Validate required info
        if "num_features" not in info:
            raise ValueError(f"Dataset info missing 'num_features': {info}")
        if "num_classes" not in info:
            raise ValueError(f"Dataset info missing 'num_classes': {info}")

        logger.info(f"Dataset info: {info}")
        return datamodule, info
    except Exception as e:
        logger.error(f"Failed to prepare datamodule for survey '{survey}': {e}")
        raise


def prepare_model(
    survey: str,
    task: str,
    model_type: str,
    info: dict,
    config: dict,
) -> Any:
    hidden_dim = int(config.get("hidden_dim", 128))
    num_layers = int(config.get("num_layers", 3))
    dropout = float(config.get("dropout", 0.1))
    learning_rate = float(config.get("learning_rate", 1e-3))
    # Check for HeteroData metadata
    metadata = None
    if (
        info.get("pyg_type") == "HeteroData"
        and "node_types" in info
        and "edge_types" in info
    ):
        metadata = (info["node_types"], info["edge_types"])
    if task == "node_classification":
        if survey == "gaia":
            return create_stellar_model(
                num_features=info["num_features"],
                num_classes=info["num_classes"],
                conv_type=model_type,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                task=task,  # Pass task explicitly
                metadata=metadata,
            )
        else:
            return AstroModel(
                num_features=info["num_features"],
                num_classes=info["num_classes"],
                conv_type=model_type,
                task=task,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                learning_rate=learning_rate,
                metadata=metadata,
            )
    elif task == "graph_classification":
        return create_cosmic_web_model(
            num_features=info["num_features"],
            num_classes=info["num_classes"],
            conv_type=model_type,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate,
            task=task,  # Pass task explicitly
            metadata=metadata,
        )
    else:
        raise ValueError(f"Unsupported task: {task}")


def train_model(
    survey: str,
    task: str = "node_classification",
    model_type: str = "gcn",
    run_name: str = "",
    config: Optional[dict] = None,
    **kwargs,
) -> Dict[str, Any]:
    # Defensive: convert Namespace to dict if needed
    if config is not None and hasattr(config, "__dict__"):
        config = vars(config)
    if config is None:
        config = get_combined_config(survey, task)
    model_type = str(model_type or config.get("conv_type", "gcn"))
    experiment_name = str(config.get("experiment_name", "astro_gnn"))
    devices = config.get("devices", "auto")
    accelerator = str(config.get("accelerator", "auto"))
    gradient_clip_val = float(config.get("gradient_clip_val", 1.0))

    # Enable Tensor Core optimization for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

    datamodule, info = prepare_datamodule(
        survey=survey,
        task=task,
        config=config,
    )

    # Debug: Check if dataset has valid data
    try:
        sample_batch = next(iter(datamodule.train_dataloader()))
        logger.info(f"Sample batch type: {type(sample_batch)}")
        logger.info(
            f"Sample batch keys: {list(sample_batch.keys()) if hasattr(sample_batch, 'keys') else 'No keys'}"
        )

        # Check if we're using HeteroData
        if hasattr(sample_batch, "node_types"):
            logger.info(
                f"HeteroData detected with node types: {sample_batch.node_types}"
            )
            logger.info(f"Edge types: {sample_batch.edge_types}")

            # Log info for each node type
            for node_type in sample_batch.node_types:
                if hasattr(sample_batch[node_type], "x"):
                    x = sample_batch[node_type].x
                    logger.info(
                        f"Node type '{node_type}' x shape: {x.shape if x is not None else 'None'}"
                    )
                    logger.info(
                        f"Node type '{node_type}' x device: {x.device if x is not None else 'None'}"
                    )
                    if x is not None:
                        logger.info(f"Node type '{node_type}' x numel: {x.numel()}")
        else:
            # Regular Data
            if hasattr(sample_batch, "x"):
                logger.info(
                    f"Sample batch.x shape: {sample_batch.x.shape if sample_batch.x is not None else 'None'}"
                )
                logger.info(
                    f"Sample batch.x device: {sample_batch.x.device if sample_batch.x is not None else 'None'}"
                )
                logger.info(
                    f"Sample batch.x dtype: {sample_batch.x.dtype if sample_batch.x is not None else 'None'}"
                )
                if sample_batch.x is not None:
                    logger.info(f"Sample batch.x numel: {sample_batch.x.numel()}")
            else:
                logger.warning("Sample batch has no 'x' attribute!")

            # Check edge_index
            if hasattr(sample_batch, "edge_index"):
                logger.info(
                    f"Sample batch.edge_index shape: {sample_batch.edge_index.shape if sample_batch.edge_index is not None else 'None'}"
                )
            else:
                logger.warning("Sample batch has no 'edge_index' attribute!")

    except Exception as e:
        logger.error(f"Error checking sample batch: {e}")
        raise

    model = prepare_model(
        survey=survey,
        task=task,
        model_type=model_type,
        info=info,
        config=config,
    )

    # Ensure model is on CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to CUDA")

    # Disable torch.compile on Windows or if disabled in config
    if config.get("compile_model", False) and platform.system() != "Windows":
        if hasattr(torch, "compile"):
            compile_mode = config.get("compile_mode", "default")
            try:
                model = torch.compile(model, mode=compile_mode, dynamic=True)
                logger.info(f"Model compiled with mode: {compile_mode}")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed: {e}. Continuing without compilation."
                )
        else:
            logger.warning("torch.compile not available in this PyTorch version.")
    else:
        logger.info("torch.compile disabled (Windows or config setting)")

    trainer = AstroTrainer(
        experiment_name=experiment_name,
        run_name=run_name or None,
        max_epochs=int(config.get("max_epochs", 100)),
        accelerator=accelerator,
        devices=devices,
        precision=str(config.get("precision", "32-true")),
        gradient_clip_val=gradient_clip_val,
        **kwargs,
    )
    # Get DataLoaders directly from datamodule
    train_dataloader = datamodule.train_dataloader()
    val_dataloader = datamodule.val_dataloader()
    test_dataloader = datamodule.test_dataloader()

    # Pass DataLoaders explicitly to bypass Lightning's validation
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    test_results = trainer.test(model, dataloaders=test_dataloader)
    return {
        "model": model,
        "trainer": trainer,
        "test_results": test_results,
        "info": info,
    }


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    # Set default tensor type to CUDA if available for better performance
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.set_float32_matmul_precision("medium")

    try:
        results = train_model(
            survey="gaia",
            task="node_classification",
        )
        print("Test results:", results["test_results"])
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    main()
