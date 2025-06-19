# Training_Schemas Module

Auto-generated documentation for `schemas.training_schemas`

## LightningConfigSchema

Configuration schema for PyTorch Lightning.

### Parameters

**`accelerator`** *(string)* = `auto`
  Accelerator type (auto, cpu, gpu, tpu)

**`devices`** *(Union[integer, string, array])* = `auto`
  Number or list of devices to use

**`precision`** *(Union[integer, string])* = `32`
  Training precision (16, 32, 64, bf16)

**`strategy`** *(Optional[string])* = `None`
  Training strategy (ddp, fsdp, etc.)

**`max_epochs`** *(integer)* = `100`
  Maximum number of epochs
  *≥1, ≤10000*

**`gradient_clip_val`** *(Optional[number])* = `None`
  Gradient clipping value

**`accumulate_grad_batches`** *(integer)* = `1`
  Number of batches to accumulate gradients
  *≥1*

### Usage

```python
from docs.auto.schemas.data_schemas import LightningConfigSchema

config = LightningConfigSchema(

    # Optional parameters:
    # accelerator="example"
    # devices=None
    # precision=None
    # strategy=None
    # max_epochs=1
    # gradient_clip_val=None
    # accumulate_grad_batches=1
)
```

## MLflowConfigSchema

Configuration schema for MLflow logging.

### Parameters

**`experiment_name`** *(string)*
  MLflow experiment name

**`run_name`** *(Optional[string])* = `None`
  MLflow run name

**`tracking_uri`** *(Optional[string])* = `None`
  MLflow tracking server URI

**`log_model`** *(boolean)* = `True`
  Whether to log the model

**`log_artifacts`** *(boolean)* = `True`
  Whether to log artifacts

**`tags`** *(Optional[object])* = `None`
  Tags for the run

### Usage

```python
from docs.auto.schemas.data_schemas import MLflowConfigSchema

config = MLflowConfigSchema(
    experiment_name="example"

    # Optional parameters:
    # run_name=None
    # tracking_uri=None
    # log_model=True
    # log_artifacts=True
    # tags=None
)
```

## OptimizerConfigSchema

Configuration schema for optimizers.

### Parameters

**`optimizer_type`** *(string)* = `adam`
  Optimizer type (adam, sgd, adamw, etc.)

**`learning_rate`** *(number)* = `0.001`
  Learning rate
  *≤1.0, >0.0*

**`weight_decay`** *(number)* = `0.0001`
  Weight decay
  *≥0.0, ≤1.0*

**`momentum`** *(Optional[number])* = `None`
  Momentum (for SGD)

**`betas`** *(Optional[array])* = `[0.9, 0.999]`
  Beta parameters (for Adam)

### Usage

```python
from docs.auto.schemas.data_schemas import OptimizerConfigSchema

config = OptimizerConfigSchema(

    # Optional parameters:
    # optimizer_type="example"
    # learning_rate=1.0
    # weight_decay=0.0
    # momentum=None
    # betas=None
)
```

## OptunaConfigSchema

Configuration schema for Optuna hyperparameter optimization.

### Parameters

**`n_trials`** *(integer)* = `100`
  Number of optimization trials
  *≥1, ≤10000*

**`timeout`** *(Optional[integer])* = `None`
  Timeout in seconds

**`study_name`** *(Optional[string])* = `None`
  Name of the study

**`storage`** *(Optional[string])* = `None`
  Storage backend URL

**`sampler`** *(string)* = `tpe`
  Sampling algorithm (tpe, random, cmaes)

**`pruner`** *(string)* = `median`
  Pruning algorithm (median, hyperband, none)

### Usage

```python
from docs.auto.schemas.data_schemas import OptunaConfigSchema

config = OptunaConfigSchema(

    # Optional parameters:
    # n_trials=1
    # timeout=None
    # study_name=None
    # storage=None
    # sampler="example"
    # pruner="example"
)
```

## SchedulerConfigSchema

Configuration schema for learning rate schedulers.

### Parameters

**`scheduler_type`** *(string)* = `plateau`
  Scheduler type (plateau, cosine, step, etc.)

**`factor`** *(number)* = `0.5`
  Factor to reduce learning rate
  *>0.0, <1.0*

**`patience`** *(integer)* = `5`
  Patience for plateau scheduler
  *≥1, ≤100*

**`min_lr`** *(number)* = `1e-06`
  Minimum learning rate
  *>0.0*

### Usage

```python
from docs.auto.schemas.data_schemas import SchedulerConfigSchema

config = SchedulerConfigSchema(

    # Optional parameters:
    # scheduler_type="example"
    # factor=1.0
    # patience=1
    # min_lr=1.0
)
```

## TrainingConfigSchema

Configuration schema for training parameters.

### Parameters

**`max_epochs`** *(integer)* = `100`
  Maximum number of training epochs
  *≥1, ≤10000*

**`batch_size`** *(integer)* = `32`
  Training batch size
  *≥1, ≤10000*

**`learning_rate`** *(number)* = `0.001`
  Initial learning rate
  *≤1.0, >0.0*

**`weight_decay`** *(number)* = `0.0001`
  Weight decay for regularization
  *≥0.0, ≤1.0*

**`patience`** *(integer)* = `10`
  Early stopping patience
  *≥1, ≤100*

**`min_delta`** *(number)* = `0.0001`
  Minimum change for early stopping
  *≥0.0*

### Usage

```python
from docs.auto.schemas.data_schemas import TrainingConfigSchema

config = TrainingConfigSchema(

    # Optional parameters:
    # max_epochs=1
    # batch_size=1
    # learning_rate=1.0
    # weight_decay=0.0
    # patience=1
    # min_delta=0.0
)
```
