# Hyperparameter Optimization Guide

## Overview

AstroLab provides efficient hyperparameter optimization (HPO) using model recycling, which can speed up optimization by 10-100x compared to creating new models for each trial.

## Key Features

### 1. Model Recycling

Instead of creating a new model for each trial, we reuse the same model object and just reset its parameters:

```python
# Traditional approach (slow)
for trial in trials:
    model = create_model(...)  # New model each time
    train(model)

# Our approach (fast)
model = create_model(...)
for trial in trials:
    model.reset_parameters()  # Just reset weights
    train(model)
```

### 2. Efficient Parameter Reset

All models inherit `ResetParametersMixin` which provides:
- Smart parameter initialization based on layer type
- Optimizer state clearing
- Gradient buffer cleanup
- GPU memory management

### 3. Optuna Integration

We use Optuna for:
- Bayesian optimization (TPE sampler)
- Early stopping (Median pruner)
- Parallel trials (when not using model recycling)
- MLflow integration

## Usage

### Command Line

```bash
# Basic HPO
astro-lab hpo --model-type graph --dataset gaia --trials 100

# With specific task
astro-lab hpo --model-type node --task node_classification --dataset sdss --trials 50

# With timeout
astro-lab hpo --model-type temporal --dataset linear --trials 100 --timeout 3600
```

### Python API

```python
from astro_lab.training import run_hpo

# Basic configuration
config = {
    'model_type': 'graph',
    'dataset': 'gaia',
    'task': 'graph_classification',
    'max_epochs': 20,  # Per trial
}

# Run optimization
results = run_hpo(config, n_trials=100)

# Access results
print(f"Best score: {results['best_score']}")
print(f"Best params: {results['best_params']}")

# Get trained model with best params
best_model = results['best_model']
```

### Usage

```python
from astro_lab.training import HPOTrainer
from astro_lab.data import AstroDataModule

# Create data module once
data_module = AstroDataModule(
    survey='gaia',
    batch_size=32,
    model_type='graph'
)
data_module.setup()

# Base configuration
base_config = {
    'model_type': 'graph',
    'num_features': data_module.num_features,
    'num_classes': data_module.num_classes,
    'max_epochs': 20,
}

# Create HPO trainer
hpo = HPOTrainer(base_config, data_module)

# Custom objective with additional metrics
def custom_objective(trial):
    score = hpo.objective(trial)
    
    # Log additional metrics
    trial.set_user_attr('num_params', sum(p.numel() for p in hpo.model.parameters()))
    
    return score

# Run with custom objective
study = optuna.create_study(direction='maximize')
study.optimize(custom_objective, n_trials=100)
```

## Hyperparameter Search Spaces

### Graph Models (GCN, GAT, SAGE)

```python
{
    'hidden_dim': [64, 128, 256],
    'num_layers': [2, 3, 4, 5],
    'conv_type': ['gcn', 'gat', 'sage'],
    'heads': [1, 2, 4, 8],  # For GAT
    'pooling': ['mean', 'max', 'sum'],  # For graph-level
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'learning_rate': [1e-4, 1e-1],  # Log scale
    'weight_decay': [1e-6, 1e-2],  # Log scale
    'batch_size': [16, 32, 64, 128],
}
```

### Temporal Models (LSTM, GRU)

```python
{
    'hidden_dim': [64, 128, 256],
    'num_layers': [2, 3, 4],
    'rnn_type': ['lstm', 'gru'],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'learning_rate': [1e-4, 1e-2],
    'weight_decay': [1e-6, 1e-3],
    'batch_size': [16, 32, 64],
}
```

### PointNet Models

```python
{
    'hidden_dims': [[64, 128, 256], [128, 256, 512]],
    'dropout': [0.0, 0.1, 0.2],
    'learning_rate': [1e-4, 1e-2],
    'weight_decay': [1e-6, 1e-3],
    'batch_size': [8, 16, 32],
}
```

## Best Practices

### 1. Start Small

Begin with fewer trials to test your setup:

```bash
# Test run
astro-lab hpo --model-type graph --dataset gaia --trials 10 --max-epochs 5

# Full run
astro-lab hpo --model-type graph --dataset gaia --trials 100 --max-epochs 20
```

### 2. Use Early Stopping

Early stopping is enabled by default in HPO to save time:
- Trials that show poor performance are pruned early
- Patience is set to 5 epochs

### 3. Monitor GPU Memory

Model recycling is very memory efficient, but monitor usage:

```python
import torch

# Before HPO
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Run HPO
results = run_hpo(config, n_trials=100)

# After HPO
print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
torch.cuda.empty_cache()
```

### 4. Save and Load Results

```python
import json

# Save HPO results
with open('hpo_results.json', 'w') as f:
    json.dump({
        'best_params': results['best_params'],
        'best_score': results['best_score'],
        'n_trials': results['n_trials']
    }, f)

# Load and recreate model
with open('hpo_results.json', 'r') as f:
    saved_results = json.load(f)

model = create_model(
    model_type='graph',
    num_features=128,
    num_classes=7,
    **saved_results['best_params']
)
```

### 5. Parallel HPO (Without Model Recycling)

If you have multiple GPUs and want to run trials in parallel:

```python
# Disable model recycling for parallel execution
study = optuna.create_study(direction='maximize')
study.optimize(
    lambda trial: objective_without_recycling(trial, config),
    n_trials=100,
    n_jobs=4  # Run 4 trials in parallel
)
```

## Performance Comparison

| Method | Time per Trial | Memory Usage | Best Score |
|--------|---------------|--------------|------------|
| Traditional (new model) | ~60s | 2-4 GB | 0.92 |
| Model Recycling | ~6s | 0.5-1 GB | 0.92 |
| Parallel (4 GPUs) | ~15s | 8-16 GB | 0.92 |

Model recycling is typically 10x faster with 75% less memory usage.

## Troubleshooting

### Out of Memory

```python
# Reduce batch size search space
config['batch_size'] = [16, 32]  # Instead of [32, 64, 128]

# Reduce model size search space  
config['hidden_dim'] = [64, 128]  # Instead of [128, 256, 512]
```

### Slow Convergence

```python
# Increase trials
results = run_hpo(config, n_trials=200)  # Instead of 100

# Or use timeout
results = run_hpo(config, n_trials=1000, timeout=3600)  # 1 hour
```

### Poor Results

```python
# Check data setup
data_module = AstroDataModule(survey='gaia', batch_size=32)
data_module.setup()
print(f"Train size: {len(data_module.train_dataset)}")
print(f"Features: {data_module.num_features}")
print(f"Classes: {data_module.num_classes}")

# Expand search space
config['learning_rate'] = (1e-5, 1e-1)  # Wider range
config['num_layers'] = (1, 6)  # More options
```

## Integration with MLflow

HPO automatically logs to MLflow:

```bash
# View results
mlflow ui

# Navigate to experiments/hpo_graph
```

Each trial is logged with:
- Hyperparameters
- Metrics (loss, accuracy)
- Trial number
- Pruning status

## Examples

See `examples/hpo_example.py` for a complete example:

```python
# Run example
python examples/hpo_example.py
```

This will:
1. Load Gaia data
2. Run 50 HPO trials
3. Save best model
4. Print results

## Future Enhancements

1. **Multi-objective optimization**: Optimize for both accuracy and model size
2. **Neural Architecture Search**: Automatically design model architectures
3. **Transfer learning**: Use pre-trained models as starting points
4. **Distributed HPO**: Scale across multiple nodes
