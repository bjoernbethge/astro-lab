# Comprehensive Cheat Sheet: PyTorch Lightning + MLflow + PyTorch Geometric + TensorDict + Marimo

## PyTorch Lightning

### Key Classes and Parameters

| Class/Method | Key Parameters | Description |
|--------------|----------------|-------------|
| **LightningModule** | | Base class for all models |
| `training_step()` | `batch`, `batch_idx` | Define training logic, return loss |
| `validation_step()` | `batch`, `batch_idx` | Validation logic, use `self.log()` |
| `configure_optimizers()` | None | Return optimizer(s) and scheduler(s) |
| **Trainer** | | Orchestrates training |
| | `accelerator` | `"gpu"`, `"cpu"`, `"tpu"`, `"hpu"` |
| | `devices` | `1`, `4`, `[0,1,2,3]`, `"auto"` |
| | `strategy` | `"ddp"`, `"fsdp"`, `"deepspeed"` |
| | `precision` | `"32-true"`, `"16-mixed"`, `"bf16-mixed"` |
| | `max_epochs` | Number of training epochs |
| | `callbacks` | List of callbacks |
| | `logger` | Experiment logger(s) |
| **Callbacks** | | |
| `ModelCheckpoint` | `monitor`, `save_top_k`, `mode` | Save best models |
| `EarlyStopping` | `monitor`, `patience`, `mode` | Stop on plateau |
| `LearningRateMonitor` | `logging_interval` | Track LR changes |

### Special Features
- **Automatic optimization**: No need to call `zero_grad()`, `backward()`, `step()`
- **Multi-GPU training**: Just set `devices=4` for DDP
- **Mixed precision**: Enable with `precision="16-mixed"`
- **Gradient accumulation**: `accumulate_grad_batches=4`
- **Built-in profiling**: `profiler="pytorch"`

## MLflow

### Key Components and Parameters

| Component | Key Functions | Parameters |
|-----------|---------------|------------|
| **Tracking** | | |
| `mlflow.start_run()` | Initialize run | `run_name`, `experiment_id` |
| `mlflow.log_params()` | Log parameters | Dictionary of params |
| `mlflow.log_metrics()` | Log metrics | Dictionary, `step` |
| `mlflow.log_artifact()` | Log files | `local_path`, `artifact_path` |
| **Projects** | | |
| MLproject file | Define entry points | `name`, `python_env`, `entry_points` |
| `mlflow.projects.run()` | Run project | `uri`, `parameters`, `backend` |
| **Models** | | |
| `mlflow.<flavor>.log_model()` | Log model | `artifact_path`, `signature`, `input_example` |
| `mlflow.<flavor>.load_model()` | Load model | `model_uri` |
| **Registry** | | |
| `mlflow.register_model()` | Register model | `model_uri`, `name` |
| `transition_model_version_stage()` | Change stage | `name`, `version`, `stage` |

### Special Features
- **Autologging**: `mlflow.pytorch.autolog()` - automatic parameter/metric logging
- **Model signatures**: Schema validation for inputs/outputs
- **Model flavors**: Support for sklearn, pytorch, tensorflow, transformers, etc.
- **Deployment**: Direct deployment to SageMaker, AzureML, Kubernetes

## PyTorch Geometric

### Key Classes and Parameters

| Class | Key Parameters | Use Case |
|-------|----------------|----------|
| **Data** | | Graph representation |
| | `x` | Node features `[num_nodes, num_features]` |
| | `edge_index` | Edges in COO format `[2, num_edges]` |
| | `edge_attr` | Edge features `[num_edges, num_edge_features]` |
| | `y` | Labels (node/graph level) |
| **DataLoader** | `batch_size`, `shuffle` | Batch multiple graphs |
| **NeighborLoader** | `num_neighbors`, `batch_size` | Sample neighbors for large graphs |
| **MessagePassing** | `aggr`, `flow` | Base class for GNN layers |
| **GNN Layers** | | |
| `GCNConv` | `in_channels`, `out_channels` | Graph Convolutional Network |
| `GATConv` | `heads`, `concat` | Graph Attention Network |
| `SAGEConv` | `aggr`, `normalize` | GraphSAGE |
| `GINConv` | `nn`, `eps` | Graph Isomorphism Network |

### Special Features
- **Automatic batching**: Diagonal block-diagonal adjacency matrices
- **Heterogeneous graphs**: `HeteroData` for multi-relational data
- **Graph pooling**: `TopKPooling`, `SAGPooling`, `global_mean_pool`
- **Explainability**: `GNNExplainer`, `PGExplainer`
- **Large-scale support**: `NeighborLoader`, `ClusterLoader`, `GraphSAINT`

## TensorDict

### Key Classes and Operations

| Class/Method | Parameters | Description |
|--------------|------------|-------------|
| **TensorDict** | | Dictionary-like tensor container |
| | `source` | Initial data dict |
| | `batch_size` | Leading dimensions |
| | `device` | Device placement |
| **Operations** | | |
| `select()` | Keys to keep | Extract subset |
| `exclude()` | Keys to remove | Remove keys |
| `apply()` | Function | Apply to all tensors |
| `map()` | Function, batch_size | Transform with new batch |
| **Memory** | | |
| `memmap_()` | `prefix` | Memory-mapped storage |
| `share_memory_()` | None | Multi-process sharing |
| `consolidate()` | `filename` | Fast serialization |
| **TensorDictModule** | `in_keys`, `out_keys` | Wrap nn.Module |

### Special Features
- **Lazy evaluation**: `LazyStackedTensorDict` for deferred computation
- **Zero-copy operations**: Efficient indexing and slicing
- **Nested structures**: Hierarchical data organization
- **PyTorch integration**: Works seamlessly with DataLoader
- **Memory efficiency**: Memory-mapped tensors for large datasets

## Marimo

### Key Components and Parameters

| Component | Function | Key Features |
|-----------|----------|--------------|
| **Reactive Cells** | `@app.cell` | Auto-execution on dependency change |
| **UI Elements** | | |
| `mo.ui.slider` | `start`, `stop`, `step` | Numeric input |
| `mo.ui.text` | `placeholder`, `label` | Text input |
| `mo.ui.dropdown` | `options`, `value` | Selection |
| `mo.ui.button` | `on_click`, `label` | Interactive button |
| `mo.ui.dataframe` | `data` | Interactive dataframe |
| **Composite UI** | | |
| `mo.ui.array` | List of elements | Group multiple inputs |
| `mo.ui.form` | Submit gate | Batch updates |
| **Deployment** | | |
| `marimo edit` | Development mode | Interactive editing |
| `marimo run` | App mode | Read-only web app |
| `python notebook.py` | Script mode | CLI execution |

### Special Features
- **Pure Python files**: Version control friendly `.py` format
- **Automatic dependency tracking**: No manual cell ordering
- **No hidden state**: Deleting cells removes variables
- **Built-in SQL support**: Native DuckDB integration
- **WASM export**: Browser-only execution

## Integration Patterns

### PyTorch Lightning + MLflow

```python
import mlflow
from pytorch_lightning.loggers import MLFlowLogger

# Setup
mlflow.pytorch.autolog()
mlf_logger = MLFlowLogger(
    experiment_name="my_experiment",
    tracking_uri="file:./mlruns"
)

# Training
trainer = pl.Trainer(
    logger=mlf_logger,
    max_epochs=100
)

# In LightningModule
def training_step(self, batch, batch_idx):
    loss = self.compute_loss(batch)
    self.log("train_loss", loss)  # Automatically logged to MLflow
    return loss
```

### PyTorch Lightning + PyTorch Geometric

```python
class GNNLightningModule(pl.LightningModule):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, training=self.training)
        return self.conv2(x, edge_index)
    
    def training_step(self, batch, batch_idx):
        # batch is a PyG Data object
        out = self(batch.x, batch.edge_index)
        loss = F.cross_entropy(out[batch.train_mask], 
                              batch.y[batch.train_mask])
        self.log("train_loss", loss)
        return loss
```

### TensorDict + PyTorch Lightning

```python
from tensordict.nn import TensorDictModule

class TensorDictLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = TensorDictModule(
            nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2)),
            in_keys=["features"],
            out_keys=["logits"]
        )
    
    def training_step(self, batch: TensorDict, batch_idx):
        # batch is a TensorDict
        output = self.model(batch)
        loss = F.cross_entropy(output["logits"], batch["labels"])
        self.log("train_loss", loss)
        return loss
```

### Marimo + MLflow Interactive Dashboard

```python
import marimo as mo
import mlflow
import pandas as pd

# Interactive experiment selector
experiments = mlflow.search_experiments()
exp_selector = mo.ui.dropdown(
    options=[e.name for e in experiments],
    label="Select Experiment"
)

# Reactive metrics visualization
@mo.reactive
def show_runs():
    if exp_selector.value:
        runs = mlflow.search_runs(experiment_names=[exp_selector.value])
        return mo.ui.plotly(
            px.scatter(runs, x="metrics.loss", y="metrics.accuracy", 
                      hover_data=["params.learning_rate"])
        )

mo.md(f"""
# MLflow Experiment Dashboard
{exp_selector}
{show_runs()}
""")
```

### PyTorch Geometric + TensorDict for Heterogeneous Graphs

```python
# Convert HeteroData to TensorDict
def hetero_to_tensordict(hetero_data):
    td = TensorDict({
        "nodes": {},
        "edges": {}
    })
    
    for node_type in hetero_data.node_types:
        td["nodes"][node_type] = hetero_data[node_type].x
    
    for edge_type in hetero_data.edge_types:
        src, rel, dst = edge_type
        td["edges"][f"{src}_{rel}_{dst}"] = hetero_data[edge_type].edge_index
    
    return td

# Use in training
hetero_td = hetero_to_tensordict(hetero_data)
hetero_td = hetero_td.to("cuda")  # Move entire structure to GPU
```

## Best Practices for Integration

### 1. Unified Configuration Management

```python
# config.yaml
training:
  max_epochs: 100
  batch_size: 32
  learning_rate: 0.001
  
model:
  hidden_dim: 64
  num_layers: 3
  
mlflow:
  experiment_name: "gnn_experiments"
  tracking_uri: "sqlite:///mlflow.db"
```

### 2. Custom Callbacks for Integration

```python
class IntegratedLoggingCallback(pl.Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Log TensorDict statistics
        if isinstance(batch, TensorDict):
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor):
                    mlflow.log_metric(f"batch_{key}_mean", 
                                    tensor.float().mean().item(), 
                                    step=trainer.global_step)
```

### 3. Memory-Efficient Data Pipeline

```python
# Use TensorDict with memory mapping for large datasets
dataset = TensorDict({
    "features": MemoryMappedTensor.empty((1000000, 128)),
    "labels": MemoryMappedTensor.empty((1000000,), dtype=torch.long)
}, batch_size=[1000000])

# PyG NeighborLoader for large graphs
loader = NeighborLoader(
    data,
    num_neighbors=[15, 10, 5],  # 3-hop sampling
    batch_size=1024,
    num_workers=4
)
```

### 4. Production Deployment Pattern

```python
# Train with Lightning + MLflow
trainer = pl.Trainer(logger=MLFlowLogger())
trainer.fit(model)

# Register model
mlflow.pytorch.log_model(model, "model", 
                        registered_model_name="GNN_Production")

# Deploy from registry
client = mlflow.tracking.MlflowClient()
model_version = client.get_latest_versions("GNN_Production", 
                                          stages=["Production"])[0]
model_uri = f"models:/GNN_Production/{model_version.version}"

# Create Marimo app for serving
@mo.app
def serve():
    model = mlflow.pytorch.load_model(model_uri)
    input_data = mo.ui.file(label="Upload graph data")
    
    if input_data.value:
        predictions = model(process_input(input_data.value))
        mo.md(f"Predictions: {predictions}")
```

## Performance Tips

1. **Use TensorDict consolidation** for faster serialization
2. **Enable PyTorch compile** mode: `model = torch.compile(model)`
3. **Use sparse operations** in PyG for large graphs
4. **Leverage Marimo's caching**: `@mo.cache` for expensive operations
5. **MLflow autolog with limits**: `mlflow.pytorch.autolog(log_every_n_step=100)`

This cheat sheet provides a comprehensive reference for integrating these powerful frameworks, emphasizing practical patterns and features that accelerate ML development while avoiding common pitfalls.