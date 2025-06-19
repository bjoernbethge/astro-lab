# register

Part of `torch_geometric.graphgym`
Module: `torch_geometric.graphgym.register`

## Functions (17)

### `register_act(key: str, module: Any = None)`

Registers an activation function in GraphGym.

### `register_base(mapping: Dict[str, Any], key: str, module: Any = None) -> Optional[Callable]`

Base function for registering a module in GraphGym.

Args:
    mapping (dict): :python:`Python` dictionary to register the module.
        hosting all the registered modules
    key (str): The name of the module.
    module (any, optional): The module. If set to :obj:`None`, will return
        a decorator to register a module.

### `register_config(key: str, module: Any = None)`

Registers a configuration group in GraphGym.

### `register_dataset(key: str, module: Any = None)`

Registers a dataset in GraphGym.

### `register_edge_encoder(key: str, module: Any = None)`

Registers an edge feature encoder in GraphGym.

### `register_head(key: str, module: Any = None)`

Registers a GNN prediction head in GraphGym.

### `register_layer(key: str, module: Any = None)`

Registers a GNN layer in GraphGym.

### `register_loader(key: str, module: Any = None)`

Registers a data loader in GraphGym.

### `register_loss(key: str, module: Any = None)`

Registers a loss function in GraphGym.

### `register_metric(key: str, module: Any = None)`

Register a metric function in GraphGym.

### `register_network(key: str, module: Any = None)`

Registers a GNN model in GraphGym.

### `register_node_encoder(key: str, module: Any = None)`

Registers a node feature encoder in GraphGym.

### `register_optimizer(key: str, module: Any = None)`

Registers an optimizer in GraphGym.

### `register_pooling(key: str, module: Any = None)`

Registers a GNN global pooling/readout layer in GraphGym.

### `register_scheduler(key: str, module: Any = None)`

Registers a learning rate scheduler in GraphGym.

### `register_stage(key: str, module: Any = None)`

Registers a customized GNN stage in GraphGym.

### `register_train(key: str, module: Any = None)`

Registers a training function in GraphGym.

## Classes (1)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.
