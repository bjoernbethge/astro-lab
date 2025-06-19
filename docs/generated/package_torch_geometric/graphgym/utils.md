# utils

Part of `torch_geometric.graphgym`
Module: `torch_geometric.graphgym.utils`

## Functions (12)

### `agg_batch(dir, metric_best='auto')`

Aggregate across results from multiple experiments via grid search.

Args:
    dir (str): Directory of the results, containing multiple experiments
    metric_best (str, optional): The metric for selecting the best
    validation performance. Options: auto, accuracy, auc.

### `agg_runs(dir, metric_best='auto')`

Aggregate over different random seeds of a single experiment.

Args:
    dir (str): Directory of the results, containing 1 experiment
    metric_best (str, optional): The metric for selecting the best
    validation performance. Options: auto, accuracy, auc.

### `auto_select_device()`

Auto select device for the current experiment.

### `dict_list_to_json(dict_list, fname)`

Dump a list of :python:`Python` dictionaries to a JSON file.

Args:
    dict_list (list of dict): List of :python:`Python` dictionaries.
    fname (str): the output file name.

### `dict_to_json(dict, fname)`

Dump a :python:`Python` dictionary to a JSON file.

Args:
    dict (dict): The :python:`Python` dictionary.
    fname (str): The output file name.

### `dict_to_tb(dict, writer, epoch)`

Add a dictionary of statistics to a Tensorboard writer.

Args:
    dict (dict): Statistics of experiments, the keys are attribute names,
    the values are the attribute values
    writer: Tensorboard writer object
    epoch (int): The current epoch

### `get_current_gpu_usage()`

Get the current GPU memory usage.

### `is_ckpt_epoch(cur_epoch)`

Determines if the model should be evaluated at the current epoch.

### `is_eval_epoch(cur_epoch)`

Determines if the model should be evaluated at the current epoch.

### `makedirs_rm_exist(dir)`

Make a directory, remove any existing data.

Args:
    dir (str): The directory to be created.

### `match_baseline_cfg(cfg_dict, cfg_dict_baseline, verbose=True)`

Match the computational budget of a given baseline model. The current
configuration dictionary will be modifed and returned.

Args:
    cfg_dict (dict): Current experiment's configuration
    cfg_dict_baseline (dict): Baseline configuration
    verbose (str, optional): If printing matched paramter conunts

### `params_count(model)`

Computes the number of parameters.

Args:
    model (nn.Module): PyTorch model

## Classes (1)

### `dummy_context`

Default context manager that does nothing.
