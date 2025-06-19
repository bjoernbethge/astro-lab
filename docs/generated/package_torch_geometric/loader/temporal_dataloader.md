# temporal_dataloader

Part of `torch_geometric.loader`
Module: `torch_geometric.loader.temporal_dataloader`

## Classes (2)

### `TemporalData`

A data object composed by a stream of events describing a temporal
graph.
The :class:`~torch_geometric.data.TemporalData` object can hold a list of
events (that can be understood as temporal edges in a graph) with
structured messages.
An event is composed by a source node, a destination node, a timestamp
and a message. Any *Continuous-Time Dynamic Graph* (CTDG) can be
represented with these four values.

In general, :class:`~torch_geometric.data.TemporalData` tries to mimic
the behavior of a regular :python:`Python` dictionary.
In addition, it provides useful functionality for analyzing graph
structures, and provides basic PyTorch tensor functionalities.

.. code-block:: python

    from torch import Tensor
    from torch_geometric.data import TemporalData

    events = TemporalData(
        src=Tensor([1,2,3,4]),
        dst=Tensor([2,3,4,5]),
        t=Tensor([1000,1010,1100,2000]),
        msg=Tensor([1,1,0,0])
    )

    # Add additional arguments to `events`:
    events.y = Tensor([1,1,0,0])

    # It is also possible to set additional arguments in the constructor
    events = TemporalData(
        ...,
        y=Tensor([1,1,0,0])
    )

    # Get the number of events:
    events.num_events
    >>> 4

    # Analyzing the graph structure:
    events.num_nodes
    >>> 5

    # PyTorch tensor functionality:
    events = events.pin_memory()
    events = events.to('cuda:0', non_blocking=True)

Args:
    src (torch.Tensor, optional): A list of source nodes for the events
        with shape :obj:`[num_events]`. (default: :obj:`None`)
    dst (torch.Tensor, optional): A list of destination nodes for the
        events with shape :obj:`[num_events]`. (default: :obj:`None`)
    t (torch.Tensor, optional): The timestamps for each event with shape
        :obj:`[num_events]`. (default: :obj:`None`)
    msg (torch.Tensor, optional): Messages feature matrix with shape
        :obj:`[num_events, num_msg_features]`. (default: :obj:`None`)
    **kwargs (optional): Additional attributes.

.. note::
    The shape of :obj:`src`, :obj:`dst`, :obj:`t` and the first dimension
    of :obj`msg` should be the same (:obj:`num_events`).

#### Methods

- **`index_select(self, idx: Any) -> 'TemporalData'`**

- **`stores_as(self, data: 'TemporalData')`**

- **`to_dict(self) -> Dict[str, Any]`**
  Returns a dictionary of stored key/value pairs.

### `TemporalDataLoader`

A data loader which merges succesive events of a
:class:`torch_geometric.data.TemporalData` to a mini-batch.

Args:
    data (TemporalData): The :obj:`~torch_geometric.data.TemporalData`
        from which to load the data.
    batch_size (int, optional): How many samples per batch to load.
        (default: :obj:`1`)
    neg_sampling_ratio (float, optional): The ratio of sampled negative
        destination nodes to the number of postive destination nodes.
        (default: :obj:`0.0`)
    **kwargs (optional): Additional arguments of
        :class:`torch.utils.data.DataLoader`.
