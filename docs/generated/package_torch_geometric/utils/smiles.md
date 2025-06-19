# smiles

Part of `torch_geometric.utils`
Module: `torch_geometric.utils.smiles`

## Functions (4)

### `from_rdmol(mol: Any) -> 'torch_geometric.data.Data'`

Converts a :class:`rdkit.Chem.Mol` instance to a
:class:`torch_geometric.data.Data` instance.

Args:
    mol (rdkit.Chem.Mol): The :class:`rdkit` molecule.

### `from_smiles(smiles: str, with_hydrogen: bool = False, kekulize: bool = False) -> 'torch_geometric.data.Data'`

Converts a SMILES string to a :class:`torch_geometric.data.Data`
instance.

Args:
    smiles (str): The SMILES string.
    with_hydrogen (bool, optional): If set to :obj:`True`, will store
        hydrogens in the molecule graph. (default: :obj:`False`)
    kekulize (bool, optional): If set to :obj:`True`, converts aromatic
        bonds to single/double bonds. (default: :obj:`False`)

### `to_rdmol(data: 'torch_geometric.data.Data', kekulize: bool = False) -> Any`

Converts a :class:`torch_geometric.data.Data` instance to a
:class:`rdkit.Chem.Mol` instance.

Args:
    data (torch_geometric.data.Data): The molecular graph data.
    kekulize (bool, optional): If set to :obj:`True`, converts aromatic
        bonds to single/double bonds. (default: :obj:`False`)

### `to_smiles(data: 'torch_geometric.data.Data', kekulize: bool = False) -> str`

Converts a :class:`torch_geometric.data.Data` instance to a SMILES
string.

Args:
    data (torch_geometric.data.Data): The molecular graph.
    kekulize (bool, optional): If set to :obj:`True`, converts aromatic
        bonds to single/double bonds. (default: :obj:`False`)

## Classes (1)

### `Any`

Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.
