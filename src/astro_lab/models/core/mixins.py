"""
Model Mixins for Enhanced Functionality
======================================

Reusable mixins that add specific capabilities to models.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch
from tensordict import TensorDict


class VisualizationMixin:
    """
    Mixin for model visualization capabilities.
    
    Provides methods for creating various visualizations of model
    predictions, features, and attention patterns.
    """
    
    def create_visualization(
        self,
        data: Dict[str, Any],
        viz_type: str = "3d_scatter",
        backend: str = "plotly",
        **kwargs,
    ) -> Any:
        """
        Create visualization based on the specified backend.
        
        Args:
            data: Dictionary with visualization data
            viz_type: Type of visualization
            backend: Visualization backend ("plotly", "matplotlib", "cosmograph")
            **kwargs: Additional arguments for the visualization
            
        Returns:
            Visualization object (specific to backend)
        """
        
        if backend == "plotly":
            return self._create_plotly_viz(data, viz_type, **kwargs)
        elif backend == "matplotlib":
            return self._create_matplotlib_viz(data, viz_type, **kwargs)
        elif backend == "cosmograph":
            return self._create_cosmograph_viz(data, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")
            
    def _create_plotly_viz(
        self,
        data: Dict[str, Any],
        viz_type: str,
        **kwargs,
    ) -> Any:
        """Create Plotly visualization."""
        import plotly.graph_objects as go
        
        if viz_type == "3d_scatter":
            positions = data["positions"]
            colors = data.get("predictions", np.zeros(len(positions)))
            
            fig = go.Figure(data=[go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color=colors,
                    colorscale=kwargs.get("colorscale", "Viridis"),
                    showscale=True,
                ),
                text=data.get("labels", None),
                hovertemplate='<b>%{text}</b><br>' +
                             'X: %{x:.2f}<br>' +
                             'Y: %{y:.2f}<br>' +
                             'Z: %{z:.2f}<br>' +
                             '<extra></extra>',
            )])
            
            fig.update_layout(
                title=kwargs.get("title", "3D Point Cloud"),
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                ),
                width=kwargs.get("width", 800),
                height=kwargs.get("height", 600),
            )
            
            return fig
            
        elif viz_type == "attention_heatmap":
            attention = data["attention"]
            
            fig = go.Figure(data=go.Heatmap(
                z=attention,
                colorscale=kwargs.get("colorscale", "Hot"),
                showscale=True,
            ))
            
            fig.update_layout(
                title=kwargs.get("title", "Attention Weights"),
                xaxis_title="Target",
                yaxis_title="Source",
            )
            
            return fig
            
    def _create_matplotlib_viz(
        self,
        data: Dict[str, Any],
        viz_type: str,
        **kwargs,
    ) -> Any:
        """Create Matplotlib visualization."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        if viz_type == "3d_scatter":
            fig = plt.figure(figsize=kwargs.get("figsize", (10, 8)))
            ax = fig.add_subplot(111, projection='3d')
            
            positions = data["positions"]
            colors = data.get("predictions", np.zeros(len(positions)))
            
            scatter = ax.scatter(
                positions[:, 0],
                positions[:, 1],
                positions[:, 2],
                c=colors,
                cmap=kwargs.get("cmap", "viridis"),
                s=kwargs.get("size", 10),
            )
            
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(kwargs.get("title", "3D Point Cloud"))
            
            plt.colorbar(scatter)
            return fig
            
    def _create_cosmograph_viz(
        self,
        data: Dict[str, Any],
        **kwargs,
    ) -> Any:
        """Create Cosmograph visualization for large-scale data."""
        # This would integrate with the Cosmograph widget
        # Placeholder for now
        return {
            "type": "cosmograph",
            "data": data,
            "config": kwargs,
        }
        
    def plot_feature_importance(
        self,
        batch: Union[Data, Batch, TensorDict],
        method: str = "gradient",
        **kwargs,
    ) -> Any:
        """
        Plot feature importance for the given batch.
        
        Args:
            batch: Input data
            method: Method for computing importance ("gradient", "attention", "permutation")
            **kwargs: Additional plotting arguments
            
        Returns:
            Visualization object
        """
        
        if method == "gradient":
            importance = self._compute_gradient_importance(batch)
        elif method == "attention":
            importance = self._compute_attention_importance(batch)
        elif method == "permutation":
            importance = self._compute_permutation_importance(batch)
        else:
            raise ValueError(f"Unknown importance method: {method}")
            
        # Create bar plot
        import plotly.graph_objects as go
        
        feature_names = kwargs.get("feature_names", [f"Feature {i}" for i in range(len(importance))])
        
        fig = go.Figure(data=[go.Bar(
            x=feature_names,
            y=importance,
            marker_color=kwargs.get("color", "lightblue"),
        )])
        
        fig.update_layout(
            title=kwargs.get("title", f"Feature Importance ({method})"),
            xaxis_title="Features",
            yaxis_title="Importance",
            width=kwargs.get("width", 800),
            height=kwargs.get("height", 500),
        )
        
        return fig
        
    def _compute_gradient_importance(self, batch: Union[Data, Batch, TensorDict]) -> np.ndarray:
        """Compute gradient-based feature importance."""
        # Placeholder - would compute actual gradients
        num_features = batch.x.shape[-1] if hasattr(batch, 'x') else 10
        return np.random.rand(num_features)
        
    def _compute_attention_importance(self, batch: Union[Data, Batch, TensorDict]) -> np.ndarray:
        """Compute attention-based feature importance."""
        # Placeholder - would aggregate attention weights
        num_features = batch.x.shape[-1] if hasattr(batch, 'x') else 10
        return np.random.rand(num_features)
        
    def _compute_permutation_importance(self, batch: Union[Data, Batch, TensorDict]) -> np.ndarray:
        """Compute permutation-based feature importance."""
        # Placeholder - would perform permutation tests
        num_features = batch.x.shape[-1] if hasattr(batch, 'x') else 10
        return np.random.rand(num_features)


class InterpretabilityMixin:
    """
    Mixin for model interpretability.
    
    Provides methods for understanding model decisions through
    various explainability techniques.
    """
    
    def explain_prediction(
        self,
        batch: Union[Data, Batch, TensorDict],
        method: str = "integrated_gradients",
        target_class: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Explain model predictions using various methods.
        
        Args:
            batch: Input data
            method: Explanation method
            target_class: Target class to explain (None for predicted class)
            **kwargs: Method-specific arguments
            
        Returns:
            Dictionary with explanation results
        """
        
        if method == "integrated_gradients":
            return self._integrated_gradients(batch, target_class, **kwargs)
        elif method == "grad_cam":
            return self._grad_cam(batch, target_class, **kwargs)
        elif method == "shap":
            return self._shap_explanation(batch, target_class, **kwargs)
        elif method == "attention":
            return self._attention_explanation(batch, **kwargs)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
            
    def _integrated_gradients(
        self,
        batch: Union[Data, Batch, TensorDict],
        target_class: Optional[int] = None,
        steps: int = 50,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute integrated gradients."""
        
        # Get input features
        if isinstance(batch, (Data, Batch)):
            x = batch.x
        else:
            x = batch.get("features", batch.get("x"))
            
        # Create baseline (zeros)
        baseline = torch.zeros_like(x)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps).to(x.device)
        interpolated_inputs = []
        
        for alpha in alphas:
            interpolated = baseline + alpha * (x - baseline)
            interpolated_inputs.append(interpolated)
            
        interpolated_inputs = torch.stack(interpolated_inputs)
        
        # Compute gradients
        interpolated_inputs.requires_grad_(True)
        
        # Forward pass for all interpolated inputs
        outputs = []
        for i in range(steps):
            # Create batch with interpolated input
            if isinstance(batch, (Data, Batch)):
                batch_i = batch.clone()
                batch_i.x = interpolated_inputs[i]
            else:
                batch_i = batch.copy()
                batch_i["features"] = interpolated_inputs[i]
                
            output = self.forward(batch_i)
            outputs.append(output)
            
        outputs = torch.stack(outputs)
        
        # Select target class
        if target_class is None:
            target_class = outputs[-1].argmax(dim=-1)
            
        # Compute gradients
        grads = []
        for i in range(steps):
            if outputs[i].dim() > 1:
                target_output = outputs[i][:, target_class].sum()
            else:
                target_output = outputs[i].sum()
                
            grad = torch.autograd.grad(target_output, interpolated_inputs, retain_graph=True)[0][i]
            grads.append(grad)
            
        grads = torch.stack(grads)
        
        # Integrated gradients
        avg_grads = grads.mean(dim=0)
        integrated_grads = (x - baseline) * avg_grads
        
        return {
            "attributions": integrated_grads,
            "convergence_delta": (grads[-1] - grads[0]).abs().mean(),
        }
        
    def _grad_cam(
        self,
        batch: Union[Data, Batch, TensorDict],
        target_class: Optional[int] = None,
        layer_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute Grad-CAM for graph data."""
        
        # Hook to capture activations and gradients
        activations = {}
        gradients = {}
        
        def forward_hook(module, input, output):
            activations["target"] = output
            
        def backward_hook(module, grad_input, grad_output):
            gradients["target"] = grad_output[0]
            
        # Register hooks
        if layer_name is None:
            # Use last conv layer by default
            target_layer = None
            for name, module in self.named_modules():
                if "conv" in name.lower():
                    target_layer = module
                    layer_name = name
        else:
            target_layer = dict(self.named_modules())[layer_name]
            
        if target_layer is None:
            raise ValueError("No convolutional layer found")
            
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_backward_hook(backward_hook)
        
        try:
            # Forward pass
            output = self.forward(batch)
            
            # Select target class
            if target_class is None:
                target_class = output.argmax(dim=-1)
                
            # Backward pass
            self.zero_grad()
            if output.dim() > 1:
                target_output = output[:, target_class].sum()
            else:
                target_output = output.sum()
            target_output.backward()
            
            # Compute Grad-CAM
            activation = activations["target"]
            gradient = gradients["target"]
            
            # Global average pooling of gradients
            weights = gradient.mean(dim=tuple(range(2, gradient.dim())))
            
            # Weighted combination of activation maps
            cam = torch.zeros(activation.shape[0], activation.shape[2:])
            for i in range(weights.shape[1]):
                cam += weights[:, i].unsqueeze(-1) * activation[:, i]
                
            # ReLU
            cam = torch.relu(cam)
            
            # Normalize
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
        finally:
            forward_handle.remove()
            backward_handle.remove()
            
        return {
            "cam": cam,
            "layer_name": layer_name,
        }
        
    def _shap_explanation(
        self,
        batch: Union[Data, Batch, TensorDict],
        target_class: Optional[int] = None,
        num_samples: int = 100,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute SHAP values (simplified version)."""
        
        # This is a simplified implementation
        # In practice, you'd use the SHAP library
        
        # Get features
        if isinstance(batch, (Data, Batch)):
            x = batch.x
        else:
            x = batch.get("features", batch.get("x"))
            
        num_features = x.shape[-1]
        shap_values = torch.zeros_like(x)
        
        # Sample random coalitions
        for _ in range(num_samples):
            # Random subset of features
            mask = torch.rand(num_features) > 0.5
            
            # Compute marginal contribution
            # (simplified - actual SHAP is more complex)
            masked_x = x * mask
            
            if isinstance(batch, (Data, Batch)):
                batch_masked = batch.clone()
                batch_masked.x = masked_x
            else:
                batch_masked = batch.copy()
                batch_masked["features"] = masked_x
                
            output_masked = self.forward(batch_masked)
            
            # Attribute contribution to present features
            shap_values += output_masked.unsqueeze(-1) * mask
            
        shap_values /= num_samples
        
        return {
            "shap_values": shap_values,
            "feature_importance": shap_values.abs().mean(dim=0),
        }
        
    def _attention_explanation(
        self,
        batch: Union[Data, Batch, TensorDict],
        aggregate: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Extract attention weights for explanation."""
        
        attention_weights = []
        
        # Hook to capture attention weights
        def attention_hook(module, input, output):
            if hasattr(module, "attention_weights"):
                attention_weights.append(module.attention_weights)
                
        # Register hooks on attention layers
        handles = []
        for name, module in self.named_modules():
            if "attention" in name.lower() or "attn" in name.lower():
                handle = module.register_forward_hook(attention_hook)
                handles.append(handle)
                
        try:
            # Forward pass
            _ = self.forward(batch)
            
            # Aggregate attention weights
            if attention_weights and aggregate:
                # Average across layers
                aggregated = torch.stack(attention_weights).mean(dim=0)
            else:
                aggregated = attention_weights
                
        finally:
            for handle in handles:
                handle.remove()
                
        return {
            "attention_weights": aggregated,
            "num_layers": len(attention_weights),
        }


class EfficientProcessingMixin:
    """
    Mixin for efficient processing of large-scale data.
    
    Provides methods for memory-efficient operations, chunked processing,
    and optimized computation patterns.
    """
    
    def process_in_chunks(
        self,
        data: Union[torch.Tensor, Data, Batch, TensorDict],
        chunk_size: int = 10000,
        dim: int = 0,
        fn: Optional[callable] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Process data in chunks to manage memory usage.
        
        Args:
            data: Input data to process
            chunk_size: Size of each chunk
            dim: Dimension along which to chunk
            fn: Function to apply to each chunk (default: self.forward)
            **kwargs: Additional arguments for the processing function
            
        Returns:
            Processed results (concatenated or as list)
        """
        
        if fn is None:
            fn = self.forward
            
        # Handle different input types
        if isinstance(data, torch.Tensor):
            return self._process_tensor_chunks(data, chunk_size, dim, fn, **kwargs)
        elif isinstance(data, (Data, Batch)):
            return self._process_pyg_chunks(data, chunk_size, fn, **kwargs)
        elif isinstance(data, TensorDict):
            return self._process_tensordict_chunks(data, chunk_size, fn, **kwargs)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
            
    def _process_tensor_chunks(
        self,
        tensor: torch.Tensor,
        chunk_size: int,
        dim: int,
        fn: callable,
        **kwargs,
    ) -> torch.Tensor:
        """Process tensor in chunks."""
        
        chunks = tensor.split(chunk_size, dim=dim)
        results = []
        
        for chunk in chunks:
            with torch.autocast(device_type="cuda", enabled=hasattr(self, "use_amp") and self.use_amp):
                result = fn(chunk, **kwargs)
                results.append(result)
                
        return torch.cat(results, dim=dim)
        
    def _process_pyg_chunks(
        self,
        data: Union[Data, Batch],
        chunk_size: int,
        fn: callable,
        **kwargs,
    ) -> List[torch.Tensor]:
        """Process PyG data in chunks."""
        
        num_nodes = data.x.shape[0]
        results = []
        
        for start_idx in range(0, num_nodes, chunk_size):
            end_idx = min(start_idx + chunk_size, num_nodes)
            
            # Create sub-graph
            if hasattr(data, "batch"):
                # Handle batched data
                node_mask = torch.zeros(num_nodes, dtype=torch.bool)
                node_mask[start_idx:end_idx] = True
                
                # Get sub-batch
                sub_data = data.subgraph(node_mask)
            else:
                # Simple slicing for single graph
                sub_data = Data(
                    x=data.x[start_idx:end_idx],
                    edge_index=data.edge_index,  # Note: edges might need filtering
                    edge_attr=data.edge_attr if hasattr(data, "edge_attr") else None,
                )
                
            result = fn(sub_data, **kwargs)
            results.append(result)
            
        return results
        
    def _process_tensordict_chunks(
        self,
        td: TensorDict,
        chunk_size: int,
        fn: callable,
        **kwargs,
    ) -> List[torch.Tensor]:
        """Process TensorDict in chunks."""
        
        # Determine size from first tensor
        first_key = list(td.keys())[0]
        num_items = td[first_key].shape[0]
        
        results = []
        
        for start_idx in range(0, num_items, chunk_size):
            end_idx = min(start_idx + chunk_size, num_items)
            
            # Create sub-tensordict
            sub_td = TensorDict({
                key: value[start_idx:end_idx] 
                for key, value in td.items()
            }, batch_size=td.batch_size)
            
            result = fn(sub_td, **kwargs)
            results.append(result)
            
        return results
        
    def enable_memory_efficient_mode(self):
        """Enable memory-efficient processing mode."""
        
        # Enable gradient checkpointing if available
        if hasattr(self, "gradient_checkpointing_enable"):
            self.gradient_checkpointing_enable()
            
        # Set to evaluation mode for inference
        self.eval()
        
        # Enable mixed precision
        self.use_amp = True
        
        # Reduce batch size recommendations
        if hasattr(self, "max_points_per_batch"):
            self.max_points_per_batch = self.max_points_per_batch // 2
            
    def estimate_memory_usage(
        self,
        batch_size: int,
        num_features: int,
        num_nodes: int,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Estimate memory usage for given input size.
        
        Returns:
            Dictionary with memory estimates in MB
        """
        
        # Count parameters
        num_params = sum(p.numel() for p in self.parameters())
        param_memory = num_params * 4 / 1024 / 1024  # Assuming float32
        
        # Estimate activation memory
        # This is a rough estimate - actual usage depends on architecture
        hidden_dim = getattr(self, "hidden_dim", 256)
        num_layers = getattr(self, "num_layers", 3)
        
        activation_memory = (
            batch_size * num_nodes * hidden_dim * num_layers * 4
        ) / 1024 / 1024
        
        # Estimate gradient memory (similar to parameters)
        gradient_memory = param_memory
        
        # Estimate input/output memory
        io_memory = (
            batch_size * num_nodes * (num_features + self.num_classes) * 4
        ) / 1024 / 1024
        
        total_memory = param_memory + activation_memory + gradient_memory + io_memory
        
        return {
            "parameters": param_memory,
            "activations": activation_memory,
            "gradients": gradient_memory,
            "input_output": io_memory,
            "total": total_memory,
        }
        
    @torch.no_grad()
    def benchmark_performance(
        self,
        input_sizes: List[Tuple[int, int]],  # [(num_nodes, num_features), ...]
        num_iterations: int = 10,
        device: str = "cuda",
    ) -> Dict[str, List[float]]:
        """
        Benchmark model performance on different input sizes.
        
        Args:
            input_sizes: List of (num_nodes, num_features) tuples
            num_iterations: Number of iterations per size
            device: Device to run on
            
        Returns:
            Dictionary with timing results
        """
        
        import time
        
        results = {
            "input_sizes": input_sizes,
            "forward_times": [],
            "throughput": [],
        }
        
        self.to(device)
        self.eval()
        
        for num_nodes, num_features in input_sizes:
            # Create dummy input
            if hasattr(self, "_create_dummy_input"):
                dummy_input = self._create_dummy_input(num_nodes, num_features, device)
            else:
                dummy_input = Data(
                    x=torch.randn(num_nodes, num_features, device=device),
                    edge_index=torch.randint(0, num_nodes, (2, num_nodes * 10), device=device),
                )
                
            # Warmup
            for _ in range(3):
                _ = self.forward(dummy_input)
                
            # Benchmark
            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.time()
            
            for _ in range(num_iterations):
                _ = self.forward(dummy_input)
                
            torch.cuda.synchronize() if device == "cuda" else None
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_iterations
            throughput = num_nodes / avg_time
            
            results["forward_times"].append(avg_time)
            results["throughput"].append(throughput)
            
        return results


class CheckpointingMixin:
    """
    Mixin for gradient checkpointing support.
    
    Enables training of larger models with limited memory.
    """
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory-efficient training."""
        self._gradient_checkpointing = True
        
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing."""
        self._gradient_checkpointing = False
        
    def checkpoint_forward(self, module: nn.Module, *args, **kwargs):
        """
        Forward with optional gradient checkpointing.
        
        Args:
            module: Module to run
            *args: Positional arguments for module
            **kwargs: Keyword arguments for module
            
        Returns:
            Module output
        """
        
        if self.training and getattr(self, "_gradient_checkpointing", False):
            return torch.utils.checkpoint.checkpoint(module, *args, **kwargs)
        else:
            return module(*args, **kwargs)
