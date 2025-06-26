"""
Custom Lightning Callbacks for AstroLab
======================================

Custom callbacks for enhanced training control and monitoring.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

from lightning.pytorch.callbacks import ModelCheckpoint


class SafeModelCheckpoint(ModelCheckpoint):
    """
    ModelCheckpoint that ensures MLflow-compatible filenames.
    
    This checkpoint callback sanitizes filenames to avoid issues with
    MLflow artifact logging by replacing problematic characters.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with parent arguments."""
        super().__init__(*args, **kwargs)
    
    def format_checkpoint_name(
        self, 
        metrics: Dict[str, Any], 
        filename: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate checkpoint filename with MLflow-safe formatting.
        
        Args:
            metrics: Dictionary of metrics to include in filename
            filename: Optional filename template
            **kwargs: Additional arguments
            
        Returns:
            Sanitized checkpoint filename
        """
        # Get the formatted filename from parent
        checkpoint_name = super().format_checkpoint_name(metrics, filename, **kwargs)
        
        # Sanitize for MLflow: replace dots in metric values
        # This regex finds patterns like "1.234" and replaces the dot with underscore
        sanitized_name = re.sub(r'(\d+)\.(\d+)', r'\1_\2', checkpoint_name)
        
        # Also ensure no double dots or problematic characters
        sanitized_name = sanitized_name.replace('..', '_')
        
        return sanitized_name
    
    def _format_checkpoint_name(
        self,
        filename: Optional[str],
        metrics: Dict[str, Any],
        prefix: str = "",
        auto_insert_metric_name: bool = True
    ) -> str:
        """
        Format checkpoint name with sanitization.
        
        This method is called by Lightning internally.
        """
        # First, let the parent class format the name
        if hasattr(super(), '_format_checkpoint_name'):
            # For newer Lightning versions
            formatted = super()._format_checkpoint_name(
                filename, metrics, prefix, auto_insert_metric_name
            )
        else:
            # For older versions, use format_checkpoint_name
            formatted = self.format_checkpoint_name(metrics, filename)
        
        # Sanitize the formatted name
        # Replace dots in numeric values with underscores
        formatted = re.sub(r'(\d+)\.(\d+)', r'\1_\2', formatted)
        
        # Ensure no problematic patterns
        formatted = formatted.replace('..', '_')
        formatted = formatted.replace(' ', '_')
        
        return formatted
