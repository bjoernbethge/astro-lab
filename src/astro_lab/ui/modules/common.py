"""
Common UI Components - Reusable building blocks
==============================================

DRY components for consistent UI across all modules.
"""

import marimo as mo
from typing import Any, Dict, List, Optional, Callable, Union
import torch
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# UI Component Builders
# =============================================================================

def create_status_card(
    title: str,
    items: Dict[str, str],
    icon: str = "📊",
    style: Optional[str] = None
) -> mo.Html:
    """Create a consistent status card."""
    default_style = "background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 8px; margin-bottom: 1rem;"
    
    if style:
        card_style = f"{default_style} {style}"
    else:
        card_style = default_style
    
    items_html = ""
    for key, value in items.items():
        items_html += f"<div>{key}: <strong>{value}</strong></div>\n"
    
    return mo.Html(f"""
    <div style="{card_style}">
        <h3 style="margin: 0 0 0.5rem 0;">{icon} {title}</h3>
        <div style="font-size: 0.9rem; line-height: 1.6;">
            {items_html}
        </div>
    </div>
    """)


def create_progress_bar(
    value: float,
    max_value: float = 100.0,
    label: str = "",
    color: str = "#4CAF50",
    show_percentage: bool = True
) -> mo.Html:
    """Create a progress bar."""
    percentage = (value / max_value) * 100 if max_value > 0 else 0
    
    percentage_text = f"{percentage:.1f}%" if show_percentage else ""
    
    return mo.Html(f"""
    <div style="margin: 1rem 0;">
        {f'<h4>{label}</h4>' if label else ''}
        <div style="background: #e0e0e0; height: 30px; border-radius: 15px;">
            <div style="background: {color}; width: {percentage}%; height: 100%; 
                        border-radius: 15px; display: flex; align-items: center; 
                        justify-content: center; color: white; font-weight: bold;">
                {percentage_text}
            </div>
        </div>
    </div>
    """)


def create_metric_card(
    title: str,
    value: Union[str, float, int],
    subtitle: Optional[str] = None,
    trend: Optional[str] = None,  # "up", "down", "stable"
    color: str = "#4CAF50"
) -> mo.Html:
    """Create a metric display card."""
    trend_icon = ""
    if trend == "up":
        trend_icon = "📈"
    elif trend == "down":
        trend_icon = "📉"
    elif trend == "stable":
        trend_icon = "➡️"
    
    # Format value if numeric
    if isinstance(value, (int, float)):
        if value >= 1_000_000:
            display_value = f"{value/1_000_000:.2f}M"
        elif value >= 1_000:
            display_value = f"{value/1_000:.1f}K"
        else:
            display_value = f"{value:.2f}" if isinstance(value, float) else str(value)
    else:
        display_value = value
    
    return mo.Html(f"""
    <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; 
                border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
                text-align: center;">
        <h3 style="margin: 0; color: {color};">{display_value} {trend_icon}</h3>
        <div style="margin-top: 0.5rem; font-weight: bold;">{title}</div>
        {f'<div style="margin-top: 0.25rem; opacity: 0.7; font-size: 0.9rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """)


def create_info_banner(
    message: str,
    kind: str = "info",  # "info", "success", "warning", "error"
    dismissible: bool = True
) -> mo.Html:
    """Create an information banner."""
    colors = {
        "info": ("#2196F3", "ℹ️"),
        "success": ("#4CAF50", "✅"),
        "warning": ("#FF9800", "⚠️"),
        "error": ("#F44336", "❌"),
    }
    
    color, icon = colors.get(kind, ("#2196F3", "ℹ️"))
    
    dismiss_button = """
    <button onclick="this.parentElement.style.display='none'" 
            style="background: none; border: none; color: white; 
                   cursor: pointer; font-size: 1.2rem; float: right;">
        ×
    </button>
    """ if dismissible else ""
    
    return mo.Html(f"""
    <div style="background: {color}; color: white; padding: 1rem; 
                border-radius: 8px; margin: 1rem 0;">
        {dismiss_button}
        <div style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="font-size: 1.5rem;">{icon}</span>
            <span>{message}</span>
        </div>
    </div>
    """)


def create_action_button(
    label: str,
    icon: str = "",
    kind: str = "primary",  # "primary", "secondary", "success", "danger"
    on_click: Optional[Callable] = None,
    disabled: bool = False,
    full_width: bool = False
) -> mo.ui.button:
    """Create a styled action button."""
    button_label = f"{icon} {label}" if icon else label
    
    return mo.ui.button(
        button_label,
        on_click=on_click,
        kind=kind,
        disabled=disabled,
        full_width=full_width,
    )


def create_loading_spinner(
    message: str = "Loading...",
    size: str = "medium"  # "small", "medium", "large"
) -> mo.Html:
    """Create a loading spinner."""
    sizes = {
        "small": "20px",
        "medium": "40px",
        "large": "60px",
    }
    
    spinner_size = sizes.get(size, "40px")
    
    return mo.Html(f"""
    <div style="text-align: center; padding: 2rem;">
        <style>
            @keyframes spin {{
                0% {{ transform: rotate(0deg); }}
                100% {{ transform: rotate(360deg); }}
            }}
        </style>
        <div style="border: 4px solid rgba(255,255,255,0.1); 
                    border-top: 4px solid #667eea; 
                    border-radius: 50%; 
                    width: {spinner_size}; 
                    height: {spinner_size};
                    animation: spin 1s linear infinite;
                    margin: 0 auto;">
        </div>
        <div style="margin-top: 1rem; color: #a0a0a0;">{message}</div>
    </div>
    """)


# =============================================================================
# Layout Helpers
# =============================================================================

def create_grid_layout(
    items: List[mo.Html],
    columns: int = 3,
    gap: str = "1rem"
) -> mo.Html:
    """Create a responsive grid layout."""
    grid_items = "".join([f"<div>{item._repr_html_()}</div>" for item in items])
    
    return mo.Html(f"""
    <div style="display: grid; 
                grid-template-columns: repeat({columns}, 1fr); 
                gap: {gap};">
        {grid_items}
    </div>
    """)


def create_tab_content(
    tabs: Dict[str, mo.Html],
    default_tab: Optional[str] = None
) -> mo.ui.tabs:
    """Create consistent tab layout."""
    return mo.ui.tabs(tabs, value=default_tab)


# =============================================================================
# Data Display Helpers
# =============================================================================

def format_number(
    value: Union[int, float],
    precision: int = 2,
    use_scientific: bool = False
) -> str:
    """Format numbers for display."""
    if use_scientific or abs(value) > 1e6 or (abs(value) < 1e-3 and value != 0):
        return f"{value:.{precision}e}"
    elif isinstance(value, int):
        return f"{value:,}"
    else:
        return f"{value:,.{precision}f}"


def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_duration(seconds: float) -> str:
    """Format duration to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


# =============================================================================
# State Management Helpers
# =============================================================================

def create_state_manager(
    initial_state: Dict[str, Any]
) -> tuple:
    """Create a state manager with getters and setters."""
    state, set_state = mo.state(initial_state)
    
    def get(key: str, default: Any = None) -> Any:
        """Get state value."""
        return state().get(key, default)
    
    def set(key: str, value: Any) -> None:
        """Set state value."""
        set_state(lambda s: {**s, key: value})
    
    def update(updates: Dict[str, Any]) -> None:
        """Update multiple state values."""
        set_state(lambda s: {**s, **updates})
    
    return state, get, set, update


# =============================================================================
# Error Handling
# =============================================================================

def safe_execute(
    func: Callable,
    error_message: str = "An error occurred",
    show_traceback: bool = False
) -> Optional[Any]:
    """Safely execute a function with error handling."""
    try:
        return func()
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}")
        
        if show_traceback:
            import traceback
            tb = traceback.format_exc()
            mo.output.append(mo.md(f"""
            ❌ **{error_message}**
            
            ```python
            {tb}
            ```
            """))
        else:
            mo.output.append(create_info_banner(
                f"{error_message}: {str(e)}",
                kind="error"
            ))
        
        return None


# =============================================================================
# Validation Helpers
# =============================================================================

def validate_data_loaded(dm: Any) -> bool:
    """Validate that data is loaded."""
    if dm is None:
        mo.output.append(create_info_banner(
            "No data loaded. Please load data first.",
            kind="warning"
        ))
        return False
    return True


def validate_gpu_available() -> bool:
    """Validate GPU availability."""
    if not torch.cuda.is_available():
        mo.output.append(create_info_banner(
            "GPU not available. Running on CPU.",
            kind="warning"
        ))
        return False
    return True


# =============================================================================
# Export all components
# =============================================================================

__all__ = [
    # UI Components
    "create_status_card",
    "create_progress_bar",
    "create_metric_card",
    "create_info_banner",
    "create_action_button",
    "create_loading_spinner",
    # Layout
    "create_grid_layout",
    "create_tab_content",
    # Formatting
    "format_number",
    "format_bytes",
    "format_duration",
    # State Management
    "create_state_manager",
    # Error Handling
    "safe_execute",
    # Validation
    "validate_data_loaded",
    "validate_gpu_available",
]
