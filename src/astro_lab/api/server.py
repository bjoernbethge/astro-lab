"""
AstroLab API Server
==================

FastAPI server for AstroLab backend functionality.
Provides endpoints for data loading, model training, and visualization.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from astro_lab.data import AstroDataModule, list_available_catalogs
from astro_lab.models.core import list_lightning_models, list_presets
from astro_lab.training import AstroTrainer
from astro_lab.cli.config import load_and_prepare_training_config

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="AstroLab API",
    description="API for astronomical data analysis with Graph Neural Networks",
    version="2.0.0",
)

# CORS middleware für UI-Kommunikation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Für Entwicklung - in Produktion einschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models für Request/Response
class TrainingRequest(BaseModel):
    """Training request model."""
    preset: Optional[str] = None
    model: Optional[str] = None
    dataset: str = "gaia"
    max_epochs: int = Field(default=50, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.001, gt