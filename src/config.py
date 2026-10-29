"""
Módulo de Configuración Global del Proyecto.
Define rutas, constantes estadísticas, parámetros de modelado y estilos de visualización.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List

# Rutas del Sistema
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = BASE_DIR / "data.xlsx"
PLOTS_DIR: Path = BASE_DIR / "plots"
NOTEBOOK_DIR: Path = BASE_DIR / "notebook"
INFORME_PATH: Path = BASE_DIR / "INFORME_EJECUTIVO_REGRESION.md"
PRESENTACION_PATH: Path = BASE_DIR / "presentacion_ejecutiva.html"

# Asegurar existencia de directorios de salida
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

# Nombres de Columnas del Conjunto de Datos
ID_COL: str = "id"
FEATURE_COL: str = "edad"
TARGET_COL: str = "ingreso_mensual"
EDUCATION_COL: str = "nivel_edu"
GENDER_COL: str = "sex"
YEARS_SCHOOL_COL: str = "anios_esc"

@dataclass(frozen=True)
class ModelConfig:
    """Configuración de parámetros de modelado y experimentación."""
    random_state: int = 42
    test_size: float = 0.20
    k_folds: int = 10
    confidence_level: float = 0.95
    target_ages: List[int] = field(default_factory=lambda: [18, 22, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])

@dataclass(frozen=True)
class PlotConfig:
    """Configuración gráfica para plots profesionales."""
    dpi: int = 300
    style: str = "seaborn-v0_8-whitegrid"
    primary_color: str = "#1A365D"      # Azul marino profundo
    secondary_color: str = "#2B6CB0"    # Azul corporativo
    accent_color: str = "#DD6B20"       # Naranja cobrizo
    danger_color: str = "#E53E3E"       # Rojo coral
    success_color: str = "#38A169"      # Verde esmeralda
    background_color: str = "#F7FAFC"   # Gris muy claro
    font_family: str = "sans-serif"
    figsize_wide: tuple = (12, 6)
    figsize_square: tuple = (8, 8)
    figsize_grid: tuple = (14, 10)

MODEL_CONFIG = ModelConfig()
PLOT_CONFIG = PlotConfig()
