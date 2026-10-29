"""
Módulo de Carga, Validación e Inspección de Datos.
Implementa validaciones de esquema, integridad y cálculo de estadísticas descriptivas.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.config import (
    DATA_PATH,
    FEATURE_COL,
    TARGET_COL,
    EDUCATION_COL,
    GENDER_COL,
    YEARS_SCHOOL_COL,
    ID_COL
)


class DataLoader:
    """Clase responsable de la ingesta y validación de calidad de datos."""

    REQUIRED_COLUMNS = [ID_COL, FEATURE_COL, EDUCATION_COL, TARGET_COL, GENDER_COL, YEARS_SCHOOL_COL]

    def __init__(self, file_path: Path = DATA_PATH):
        self.file_path = file_path
        self._raw_df: pd.DataFrame | None = None
        self._clean_df: pd.DataFrame | None = None

    def load_data(self, sheet_name: str | int = 0) -> pd.DataFrame:
        """Carga el dataset desde archivo Excel y valida estructura."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos en: {self.file_path}")

        df = pd.read_excel(self.file_path, sheet_name=sheet_name)
        
        # Validar esquema de columnas
        missing_cols = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Faltan columnas requeridas en el archivo: {missing_cols}")

        self._raw_df = df.copy()
        self._clean_df = self._clean_data(df)
        return self._clean_df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica reglas de negocio e integridad sobre los datos crudos."""
        cleaned = df.copy()

        # Filtrar posibles filas totalmente nulas
        cleaned = cleaned.dropna(subset=[FEATURE_COL, TARGET_COL])

        # Asegurar tipos de datos numéricos consistentes
        cleaned[FEATURE_COL] = pd.to_numeric(cleaned[FEATURE_COL], errors="coerce")
        cleaned[TARGET_COL] = pd.to_numeric(cleaned[TARGET_COL], errors="coerce")
        cleaned[YEARS_SCHOOL_COL] = pd.to_numeric(cleaned[YEARS_SCHOOL_COL], errors="coerce")

        # Eliminar registros con valores inválidos (edades <= 0, ingresos < 0)
        cleaned = cleaned[(cleaned[FEATURE_COL] > 0) & (cleaned[TARGET_COL] >= 0)]

        # Limpiar espacios en blanco en cadenas de texto
        cleaned[EDUCATION_COL] = cleaned[EDUCATION_COL].astype(str).str.strip()
        cleaned[GENDER_COL] = cleaned[GENDER_COL].astype(str).str.strip()

        return cleaned

    def get_summary_statistics(self) -> Dict[str, Dict[str, float]]:
        """Calcula estadísticas descriptivas univariadas completas."""
        if self._clean_df is None:
            raise RuntimeError("Debe cargar los datos antes de solicitar estadísticas.")

        stats_dict = {}
        numeric_cols = [FEATURE_COL, TARGET_COL, YEARS_SCHOOL_COL]

        for col in numeric_cols:
            series = self._clean_df[col]
            q25 = float(series.quantile(0.25))
            q75 = float(series.quantile(0.75))
            stats_dict[col] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "median": float(series.median()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": q25,
                "q75": q75,
                "iqr": q75 - q25,
                "skewness": float(series.skew()),
                "kurtosis": float(series.kurtosis()),
                "cv_percent": float((series.std() / series.mean()) * 100 if series.mean() != 0 else 0)
            }

        return stats_dict

    def get_categorical_distributions(self) -> Dict[str, Dict[str, int]]:
        """Devuelve conteos y distribuciones de las variables categóricas."""
        if self._clean_df is None:
            raise RuntimeError("Debe cargar los datos primero.")

        return {
            GENDER_COL: self._clean_df[GENDER_COL].value_counts().to_dict(),
            EDUCATION_COL: self._clean_df[EDUCATION_COL].value_counts().to_dict()
        }

    def get_feature_target_split(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Extrae el predictor (edad) y el objetivo (ingreso_mensual)."""
        if self._clean_df is None:
            raise RuntimeError("Debe cargar los datos primero.")
        return self._clean_df[[FEATURE_COL]], self._clean_df[TARGET_COL]
