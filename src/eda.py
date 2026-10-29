"""
Módulo de Análisis Exploratorio de Datos (EDA).
Realiza análisis univariado, bivariado, estratificación por cohortes de edad y correlaciones.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from src.config import (
    FEATURE_COL,
    TARGET_COL,
    EDUCATION_COL,
    GENDER_COL,
    YEARS_SCHOOL_COL
)


class ExploratoryAnalysis:
    """Ejecuta análisis econométricos y estadísticos descriptivos."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def compute_correlations(self) -> pd.DataFrame:
        """Calcula matriz de correlación de Pearson y Spearman."""
        numeric_cols = [FEATURE_COL, TARGET_COL, YEARS_SCHOOL_COL]
        pearson = self.df[numeric_cols].corr(method="pearson")
        spearman = self.df[numeric_cols].corr(method="spearman")
        return pd.concat({"pearson": pearson, "spearman": spearman}, axis=1)

    def analyze_age_cohorts(self) -> pd.DataFrame:
        """Estratifica la población en cohortes etarias y calcula métricas salariales."""
        bins = [10, 25, 35, 45, 55, 65, 100]
        labels = [
            "12 - 25 años (Jóvenes)",
            "26 - 35 años (Adultos Jóvenes)",
            "36 - 45 años (Madurez Temprana)",
            "46 - 55 años (Plena Madurez)",
            "56 - 65 años (Pre-Jubilación)",
            "66+ años (Tercera Edad)"
        ]
        
        cohort_df = self.df.copy()
        cohort_df["cohorte_edad"] = pd.cut(cohort_df[FEATURE_COL], bins=bins, labels=labels, right=True)
        
        summary = cohort_df.groupby("cohorte_edad", observed=False).agg(
            n_personas=(TARGET_COL, "count"),
            ingreso_promedio=(TARGET_COL, "mean"),
            ingreso_mediana=(TARGET_COL, "median"),
            ingreso_std=(TARGET_COL, "std"),
            ingreso_min=(TARGET_COL, "min"),
            ingreso_max=(TARGET_COL, "max"),
            anios_esc_promedio=(YEARS_SCHOOL_COL, "mean")
        ).reset_index()

        summary["porcentaje_poblacion"] = (summary["n_personas"] / len(self.df)) * 100
        return summary

    def analyze_by_gender(self) -> pd.DataFrame:
        """Calcula brecha salarial y estadísticas por género."""
        gender_summary = self.df.groupby(GENDER_COL).agg(
            n_personas=(TARGET_COL, "count"),
            edad_promedio=(FEATURE_COL, "mean"),
            ingreso_promedio=(TARGET_COL, "mean"),
            ingreso_mediana=(TARGET_COL, "median"),
            ingreso_std=(TARGET_COL, "std"),
            anios_esc_promedio=(YEARS_SCHOOL_COL, "mean")
        ).reset_index()

        gender_summary["porcentaje"] = (gender_summary["n_personas"] / len(self.df)) * 100
        return gender_summary

    def analyze_by_education(self) -> pd.DataFrame:
        """Calcula estadísticas de ingreso y edad según nivel educativo."""
        edu_summary = self.df.groupby(EDUCATION_COL).agg(
            n_personas=(TARGET_COL, "count"),
            edad_promedio=(FEATURE_COL, "mean"),
            ingreso_promedio=(TARGET_COL, "mean"),
            ingreso_mediana=(TARGET_COL, "median"),
            anios_esc_promedio=(YEARS_SCHOOL_COL, "mean")
        ).reset_index().sort_values(by="ingreso_promedio", ascending=False)

        edu_summary["porcentaje"] = (edu_summary["n_personas"] / len(self.df)) * 100
        return edu_summary

    def detect_outliers_iqr(self, column: str) -> Dict[str, Any]:
        """Detecta valores atípicos usando el método del Rango Intercuartílico (Tukey)."""
        series = self.df[column]
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return {
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "n_outliers": int(len(outliers)),
            "outliers_percentage": float((len(outliers) / len(series)) * 100),
            "max_value": float(series.max()),
            "min_value": float(series.min())
        }
