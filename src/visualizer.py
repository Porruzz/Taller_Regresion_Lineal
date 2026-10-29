"""
Módulo de Visualización Gráfica Profesional para Inteligencia Artificial.
Genera figuras de alta resolución (300 DPI) para publicación académica y ejecutiva.
"""

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

from src.config import (
    PLOTS_DIR,
    PLOT_CONFIG,
    FEATURE_COL,
    TARGET_COL
)
from src.models import SimpleLinearModel, PolynomialModel, GradientDescentRegressor


class Visualizer:
    """Generador de gráficos estadísticos y econométricos de alta calidad."""

    def __init__(self, output_dir: Path = PLOTS_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_style()

    def _setup_style(self):
        """Configura el tema global y paleta de matplotlib / seaborn."""
        sns.set_theme(style="whitegrid", palette="deep")
        plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial"]
        plt.rcParams["axes.edgecolor"] = "#CBD5E0"
        plt.rcParams["axes.linewidth"] = 0.8
        plt.rcParams["grid.color"] = "#E2E8F0"
        plt.rcParams["grid.linestyle"] = "--"
        plt.rcParams["grid.alpha"] = 0.7

    def plot_distributions(self, df: pd.DataFrame) -> Path:
        """Figura 1: Distribuciones y Boxplots de Edad e Ingreso Mensual."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Histograma Edad
        sns.histplot(
            df[FEATURE_COL], kde=True, ax=axes[0, 0],
            color=PLOT_CONFIG.secondary_color, bins=35, edgecolor="white", alpha=0.8
        )
        edad_mean = df[FEATURE_COL].mean()
        edad_median = df[FEATURE_COL].median()
        axes[0, 0].axvline(edad_mean, color=PLOT_CONFIG.danger_color, linestyle="--", linewidth=2, label=f"Media: {edad_mean:.1f} años")
        axes[0, 0].axvline(edad_median, color=PLOT_CONFIG.accent_color, linestyle=":", linewidth=2, label=f"Mediana: {edad_median:.1f} años")
        axes[0, 0].set_title("Distribución de la Edad", fontsize=13, fontweight="bold", pad=10)
        axes[0, 0].set_xlabel("Edad (Años)")
        axes[0, 0].set_ylabel("Frecuencia Absoluta")
        axes[0, 0].legend(frameon=True, facecolor="white")

        # 2. Boxplot Edad
        sns.boxplot(
            x=df[FEATURE_COL], ax=axes[0, 1],
            color="#90CDF4", flierprops={"marker": "o", "color": PLOT_CONFIG.danger_color, "alpha": 0.5}
        )
        axes[0, 1].set_title("Diagrama de Caja: Edad", fontsize=13, fontweight="bold", pad=10)
        axes[0, 1].set_xlabel("Edad (Años)")

        # 3. Histograma Ingreso Mensual
        sns.histplot(
            df[TARGET_COL], kde=True, ax=axes[1, 0],
            color=PLOT_CONFIG.primary_color, bins=40, edgecolor="white", alpha=0.8
        )
        ing_mean = df[TARGET_COL].mean()
        ing_median = df[TARGET_COL].median()
        axes[1, 0].axvline(ing_mean, color=PLOT_CONFIG.danger_color, linestyle="--", linewidth=2, label=f"Media: ${ing_mean:,.0f}")
        axes[1, 0].axvline(ing_median, color=PLOT_CONFIG.accent_color, linestyle=":", linewidth=2, label=f"Mediana: ${ing_median:,.0f}")
        axes[1, 0].set_title("Distribución del Ingreso Mensual (Sesgo Positivo)", fontsize=13, fontweight="bold", pad=10)
        axes[1, 0].set_xlabel("Ingreso Mensual ($)")
        axes[1, 0].set_ylabel("Frecuencia Absoluta")
        axes[1, 0].legend(frameon=True, facecolor="white")

        # 4. Boxplot Ingreso Mensual
        sns.boxplot(
            x=df[TARGET_COL], ax=axes[1, 1],
            color="#BEE3F8", flierprops={"marker": "o", "color": PLOT_CONFIG.accent_color, "alpha": 0.4, "markersize": 3}
        )
        axes[1, 1].set_title("Diagrama de Caja: Ingreso Mensual (Outliers Superiores)", fontsize=13, fontweight="bold", pad=10)
        axes[1, 1].set_xlabel("Ingreso Mensual ($)")

        plt.suptitle("Análisis Exploratorio Univariado de Variables Clave (N = 10,280)", fontsize=16, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        file_path = self.output_dir / "01_distribucion_variables.png"
        fig.savefig(file_path, dpi=PLOT_CONFIG.dpi)
        plt.close(fig)
        return file_path

    def plot_linear_regression(self, df: pd.DataFrame, model: SimpleLinearModel) -> Path:
        """Figura 2: Dispersión con Recta OLS, Banda de Confianza (95%) y Banda de Predicción (95%)."""
        fig, ax = plt.subplots(figsize=PLOT_CONFIG.figsize_wide)

        ax.scatter(
            df[FEATURE_COL], df[TARGET_COL],
            alpha=0.20, color=PLOT_CONFIG.secondary_color, edgecolors="none", s=25, label="Observaciones Reales (N=10,280)"
        )

        x_grid = np.linspace(df[FEATURE_COL].min(), df[FEATURE_COL].max(), 300)
        intervals_df = model.predict_with_intervals(x_grid, alpha=0.05)

        ax.plot(
            x_grid, intervals_df["prediccion_ingreso"],
            color=PLOT_CONFIG.danger_color, linewidth=2.8, label="Recta de Regresión OLS"
        )

        ax.fill_between(
            x_grid, intervals_df["ic_media_inferior"], intervals_df["ic_media_superior"],
            color=PLOT_CONFIG.danger_color, alpha=0.35, label="I.C. 95% para la Media E[Y|X]"
        )

        ax.plot(x_grid, intervals_df["ip_individual_superior"], color="#805AD5", linestyle="--", linewidth=1.4, label="I.P. 95% Individual (Límite Superior)")
        ax.plot(x_grid, intervals_df["ip_individual_inferior"], color="#805AD5", linestyle="--", linewidth=1.4, label="I.P. 95% Individual (Límite Inferior)")

        res = model.results
        stats_text = (
            f"Ecuación: {res.equation_str}\n"
            f"Pendiente (β₁): ${res.slope:.2f} (p < 0.001)\n"
            f"Intercepto (β₀): ${res.intercept:,.2f}\n"
            f"R²: {res.r_squared:.4f} | R²-Adj: {res.adj_r_squared:.4f}\n"
            f"F-stat: {res.f_statistic:.2f} (p = {res.f_p_value:.4e})\n"
            f"Error Estándar Residual (Se): ${res.residual_std_error:,.2f}"
        )
        ax.text(
            0.03, 0.95, stats_text, transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#A0AEC0", alpha=0.95),
            fontsize=10.5, family="monospace"
        )

        ax.set_title("Ajuste del Modelo de Regresión Lineal Simple: Ingreso Mensual vs. Edad", fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel("Edad (Años)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Ingreso Mensual ($)", fontsize=12, fontweight="bold")
        ax.set_ylim(-1000, 45000)
        ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#CBD5E0")

        plt.tight_layout()
        file_path = self.output_dir / "02_dispersion_regresion_lineal.png"
        fig.savefig(file_path, dpi=PLOT_CONFIG.dpi)
        plt.close(fig)
        return file_path

    def plot_residual_diagnostics(self, sm_results) -> Path:
        """Figura 3: Diagnóstico Cuádruple de Residuos de Gauss-Márkov."""
        residuals = sm_results.resid
        fitted = sm_results.fittedvalues
        standardized_resid = sm_results.get_influence().resid_studentized_internal

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # 1. Residuos vs Valores Ajustados
        axes[0, 0].scatter(fitted, residuals, alpha=0.25, color=PLOT_CONFIG.secondary_color, s=20)
        axes[0, 0].axhline(0, color=PLOT_CONFIG.danger_color, linestyle="--", linewidth=1.5)
        lowess = sm.nonparametric.lowess(residuals, fitted, frac=0.3)
        axes[0, 0].plot(lowess[:, 0], lowess[:, 1], color=PLOT_CONFIG.accent_color, linewidth=2.5, label="Tendencia Local (LOWESS)")
        axes[0, 0].set_title("1. Residuos vs. Valores Ajustados (Linealidad)", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Valores Ajustados (Ingreso Predicho)")
        axes[0, 0].set_ylabel("Residuos Ordinarios")
        axes[0, 0].legend(loc="upper right", frameon=True, facecolor="white")

        # 2. Normal Q-Q Plot
        (osm, osr), (slope, intercept, r) = stats.probplot(standardized_resid, dist="norm")
        axes[0, 1].scatter(osm, osr, alpha=0.30, color=PLOT_CONFIG.primary_color, s=20)
        axes[0, 1].plot(osm, slope * np.array(osm) + intercept, color=PLOT_CONFIG.danger_color, linewidth=2, label=f"Referencia Normal (R={r:.3f})")
        axes[0, 1].set_title("2. Gráfico Normal Q-Q (Prueba de Normalidad)", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Cuantiles Teóricos Normales")
        axes[0, 1].set_ylabel("Residuos Estandarizados")
        axes[0, 1].legend(loc="upper left", frameon=True, facecolor="white")

        # 3. Scale-Location
        root_abs_resid = np.sqrt(np.abs(standardized_resid))
        axes[1, 0].scatter(fitted, root_abs_resid, alpha=0.25, color=PLOT_CONFIG.secondary_color, s=20)
        lowess_sl = sm.nonparametric.lowess(root_abs_resid, fitted, frac=0.3)
        axes[1, 0].plot(lowess_sl[:, 0], lowess_sl[:, 1], color=PLOT_CONFIG.danger_color, linewidth=2.5, label="Tendencia de Varianza")
        axes[1, 0].set_title("3. Scale-Location (Homocedasticidad)", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Valores Ajustados")
        axes[1, 0].set_ylabel(r"$\sqrt{|\text{Residuos Estandarizados}|}$")
        axes[1, 0].legend(loc="upper right", frameon=True, facecolor="white")

        # 4. Histograma y KDE de los Residuos
        sns.histplot(residuals, kde=True, ax=axes[1, 1], color=PLOT_CONFIG.primary_color, bins=45, stat="density", alpha=0.7)
        x_norm = np.linspace(residuals.min(), residuals.max(), 300)
        norm_curve = stats.norm.pdf(x_norm, loc=residuals.mean(), scale=residuals.std())
        axes[1, 1].plot(x_norm, norm_curve, color=PLOT_CONFIG.danger_color, linewidth=2.2, label="Densidad Teórica Gaussiana")
        axes[1, 1].set_xlim(-15000, 30000)
        axes[1, 1].set_title("4. Distribución de Residuos vs. Curva Normal", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Valor del Residuo ($)")
        axes[1, 1].set_ylabel("Densidad")
        axes[1, 1].legend(loc="upper right", frameon=True, facecolor="white")

        plt.suptitle("Batería de Diagnóstico Econométrico de Supuestos de Gauss-Márkov", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        file_path = self.output_dir / "03_diagnostico_residuos.png"
        fig.savefig(file_path, dpi=PLOT_CONFIG.dpi)
        plt.close(fig)
        return file_path

    def plot_polynomial_comparison(
        self,
        df: pd.DataFrame,
        linear_model: SimpleLinearModel,
        poly_model: PolynomialModel
    ) -> Path:
        """Figura 4: Comparación Lineal vs. Polinomial Cuadrática (Curva de Ciclo de Vida Minceriana)."""
        fig, ax = plt.subplots(figsize=PLOT_CONFIG.figsize_wide)

        age_means = df.groupby(FEATURE_COL)[TARGET_COL].mean().reset_index()
        ax.scatter(
            age_means[FEATURE_COL], age_means[TARGET_COL],
            color="#A0AEC0", s=40, zorder=3, alpha=0.8, label="Ingreso Promedio Empírico por Año de Edad"
        )

        x_grid = np.linspace(15, 80, 200)
        y_lin = linear_model.predict(x_grid)
        y_poly = poly_model.predict(x_grid)

        ax.plot(x_grid, y_lin, color=PLOT_CONFIG.secondary_color, linestyle="--", linewidth=2.5, label="Modelo Lineal Simple: β₁ = +$16.05/año (R² = 0.15%)")
        ax.plot(x_grid, y_poly, color=PLOT_CONFIG.accent_color, linewidth=3.2, label="Modelo Polinomial Cuadrático: R² = 2.80% (p < 10⁻⁶⁴)")

        peak_age = poly_model.peak_age
        peak_income = poly_model.peak_income
        ax.scatter([peak_age], [peak_income], color=PLOT_CONFIG.danger_color, s=140, zorder=5, edgecolor="white", linewidth=2)
        ax.axvline(peak_age, color=PLOT_CONFIG.danger_color, linestyle=":", linewidth=1.8)
        ax.annotate(
            f"Vértice de Ingresos Máximos:\nEdad Pico = {peak_age:.1f} años\nIngreso Predicho = ${peak_income:,.2f}",
            xy=(peak_age, peak_income),
            xytext=(peak_age + 4, peak_income + 900),
            arrowprops=dict(facecolor=PLOT_CONFIG.danger_color, shrink=0.08, width=1.5, headwidth=7),
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF5F5", edgecolor=PLOT_CONFIG.danger_color, alpha=0.95)
        )

        ax.set_title("Comparación Econométrica: Modelo Lineal vs. Curva Cuadrática de Ciclo de Vida Salarial", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Edad (Años)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Ingreso Mensual ($)", fontsize=12, fontweight="bold")
        ax.set_ylim(2000, 11000)
        ax.legend(loc="lower center", frameon=True, facecolor="white", edgecolor="#CBD5E0")

        plt.tight_layout()
        file_path = self.output_dir / "04_curva_ciclo_vida_polinomial.png"
        fig.savefig(file_path, dpi=PLOT_CONFIG.dpi)
        plt.close(fig)
        return file_path

    def plot_cross_validation_performance(self, cv_results: Dict[str, Any], split_results: Dict[str, float]) -> Path:
        """Figura 5: Validación Cruzada en 10 Pliegues y Comparativa de Métricas."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        folds = [f"F{i+1}" for i in range(len(cv_results["cv_rmse_scores"]))]
        scores = cv_results["cv_rmse_scores"]
        
        bars = axes[0].bar(folds, scores, color=PLOT_CONFIG.secondary_color, edgecolor="#2B6CB0", alpha=0.85)
        mean_rmse = cv_results["mean_rmse"]
        axes[0].axhline(mean_rmse, color=PLOT_CONFIG.danger_color, linestyle="--", linewidth=2, label=f"Media CV: ${mean_rmse:,.1f}")
        axes[0].fill_between(
            [-0.5, len(folds)-0.5],
            mean_rmse - cv_results["std_rmse"],
            mean_rmse + cv_results["std_rmse"],
            color=PLOT_CONFIG.danger_color, alpha=0.15, label=f"±1 Desv. Est. (${cv_results['std_rmse']:.1f})"
        )
        axes[0].set_title("Estabilidad del Error Fuera de Muestra (10-Fold CV)", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Pliegue de Validación (Fold)")
        axes[0].set_ylabel("RMSE ($)")
        axes[0].set_ylim(4500, 7500)
        axes[0].legend(frameon=True, facecolor="white")

        categories = ["Train Set (80%)", "Test Set (20%)", "10-Fold CV"]
        rmse_values = [split_results["train_rmse"], split_results["test_rmse"], cv_results["mean_rmse"]]
        mae_values = [split_results["train_mae"], split_results["test_mae"], cv_results["mean_mae"]]

        x = np.arange(len(categories))
        width = 0.35

        rects1 = axes[1].bar(x - width/2, rmse_values, width, label="RMSE ($)", color=PLOT_CONFIG.primary_color, alpha=0.85)
        rects2 = axes[1].bar(x + width/2, mae_values, width, label="MAE ($)", color=PLOT_CONFIG.accent_color, alpha=0.85)

        for rect in rects1:
            h = rect.get_height()
            axes[1].annotate(f"${h:,.0f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                             textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            axes[1].annotate(f"${h:,.0f}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 4),
                             textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        axes[1].set_title("Comparativa de Errores: Train vs. Test vs. Validación Cruzada", fontsize=13, fontweight="bold")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories, fontweight="bold")
        axes[1].set_ylabel("Error en Unidades Monetarias ($)")
        axes[1].set_ylim(0, 7500)
        axes[1].legend(frameon=True, facecolor="white")

        plt.suptitle("Validación de Robustez y Generalización del Modelo", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        file_path = self.output_dir / "05_validacion_cruzada_kfold.png"
        fig.savefig(file_path, dpi=PLOT_CONFIG.dpi)
        plt.close(fig)
        return file_path

    def plot_key_age_predictions(self, predictions_df: pd.DataFrame) -> Path:
        """Figura 6: Predicciones Puntuales e Intervalos de Confianza para Edades Clave."""
        fig, ax = plt.subplots(figsize=PLOT_CONFIG.figsize_wide)

        ages = predictions_df["edad"]
        preds = predictions_df["prediccion_ingreso"]
        ci_lower = predictions_df["ic_media_inferior"]
        ci_upper = predictions_df["ic_media_superior"]

        yerr = [preds - ci_lower, ci_upper - preds]

        bars = ax.bar(
            [str(int(a)) for a in ages], preds,
            yerr=yerr, capsize=5, color="#3182CE", edgecolor="#2B6CB0", alpha=0.85,
            error_kw=dict(ecolor=PLOT_CONFIG.danger_color, lw=2, capthick=2)
        )

        for bar, pred in zip(bars, preds):
            h = bar.get_height()
            ax.annotate(
                f"${pred:,.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, h / 2),
                ha="center", va="center", color="white", fontweight="bold", fontsize=10, rotation=90
            )

        ax.set_title("Predicción Puntual de Ingreso Mensual según la Edad con Margen de Error I.C. 95%", fontsize=14, fontweight="bold", pad=12)
        ax.set_xlabel("Edad del Individuo (Años)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Ingreso Mensual Predicho ($)", fontsize=12, fontweight="bold")
        ax.set_ylim(5500, 8500)

        plt.tight_layout()
        file_path = self.output_dir / "06_prediccion_edades_clave.png"
        fig.savefig(file_path, dpi=PLOT_CONFIG.dpi)
        plt.close(fig)
        return file_path

    def plot_gradient_descent_learning(
        self,
        gd_model: GradientDescentRegressor,
        linear_model: SimpleLinearModel
    ) -> Path:
        """
        Figura 7 (Inteligencia Artificial / Machine Learning):
        Curva de Convergencia de la Función de Costo J(theta) y Comparación de Aprendizaje
        entre el Descenso del Gradiente y la Solución Analítica de la Ecuación Normal.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        history = gd_model.history
        epochs = np.arange(1, history.epochs + 1)
        loss = history.loss_history

        # 1. Curva de Costo J(theta) vs Épocas (Learning Curve)
        axes[0].plot(epochs, loss, color=PLOT_CONFIG.accent_color, linewidth=2.5, label=f"J(θ) [α = {history.learning_rate}]")
        axes[0].scatter([history.epochs], [history.final_loss], color=PLOT_CONFIG.danger_color, s=80, zorder=5)
        axes[0].annotate(
            f"Pérdida Final:\n{history.final_loss:,.0f}",
            xy=(history.epochs, history.final_loss),
            xytext=(history.epochs - 120, history.final_loss + 2e6),
            arrowprops=dict(facecolor=PLOT_CONFIG.danger_color, shrink=0.08, width=1.5, headwidth=6),
            fontsize=9.5, fontweight="bold"
        )
        axes[0].set_title("Curva de Aprendizaje de IA: Función de Costo J(θ) vs. Épocas", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("Épocas de Entrenamiento (Epochs)")
        axes[0].set_ylabel("Costo J(θ) = MSE / 2")
        axes[0].legend(frameon=True, facecolor="white")

        # 2. Comparación de Parámetros Aprendidos (GD vs Analítico)
        methods = ["Solución Analítica (OLS)", "Descenso del Gradiente (IA)"]
        slopes = [linear_model.results.slope, gd_model.slope_]
        intercepts = [linear_model.results.intercept, gd_model.intercept_]

        x = np.arange(len(methods))
        width = 0.35

        ax2 = axes[1]
        bars1 = ax2.bar(x - width/2, intercepts, width, label="Intercepto (β₀)", color=PLOT_CONFIG.primary_color, alpha=0.85)
        
        # Eje secundario para la pendiente (por la escala)
        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x + width/2, slopes, width, label="Pendiente (β₁)", color=PLOT_CONFIG.danger_color, alpha=0.85)

        for b in bars1:
            h = b.get_height()
            ax2.annotate(f"${h:,.2f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                         textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        for b in bars2:
            h = b.get_height()
            ax2_twin.annotate(f"${h:.2f}", xy=(b.get_x() + b.get_width()/2, h), xytext=(0, 4),
                              textcoords="offset points", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

        ax2.set_title("Convergencia de Parámetros: OLS vs. Descenso del Gradiente", fontsize=12, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(methods, fontweight="bold")
        ax2.set_ylabel("Intercepto ($)", color=PLOT_CONFIG.primary_color, fontweight="bold")
        ax2_twin.set_ylabel("Pendiente ($/año)", color=PLOT_CONFIG.danger_color, fontweight="bold")
        ax2.set_ylim(0, 8500)
        ax2_twin.set_ylim(0, 25)
        ax2_twin.grid(False)

        plt.suptitle("Optimización mediante Aprendizaje Automático: Descenso del Gradiente (Batch GD)", fontsize=15, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        file_path = self.output_dir / "07_aprendizaje_gradiente_descendente.png"
        fig.savefig(file_path, dpi=PLOT_CONFIG.dpi)
        plt.close(fig)
        return file_path
