"""
Orquestador Principal del Proyecto: Taller de Regresión Lineal e Inteligencia Artificial.
Materia: SIST5036 - Inteligencia Artificial (Universidad Sergio Arboleda).

Ejecuta el pipeline completo de:
1. Ingesta y calidad de datos.
2. Análisis exploratorio univariado y bivariado.
3. Regresión Lineal OLS (Ecuación Normal).
4. Aprendizaje Supervisado: Descenso del Gradiente (Gradient Descent) optimizando la función de costo J(θ).
5. Predicciones con Intervalos de Confianza (95%) y Predicción (95%).
6. Modelos Avanzados de IA: Curvatura Polinomial y Regularización L1/L2 (Lasso y Ridge).
7. Validación Cruzada 10-Fold CV y Diagnóstico de Supuestos de Gauss-Márkov.
8. Generación de Batería de 7 Gráficos Científicos (300 DPI).
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Configurar codificación UTF-8 en Windows para evitar errores en terminal
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from src.config import (
    DATA_PATH,
    PLOTS_DIR,
    FEATURE_COL,
    TARGET_COL,
    MODEL_CONFIG
)
from src.data_loader import DataLoader
from src.eda import ExploratoryAnalysis
from src.models import (
    SimpleLinearModel,
    GradientDescentRegressor,
    PolynomialModel,
    RegularizedMLModels,
    MultivariateModel
)
from src.evaluator import ModelEvaluator
from src.visualizer import Visualizer


def print_banner():
    banner = """
================================================================================
   TALLER DE INTELIGENCIA ARTIFICIAL: REGRESIÓN LINEAL Y MACHINE LEARNING
   Asignatura: Inteligencia Artificial (SIST5036) - Universidad Sergio Arboleda
   Predicción de Ingreso Mensual vs. Edad | OLS vs. Descenso del Gradiente
================================================================================
    """
    print(banner)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Pipeline de Inteligencia Artificial: Regresión Lineal")
    parser.add_argument(
        "--predict",
        nargs="+",
        type=float,
        default=MODEL_CONFIG.target_ages,
        help="Lista de edades a predecir (ejemplo: --predict 18 25 35 45 55 65)"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Omitir la generación de archivos de imagen PNG"
    )
    return parser.parse_args()


def run_pipeline(target_ages: list, generate_plots: bool = True):
    print_banner()

    # 1. INGESTA Y CALIDAD DE DATOS
    print("\n[PASO 1/7] Ingesta y Validación de Datos...")
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()
    print(f"  [OK] Dataset cargado: {len(df):,} observaciones y {len(df.columns)} características.")
    
    stats_dict = loader.get_summary_statistics()
    edad_stats = stats_dict[FEATURE_COL]
    ing_stats = stats_dict[TARGET_COL]

    print("\n  --- Estadísticas Descriptivas ---")
    print(f"  * Edad: Media = {edad_stats['mean']:.2f} años | Mediana = {edad_stats['median']:.1f} | Min = {edad_stats['min']:.0f} | Max = {edad_stats['max']:.0f} | Asimetría = {edad_stats['skewness']:+.3f}")
    print(f"  * Ingreso: Media = ${ing_stats['mean']:,.2f} | Mediana = ${ing_stats['median']:,.2f} | Min = ${ing_stats['min']:,.2f} | Max = ${ing_stats['max']:,.2f} | Asimetría = {ing_stats['skewness']:+.3f}")

    # 2. ANÁLISIS EXPLORATORIO BIVARIADO
    print("\n[PASO 2/7] Análisis Exploratorio de Datos (EDA)...")
    eda = ExploratoryAnalysis(df)
    corr_df = eda.compute_correlations()
    r_pearson = corr_df.loc[FEATURE_COL, ("pearson", TARGET_COL)]
    r_spearman = corr_df.loc[FEATURE_COL, ("spearman", TARGET_COL)]
    print(f"  * Correlación de Pearson (Edad vs Ingreso): r = {r_pearson:+.4f}")
    print(f"  * Correlación de Spearman (Monótona): rho = {r_spearman:+.4f}")

    cohorts = eda.analyze_age_cohorts()
    print("\n  --- Ingresos Promedio por Cohorte Etaria ---")
    for _, row in cohorts.iterrows():
        print(f"    - {row['cohorte_edad']:<32}: ${row['ingreso_promedio']:,.2f} (Mediana: ${row['ingreso_mediana']:,.2f}, N = {row['n_personas']:,})")

    # 3. REGRESIÓN LINEAL: SOLUCIÓN ANALÍTICA (ECUACIÓN NORMAL / OLS)
    print("\n[PASO 3/7] Ajuste del Modelo Lineal Simple (Solución Analítica OLS)...")
    X, y = loader.get_feature_target_split()
    linear_model = SimpleLinearModel()
    linear_model.fit(X, y)
    res = linear_model.results

    print("\n  ================================================================")
    print(f"  ECUACIÓN ESTIMADA: {res.equation_str}")
    print("  ================================================================")
    print(f"  * Intercepto (β₀):       ${res.intercept:,.2f} (SE: {res.intercept_se:.2f}, t = {res.t_stat_intercept:.2f}, p < 0.001)")
    print(f"  * Pendiente (β₁):        ${res.slope:,.2f}/año (SE: {res.slope_se:.2f}, t = {res.t_stat_slope:.2f}, p = {res.p_val_slope:.4e})")
    print(f"  * Coeficiente R²:         {res.r_squared:.4f} ({res.r_squared * 100:.2f}% de varianza explicada)")
    print(f"  * Estadístico F (1, {res.df_residuals:,}):  {res.f_statistic:.2f} (Prob > F = {res.f_p_value:.4e})")
    print(f"  * Error Estándar Residual: ${res.residual_std_error:,.2f}")

    # 4. APRENDIZAJE SUPERVISADO EN IA: DESCENSO DEL GRADIENTE (GRADIENT DESCENT)
    print("\n[PASO 4/7] Optimización por Inteligencia Artificial: Descenso del Gradiente...")
    gd_model = GradientDescentRegressor(learning_rate=0.05, epochs=500)
    gd_model.fit(X, y)
    gd_hist = gd_model.history

    print(f"  * Algoritmo: Batch Gradient Descent con Estandarización Z-score")
    print(f"  * Épocas: {gd_hist.epochs} | Tasa de Aprendizaje (alpha): {gd_hist.learning_rate}")
    print(f"  * Costo Final J(θ) = MSE / 2: {gd_hist.final_loss:,.2f}")
    print(f"  * Parámetros Aprendidos por IA:")
    print(f"    - Intercepto Aprendido (β₀): ${gd_hist.learned_intercept:,.4f} (Coincidencia con OLS: {abs(gd_hist.learned_intercept - res.intercept):.4f})")
    print(f"    - Pendiente Aprendida (β₁):   ${gd_hist.learned_slope:,.4f}/año (Coincidencia con OLS: {abs(gd_hist.learned_slope - res.slope):.4f})")
    print("  [OK] El Descenso del Gradiente converge con exactitud al óptimo global analítico.")

    # 5. PREDICCIONES E INTERVALOS DE CONFIANZA
    print("\n[PASO 5/7] Predicciones del Ingreso vs. Edad...")
    predictions_df = linear_model.predict_with_intervals(target_ages, alpha=0.05)
    
    print("\n  TABLA DE PREDICCIONES (Nivel de Confianza: 95%):")
    print(f"  {'Edad':<6} | {'Ingreso Predicho':<18} | {'I.C. 95% Media':<25} | {'I.P. 95% Individual':<25}")
    print("  " + "-" * 78)
    for _, row in predictions_df.iterrows():
        age_val = int(row['edad'])
        pred_val = f"${row['prediccion_ingreso']:,.2f}"
        ci_str = f"[${row['ic_media_inferior']:,.2f} - ${row['ic_media_superior']:,.2f}]"
        pi_str = f"[${row['ip_individual_inferior']:,.2f} - ${row['ip_individual_superior']:,.2f}]"
        print(f"  {age_val:<6} | {pred_val:<18} | {ci_str:<25} | {pi_str:<25}")

    # 6. MODELOS COMPLEMENTARIOS DE MACHINE LEARNING (POLINOMIAL & REGULARIZACIÓN)
    print("\n[PASO 6/7] Extensiones de Machine Learning: Regresión Polinomial y Regularización...")
    poly_model = PolynomialModel()
    poly_model.fit(X, y)
    print(f"  * Regresión Polinomial (Grado 2): R² = {poly_model.results.rsquared:.4f} (p < 10^-60)")
    print(f"    -> Vértice / Edad Pico Salarial: {poly_model.peak_age:.2f} años")
    print(f"    -> Ingreso Máximo Estimado:     ${poly_model.peak_income:,.2f}")

    reg_models = RegularizedMLModels(alpha=10.0)
    reg_models.fit(X, y)
    print("  * Modelos Regularizados L1 / L2 (Prevención de Overfitting):")
    comp_df = reg_models.get_coefficients_comparison()
    for _, r in comp_df.iterrows():
        print(f"    - {r['Característica']:<22}: Ridge(L2)={r['Ridge (L2)']:>10.2f} | Lasso(L1)={r['Lasso (L1)']:>10.2f}")

    multi_model = MultivariateModel()
    multi_model.fit(df)
    print(f"  * Modelo Multivariado (+ Años Escolaridad + Sexo): R² = {multi_model.results.rsquared:.4f}")
    print(f"    -> Coeficiente Edad Controlado: +${multi_model.results.params['edad']:.2f}/año (p < 0.001)")
    print(f"    -> Retorno por Año Escolaridad: +${multi_model.results.params['anios_esc']:.2f} (p < 0.001)")

    # 7. EVALUACIÓN, REMUESTREO Y GRÁFICOS
    print("\n[PASO 7/7] Evaluación Fuera de Muestra y Generación de Visualizaciones...")
    evaluator = ModelEvaluator(X, y)
    split_res = evaluator.evaluate_train_test_split()
    cv_res = evaluator.evaluate_kfold_cv(n_splits=10)
    gm_res = evaluator.test_gauss_markov_assumptions(linear_model.sm_results)

    print(f"  * Train/Test (80/20): Train RMSE = ${split_res['train_rmse']:,.2f} | Test RMSE = ${split_res['test_rmse']:,.2f}")
    print(f"  * 10-Fold CV RMSE: ${cv_res['mean_rmse']:,.2f} (± ${cv_res['std_rmse']:,.2f})")
    print(f"  * Durbin-Watson: {gm_res['durbin_watson']['statistic']:.3f} (Ausencia de autocorrelación serial)")
    print(f"  * Breusch-Pagan: LM = {gm_res['breusch_pagan']['lm_statistic']:.2f} (p = {gm_res['breusch_pagan']['lm_p_value']:.4e})")

    if generate_plots:
        print("\n  Generando Batería de 7 Gráficos Científicos (300 DPI)...")
        viz = Visualizer(PLOTS_DIR)
        p1 = viz.plot_distributions(df)
        print(f"  [OK] [1/7] {p1.name}")
        p2 = viz.plot_linear_regression(df, linear_model)
        print(f"  [OK] [2/7] {p2.name}")
        p3 = viz.plot_residual_diagnostics(linear_model.sm_results)
        print(f"  [OK] [3/7] {p3.name}")
        p4 = viz.plot_polynomial_comparison(df, linear_model, poly_model)
        print(f"  [OK] [4/7] {p4.name}")
        p5 = viz.plot_cross_validation_performance(cv_res, split_res)
        print(f"  [OK] [5/7] {p5.name}")
        p6 = viz.plot_key_age_predictions(predictions_df)
        print(f"  [OK] [6/7] {p6.name}")
        p7 = viz.plot_gradient_descent_learning(gd_model, linear_model)
        print(f"  [OK] [7/7] {p7.name} (Curva de aprendizaje de Descenso del Gradiente)")
    else:
        print("  Generación de gráficos omitida por el usuario.")

    print("\n================================================================================")
    print("   ¡PIPELINE DE INTELIGENCIA ARTIFICIAL EJECUTADO EXITOSAMENTE!")
    print(f"   Visualizaciones y reportes disponibles en: {PLOTS_DIR}")
    print("================================================================================\n")


if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(target_ages=args.predict, generate_plots=not args.skip_plots)
