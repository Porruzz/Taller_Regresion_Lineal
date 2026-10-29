"""
================================================================================
TALLER DE INTELIGENCIA ARTIFICIAL: REGRESIÓN LINEAL Y APRENDIZAJE SUPERVISADO
Asignatura: Inteligencia Artificial (SIST5036)
Tema: Modelo de Regresión Lineal, Descenso del Gradiente (Batch GD),
      Regresión Polinomial y Regularización L1/L2 (Ridge y Lasso).
================================================================================
Fachada Principal (Facade Pattern - GoF):
Coordina la ejecución de los módulos desacoplados bajo el paquete 'src',
proporcionando una interfaz unificada para consola, predicción de ingresos y
exportación de gráficos científicos a 300 DPI.
================================================================================
"""

import sys
from main import run_pipeline, parse_arguments

if __name__ == "__main__":
    args = parse_arguments()
    run_pipeline(target_ages=args.predict, generate_plots=not args.skip_plots)
