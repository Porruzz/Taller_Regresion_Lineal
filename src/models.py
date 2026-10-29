"""
Módulo de Modelación Econométrica y de Machine Learning (Inteligencia Artificial).
Implementa:
1. Solución Analítica OLS (Ecuación Normal y Mínimos Cuadrados Ordinarios).
2. Algoritmo de Aprendizaje Supervisado: Descenso del Gradiente (Batch Gradient Descent from scratch).
3. Intervalos de Confianza y Predicción Estadísticos.
4. Modelo Polinomial Cuadrático (Curva de Ciclo de Vida Minceriana).
5. Modelos Regularizados de IA (Ridge L2 y Lasso L1) para control de Sobreajuste (Overfitting).
6. Regresión Multivariable de Control.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler


@dataclass
class RegressionResults:
    """Contenedor de resultados analíticos del modelo de regresión."""
    intercept: float
    slope: float
    intercept_se: float
    slope_se: float
    t_stat_slope: float
    p_val_slope: float
    t_stat_intercept: float
    p_val_intercept: float
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_p_value: float
    residual_std_error: float
    n_obs: int
    df_residuals: int
    equation_str: str


@dataclass
class GradientDescentHistory:
    """Historial de entrenamiento del algoritmo de Descenso del Gradiente."""
    epochs: int
    learning_rate: float
    final_loss: float
    loss_history: List[float] = field(default_factory=list)
    theta0_history: List[float] = field(default_factory=list)
    theta1_history: List[float] = field(default_factory=list)
    learned_intercept: float = 0.0
    learned_slope: float = 0.0


class SimpleLinearModel:
    """
    Modelo de Regresión Lineal OLS analítico (Ecuación Normal).
    Relaciona la Edad (X) con el Ingreso Mensual (Y).
    """

    def __init__(self):
        self.fitted = False
        self.results: RegressionResults | None = None
        self.sm_model = None
        self.sm_results = None
        self.sklearn_model = LinearRegression()
        
        # Parámetros analíticos para intervalos
        self.x_mean: float = 0.0
        self.y_mean: float = 0.0
        self.ss_xx: float = 0.0
        self.s_e: float = 0.0
        self.n: int = 0

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "SimpleLinearModel":
        """Ajusta el modelo analítica y estadísticamente."""
        x_vals = np.asarray(X).ravel()
        y_vals = np.asarray(y).ravel()

        self.n = len(x_vals)
        self.x_mean = float(np.mean(x_vals))
        self.y_mean = float(np.mean(y_vals))

        # Sumas de cuadrados
        diff_x = x_vals - self.x_mean
        diff_y = y_vals - self.y_mean
        self.ss_xx = float(np.sum(diff_x ** 2))
        ss_yy = float(np.sum(diff_y ** 2))
        ss_xy = float(np.sum(diff_x * diff_y))

        # Coeficientes OLS
        slope = ss_xy / self.ss_xx
        intercept = self.y_mean - slope * self.x_mean

        # Residuos y Varianza Residual
        y_pred = intercept + slope * x_vals
        residuals = y_vals - y_pred
        sse = float(np.sum(residuals ** 2))
        df_res = self.n - 2
        self.s_e = np.sqrt(sse / df_res)

        # Errores estándar de los coeficientes
        slope_se = self.s_e / np.sqrt(self.ss_xx)
        intercept_se = self.s_e * np.sqrt((1.0 / self.n) + (self.x_mean ** 2 / self.ss_xx))

        # Estadísticos t y p-values
        t_slope = slope / slope_se
        p_slope = 2 * (1 - stats.t.cdf(np.abs(t_slope), df=df_res))

        t_intercept = intercept / intercept_se
        p_intercept = 2 * (1 - stats.t.cdf(np.abs(t_intercept), df=df_res))

        # R-cuadrado y F-statistic
        ssr = ss_yy - sse
        r_squared = ssr / ss_yy
        adj_r_squared = 1 - ((1 - r_squared) * (self.n - 1) / df_res)
        f_stat = (ssr / 1) / (sse / df_res)
        f_p_val = 1 - stats.f.cdf(f_stat, dfn=1, dfd=df_res)

        eq_str = f"Ingreso = {intercept:,.2f} + {slope:.2f} * Edad"

        self.results = RegressionResults(
            intercept=intercept,
            slope=slope,
            intercept_se=intercept_se,
            slope_se=slope_se,
            t_stat_slope=t_slope,
            p_val_slope=p_slope,
            t_stat_intercept=t_intercept,
            p_val_intercept=p_intercept,
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            f_statistic=f_stat,
            f_p_value=f_p_val,
            residual_std_error=self.s_e,
            n_obs=self.n,
            df_residuals=df_res,
            equation_str=eq_str
        )

        # Ajuste Statsmodels para diagnósticos avanzados
        X_sm = sm.add_constant(x_vals)
        self.sm_model = sm.OLS(y_vals, X_sm)
        self.sm_results = self.sm_model.fit()

        # Ajuste Scikit-Learn
        self.sklearn_model.fit(x_vals.reshape(-1, 1), y_vals)

        self.fitted = True
        return self

    def predict(self, ages: List[float] | np.ndarray) -> np.ndarray:
        """Predice el ingreso mensual esperado para un arreglo de edades."""
        if not self.fitted:
            raise RuntimeError("El modelo debe ser entrenado antes de predecir.")
        ages_arr = np.asarray(ages, dtype=float)
        return self.results.intercept + self.results.slope * ages_arr

    def predict_with_intervals(
        self,
        ages: List[float] | np.ndarray,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Calcula predicciones puntuales, Intervalo de Confianza para la Media (CI)
        e Intervalo de Predicción para Nuevos Individuos (PI) al nivel (1 - alpha).
        """
        if not self.fitted:
            raise RuntimeError("El modelo debe ser entrenado antes de generar intervalos.")

        ages_arr = np.asarray(ages, dtype=float)
        y_pred = self.predict(ages_arr)
        
        t_crit = stats.t.ppf(1 - alpha / 2, df=self.results.df_residuals)

        # Término de dispersión respecto a la media
        term_dist = (ages_arr - self.x_mean) ** 2 / self.ss_xx

        # Error estándar de la media esperada E[Y|X0]
        se_mean = self.s_e * np.sqrt((1.0 / self.n) + term_dist)
        ci_lower = y_pred - t_crit * se_mean
        ci_upper = y_pred + t_crit * se_mean

        # Error estándar de una predicción individual Y_new|X0
        se_pred = self.s_e * np.sqrt(1.0 + (1.0 / self.n) + term_dist)
        pi_lower = y_pred - t_crit * se_pred
        pi_upper = y_pred + t_crit * se_pred

        return pd.DataFrame({
            "edad": ages_arr,
            "prediccion_ingreso": np.round(y_pred, 2),
            "ic_media_inferior": np.round(ci_lower, 2),
            "ic_media_superior": np.round(ci_upper, 2),
            "ip_individual_inferior": np.round(np.clip(pi_lower, 0, None), 2),
            "ip_individual_superior": np.round(pi_upper, 2)
        })


class GradientDescentRegressor:
    """
    Algoritmo de Inteligencia Artificial / Aprendizaje Supervisado:
    Descenso del Gradiente por Lotes (Batch Gradient Descent) implementado desde cero.
    Minimiza la Función de Costo J(theta) = 1/(2m) * sum((h(x) - y)^2).
    """

    def __init__(self, learning_rate: float = 0.05, epochs: int = 500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.history: GradientDescentHistory | None = None
        self.fitted = False
        
        # Parámetros aprendidos (en espacio estandarizado y original)
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.intercept_ = 0.0
        self.slope_ = 0.0
        
        # Parámetros de escalamiento
        self.mu_x = 0.0
        self.std_x = 1.0

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "GradientDescentRegressor":
        x_raw = np.asarray(X).ravel()
        y_raw = np.asarray(y).ravel()
        m = len(y_raw)

        # 1. Estandarización de características (Z-score Scaling) para convergencia óptima
        self.mu_x = float(np.mean(x_raw))
        self.std_x = float(np.std(x_raw))
        z = (x_raw - self.mu_x) / self.std_x

        # Inicialización de pesos sinápticos / parámetros theta
        self.theta0 = 0.0
        self.theta1 = 0.0

        loss_hist = []
        t0_hist = []
        t1_hist = []

        for epoch in range(self.epochs):
            # Hipótesis lineal: h_theta(z) = theta0 + theta1 * z
            y_pred = self.theta0 + self.theta1 * z
            error = y_pred - y_raw
            
            # Función de Costo (Mean Squared Error Loss)
            loss = (1.0 / (2.0 * m)) * np.sum(error ** 2)
            loss_hist.append(float(loss))
            t0_hist.append(float(self.theta0))
            t1_hist.append(float(self.theta1))

            # Gradientes parciales: dJ/dtheta
            grad0 = (1.0 / m) * np.sum(error)
            grad1 = (1.0 / m) * np.sum(error * z)

            # Actualización iterativa de pesos
            self.theta0 -= self.learning_rate * grad0
            self.theta1 -= self.learning_rate * grad1

        # Transformación al espacio original de variables (des-estandarización)
        # h(x) = theta0 + theta1 * ((x - mu)/std) = (theta0 - theta1*mu/std) + (theta1/std) * x
        self.slope_ = self.theta1 / self.std_x
        self.intercept_ = self.theta0 - self.slope_ * self.mu_x

        self.history = GradientDescentHistory(
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            final_loss=loss_hist[-1],
            loss_history=loss_hist,
            theta0_history=t0_hist,
            theta1_history=t1_hist,
            learned_intercept=self.intercept_,
            learned_slope=self.slope_
        )

        self.fitted = True
        return self

    def predict(self, ages: List[float] | np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("El modelo debe ser entrenado antes de predecir.")
        ages_arr = np.asarray(ages, dtype=float)
        return self.intercept_ + self.slope_ * ages_arr


class PolynomialModel:
    """
    Modelo Polinomial Cuadrático (Grado 2):
    Ingreso = beta_0 + beta_1 * Edad + beta_2 * Edad^2
    Modela la concavidad del ciclo de vida y estima la edad pico de ingresos.
    """

    def __init__(self):
        self.fitted = False
        self.results = None
        self.peak_age: float = 0.0
        self.peak_income: float = 0.0

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "PolynomialModel":
        x_vals = np.asarray(X).ravel()
        y_vals = np.asarray(y).ravel()

        X_mat = np.column_stack([np.ones_like(x_vals), x_vals, x_vals ** 2])
        sm_model = sm.OLS(y_vals, X_mat)
        self.results = sm_model.fit()

        beta0, beta1, beta2 = self.results.params
        
        # Edad pico en el vértice de la parábola: -b / (2a)
        if beta2 < 0:
            self.peak_age = float(-beta1 / (2 * beta2))
            self.peak_income = float(beta0 + beta1 * self.peak_age + beta2 * (self.peak_age ** 2))
        else:
            self.peak_age = np.nan
            self.peak_income = np.nan

        self.fitted = True
        return self

    def predict(self, ages: List[float] | np.ndarray) -> np.ndarray:
        ages_arr = np.asarray(ages, dtype=float)
        beta0, beta1, beta2 = self.results.params
        return beta0 + beta1 * ages_arr + beta2 * (ages_arr ** 2)


class RegularizedMLModels:
    """
    Modelos de Aprendizaje Supervisado con Regularización (Semana 10 de IA):
    Ridge (L2) y Lasso (L1) para control de sobreajuste y reducción de varianza.
    """

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.ridge_model = Ridge(alpha=alpha)
        self.lasso_model = Lasso(alpha=alpha)
        self.fitted = False

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "RegularizedMLModels":
        x_arr = np.asarray(X).reshape(-1, 1)
        y_arr = np.asarray(y).ravel()

        # Construir matriz polinomial de grado 2 para evaluar regularización
        X_poly = np.column_stack([x_arr, x_arr ** 2])
        X_scaled = self.scaler.fit_transform(X_poly)

        self.ridge_model.fit(X_scaled, y_arr)
        self.lasso_model.fit(X_scaled, y_arr)
        self.fitted = True
        return self

    def get_coefficients_comparison(self) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Modelos no entrenados.")
        return pd.DataFrame({
            "Característica": ["Edad (Lineal)", "Edad^2 (Cuadrático)"],
            "Ridge (L2)": self.ridge_model.coef_,
            "Lasso (L1)": self.lasso_model.coef_
        })


class MultivariateModel:
    """
    Modelo de Regresión Múltiple:
    Ingreso = beta_0 + beta_1 * Edad + beta_2 * Años_Educacion + beta_3 * Sexo_Mujer
    Explica el sesgo de variable omitida en el modelo simple.
    """

    def __init__(self):
        self.fitted = False
        self.results = None

    def fit(self, df: pd.DataFrame) -> "MultivariateModel":
        import statsmodels.formula.api as smf
        formula = "ingreso_mensual ~ edad + anios_esc + C(sex, Treatment(reference='Hombre'))"
        self.results = smf.ols(formula=formula, data=df).fit()
        self.fitted = True
        return self
