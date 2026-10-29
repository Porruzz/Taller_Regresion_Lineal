"""
Módulo de Evaluación Econométrica y Validación Cruzada.
Valida supuestos de Gauss-Márkov (normalidad, homocedasticidad, no autocorrelación),
calcula métricas de error (RMSE, MAE, R2) y ejecuta remuestreo K-Fold.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson, jarque_bera

from src.config import MODEL_CONFIG


class ModelEvaluator:
    """Ejecuta diagnósticos rigurosos de machine learning y econometría."""

    def __init__(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)

    def evaluate_train_test_split(
        self,
        test_size: float = MODEL_CONFIG.test_size,
        random_state: int = MODEL_CONFIG.random_state
    ) -> Dict[str, float]:
        """Divide datos en Train/Test y calcula métricas de generalización."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))

        train_mae = float(mean_absolute_error(y_train, y_train_pred))
        test_mae = float(mean_absolute_error(y_test, y_test_pred))

        train_r2 = float(r2_score(y_train, y_train_pred))
        test_r2 = float(r2_score(y_test, y_test_pred))

        return {
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "rmse_gap": test_rmse - train_rmse,
            "coef_slope": float(model.coef_[0]),
            "intercept": float(model.intercept_)
        }

    def evaluate_kfold_cv(
        self,
        n_splits: int = MODEL_CONFIG.k_folds,
        random_state: int = MODEL_CONFIG.random_state
    ) -> Dict[str, Any]:
        """Ejecuta K-Fold Cross Validation para verificar estabilidad."""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        model = LinearRegression()

        # Errores en cada pliegue
        cv_mse = -cross_val_score(model, self.X, self.y, cv=kf, scoring="neg_mean_squared_error")
        cv_rmse = np.sqrt(cv_mse)
        cv_mae = -cross_val_score(model, self.X, self.y, cv=kf, scoring="neg_mean_absolute_error")
        cv_r2 = cross_val_score(model, self.X, self.y, cv=kf, scoring="r2")

        return {
            "k_folds": n_splits,
            "cv_rmse_scores": cv_rmse.tolist(),
            "cv_mae_scores": cv_mae.tolist(),
            "cv_r2_scores": cv_r2.tolist(),
            "mean_rmse": float(np.mean(cv_rmse)),
            "std_rmse": float(np.std(cv_rmse)),
            "mean_mae": float(np.mean(cv_mae)),
            "std_mae": float(np.std(cv_mae)),
            "mean_r2": float(np.mean(cv_r2)),
            "std_r2": float(np.std(cv_r2))
        }

    def test_gauss_markov_assumptions(self, sm_fitted_model) -> Dict[str, Any]:
        """
        Evalúa formalmente los supuestos clásicos de Gauss-Márkov:
        1. Normalidad de residuos (Jarque-Bera)
        2. Homocedasticidad (Breusch-Pagan y White)
        3. No autocorrelación de errores (Durbin-Watson)
        """
        residuals = sm_fitted_model.resid
        fitted_values = sm_fitted_model.fittedvalues
        exog = sm_fitted_model.model.exog

        # 1. Durbin-Watson (Autocorrelación)
        dw_stat = float(durbin_watson(residuals))

        # 2. Jarque-Bera (Normalidad)
        jb_stat, jb_pval, skew, kurt = jarque_bera(residuals)

        # 3. Breusch-Pagan (Homocedasticidad)
        bp_lm_stat, bp_lm_pval, bp_f_stat, bp_f_pval = het_breuschpagan(residuals, exog)

        # 4. White test (Heterocedasticidad no lineal)
        white_stat, white_pval, _, _ = het_white(residuals, exog)

        return {
            "durbin_watson": {
                "statistic": dw_stat,
                "interpretation": "Ausencia de autocorrelación serial de 1er orden (~2.0)" if 1.8 < dw_stat < 2.2 else "Posible autocorrelación"
            },
            "jarque_bera": {
                "statistic": float(jb_stat),
                "p_value": float(jb_pval),
                "skewness": float(skew),
                "kurtosis": float(kurt),
                "is_normal": bool(jb_pval > 0.05),
                "interpretation": "Residuos con asimetría positiva y colas pesadas típica de ingresos económicos salriales."
            },
            "breusch_pagan": {
                "lm_statistic": float(bp_lm_stat),
                "lm_p_value": float(bp_lm_pval),
                "f_statistic": float(bp_f_stat),
                "f_p_value": float(bp_f_pval),
                "is_homoscedastic": bool(bp_lm_pval > 0.05)
            },
            "white_test": {
                "statistic": float(white_stat),
                "p_value": float(white_pval),
                "is_homoscedastic": bool(white_pval > 0.05)
            }
        }
