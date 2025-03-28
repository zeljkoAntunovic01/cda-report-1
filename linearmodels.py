from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import root_mean_squared_error
import numpy as np
import pandas as pd
from data import DataProcessor
from visualizer import visualize_linear_models_estimates

TRAIN_DATA_DIR = "data/case1Data.csv"
XNEW_DATA_DIR = "data/case1Data_Xnew.csv"
RESULTS_DIR = "results/linearmodels/"

def nested_cv_ols(df: pd.DataFrame, target_col='y', outer_cv_folds=10):
    outer_cv = KFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
    rmse_scores = []
    processor = DataProcessor()

    for train_idx, test_idx in outer_cv.split(df):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()

        # Separate target
        y_train = df_train[target_col]
        y_test = df_test[target_col]
        X_train_raw = df_train.drop(columns=[target_col])
        X_test_raw = df_test.drop(columns=[target_col])

        # Process
        X_train = processor.preprocess_train_data(X_train_raw)
        X_test = processor.preprocess_test_data(X_test_raw)

        # Train and evaluate
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        rmse_scores.append(rmse)
        processor.reset()

    avg_rmse = np.mean(rmse_scores)
    print(f"OLS | Estimated RMSE: {avg_rmse:.4f}")
    return avg_rmse, rmse_scores

def nested_cv_ridge(df: pd.DataFrame, target_col='y', outer_cv_folds=10, inner_cv_folds=5, alphas=None):
    if alphas is None:
        alphas = np.logspace(-3, 3, 30)

    outer_cv = KFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
    rmse_scores = []
    selected_alphas = []
    processor = DataProcessor()

    for train_idx, test_idx in outer_cv.split(df):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()

        y_train = df_train[target_col]
        y_test = df_test[target_col]
        X_train_raw = df_train.drop(columns=[target_col])
        X_test_raw = df_test.drop(columns=[target_col])

        X_train = processor.preprocess_train_data(X_train_raw)
        X_test = processor.preprocess_test_data(X_test_raw)

        grid = GridSearchCV(Ridge(), {'alpha': alphas}, cv=inner_cv_folds, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        selected_alphas.append(best_model.alpha)

        y_pred = best_model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        rmse_scores.append(rmse)
        processor.reset()

    avg_rmse = np.mean(rmse_scores)
    final_alpha = np.mean(selected_alphas)

    print(f"Ridge | Estimated RMSE: {avg_rmse:.4f}")
    print(f"Ridge | Final selected alpha (mean across folds): {final_alpha:.5f}")

    return avg_rmse, rmse_scores, final_alpha

def nested_cv_lasso(df: pd.DataFrame, target_col='y', outer_cv_folds=10, inner_cv_folds=5, alphas=None):
    if alphas is None:
        alphas = np.logspace(-3, 3, 30)

    outer_cv = KFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
    rmse_scores = []
    selected_alphas = []
    processor = DataProcessor()

    for train_idx, test_idx in outer_cv.split(df):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()

        y_train = df_train[target_col]
        y_test = df_test[target_col]
        X_train_raw = df_train.drop(columns=[target_col])
        X_test_raw = df_test.drop(columns=[target_col])

        X_train = processor.preprocess_train_data(X_train_raw)
        X_test = processor.preprocess_test_data(X_test_raw)

        grid = GridSearchCV(Lasso(max_iter=10000, tol=1e-2), {'alpha': alphas}, cv=inner_cv_folds, scoring='neg_mean_squared_error')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        selected_alphas.append(best_model.alpha)

        y_pred = best_model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        rmse_scores.append(rmse)
        processor.reset()

    avg_rmse = np.mean(rmse_scores)
    final_alpha = np.mean(selected_alphas)

    print(f"Lasso | Estimated RMSE: {avg_rmse:.4f}")
    print(f"Lasso | Final selected alpha (mean across folds): {final_alpha:.5f}")

    return avg_rmse, rmse_scores, final_alpha

def nested_cv_elasticnet(df: pd.DataFrame, target_col='y', outer_cv_folds=10, inner_cv_folds=5,
                         alphas=None, l1_ratios=None):
    if alphas is None:
        alphas = np.logspace(-3, 3, 10)
    if l1_ratios is None:
        l1_ratios = [0.1, 0.5, 0.9]

    outer_cv = KFold(n_splits=outer_cv_folds, shuffle=True, random_state=42)
    rmse_scores = []
    selected_alphas = []
    selected_l1_ratios = []
    processor = DataProcessor()

    for train_idx, test_idx in outer_cv.split(df):
        df_train = df.iloc[train_idx].copy()
        df_test = df.iloc[test_idx].copy()

        y_train = df_train[target_col]
        y_test = df_test[target_col]
        X_train_raw = df_train.drop(columns=[target_col])
        X_test_raw = df_test.drop(columns=[target_col])

        X_train = processor.preprocess_train_data(X_train_raw)
        X_test = processor.preprocess_test_data(X_test_raw)

        grid = GridSearchCV(
            ElasticNet(max_iter=10000, tol=1e-2),
            {'alpha': alphas, 'l1_ratio': l1_ratios},
            cv=inner_cv_folds,
            scoring='neg_mean_squared_error'
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        selected_alphas.append(best_model.alpha)
        selected_l1_ratios.append(best_model.l1_ratio)

        y_pred = best_model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        rmse_scores.append(rmse)
        processor.reset()

    avg_rmse = np.mean(rmse_scores)
    final_alpha = np.mean(selected_alphas)
    final_l1_ratio = np.mean(selected_l1_ratios)

    print(f"ElasticNet | Estimated RMSE: {avg_rmse:.4f}")
    print(f"ElasticNet | Final selected alpha (mean): {final_alpha:.5f}")
    print(f"ElasticNet | Final selected l1_ratio (mean): {final_l1_ratio:.3f}")

    return avg_rmse, rmse_scores, final_alpha, final_l1_ratio
 

if __name__ == "__main__":
    df = pd.read_csv(TRAIN_DATA_DIR)
    target_col = "y"
    outer_cv_folds = 10
    inner_cv_folds = 5
    alphas = np.logspace(-3, 3, 30)
    l1_ratios = [0.1, 0.5, 0.9]
    linear_rmses = dict()

    # Nested CV for OLS
    ols_rmse, ols_scores = nested_cv_ols(df, target_col, outer_cv_folds)
    linear_rmses.update({"OLS": ols_rmse})

    # Nested CV for Ridge
    ridge_rmse, ridge_scores, ridge_alpha = nested_cv_ridge(df, target_col, outer_cv_folds, inner_cv_folds, alphas)
    linear_rmses.update({"Ridge": ridge_rmse})

    # Nested CV for Lasso
    lasso_rmse, lasso_scores, lasso_alpha = nested_cv_lasso(df, target_col, outer_cv_folds, inner_cv_folds, alphas)
    linear_rmses.update({"Lasso": lasso_rmse})

    # Nested CV for ElasticNet
    elasticnet_rmse, elasticnet_scores, elasticnet_alpha, elasticnet_l1_ratio = nested_cv_elasticnet(
        df, target_col, outer_cv_folds, inner_cv_folds, alphas, l1_ratios
    )
    linear_rmses.update({"ElasticNet": elasticnet_rmse})

    # Find the best model
    best_model = min(linear_rmses, key=linear_rmses.get)

    # Save RMSE summary
    summary_data = [
        ["OLS", ols_rmse, None, None],
        ["Ridge", ridge_rmse, ridge_alpha, None],
        ["Lasso", lasso_rmse, lasso_alpha, None],
        ["ElasticNet", elasticnet_rmse, elasticnet_alpha, elasticnet_l1_ratio]
    ]

    summary_df = pd.DataFrame(summary_data, columns=["Model", "RMSE", "Alpha", "L1_Ratio"])
    summary_df.to_csv(f"{RESULTS_DIR}rmse_estimates.csv", index=False)
    print(f"Saved model performance summary to {RESULTS_DIR}rmse_estimates.csv")

    # Refit best model on full dataset
    df_full = pd.read_csv(TRAIN_DATA_DIR)
    X_full_raw = df_full.drop(columns=["y"])
    y_full = df_full["y"]

    processor = DataProcessor()
    X_full = processor.preprocess_train_data(X_full_raw)

    if best_model == "OLS":
        final_model = LinearRegression()
    elif best_model == "Ridge":
        final_model = Ridge(alpha=ridge_alpha)
    elif best_model == "Lasso":
        final_model = Lasso(alpha=lasso_alpha, max_iter=10000)
    elif best_model == "ElasticNet":
        final_model = ElasticNet(alpha=elasticnet_alpha, l1_ratio=elasticnet_l1_ratio, max_iter=10000)
    else:
        raise ValueError("Unknown best model")

    final_model.fit(X_full, y_full)
    print(f"Trained final {best_model} model on full dataset")

    # Load and preprocess x_new, make predictions
    xnew_df = pd.read_csv(XNEW_DATA_DIR)
    X_new = processor.preprocess_test_data(xnew_df)

    y_new_pred = final_model.predict(X_new)

    # Save predictions
    pred_df = pd.DataFrame({"Prediction": y_new_pred})
    pred_df.to_csv(f"{RESULTS_DIR}{best_model}_model_predictions.csv", index=False)
    print(f"Saved predictions of {best_model} model to {RESULTS_DIR}{best_model}_model_predictions.csv")

    visualize_linear_models_estimates(f"{RESULTS_DIR}rmse_estimates.csv")