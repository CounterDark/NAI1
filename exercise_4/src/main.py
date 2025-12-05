"""
Project: Seeds and Mushroom Classification
Authors: Mateusz Anikiej and Aleksander Kunkowski

Description:
This script trains Decision Tree and SVM classifiers on two datasets:
1. Wheat Seeds
2. Mushrooms

It evaluates the models, saves the results to text files, and generates
visualizations of the decision boundaries. It also demonstrates predictions
on sample data.

Usage:
    python src/main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from typing import Tuple, List, Dict, Any, Optional, Union

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator

# Constants
RANDOM_STATE = 0


class ClassificationApp:
    """
    Main application class for training and evaluating classifiers on Seeds
    and Mushroom datasets.
    """

    def __init__(self):
        """Initialize the application with paths and configuration."""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, "..", "data")
        self.models_dir = os.path.join(self.base_dir, "..", "models")

        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)

    def load_seeds(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load the Seeds dataset.

        Returns:
            X: Feature matrix.
            y: Target vector.
            feature_names: List of feature names.
        """
        path = os.path.join(self.data_dir, "seeds_dataset.txt")
        df = pd.read_csv(path, header=None, sep="\t")

        if df.shape[1] == 8:
            X = np.array(df.iloc[:, :7].values)
            y = np.array(df.iloc[:, 7].values)
        else:
            raise ValueError("Unexpected seeds file shape: " + str(df.shape))

        feature_names = [
            "Area A",
            "Perimeter P",
            "Compactness C",
            "Length of kernel",
            "Width of kernel",
            "Asymmetry coefficient",
            "Length of kernel groove",
        ]
        return X, y, feature_names

    def load_mushrooms(self) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
        """
        Load the Mushroom dataset.

        Returns:
            X_df: DataFrame containing features.
            y: Target vector (0=edible, 1=poisonous).
            feature_names: List of feature names.
        """
        path = os.path.join(self.data_dir, "agaricus-lepiota.txt")
        column_names = [
            "class",
            "cap-shape",
            "cap-surface",
            "cap-color",
            "bruises",
            "odor",
            "gill-attachment",
            "gill-spacing",
            "gill-size",
            "gill-color",
            "stalk-shape",
            "stalk-root",
            "stalk-surface-above-ring",
            "stalk-surface-below-ring",
            "stalk-color-above-ring",
            "stalk-color-below-ring",
            "veil-type",
            "veil-color",
            "ring-number",
            "ring-type",
            "spore-print-color",
            "population",
            "habitat",
        ]

        df = pd.read_csv(path, header=None, names=column_names)

        # Change classification to {0,1}: edible=e -> 0, poisonous=p -> 1
        y = np.array((df["class"] == "p").astype(int).values)
        X_df = df.drop(columns=["class"])
        feature_names = list(X_df.columns)

        return X_df, y, feature_names

    def _load_model(
        self, key: str, dataset_name: str
    ) -> Tuple[Optional[BaseEstimator], Optional[Dict[str, Any]]]:
        """
        Helper to load a saved model and its parameters.

        Args:
            key: Identifier for the model.
            dataset_name: Name of the dataset.

        Returns:
            Tuple containing the loaded model (or None) and parameters (or None).
        """
        file_path = os.path.join(self.models_dir, f"{dataset_name}_{key}.joblib")
        model = None
        params = None

        if os.path.exists(file_path):
            print(f"Loading model from {file_path}")
            model = joblib.load(file_path)

            if key == "svc_full":
                params_file_path = os.path.join(
                    self.models_dir, f"{dataset_name}_{key}_grid_best_params.joblib"
                )
                if os.path.exists(params_file_path):
                    params = joblib.load(params_file_path)

        return model, params

    def _save_models(self, results: Dict[str, Any], dataset_name: str) -> None:
        """
        Helper to save trained models and parameters.

        Args:
            results: Dictionary containing training results and models.
            dataset_name: Name of the dataset.
        """
        for key, val in results.items():
            model = val["model"]
            file_path = os.path.join(self.models_dir, f"{dataset_name}_{key}.joblib")
            joblib.dump(model, file_path)
            print(f"Saved model: {file_path}")

            if key == "svc_full" and "grid_best_params" in val:
                params = val["grid_best_params"]
                file_path = os.path.join(
                    self.models_dir, f"{dataset_name}_{key}_grid_best_params.joblib"
                )
                joblib.dump(params, file_path)
                print(f"Saved grid params: {file_path}")

    def _print_and_save_report(self, report_str: str, out_path: str) -> None:
        """Print a report string and append it to a file."""
        print(report_str)
        with open(out_path, "a", encoding="utf8") as f:
            f.write(report_str + "\n\n")

    def train_eval_pipeline(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_names: List[str],
        dataset_name: str,
        categorical: bool = False,
    ) -> Dict[str, Any]:
        """
        Train and evaluate Decision Tree and SVM classifiers.

        Args:
            X: Features (numeric array or DataFrame).
            y: Target labels.
            feature_names: List of feature names.
            dataset_name: Name of the dataset (for saving files).
            categorical: Boolean indicating if data is categorical.

        Returns:
            Dictionary containing results, data splits, and feature importances.
        """
        results: Dict[str, Any] = {}

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
        )

        # Prepare data for Decision Tree
        if categorical:
            ordinal = ColumnTransformer(
                [
                    (
                        "ord",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value", unknown_value=-1
                        ),
                        list(range(X_train.shape[1])),
                    )
                ]
            )
            X_train_dt = ordinal.fit_transform(X_train)
            X_test_dt = ordinal.transform(X_test)
        else:
            X_train_dt = np.array(X_train)
            X_test_dt = np.array(X_test)

        # --- Decision Tree ---
        dt_key = "decision_tree_full"
        dt_model, _ = self._load_model(dt_key, dataset_name)

        if not dt_model:
            dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8)
            dt_model.fit(X_train_dt, y_train)

        y_pred_dt = dt_model.predict(X_test_dt)

        results[dt_key] = {
            "model": dt_model,
            "report": classification_report(y_test, y_pred_dt, digits=4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_dt),
            "accuracy": accuracy_score(y_test, y_pred_dt),
        }

        importances = dt_model.feature_importances_

        # --- SVM ---
        if categorical:
            # Pipeline: One-Hot Encode -> Scale -> SVM
            svc_pipeline = Pipeline(
                [
                    (
                        "ohe",
                        ColumnTransformer(
                            [
                                (
                                    "ohe",
                                    OneHotEncoder(
                                        handle_unknown="ignore", sparse_output=False
                                    ),
                                    list(range(X_train.shape[1])),
                                )
                            ]
                        ),
                    ),
                    ("scaler", StandardScaler(with_mean=False)),
                    ("svc", SVC(random_state=RANDOM_STATE, probability=False)),
                ]
            )
        else:
            # Pipeline: Scale -> SVM
            svc_pipeline = Pipeline(
                [("scaler", StandardScaler()), ("svc", SVC(random_state=RANDOM_STATE))]
            )

        svc_key = "svc_full"
        best_svc, svc_params = self._load_model(svc_key, dataset_name)

        if not best_svc:
            param_grid = [
                {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10]},
                {
                    "svc__kernel": ["rbf"],
                    "svc__C": [0.1, 1],
                    "svc__gamma": ["scale", "auto"],
                },
                {
                    "svc__kernel": ["poly"],
                    "svc__C": [1],
                    "svc__degree": [2, 3],
                    "svc__gamma": ["scale"],
                },
                {"svc__kernel": ["sigmoid"], "svc__C": [1, 5], "svc__gamma": ["scale"]},
            ]

            gs = GridSearchCV(
                svc_pipeline, param_grid, cv=4, scoring="accuracy", n_jobs=-1, verbose=0
            )
            gs.fit(X_train, y_train)
            svc_params = gs.best_params_
            best_svc = gs.best_estimator_

        y_pred_svc = best_svc.predict(X_test)

        results[svc_key] = {
            "model": best_svc,
            "report": classification_report(y_test, y_pred_svc, digits=4),
            "confusion_matrix": confusion_matrix(y_test, y_pred_svc),
            "accuracy": accuracy_score(y_test, y_pred_svc),
            "grid_best_params": svc_params,
        }

        # Reporting
        out_file = os.path.join(
            self.base_dir, f"{dataset_name}_classification_summary.txt"
        )
        if os.path.exists(out_file):
            os.remove(out_file)

        self._print_and_save_report(f"=== Results for {dataset_name} ===", out_file)

        dt_res = results[dt_key]
        self._print_and_save_report(
            f"Decision Tree accuracy: {dt_res['accuracy']:.4f}", out_file
        )
        self._print_and_save_report(
            f"Decision Tree report:\n{dt_res['report']}", out_file
        )
        self._print_and_save_report(
            f"Decision Tree confusion matrix:\n{dt_res['confusion_matrix']}", out_file
        )

        svc_res = results[svc_key]
        self._print_and_save_report(
            f"SVM accuracy: {svc_res['accuracy']:.4f}", out_file
        )
        self._print_and_save_report(
            f"SVM best params: {svc_res['grid_best_params']}", out_file
        )
        self._print_and_save_report(f"SVM report:\n{svc_res['report']}", out_file)
        self._print_and_save_report(
            f"SVM confusion matrix:\n{svc_res['confusion_matrix']}", out_file
        )

        return {
            "results": results,
            "importances": importances,
            "X_train": X_train,
            "X_test": X_test,
            "X_test_dt": X_test_dt,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
        }

    def visualize_boundary(
        self,
        clf: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        indices: List[int],
        feature_names: List[str],
        title: str,
    ) -> None:
        """
        Visualize decision boundary for a model trained on 2 features.

        Args:
            clf: Trained classifier.
            X: Feature matrix (2 columns).
            y: Labels.
            indices: Indices of the 2 features in the original dataset.
            feature_names: List of all feature names.
            title: Plot title.
        """
        min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
        min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
        mesh_step = 0.02

        xx, yy = np.meshgrid(
            np.arange(min_x, max_x, mesh_step), np.arange(min_y, max_y, mesh_step)
        )

        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = clf.predict(grid)
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(6, 5))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="Paired", s=40)
        plt.xlabel(feature_names[indices[0]])
        plt.ylabel(feature_names[indices[1]])
        plt.title(title)
        plt.show()

    def predict_sample(
        self,
        model: BaseEstimator,
        X_sample: Union[np.ndarray, pd.DataFrame],
        y_true: Union[int, float],
        dataset_name: str,
        model_name: str,
    ) -> None:
        """
        Predict class for a single sample and save result.

        Args:
            model: Trained model.
            X_sample: Sample features (reshaped for prediction).
            y_true: True class label.
            dataset_name: Name of the dataset.
            model_name: Name of the model.
        """
        prediction = model.predict(X_sample)[0]

        output = (
            f"--- Sample Prediction ({dataset_name} - {model_name}) ---\n"
            f"Features: {X_sample}\n"
            f"True Class: {y_true}\n"
            f"Predicted Class: {prediction}\n"
        )
        print(output)

        out_path = os.path.join(self.base_dir, "sample_predictions.txt")
        with open(out_path, "a", encoding="utf8") as f:
            f.write(output + "\n")

    def run(self) -> None:
        """Execute the complete classification pipeline."""

        # Clear previous sample predictions
        sample_pred_path = os.path.join(self.base_dir, "sample_predictions.txt")
        if os.path.exists(sample_pred_path):
            os.remove(sample_pred_path)

        # --- Seeds Dataset ---
        X_seeds, y_seeds, fn_seeds = self.load_seeds()
        res_seeds = self.train_eval_pipeline(
            X_seeds, y_seeds, fn_seeds, "seeds", categorical=False
        )

        self._save_models(res_seeds["results"], "seeds")

        # Visualization (Top 2 Features)
        top2_seeds = list(np.argsort(res_seeds["importances"])[-2:][::-1])
        print(f"Seeds top 2 features: {[fn_seeds[i] for i in top2_seeds]}")

        # Prepare data for visualization (train on only top 2 features)
        X_train_s = res_seeds["X_train"]
        X_test_s = res_seeds["X_test"]
        y_train_s = res_seeds["y_train"]
        y_test_s = res_seeds["y_test"]

        X2_train = X_train_s[:, top2_seeds]
        X2_test = X_test_s[:, top2_seeds]

        # Train simple models for visualization
        dt2 = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
        dt2.fit(X2_train, y_train_s)
        self.visualize_boundary(
            dt2,
            np.vstack([X2_train, X2_test]),
            np.hstack([y_train_s, y_test_s]),
            top2_seeds,
            fn_seeds,
            "Seeds DT (Top 2 Features)",
        )

        svc2 = make_pipeline(
            StandardScaler(), SVC(kernel="rbf", random_state=RANDOM_STATE)
        )
        svc2.fit(X2_train, y_train_s)
        self.visualize_boundary(
            svc2,
            np.vstack([X2_train, X2_test]),
            np.hstack([y_train_s, y_test_s]),
            top2_seeds,
            fn_seeds,
            "Seeds SVM (Top 2 Features)",
        )

        # Sample Prediction
        sample_idx = 0
        sample_data = X_test_s[sample_idx].reshape(1, -1)
        self.predict_sample(
            res_seeds["results"]["decision_tree_full"]["model"],
            sample_data,
            y_test_s[sample_idx],
            "Seeds",
            "Decision Tree",
        )

        # --- Mushroom Dataset ---
        X_mush, y_mush, fn_mush = self.load_mushrooms()
        res_mush = self.train_eval_pipeline(
            X_mush, y_mush, fn_mush, "mushroom", categorical=True
        )

        self._save_models(res_mush["results"], "mushroom")

        # Visualization (Top 2 Features)
        top2_mush = list(np.argsort(res_mush["importances"])[-2:][::-1])
        print(f"Mushroom top 2 features: {[fn_mush[i] for i in top2_mush]}")

        # Prepare for visualization (Ordinal Encode categorical features)
        # Use original DF subset
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

        X_mush_2cols = oe.fit_transform(X_mush.iloc[:, top2_mush])

        # Split the 2-col data
        Xm_train2, Xm_test2, ym_train2, ym_test2 = train_test_split(
            X_mush_2cols,
            y_mush,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y_mush,
        )

        dt_m2 = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
        dt_m2.fit(Xm_train2, ym_train2)
        self.visualize_boundary(
            dt_m2,
            np.vstack([Xm_train2, Xm_test2]),
            np.hstack([ym_train2, ym_test2]),
            [0, 1],
            [fn_mush[i] for i in top2_mush],
            "Mushroom DT (Top 2 Features)",
        )

        svc_m2 = make_pipeline(
            StandardScaler(), SVC(kernel="rbf", random_state=RANDOM_STATE)
        )
        svc_m2.fit(Xm_train2, ym_train2)
        self.visualize_boundary(
            svc_m2,
            np.vstack([Xm_train2, Xm_test2]),
            np.hstack([ym_train2, ym_test2]),
            [0, 1],
            [fn_mush[i] for i in top2_mush],
            "Mushroom SVM (Top 2 Features)",
        )

        # Sample Prediction
        sample_idx = 0
        if "X_test_dt" in res_mush:
            sample_data = res_mush["X_test_dt"][sample_idx].reshape(1, -1)
        else:
            sample_data = res_mush["X_test"].iloc[[sample_idx]].values

        true_label = res_mush["y_test"][sample_idx]

        self.predict_sample(
            res_mush["results"]["decision_tree_full"]["model"],
            sample_data,
            true_label,
            "Mushroom",
            "Decision Tree",
        )

        # Summary File
        self._generate_summary(res_seeds["results"], res_mush["results"])

    def _generate_summary(
        self, seed_results: Dict[str, Any], mush_results: Dict[str, Any]
    ) -> None:
        """Generate a summary file for SVM kernels."""
        summary_lines = [
            "SVM Kernel summary (simple conclusions based on GridSearch metrics):\n",
            "Seeds dataset: best SVM params: "
            f"{seed_results['svc_full'].get('grid_best_params')}\n",
            f"Seeds SVM accuracy: {seed_results['svc_full']['accuracy']:.4f}\n",
            "Mushroom dataset: best SVM params: "
            f"{mush_results['svc_full'].get('grid_best_params')}\n",
            f"Mushroom SVM accuracy: {mush_results['svc_full']['accuracy']:.4f}\n",
            "\nShort notes on kernels:",
            "- linear: good for linearly separable data; simple; fewer hyperparams.",
            "- rbf: flexible; maps to infinite-dim; often good default.",
            "- poly: useful if polynomial relationships; degree controls complexity.",
            "- sigmoid: rarely best; behaves like NN activation; use sparingly.",
        ]

        out_path = os.path.join(self.base_dir, "svm_kernel_summary.txt")
        with open(out_path, "w", encoding="utf8") as f:
            f.write("\n".join(summary_lines))
        print("\nSaved svm_kernel_summary.txt with short conclusions.")


def main():
    """Entry point for the application."""
    app = ClassificationApp()
    app.run()


if __name__ == "__main__":
    main()
