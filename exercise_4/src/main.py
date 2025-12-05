import numpy as np
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt
import os
import joblib  # type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV  # type: ignore
from sklearn.tree import DecisionTreeClassifier  # type: ignore
from sklearn.svm import SVC  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # type: ignore # noqa: E501
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder  # type: ignore # noqa: E501
from sklearn.compose import ColumnTransformer  # type: ignore
from sklearn.pipeline import Pipeline, make_pipeline  # type: ignore

RANDOM_STATE = 0
MODELS_DIR = "../models"


# -------------------------
# 1) Load datasets
# -------------------------
def load_seeds(path="../data/seeds_dataset.txt"):
    """
    Expect 7 numeric features + class label (last column).
    UCI seeds file: whitespace separated, last column is class 1/2/3
    """
    df = pd.read_csv(path, header=None, sep="\t")
    # If file has 8 columns (7 features + label)
    if df.shape[1] == 8:
        X = df.iloc[:, :7].values
        y = df.iloc[:, 7].values
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


def load_mushroom(path="../data/agaricus-lepiota.txt"):
    """
    Mushroom dataset: first column is class (e=edible, p=poisonous), remaining
    are categorical features. We return DataFrame (X_df) and y (0/1) and list
    of original feature names.
    """
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

    # Import file
    df = pd.read_csv(path, header=None, names=column_names)

    # Change classification to {0,1}:
    # edible=e -> 0
    # poisonous=p -> 1
    y = (df["class"] == "p").astype(int).values

    # Remove class column
    X_df = df.drop(columns=["class"])

    feature_names = list(X_df.columns)

    return X_df, y, feature_names


# -------------------------
# 2) Utilities for reporting
# -------------------------
def print_and_save_report(report_str, out_path):
    print(report_str)
    with open(out_path, "a", encoding="utf8") as f:
        f.write(report_str + "\n\n")


# -------------------------
# 3) Train & evaluate generic pipeline
# -------------------------
def train_and_evaluate_classifiers(
    X, y, feature_names, dataset_name="dataset", categorical=False
):
    """
    X: np.array (numeric) or pd.DataFrame (categorical for mushroom)
    y: labels
    feature_names: list of names
    categorical: True if X is DataFrame with categorical features (mushroom)
    """
    results = {}

    # Split once (stratify if possible)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # Pipelines:
    # Decision Tree: we can feed numeric arrays. For categorical mushroom we
    # convert to ordinal or one-hot.
    if categorical:
        # For Decision Tree, ordinal encoding is fine; for SVM we will
        # one-hot + scaling.
        ordinal = ColumnTransformer(
            [
                (
                    "ord",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    ),
                    list(range(X_train_raw.shape[1])),
                )
            ]
        )
        X_train_dt = ordinal.fit_transform(X_train_raw)
        X_test_dt = ordinal.transform(X_test_raw)
    else:
        X_train_dt = np.array(X_train_raw)
        X_test_dt = np.array(X_test_raw)

    # --- Decision Tree trained on full features ---
    dt_key = "decision_tree_full"
    dt, dt_params = load_model(key=dt_key, dataset_name=dataset_name)
    if not dt:
        dt = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=8)
        dt.fit(X_train_dt, y_train)
    y_pred_dt = dt.predict(X_test_dt)
    report_dt = classification_report(y_test, y_pred_dt, digits=4)
    cm_dt = confusion_matrix(y_test, y_pred_dt)
    acc_dt = accuracy_score(y_test, y_pred_dt)

    results[dt_key] = {
        "model": dt,
        "report": report_dt,
        "confusion_matrix": cm_dt,
        "accuracy": acc_dt,
    }

    # Feature importances (for categorical aggregated later)
    importances = dt.feature_importances_

    # --- SVM: need numeric features, scale is helpful ---
    if categorical:
        # One-hot encode all categorical features, then scale
        preproc = ColumnTransformer(
            [
                (
                    "ohe",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    list(range(X_train_raw.shape[1])),
                )
            ]
        )
        svc_pipeline = Pipeline(
            [
                ("ohe", preproc),
                (
                    "scaler",
                    StandardScaler(with_mean=False),
                ),  # with_mean=False because sparse->dense; ok here
                ("svc", SVC(random_state=RANDOM_STATE, probability=False)),
            ]
        )
    else:
        svc_pipeline = Pipeline(
            [("scaler", StandardScaler()), ("svc", SVC(random_state=RANDOM_STATE))]
        )

    # Grid search on kernels / params (small grid for demonstration)
    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": [0.1, 1, 10]},
        {"svc__kernel": ["rbf"], "svc__C": [0.1, 1], "svc__gamma": ["scale", "auto"]},
        {
            "svc__kernel": ["poly"],
            "svc__C": [1],
            "svc__degree": [2, 3],
            "svc__gamma": ["scale"],
        },
        {"svc__kernel": ["sigmoid"], "svc__C": [1, 5], "svc__gamma": ["scale"]},
    ]

    svc_key = "svc_full"
    best_svc, svc_params = load_model(svc_key, dataset_name=dataset_name)
    if not best_svc:
        gs = GridSearchCV(
            svc_pipeline, param_grid, cv=4, scoring="accuracy", n_jobs=-1, verbose=0
        )
        gs.fit(X_train_raw, y_train)
        svc_params = gs.best_params_
        best_svc = gs.best_estimator_
    y_pred_svc = best_svc.predict(X_test_raw)
    report_svc = classification_report(y_test, y_pred_svc, digits=4)
    cm_svc = confusion_matrix(y_test, y_pred_svc)
    acc_svc = accuracy_score(y_test, y_pred_svc)

    results["svc_full"] = {
        "model": best_svc,
        "report": report_svc,
        "confusion_matrix": cm_svc,
        "accuracy": acc_svc,
        "grid_best_params": svc_params,
    }

    # Print & save summary
    out_file = f"{dataset_name}_classification_summary.txt"
    if os.path.exists(out_file):
        os.remove(out_file)

    header = f"=== Results for {dataset_name} ==="
    print_and_save_report(header, out_file)

    print_and_save_report(
        "Decision Tree (full features) accuracy: {:.4f}".format(acc_dt), out_file
    )
    print_and_save_report(
        "Decision Tree classification report:\n" + report_dt, out_file
    )
    print_and_save_report("Decision Tree confusion matrix:\n" + str(cm_dt), out_file)

    print_and_save_report(
        "SVM (best via gridsearch) accuracy: {:.4f}".format(acc_svc), out_file
    )
    print_and_save_report("SVM best params: " + str(svc_params), out_file)
    print_and_save_report("SVM classification report:\n" + report_svc, out_file)
    print_and_save_report("SVM confusion matrix:\n" + str(cm_svc), out_file)

    # Return structures for further use (importances etc.)
    return results, importances, X_train_raw, X_test_raw, y_train, y_test, feature_names


# -------------------------
# 4) Helpers for visualization
# -------------------------
def aggregate_importances_for_categorical(importances_encoded, X_df):
    """
    If the tree was trained on OneHot encoded columns, but we originally had
    categorical columns, we may want to sum importances by original categorical feature.
    For our pipeline we used ordinal for DT on mushroom, so this function might not be
    needed. We'll leave a helper stub.
    """
    # not used in current pipeline version; kept for completeness
    return importances_encoded


def visualize_decision_boundary_2features_for_model(
    clf, X_full, y, top2_indices, feature_names, title="Decision boundary"
):
    """
    clf: classifier trained on 2 features (i.e., expects 2 columns)
    X_full, y: original arrays (we'll plot data projected to top2)
    top2_indices: feature indices in original X_full
    """
    # Extract only those two columns
    X_vis = X_full[:, top2_indices]
    min_x, max_x = X_vis[:, 0].min() - 1.0, X_vis[:, 0].max() + 1.0
    min_y, max_y = X_vis[:, 1].min() - 1.0, X_vis[:, 1].max() + 1.0
    mesh_step = 0.02
    xx, yy = np.meshgrid(
        np.arange(min_x, max_x, mesh_step), np.arange(min_y, max_y, mesh_step)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, edgecolors="k", cmap=plt.cm.Paired, s=40)
    plt.xlabel(feature_names[top2_indices[0]])
    plt.ylabel(feature_names[top2_indices[1]])
    plt.title(title)
    plt.show()


# -------------------------
# 5) Utility: choose top-2 features by DecisionTree.importances_
# -------------------------
def pick_top2_features(importances, feature_names, categorical=False, X_df=None):
    """
    For numeric data: importances correspond to features. Return indices of top2.
    For categorical with one-hot encoding, you'd want to aggregate; here we assume
    importances length == len(feature_names).
    """
    idx = np.argsort(importances)[-2:][::-1]
    return list(idx)


def get_make_pipeline():
    return make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
    )


# -------------------------
# 6) Models helpers
# -------------------------
def save_models(results, dataset_name="dataset", output_dir=MODELS_DIR):
    """
    results: słownik zwrócony przez train_and_evaluate_classifiers
    dataset_name: nazwa datasetu (do folderu/plików)
    output_dir: katalog do zapisu
    """
    os.makedirs(output_dir, exist_ok=True)

    for key, val in results.items():
        model = val["model"]
        file_path = os.path.join(output_dir, f"{dataset_name}_{key}.joblib")
        joblib.dump(model, file_path)
        print(f"Zapisano model: {file_path}")
        if key == "svc_full":
            params = val["grid_best_params"]
            file_path = os.path.join(
                output_dir, f"{dataset_name}_{key}_grid_best_params.joblib"
            )
            joblib.dump(params, file_path)
            print(f"Zapisano parametry grid: {file_path}")


def load_model(key, dataset_name, output_dir=MODELS_DIR):
    """
    Helper function to load model
    :param key: Name of model type. Like 'decision_tree_full'
    :param dataset_name: Name of dataset. Like 'seeds'
    :param output_dir: Models directory
    :return: Model and params.
    """
    file_path = os.path.join(output_dir, f"{dataset_name}_{key}.joblib")
    model = None
    params = None
    if os.path.exists(file_path):
        print(f"Loading model from {file_path}")
        model = joblib.load(file_path)
        if key == "svc_full":
            params_file_path = os.path.join(
                output_dir, f"{dataset_name}_{key}_grid_best_params.joblib"
            )
            params = joblib.load(params_file_path)
    return model, params


# -------------------------
# 7) Runner for both datasets
# -------------------------
def run_all():
    # --- SEEDS dataset ---
    X_seeds, y_seeds, feature_names_seeds = load_seeds()
    (
        results_seeds,
        imp_seeds,
        X_train_s_raw,
        X_test_s_raw,
        y_train_s,
        y_test_s,
        fn_seeds,
    ) = train_and_evaluate_classifiers(
        X_seeds, y_seeds, feature_names_seeds, dataset_name="seeds", categorical=False
    )

    save_models(results_seeds, dataset_name="seeds")

    # choose top2 features (based on DecisionTree trained earlier)
    top2_seeds = pick_top2_features(imp_seeds, feature_names_seeds)
    print(
        "Seeds top2 feature indices:",
        top2_seeds,
        "names:",
        [feature_names_seeds[i] for i in top2_seeds],
    )

    # For visualization, we need a classifier trained only on these 2 features
    X2_train = X_train_s_raw[:, top2_seeds]
    X2_test = X_test_s_raw[:, top2_seeds]
    dt2 = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
    dt2.fit(X2_train, y_train_s)
    visualize_decision_boundary_2features_for_model(
        dt2,
        np.vstack([X2_train, X2_test]),
        np.hstack([y_train_s, y_test_s]),
        [0, 1],
        [feature_names_seeds[i] for i in top2_seeds],
        title=f"Seeds DT on top2: {feature_names_seeds[top2_seeds[0]]} "
        f"vs {feature_names_seeds[top2_seeds[1]]}",
    )

    # also show SVM behaviour on these two features (train simple scaled SVM)
    svc2 = get_make_pipeline()
    svc2.fit(X2_train, y_train_s)
    visualize_decision_boundary_2features_for_model(
        svc2,
        np.vstack([X2_train, X2_test]),
        np.hstack([y_train_s, y_test_s]),
        [0, 1],
        [feature_names_seeds[i] for i in top2_seeds],
        title="Seeds SVM (RBF) on top2",
    )

    # --- MUSHROOM dataset ---
    X_mush_df, y_mush, feature_names_mush = load_mushroom()
    (
        results_mush,
        imp_mush,
        X_train_m_raw,
        X_test_m_raw,
        y_train_m,
        y_test_m,
        fn_mush,
    ) = train_and_evaluate_classifiers(
        X_mush_df, y_mush, feature_names_mush, dataset_name="mushroom", categorical=True
    )

    save_models(results_mush, dataset_name="mushroom")

    # For mushroom, DecisionTree.importances_ correspond to ordinal-encoded columns
    # (one per original feature), so picking top2 is straightforward (we trained DT
    # with ordinal encoder)
    top2_mush = pick_top2_features(imp_mush, feature_names_mush)
    print(
        "Mushroom top2 feature indices:",
        top2_mush,
        "names:",
        [feature_names_mush[i] for i in top2_mush],
    )

    # Build simple visualization models trained only on those two original features.
    # For mushroom, extract the two categorical columns, ordinal-encode them to numeric
    # for plotting
    oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_mush_2cols = oe.fit_transform(X_mush_df.iloc[:, top2_mush])
    Xm_train2, Xm_test2, ym_train2, ym_test2 = train_test_split(
        X_mush_2cols, y_mush, test_size=0.25, random_state=RANDOM_STATE, stratify=y_mush
    )

    dt_m2 = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
    dt_m2.fit(Xm_train2, ym_train2)
    visualize_decision_boundary_2features_for_model(
        dt_m2,
        np.vstack([Xm_train2, Xm_test2]),
        np.hstack([ym_train2, ym_test2]),
        [0, 1],
        [feature_names_mush[i] for i in top2_mush],
        title="Mushroom DT on top2 categorical (ordinal enc)",
    )

    svc_m2 = get_make_pipeline()
    svc_m2.fit(Xm_train2, ym_train2)
    visualize_decision_boundary_2features_for_model(
        svc_m2,
        np.vstack([Xm_train2, Xm_test2]),
        np.hstack([ym_train2, ym_test2]),
        [0, 1],
        [feature_names_mush[i] for i in top2_mush],
        title="Mushroom SVM (RBF) on top2 (ordinal enc)",
    )

    # -------------------------
    # 7) Kernel summary:
    # -------------------------
    summary_lines = []
    summary_lines.append(
        "SVM Kernel summary (simple conclusions based on GridSearch metrics):\n"
    )
    summary_lines.append(
        "Seeds dataset: best SVM params: "
        + str(results_seeds["svc_full"]["grid_best_params"])
        + "\n"
    )
    summary_lines.append(
        "Seeds SVM accuracy: {:.4f}\n".format(results_seeds["svc_full"]["accuracy"])
    )
    summary_lines.append(
        "Mushroom dataset: best SVM params: "
        + str(results_mush["svc_full"]["grid_best_params"])
        + "\n"
    )
    summary_lines.append(
        "Mushroom SVM accuracy: {:.4f}\n".format(results_mush["svc_full"]["accuracy"])
    )

    summary_lines.append("\nShort notes on kernels (to include in repository README):")
    summary_lines.append(
        "- linear: good for linearly separable data; simple; fewer hyperparams."
    )
    summary_lines.append(
        "- rbf: flexible; maps to infinite-dim; often good default; gamma controls "
        "locality (large gamma -> more complex, risk overfit)."
    )
    summary_lines.append(
        "- poly: useful if polynomial relationships; degree controls complexity."
    )
    summary_lines.append(
        "- sigmoid: rarely best; behaves like NN activation; use sparingly."
    )
    summary_text = "\n".join(summary_lines)

    with open("svm_kernel_summary.txt", "w", encoding="utf8") as f:
        f.write(summary_text)

    print("\nSaved svm_kernel_summary.txt with short conclusions.")


if __name__ == "__main__":
    run_all()
