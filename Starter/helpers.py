import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score



def make_feature_pipeline(selection="all", *, k=0.5, estimator=None, step=0.2):
    """
    k is the number of columns to keep after preprocessing
    """
    if selection == "all":
        return "passthrough"
    if selection != "rfe":
        raise ValueError("selection must be 'all' or 'rfe'.")

    selector_model = (
        LogisticRegression(max_iter=2000)
        if estimator is None else clone(estimator)
    )
    return Pipeline([
        ("select", RFE(selector_model, n_features_to_select=k, step=step)),
    ])


def make_pipeline(preprocessor: ColumnTransformer, feature_pipeline, classifier):
    return Pipeline([
        ("preprocess", preprocessor),
        ("features", feature_pipeline),
        ("model", classifier),
    ])


def compare_baseline_rfe(
    X_train, y_train, X_val, y_val, *, k=7, cv=3, random_state=42,
):
    """
    Input: processed data
    k = the number of features to reteain
    cv = number of folds in cross-validation 
    """
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    results = []
    models = {}

    for selection in ["all", "rfe"]:
        preprocessor = ColumnTransformer(
            [("prepared", "passthrough", X_train.columns.tolist())],
            verbose_feature_names_out=False,
        )
        model = make_pipeline(
            preprocessor=preprocessor,
            feature_pipeline=make_feature_pipeline(
                selection, k=k, step=0.2,
                estimator=LogisticRegression(
                    max_iter=2000, random_state=random_state,
                ),
            ),
            classifier=LogisticRegression(
                max_iter=2000, random_state=random_state,
            ),
        )
        scores = cross_val_score(
            model, X_train, y_train, cv=folds,
            scoring="roc_auc", error_score="raise",
        )

        model.fit(X_train, y_train)
        models[selection] = model

        probabilities = model.predict_proba(X_val)[:, 1]
        predictions = model.predict(X_val)
        results.append({
            "method": selection,
            "n_features": model.named_steps["model"].n_features_in_,
            "cv_auc": scores.mean(),
            "cv_auc_std": scores.std(),
            "validation_auc": roc_auc_score(y_val, probabilities),
            "validation_accuracy": accuracy_score(y_val, predictions),
            "validation_f1": f1_score(y_val, predictions, zero_division=0),
        })

    comparison = (
        pd.DataFrame(results)
        .sort_values("cv_auc", ascending=False)
        .reset_index(drop=True)
    )
    return comparison, models
