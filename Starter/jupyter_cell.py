# Paste after the "Working Data" cell defining train_X, train_y, and hashes.
# The notebook has already filled missing values with constants.
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, TargetEncoder

from helpers import compare_baseline_rfe

numeric_columns = train_X.select_dtypes(include="number").columns.tolist()
categorical_columns = train_X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer(
    [
        ("numeric", StandardScaler(), numeric_columns),
        ("categorical", Pipeline([
            ("encode", TargetEncoder(target_type="binary", random_state=42)),
            ("scale", StandardScaler()),
        ]), categorical_columns),
    ],
    verbose_feature_names_out=False,
)

comparison, models = compare_baseline_rfe(
    train_X, train_y,
    preprocessor=preprocessor,
    groups=hashes,
    k=7,
    cv=GroupKFold(n_splits=5),
)

display(comparison)

print("Features retained by RFE:")
print(models["rfe"][:-1].get_feature_names_out().tolist())
