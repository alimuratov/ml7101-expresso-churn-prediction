from helpers import compare_baseline_rfe

comparison, models = compare_baseline_rfe(
    X_train, y_train,
    X_val, y_val,
    k=7,
    cv=3,
)

display(comparison)

print("Features retained by RFE:")
print(models["rfe"][:-1].get_feature_names_out().tolist())