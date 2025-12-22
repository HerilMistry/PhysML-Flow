import pandas as pd
import numpy as np

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_excel("final_foam.xlsx")

# =========================================================
# 2. PHYSICS-GUIDED REGIME LABELING
# =========================================================

def label_regime(row):
    Re = row["Reynolds_Generalized"]
    geom = row["Geometry"]
    n = row["PowerLaw_n"]

    shift = 10 * (1 - n)  # shear-thinning delay

    if geom == "Cylinder":
        if Re < 30 + shift:
            return "Creeping"
        elif Re < 47 + shift:
            return "Steady"
        elif Re < 120 + shift:
            return "Separated"
        else:
            return "Unsteady"
    else:  # Pipe
        if Re < 50 + shift:
            return "Creeping"
        else:
            return "Steady"

df["Regime"] = df.apply(label_regime, axis=1)

# =========================================================
# 3. LABEL ENCODING (CRITICAL)
# =========================================================

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["Regime"])

# =========================================================
# 4. FEATURE ENGINEERING
# =========================================================

df["ShearRate"] = df["Velocity_m_s"] / df["Diameter_m"]
df["Mu_apparent"] = (
    df["Consistency_K_Pa_s^n"] *
    (df["ShearRate"] ** (df["PowerLaw_n"] - 1))
)

X = df.drop(columns=["CaseID", "Regime"])

# =========================================================
# 5. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# =========================================================
# 6. CATBOOST (RAW CATEGORICALS)
# =========================================================

cat_features = ["Fluid", "Geometry"]

cat_model = CatBoostClassifier(
    iterations=600,
    depth=6,
    learning_rate=0.05,
    loss_function="MultiClass",
    verbose=False
)

cat_model.fit(X_train, y_train, cat_features=cat_features)

# =========================================================
# 7. ENCODE CATEGORICALS FOR RF / XGB
# =========================================================

X_train_enc = X_train.copy()
X_test_enc = X_test.copy()

for col in ["Fluid", "Geometry"]:
    le = LabelEncoder()
    X_train_enc[col] = le.fit_transform(X_train[col])
    X_test_enc[col] = le.transform(X_test[col])

rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    random_state=42
)

xgb = XGBClassifier(
    objective="multi:softprob",
    max_depth=6,
    n_estimators=300,
    learning_rate=0.05,
    tree_method="hist",
    eval_metric="mlogloss"
)

rf.fit(X_train_enc, y_train)
xgb.fit(X_train_enc, y_train)

# =========================================================
# 8. MANUAL SOFT-VOTING ENSEMBLE
# =========================================================

proba_cat = cat_model.predict_proba(X_test)
proba_rf = rf.predict_proba(X_test_enc)
proba_xgb = xgb.predict_proba(X_test_enc)

proba_ensemble = (
    0.5 * proba_cat +
    0.25 * proba_rf +
    0.25 * proba_xgb
)

# FINAL encoded prediction
y_pred = np.argmax(proba_ensemble, axis=1)

# DECODE for reporting
y_test_decoded = label_encoder.inverse_transform(y_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)

# =========================================================
# 9. EVALUATION
# =========================================================

print("\n================ CLASSIFICATION REPORT ================\n")
print(classification_report(y_test_decoded, y_pred_decoded))

print("\n================ CONFUSION MATRIX =====================\n")
print(confusion_matrix(y_test_decoded, y_pred_decoded))

# =========================================================
# 10. FEATURE IMPORTANCE (CATBOOST)
# =========================================================

print("\n================ FEATURE IMPORTANCE (CatBoost) =========\n")
for name, val in sorted(
    zip(X.columns, cat_model.get_feature_importance()),
    key=lambda x: -x[1]
):
    print(f"{name:30s} : {val:.4f}")

print("\nDONE — model trained successfully.")
