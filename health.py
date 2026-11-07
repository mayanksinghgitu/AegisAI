# ============================================================
# AegisAI - Health & Wellness Risk Prediction
# Complete ML Training Pipeline
# ============================================================

import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier


# ============================================================
# 2. LOAD DATASET
# ============================================================


df = pd.read_csv(r"C:\Users\mayan\Desktop\GENai\AegisAI_Multimodal_Health_Wellness_Risk_Dataset_300K_Synthetic.csv")
print(df.head(4))

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "Risk_Level"

y = df[TARGET]

# Remove target and leakage columns
#
# Person_ID:
#   Not useful for prediction.
#
# Wellness_Score_0_100:
#   IMPORTANT:
#   Risk_Level was generated using the wellness score.
#   Therefore using it would cause target leakage.

DROP_COLUMNS = [
    TARGET,
    "Person_ID",
    "Wellness_Score_0_100"
]

X = df.drop(columns=DROP_COLUMNS)

print("\nInput features:", X.shape[1])
print("Target:", TARGET)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 4. IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),

    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numerical_features
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 7. ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("\nTarget classes:")

for i, cls in enumerate(label_encoder.classes_):
    print(i, "=", cls)


# ============================================================
# 8. PREPROCESS TRAIN / TEST DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print("\nProcessed training shape:",
      X_train_processed.shape)

print("Processed testing shape:",
      X_test_processed.shape)


# ============================================================
# 9. TRAIN DIFFERENT MODELS
# ============================================================

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=500,
            max_depth=None,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ),

    # Artificial Neural Network (ANN)
    "ANN":
        MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            solver="adam",
            alpha=0.0005,
            batch_size=256,
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=10,
            random_state=42
        )
}

# ANN uses the same fully preprocessed, scaled feature matrix.
# This is important because neural networks generally perform better
# when numerical inputs are standardized.

results = []

trained_models = {}


# ============================================================
# 10. TRAIN CLASSICAL MODELS + ARTIFICIAL NEURAL NETWORK (ANN)
# ============================================================

for name, model in models.items():

    print("\n" + "=" * 60)
    print("TRAINING:", name)
    print("=" * 60)

    model.fit(
        X_train_processed,
        y_train_encoded
    )

    predictions = model.predict(
        X_test_processed
    )

    accuracy = accuracy_score(
        y_test_encoded,
        predictions
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test_encoded,
        predictions
    )

    macro_f1 = f1_score(
        y_test_encoded,
        predictions,
        average="macro"
    )

    weighted_f1 = f1_score(
        y_test_encoded,
        predictions,
        average="weighted"
    )

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Balanced Accuracy":
            balanced_accuracy,

        "Macro F1":
            macro_f1,

        "Weighted F1":
            weighted_f1
    })

    trained_models[name] = model

    print("Accuracy:",
          round(accuracy, 4))

    print("Balanced Accuracy:",
          round(balanced_accuracy, 4))

    print("Macro F1:",
          round(macro_f1, 4))


# ============================================================
# 11. XGBOOST (Gradient Boosting)

# ============================================================

print("\n" + "=" * 60)
print("TRAINING XGBOOST")
print("=" * 60)


# Calculate balanced sample weights

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train_encoded
)


xgb_model = XGBClassifier(

    n_estimators=700,

    max_depth=4,

    learning_rate=0.03,

    subsample=0.85,

    colsample_bytree=0.90,

    min_child_weight=2,

    reg_alpha=0.08,

    reg_lambda=2.0,

    objective="multi:softprob",

    num_class=len(
        label_encoder.classes_
    ),

    eval_metric="mlogloss",

    random_state=42,

    n_jobs=-1
)


xgb_model.fit(

    X_train_processed,

    y_train_encoded,

    sample_weight=sample_weights,

    verbose=False
)


xgb_predictions = xgb_model.predict(
    X_test_processed
)


xgb_accuracy = accuracy_score(
    y_test_encoded,
    xgb_predictions
)

xgb_balanced_accuracy = balanced_accuracy_score(
    y_test_encoded,
    xgb_predictions
)

xgb_macro_f1 = f1_score(
    y_test_encoded,
    xgb_predictions,
    average="macro"
)

xgb_weighted_f1 = f1_score(
    y_test_encoded,
    xgb_predictions,
    average="weighted"
)


results.append({

    "Model": "XGBoost",

    "Accuracy": xgb_accuracy,

    "Balanced Accuracy":
        xgb_balanced_accuracy,

    "Macro F1":
        xgb_macro_f1,

    "Weighted F1":
        xgb_weighted_f1
})


trained_models["XGBoost"] = xgb_model


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

# Rank ALL candidate models by Macro F1.
# The model with the highest Macro F1 is selected automatically.
results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
).reset_index(drop=True)


print("\n" + "=" * 60)
print("MODEL COMPARISON (INCLUDING ANN)")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# 13. SELECT BEST MODEL
# ============================================================

# ============================================================
# AUTOMATIC BEST MODEL SELECTION
# ============================================================
# Candidate models:
# Logistic Regression, Random Forest, Extra Trees, ANN, XGBoost
#
# The highest Macro F1 model is selected automatically.
# Only this selected model is saved for deployment.

best_model_name = results_df.loc[0, "Model"]
best_model = trained_models[best_model_name]

best_model_metrics = results_df.loc[0].to_dict()

print("\nBest model:",
      best_model_name)

print("\nModels evaluated:")
for model_name in results_df["Model"].tolist():
    print(" -", model_name)


# ============================================================
# 14. FINAL PREDICTIONS
# ============================================================

if best_model_name == "XGBoost":

    final_predictions = xgb_predictions

else:

    final_predictions = best_model.predict(
        X_test_processed
    )


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)


print(
    classification_report(

        y_test_encoded,

        final_predictions,

        target_names=
        label_encoder.classes_,

        digits=4
    )
)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)


cm = confusion_matrix(
    y_test_encoded,
    final_predictions
)

print(cm)


# ============================================================
# 17. SAVE COMPLETE MODEL
# ============================================================

model_bundle = {

    # ONLY THE WINNING MODEL IS SAVED
    "model":
        best_model,

    "best_model_name":
        best_model_name,

    "best_model_metrics":
        best_model_metrics,

    "preprocessor":
        preprocessor,

    "label_encoder":
        label_encoder,

    "features":
        X.columns.tolist(),

    "numeric_features":
        numerical_features,

    "categorical_features":
        categorical_features,

    "target":
        TARGET,

    "excluded_columns":
        DROP_COLUMNS
}


MODEL_PATH = "AegisAI_Health_Risk_Model.joblib"


joblib.dump(
    model_bundle,
    MODEL_PATH
)


print("\n" + "=" * 60)

print(
    "BEST MODEL SAVED SUCCESSFULLY"
)

print(
    "Selected model:",
    best_model_name
)

print(
    "Saved file:",
    MODEL_PATH
)

print(
    "Selection metric: Macro F1 =",
    round(float(best_model_metrics["Macro F1"]), 4)
)

print("=" * 60)


# ============================================================
# 18. PREDICTION FUNCTION
# ============================================================

def predict_health_risk(person_data):

    """
    Predict health risk for one person.

    The selected model may be Logistic Regression, Random Forest,
    Extra Trees, ANN, or XGBoost depending on Macro F1 score.
    All models use the same fitted preprocessing pipeline.

    person_data must contain the same
    features used during training.
    """

    model_data = joblib.load(
        MODEL_PATH
    )

    preprocessor = model_data[
        "preprocessor"
    ]

    model = model_data[
        "model"
    ]

    label_encoder = model_data[
        "label_encoder"
    ]

    features = model_data[
        "features"
    ]

    # Convert dictionary to DataFrame

    input_df = pd.DataFrame(
        [person_data]
    )

    # Ensure correct column order

    input_df = input_df[
        features
    ]

    # Preprocess

    processed_input = preprocessor.transform(
        input_df
    )

    # Prediction

    prediction = model.predict(
        processed_input
    )[0]

    # Convert numerical class
    # back to text

    risk = label_encoder.inverse_transform(
        [prediction]
    )[0]

    # Probability

    probabilities = model.predict_proba(
        processed_input
    )[0]

    probability_dict = {

        label_encoder.inverse_transform(
            [i]
        )[0]:

        round(
            float(probabilities[i]),
            4
        )

        for i in range(
            len(probabilities)
        )
    }

    return {

        "Risk_Level":
            risk,

        "Probabilities":
            probability_dict
    }


# ============================================================
# 19. TEST WITH A NEW PERSON
# ============================================================

person = {

    "Age": 22,

    "Sex": "Male",

    "Height_cm": 175,

    "Weight_kg": 68,

    "BMI": 22.2,

    "Existing_Condition": "None",

    "Medication_Status": "None",

    "Family_History": "None",

    "Facial_Expression": "Happy",

    "Stress_Level_0_10": 3,

    "Mood_Score_0_10": 8,

    "Voice_Stress_0_10": 3,

    "Voice_Tone_Score_0_10": 8,

    "Systolic_BP_mmHg": 118,

    "Diastolic_BP_mmHg": 76,

    "Heart_Rate_bpm": 68,

    "SpO2_percent": 98,

    "Body_Temperature_C": 36.7,

    "Respiratory_Rate_bpm": 15,

    "Fatigue_Score_0_10": 2,

    "Symptom_Count": 0,

    "Primary_Symptom": "None",

    "Sleep_Hours": 7.5,

    "Daily_Steps": 8500,

    "Exercise_Minutes_Week": 180,

    "Sedentary_Hours_Day": 6,

    "Smoking_Status": "Never",

    "Alcohol_Frequency": "None",

    "Water_Liters_Day": 2.5,

    "Calories_Day": 2200,

    "Protein_g_Day": 80,

    "Sugar_g_Day": 40,

    "Fruit_Veg_Servings_Day": 5,

    "AQI": 80,

    "PM2_5_ug_m3": 30,

    "Temperature_Environment_C": 27,

    "Humidity_percent": 55,

    "Noise_dB": 50,

    "UV_Index": 5
}


# ============================================================
# 20. RUN PREDICTION
# ============================================================

result = predict_health_risk(
    person
)


print("\n" + "=" * 60)
print("AEGISAI HEALTH ASSESSMENT")
print("=" * 60)

print(
    "Risk Level:",
    result["Risk_Level"]
)

print(
    "\nProbabilities:"
)

for label, probability in result[
    "Probabilities"
].items():

    print(
        f"{label}: "
        f"{probability * 100:.2f}%"
    )