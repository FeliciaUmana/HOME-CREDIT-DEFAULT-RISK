from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(
    title="Home Credit Default Risk API",
    description="Home Credit is an international constomers finance provider that focuses on lending to people with little or no credit history. the goal is to Predict whether a loan applicant will default (fail to repay), using demographics, financial history, and external credit scores.",
    version="1.0.0"
)

# ============================================================
#  Load model and feature names once when server starts
# ============================================================
try:
    model        = joblib.load("home_credit_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
    print(f"Model loaded! Expecting {len(feature_names)} features.")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


# ============================================================
#  Input Schema
# ============================================================
class ApplicantData(BaseModel):
    data: dict  # accepts any key-value pairs


# ============================================================
#  Routes
# ============================================================

@app.get("/")
def home():
    return {
        "message"   : "Home Credit Default Risk API is running!",
        "endpoints" : {
            "predict"  : "/predict  → POST",
            "features" : "/features → GET",
            "health"   : "/health   → GET",
            "docs"     : "/docs     → GET"
        }
    }


@app.get("/health")
def health():
    """Check if the API and model are loaded correctly"""
    return {
        "status"          : "healthy",
        "model_loaded"    : model is not None,
        "features_loaded" : len(feature_names)
    }


@app.get("/features")
def get_features():
    """Returns the list of features the model expects"""
    return {
        "total_features" : len(feature_names),
        "feature_names"  : feature_names
    }


@app.post("/predict")
def predict(applicant: ApplicantData):
    try:
        # Build dataframe from input
        input_df = pd.DataFrame([applicant.data])

        # Add any missing columns with NaN
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = np.nan

        # Keep only model's expected columns in correct order
        input_df = input_df[feature_names]

        # Make prediction
        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Risk level based on probability
        if probability >= 0.7:
            risk_level = "VERY HIGH RISK"
        elif probability >= 0.5:
            risk_level = "HIGH RISK"
        elif probability >= 0.3:
            risk_level = "MEDIUM RISK"
        else:
            risk_level = "LOW RISK"

        return {
            "prediction"         : int(prediction),
            "result"             : "WILL DEFAULT" if prediction == 1 else "WILL REPAY",
            "default_probability": round(float(probability), 4),
            "repay_probability"  : round(1 - float(probability), 4),
            "risk_level"         : risk_level
        }

    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing or wrong feature name: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )