from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pickle
import numpy as np

# 1. Initialize FastAPI app
app = FastAPI(title="Luminary Master Decision API")

# ==========================================
# 2. LOAD THE RANDOM FOREST & ENCODERS
# ==========================================
try:
    with open("luminary_decision_random_forest.pkl", "rb") as f:
        rf_model = pickle.load(f)
    
    with open("luminary_decision_label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
        
    print("🧠 Master Decision Engine loaded successfully.")
except Exception as e:
    print(f"Failed to load models. Error: {e}")

# ==========================================
# 3. EXACT FEATURE ORDER (From your Debug Script)
# ==========================================
EXPECTED_FEATURES = [
    'identity_verified', 
    'insurance_type', 
    'insurance_valid', 
    'vital_status', 
    'vital_risk_score', 
    'fracture_detected', 
    'fracture_region', 
    'mura_abnormal', 
    'chest_disease_detected', 
    'chest_disease_type'
]

# We know exactly which ones need text-to-number translation
ENCODED_FEATURES = [
    'insurance_type', 
    'vital_status', 
    'fracture_region', 
    'chest_disease_type'
]

# ==========================================
# 4. DEFINE THE INCOMING PAYLOAD
# ==========================================
class DiagnosticPayload(BaseModel):
    features: Dict[str, Any]

# ==========================================
# 5. THE DECISION ENDPOINT
# ==========================================
@app.post("/decide")
async def make_decision(payload: DiagnosticPayload):
    try:
        input_data = []
        
        # Loop through the exact expected feature order
        for feature_name in EXPECTED_FEATURES:
            if feature_name not in payload.features:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Missing required feature: {feature_name}"
                )
            
            raw_value = payload.features[feature_name]
            
            # Process categorical vs numerical/boolean data
            if feature_name in ENCODED_FEATURES:
                try:
                    encoded_value = encoders[feature_name].transform([str(raw_value)])[0]
                    input_data.append(encoded_value)
                except ValueError:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Unrecognized category '{raw_value}' for {feature_name}."
                    )
            else:
                # Handle Booleans safely (in case they are sent as strings like "True")
                if str(raw_value).lower() == "true":
                    input_data.append(1.0)
                elif str(raw_value).lower() == "false":
                    input_data.append(0.0)
                else:
                    # Pass raw numbers (like vital_risk_score) straight through
                    input_data.append(float(raw_value))
                
        # Convert to 2D numpy array for scikit-learn
        final_input = np.array(input_data).reshape(1, -1)
        
        # Run the Random Forest Prediction
        raw_prediction = rf_model.predict(final_input)[0]
        
        # Translate the numerical prediction back into a human-readable action
        final_action = str(raw_prediction)
        if 'target' in encoders:
            try:
                final_action = encoders['target'].inverse_transform([raw_prediction])[0]
            except:
                pass
        
        return {
            "status": "success",
            "luminary_action": final_action,
            "diagnostics_reviewed": payload.features
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Luminary Brain is online!"}