# Luminary Master Decision API

Random Forest decision engine for the Luminary (Lumy) graduation project. Takes the outputs of all other Lumy diagnostic models and returns the final recommended action.

## Files
- `main_dec.py` — FastAPI app
- `luminary_decision_random_forest.pkl` — trained Random Forest model
- `luminary_decision_label_encoders.pkl` — label encoders for categorical features

## Run locally
```bash
pip install -r requirements.txt
uvicorn main_dec:app --reload --port 8000
```
Then open `http://127.0.0.1:8000/docs` to test.

## Endpoint
- `POST /decide` — JSON body with the 10 diagnostic features, returns the recommended action.
