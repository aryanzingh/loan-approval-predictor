# 💳 LoanSense AI — Streamlit Loan Predictor

## Project Structure
```
loan_predictor/
├── app.py            ← Streamlit app
├── model.pkl         ← Exported LogisticRegression model
├── scaler.pkl        ← Exported StandardScaler
├── requirements.txt
└── README.md
```

---

## Step 1 — Export model & scaler from Google Colab

Add these lines **after** your training cell in Colab:

```python
import joblib
from google.colab import files

# Save the trained model
joblib.dump(mod, "model.pkl")
files.download("model.pkl")

# Save the scaler (required — model was trained on scaled data)
joblib.dump(scaler, "scaler.pkl")
files.download("scaler.pkl")
```

---

## Step 2 — Install & Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place `model.pkl` and `scaler.pkl` in the same folder as `app.py`,
or use the in-app file uploader when you first open the app.

---

## Features used (in order)
| # | Feature | Type |
|---|---------|------|
| 1 | no_of_dependents | int (0–5) |
| 2 | education | binary (Graduate=1, Not Graduate=0) |
| 3 | self_employed | binary (Yes=1, No=0) |
| 4 | income_annum | float |
| 5 | loan_amount | float |
| 6 | loan_term | int (months) |
| 7 | cibil_score | int (300–900) |
| 8 | total_assets | float (bank + commercial + residential + luxury) |
