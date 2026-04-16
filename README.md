# 🔧 Predictive Maintenance ML Pipeline

An end-to-end machine learning pipeline that predicts CNC machine failures from industrial sensor data. Built to demonstrate production-grade ML engineering — from raw data to a live REST API and interactive demo.

**Live Demo:** [Streamlit App](https://ml-predictive-maintenance-satvik.streamlit.app) &nbsp;|&nbsp; **Live API:** [FastAPI on Render](https://ml-predictive-maintenance-1mum.onrender.com/docs)

---

## 📋 Problem Statement

In manufacturing environments, unplanned machine failures cause costly downtime. This project builds a binary classifier that predicts whether a CNC machine will fail based on real-time sensor readings — enabling proactive maintenance before failure occurs.

**Dataset:** [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) — 10,000 samples of synthetic CNC machine sensor data from the UCI ML Repository.

---

## 🏗️ Architecture

Raw Sensor Data (CSV)
│
▼
┌─────────────────┐
│   data_loader   │  Loads & validates raw data
└────────┬────────┘
│
▼
┌─────────────────┐
│  preprocessor   │  Encodes, engineers features, scales
└────────┬────────┘
│
▼
┌─────────────────┐
│    trainer      │  Cross-validates 3 models, tunes XGBoost
└────────┬────────┘
│
▼
┌─────────────────┐
│   evaluator     │  Metrics, confusion matrix, ROC curve
└────────┬────────┘
│
▼
┌─────────────────┐
│  models/*.pkl   │  Serialized model, scaler, feature list
└────────┬────────┘
│
┌────┴────┐
▼         ▼
FastAPI    Streamlit
API        UI

---

## 📊 Model Results

| Model | F1 Score | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.29 | 0.18 | 0.87 | 0.93 |
| Random Forest | 0.78 | 0.96 | 0.66 | 0.97 |
| **XGBoost** | **0.81** | **0.81** | **0.81** | **0.98** |
| XGBoost Tuned | 0.78 | 0.75 | 0.81 | 0.98 |

**XGBoost was selected** as the best model — highest F1 score with balanced precision and recall, which is critical in predictive maintenance where both false alarms and missed failures are costly.

---

## ⚙️ Feature Engineering

Two domain-informed features were engineered from raw sensor readings:

- **`temp_difference`** = Process Temperature − Air Temperature. Captures heat dissipation stress — a key indicator of Heat Dissipation Failure.
- **`power`** = Torque × Rotational Speed. Captures mechanical power load — a key indicator of Power and Overstrain Failure.

---

## 🗂️ Project Structure

ml-predictive-maintenance/
├── data/
│   ├── raw/                  # Original dataset (not tracked in git)
│   └── processed/
├── notebooks/
│   └── plots/                # Confusion matrices, ROC curves, feature importance
├── src/
│   ├── data_loader.py        # Load and validate raw data
│   ├── preprocessor.py       # Feature engineering, encoding, scaling
│   ├── trainer.py            # Model training and cross-validation
│   ├── evaluator.py          # Metrics and plots
│   └── predictor.py          # Inference pipeline
├── models/
│   ├── best_model.pkl        # Trained XGBoost model
│   ├── scaler.pkl            # Fitted StandardScaler
│   └── feature_list.json     # Feature names in training order
├── api/
│   ├── main.py               # FastAPI app
│   └── schemas.py            # Pydantic request/response models
├── streamlit_app/
│   └── app.py                # Streamlit demo UI
├── train.py                  # Master training script
└── requirements.txt

---

## 🚀 Running Locally

**1. Clone the repository**
```bash
git clone https://github.com/SatvikSPandey/ml-predictive-maintenance.git
cd ml-predictive-maintenance
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**

Download `ai4i2020.csv` from [Kaggle](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) and place it in `data/raw/`.

**5. Train the model**
```bash
python train.py
```

**6. Start the API**
```bash
uvicorn api.main:app --reload
```

**7. Start the Streamlit UI**
```bash
streamlit run streamlit_app/app.py
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — confirms model is loaded |
| GET | `/model-info` | Returns feature list and model metadata |
| POST | `/predict` | Accepts sensor readings, returns failure prediction |

**Example request:**
```json
{
  "type": "M",
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551.0,
  "torque": 42.8,
  "tool_wear": 0.0
}
```

**Example response:**
```json
{
  "prediction": 0,
  "failure_probability": 0.0,
  "result": "NO FAILURE",
  "confidence": 1.0
}
```

---

## 🛠️ Tech Stack

- **ML:** scikit-learn, XGBoost, pandas, numpy
- **API:** FastAPI, Pydantic, Uvicorn
- **UI:** Streamlit
- **Serialization:** joblib
- **Deployment:** Render (API), Streamlit Cloud (UI)

---

---

## ☁️ AWS SageMaker Integration

This project includes a production-ready SageMaker integration with graceful fallback to the local joblib model.

### Architecture

POST /predict (FastAPI)
│
▼
src/predictor.py
│
├── SAGEMAKER_ENDPOINT_NAME set + endpoint InService?
│       YES → src/sagemaker_predictor.py → SageMaker Endpoint
│
└── NO → predict_local() → models/best_model.pkl (joblib)

### Key Files

- `src/sagemaker_trainer.py` — Full SageMaker training and deployment workflow using AWS built-in XGBoost container
- `src/sagemaker_predictor.py` — SageMaker endpoint inference client with availability check
- `src/predictor.py` — Unified prediction interface with automatic SageMaker/local routing
- `notebooks/04_sagemaker_deployment.ipynb` — Complete SageMaker deployment documentation

### To Deploy on SageMaker

**Prerequisites:**
1. AWS credentials configured: `aws configure`
2. IAM user with `AmazonSageMakerFullAccess` permissions
3. Create S3 bucket: `ml-predictive-maintenance-satvik`
4. Create SageMaker execution role in IAM

**Steps:**
```bash
# Set environment variables
export SAGEMAKER_ROLE_ARN=arn:aws:iam::<account-id>:role/SageMakerExecutionRole

# Run training job and deploy endpoint (~$0.01 for training, ~$0.115/hr for endpoint)
python src/sagemaker_trainer.py

# Set endpoint name (printed after deployment)
export SAGEMAKER_ENDPOINT_NAME=<endpoint-name>

# FastAPI automatically routes to SageMaker — no code changes needed
uvicorn api.main:app --reload
```

**Important:** Delete the endpoint after testing to avoid charges:
```python
from src.sagemaker_trainer import delete_endpoint
delete_endpoint("<endpoint-name>")
```

### Fallback Behavior

The system works in three modes automatically:
- **Local development** — no AWS config needed, uses joblib
- **Production with SageMaker** — set `SAGEMAKER_ENDPOINT_NAME`, routes automatically
- **Production without SageMaker** — falls back to joblib gracefully



## 👤 Author

**Satvik Pandey** — AI Engineer / Python Developer

[LinkedIn](https://www.linkedin.com/in/satvikpandey-433555365) | [GitHub](https://github.com/SatvikSPandey)