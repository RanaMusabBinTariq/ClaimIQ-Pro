# 🏥 ClaimIQ Pro — Intelligent Health Insurance Claims Platform

> AI-powered medical claims triage, fraud detection, patient profiling, and analytics — built with real ICD-10-CM, DRG, HCPCS, LOINC, and SNOMED CT codes.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 What is ClaimIQ Pro?

ClaimIQ Pro is a production-ready intelligent claims management platform for health insurers. It automates the most time-consuming parts of claims adjudication — triage scoring, fraud detection, and patient risk profiling — using machine learning trained on 14,000 realistic claims records.

Built from 2+ years of hands-on medical claims processing experience at MedNet Global Healthcare Solutions (a Munich Re company), combined with MSc Digital Health studies at Deggendorf Institute of Technology, Germany.

---

## 🚀 7 Modules

| Module | What it does |
|--------|-------------|
| 📊 **Dashboard** | Live KPI overview — claims volume, priority split, spend trends, anomaly flags |
| 📋 **Score a Claim** | Enter a claim → get triage score (1–100), priority, recommendation, and scoring rationale |
| 📤 **Batch Upload** | Upload CSV of up to 500 claims → auto-validate, score all, download results |
| 👤 **Patient Profile** | Full claim history, risk trajectory chart, readmission flags for any patient |
| 🛡️ **Fraud & Anomaly** | Isolation Forest model flags unusual claims for investigation |
| 📈 **Analytics** | Deep-dive: cost by category, age risk distribution, provider patterns, auth impact |
| 📖 **Code Reference** | Searchable ICD-10-CM, DRG, HCPCS, LOINC, SNOMED CT code tables |

---

## 🧠 ML Models

### Triage Scoring — Gradient Boosting Regressor
- **Accuracy:** R² = 0.98 · MAE ≈ 2.64 points
- **Training data:** 14,000 synthetic claims · 109 diagnoses
- **Features:** Claim type, diagnosis, category, age, amount, days pending, prior auth, chronic flag, missing docs, readmission, previous claims, comorbidities

| Score | Priority | Action | Turnaround |
|-------|----------|--------|------------|
| 70–100 | 🔴 High | Escalate | Same day |
| 40–69 | 🟡 Medium | Review | 3 days |
| 1–39 | 🟢 Low | Approve | 7 days |

### Anomaly / Fraud Detection — Isolation Forest
- Trained on same feature set
- Contamination rate: 8%
- Adjustable threshold slider (20–90)

---

## 🏥 Medical Code Standards

All codes are from **open / public-domain sources** — no licensing fees, no legal risk:

| Standard | Source | Licence |
|----------|--------|---------|
| **ICD-10-CM** | U.S. CMS & WHO | Public domain |
| **MS-DRG** | CMS Medicare Severity DRGs | Public domain |
| **HCPCS Level II** | CMS | Public domain |
| **LOINC** | Regenstrief Institute | Free — [LOINC licence](https://loinc.org/license/) |
| **SNOMED CT** | SNOMED International | Free in most countries via NRC |

---

## 📊 Dataset

- **14,000 claims** across 1,800 synthetic patients
- **109 real ICD-10-CM diagnoses** across 12 clinical categories
- **8 claim types:** Inpatient, Outpatient, Emergency, Pharmacy, Maternity, Day Surgery, Rehabilitation, Mental Health
- **8 insurance providers** including 8 anonymised insurance providers (Insurance Company A through H)
- **3 years** of data (2022–2024)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | Streamlit |
| ML — Triage | scikit-learn GradientBoostingRegressor |
| ML — Anomaly | scikit-learn IsolationForest |
| Data | Pandas, NumPy |
| Charts | Matplotlib |
| Serialisation | Joblib |
| Deployment | Streamlit Cloud (free) |

---

## ⚡ Quick Start

### 1. Clone
```bash
git clone https://github.com/RanaMusabBinTariq/claimiq-pro.git
cd claimiq-pro
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate data & train models (first time only)
```bash
python generate_data.py
python train_model.py
```

### 4. Run
```bash
streamlit run app.py
```

App opens at `http://localhost:8501`

---

## 🌐 Deploy Free on Streamlit Cloud

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as main file
4. Click **Deploy** — live URL in ~2 minutes

---

## 📁 Project Structure

```
claimiq-pro/
│
├── app.py                  # Main Streamlit application (7 modules)
├── medical_codes.py        # ICD-10-CM, DRG, HCPCS, LOINC, SNOMED CT reference data
├── generate_data.py        # Synthetic claims dataset generator (14,000 records)
├── train_model.py          # ML model training (Gradient Boosting + Isolation Forest)
├── requirements.txt        # Python dependencies
│
├── claims_data.csv         # Generated dataset
├── triage_model.pkl        # Trained triage model
├── anomaly_model.pkl       # Trained anomaly detection model
├── le_*.pkl                # Label encoders (5 files)
└── encoder_classes.json    # Encoder class labels
```

---

## 🎯 Use Cases

- **Health insurers** — automate claims triage and reduce manual adjudication workload
- **TPA (Third Party Administrators)** — prioritise high-value or urgent claims
- **Healthcare analytics teams** — portfolio risk analysis and cost driver reporting
- **Digital health researchers** — study claims patterns and ML applications in insurance

---

## 👤 Author

**Rana Musab Bin Tariq**
- 🌐 [GitHub](https://github.com/RanaMusabBinTariq)
- 💼 [LinkedIn](https://www.linkedin.com/in/rana-musab-bin-tariq-a70982152/)
- 📧 musabr81@gmail.com
- 🎓 MSc Digital Health · Deggendorf Institute of Technology, Germany
- 💼 Former Claims Officer — MedNet Global Healthcare Solutions (Munich Re), Dubai

---

## ⚠️ Disclaimer

This platform is built for **educational and demonstrative purposes**. The dataset is synthetic. Not intended for real clinical or financial decision-making.

---

## 📄 Licence

MIT — free to use, modify, and distribute.
