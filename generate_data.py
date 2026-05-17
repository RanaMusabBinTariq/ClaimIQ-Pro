import pandas as pd
import numpy as np
import random
import json
from medical_codes import ICD10_CONDITIONS, HCPCS_PROCEDURES, LOINC_LABS, CLAIM_TYPES, PROVIDERS, REJECTION_REASONS

np.random.seed(42)
random.seed(42)

N = 14000
diagnoses = list(ICD10_CONDITIONS.keys())
hcpcs_list = list(HCPCS_PROCEDURES.keys())
loinc_list = list(LOINC_LABS.keys())
provider_list = PROVIDERS
rejection_list = REJECTION_REASONS

PATIENT_IDS = [f"PT-{str(i).zfill(5)}" for i in range(1, 1801)]
CLAIM_IDS   = [f"CLM-{str(i).zfill(6)}" for i in range(1, N+1)]

rows = []
for i in range(N):
    diag_name  = random.choice(diagnoses)
    diag_info  = ICD10_CONDITIONS[diag_name]
    claim_type = random.choice(CLAIM_TYPES)
    patient_id = random.choice(PATIENT_IDS)
    provider   = random.choice(provider_list)
    age        = int(np.random.choice(
        [np.random.randint(1,17), np.random.randint(18,64), np.random.randint(65,90)],
        p=[0.1, 0.65, 0.25]
    ))
    gender = random.choice(["Male","Female"])

    # Claim financials
    base_amounts = {
        "Inpatient": (3000, 80000), "Emergency": (800, 25000),
        "Outpatient": (100, 3000),  "Pharmacy": (50, 8000),
        "Maternity": (2000, 20000), "Day Surgery": (1500, 15000),
        "Rehabilitation": (500, 8000), "Mental Health": (300, 5000),
    }
    lo, hi = base_amounts.get(claim_type, (200, 5000))
    claim_amount = round(np.random.lognormal(
        mean=np.log((lo+hi)/2), sigma=0.6
    ), 2)
    claim_amount = max(lo, min(hi * 2, claim_amount))

    days_submitted = int(np.random.exponential(scale=12))
    has_prior_auth = random.choices([1,0], weights=[0.72, 0.28])[0]
    missing_docs   = random.choices([0,0,0,1], weights=[0.75,0.1,0.1,0.05])[0]
    readmission    = random.choices([0,1], weights=[0.88, 0.12])[0]
    is_chronic     = int(diag_info["chronic"])
    num_prev_claims= int(np.random.poisson(3))
    comorbidities  = int(np.random.poisson(1.5))
    hcpcs          = HCPCS_PROCEDURES[random.choice(hcpcs_list)]
    loinc          = LOINC_LABS[random.choice(loinc_list)]
    drg_code       = diag_info["drg"]
    snomed_flag    = random.choice([0,1])
    month          = random.randint(1, 12)
    year           = random.choices([2022,2023,2024], weights=[0.2,0.4,0.4])[0]

    # Triage score
    score = diag_info["base_urgency"]
    if claim_type == "Emergency": score += 10
    if claim_type == "Inpatient": score += 5
    if age > 70: score += 6
    if age < 5:  score += 6
    if claim_amount > 20000: score += 8
    if claim_amount > 50000: score += 5
    if days_submitted > 30: score += 5
    if not has_prior_auth: score += 4
    if readmission: score += 7
    if missing_docs: score -= 6
    if comorbidities >= 3: score += 5
    if num_prev_claims > 6: score += 3
    score += int(np.random.randint(-4, 5))
    score = int(min(100, max(1, score)))

    if score >= 70:   priority = "High"
    elif score >= 40: priority = "Medium"
    else:             priority = "Low"

    # Recommendation
    if score >= 80 or (not has_prior_auth and claim_amount > 5000):
        recommendation = "Escalate"
    elif score >= 50 or missing_docs:
        recommendation = "Review"
    else:
        recommendation = "Approve"

    # Anomaly score (for fraud detection)
    anomaly = 0
    if claim_amount > 40000: anomaly += 30
    if days_submitted > 45:  anomaly += 25
    if not has_prior_auth and claim_type == "Inpatient": anomaly += 20
    if num_prev_claims > 8:  anomaly += 15
    if missing_docs:         anomaly += 10
    anomaly += int(np.random.randint(0, 10))
    anomaly = min(100, anomaly)

    rejection_reason = ""
    if recommendation == "Review" or recommendation == "Escalate":
        if random.random() < 0.35:
            rejection_reason = random.choice(rejection_list)

    rows.append({
        "claim_id":          CLAIM_IDS[i],
        "patient_id":        patient_id,
        "year":              year,
        "month":             month,
        "claim_type":        claim_type,
        "diagnosis":         diag_name,
        "icd10_code":        diag_info["icd10"],
        "category":          diag_info["category"],
        "drg_code":          drg_code,
        "hcpcs_code":        hcpcs,
        "loinc_code":        loinc,
        "provider":          provider,
        "patient_age":       age,
        "patient_gender":    gender,
        "claim_amount_usd":  claim_amount,
        "days_since_submission": days_submitted,
        "has_prior_auth":    has_prior_auth,
        "is_chronic":        is_chronic,
        "missing_documents": missing_docs,
        "is_readmission":    readmission,
        "num_prev_claims":   num_prev_claims,
        "comorbidities":     comorbidities,
        "snomed_flag":       snomed_flag,
        "triage_score":      score,
        "priority":          priority,
        "recommendation":    recommendation,
        "anomaly_score":     anomaly,
        "rejection_reason":  rejection_reason,
    })

df = pd.DataFrame(rows)
df.to_csv("claims_data.csv", index=False)
print(f"Generated {len(df)} claims")
print(df["priority"].value_counts())
print(df["recommendation"].value_counts())
print(f"Unique patients: {df['patient_id'].nunique()}")
print(f"Unique diagnoses: {df['diagnosis'].nunique()}")
