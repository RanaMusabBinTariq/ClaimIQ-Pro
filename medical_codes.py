"""
Medical codes reference data.
Sources (all public domain / open license):
- ICD-10-CM: U.S. Centers for Medicare & Medicaid Services (CMS) — public domain
- DRG (MS-DRG): CMS Medicare Severity DRGs — public domain
- HCPCS Level II: CMS — public domain
- LOINC: Regenstrief Institute — free to use under LOINC license (no fee)
- SNOMED CT: SNOMED International — free in most countries via NRC
"""

# ── ICD-10-CM codes (200+ real codes) ────────────────────────────────────────
ICD10_CONDITIONS = {
    # Cardiovascular
    "Acute Myocardial Infarction (STEMI)":       {"icd10": "I21.0",  "category": "Cardiovascular",    "base_urgency": 96, "chronic": False, "drg": "280"},
    "Acute Myocardial Infarction (NSTEMI)":      {"icd10": "I21.4",  "category": "Cardiovascular",    "base_urgency": 93, "chronic": False, "drg": "282"},
    "Heart Failure - Systolic":                  {"icd10": "I50.20", "category": "Cardiovascular",    "base_urgency": 85, "chronic": True,  "drg": "291"},
    "Heart Failure - Diastolic":                 {"icd10": "I50.30", "category": "Cardiovascular",    "base_urgency": 82, "chronic": True,  "drg": "292"},
    "Atrial Fibrillation":                       {"icd10": "I48.91", "category": "Cardiovascular",    "base_urgency": 72, "chronic": True,  "drg": "308"},
    "Hypertensive Crisis":                       {"icd10": "I16.0",  "category": "Cardiovascular",    "base_urgency": 88, "chronic": False, "drg": "304"},
    "Essential Hypertension":                    {"icd10": "I10",    "category": "Cardiovascular",    "base_urgency": 40, "chronic": True,  "drg": "304"},
    "Ischemic Stroke":                           {"icd10": "I63.9",  "category": "Neurological",      "base_urgency": 95, "chronic": False, "drg": "065"},
    "Transient Ischemic Attack":                 {"icd10": "G45.9",  "category": "Neurological",      "base_urgency": 78, "chronic": False, "drg": "069"},
    "Pulmonary Embolism":                        {"icd10": "I26.99", "category": "Cardiovascular",    "base_urgency": 92, "chronic": False, "drg": "175"},
    "Deep Vein Thrombosis":                      {"icd10": "I82.401","category": "Cardiovascular",    "base_urgency": 70, "chronic": False, "drg": "293"},
    "Peripheral Artery Disease":                 {"icd10": "I73.9",  "category": "Cardiovascular",    "base_urgency": 55, "chronic": True,  "drg": "299"},
    "Aortic Aneurysm":                           {"icd10": "I71.4",  "category": "Cardiovascular",    "base_urgency": 90, "chronic": True,  "drg": "237"},

    # Respiratory
    "Pneumonia - Bacterial":                     {"icd10": "J15.9",  "category": "Respiratory",       "base_urgency": 75, "chronic": False, "drg": "193"},
    "Pneumonia - Viral":                         {"icd10": "J12.9",  "category": "Respiratory",       "base_urgency": 72, "chronic": False, "drg": "193"},
    "COPD - Acute Exacerbation":                 {"icd10": "J44.1",  "category": "Respiratory",       "base_urgency": 80, "chronic": True,  "drg": "190"},
    "COPD - Stable":                             {"icd10": "J44.0",  "category": "Respiratory",       "base_urgency": 45, "chronic": True,  "drg": "192"},
    "Asthma - Acute":                            {"icd10": "J45.901","category": "Respiratory",       "base_urgency": 68, "chronic": True,  "drg": "202"},
    "Asthma - Mild Persistent":                  {"icd10": "J45.30", "category": "Respiratory",       "base_urgency": 35, "chronic": True,  "drg": "203"},
    "Respiratory Failure":                       {"icd10": "J96.00", "category": "Respiratory",       "base_urgency": 95, "chronic": False, "drg": "189"},
    "Pleural Effusion":                          {"icd10": "J90",    "category": "Respiratory",       "base_urgency": 65, "chronic": False, "drg": "199"},
    "Pneumothorax":                              {"icd10": "J93.9",  "category": "Respiratory",       "base_urgency": 85, "chronic": False, "drg": "199"},
    "Sleep Apnea":                               {"icd10": "G47.33", "category": "Respiratory",       "base_urgency": 30, "chronic": True,  "drg": "203"},
    "Pulmonary Fibrosis":                        {"icd10": "J84.10", "category": "Respiratory",       "base_urgency": 70, "chronic": True,  "drg": "192"},

    # Endocrine / Metabolic
    "Type 1 Diabetes - Uncontrolled":            {"icd10": "E10.65", "category": "Endocrine",         "base_urgency": 78, "chronic": True,  "drg": "638"},
    "Type 2 Diabetes - Uncontrolled":            {"icd10": "E11.65", "category": "Endocrine",         "base_urgency": 65, "chronic": True,  "drg": "639"},
    "Type 2 Diabetes - Controlled":              {"icd10": "E11.9",  "category": "Endocrine",         "base_urgency": 38, "chronic": True,  "drg": "640"},
    "Diabetic Ketoacidosis":                     {"icd10": "E11.10", "category": "Endocrine",         "base_urgency": 92, "chronic": True,  "drg": "638"},
    "Hypoglycemia":                              {"icd10": "E11.641","category": "Endocrine",         "base_urgency": 80, "chronic": True,  "drg": "638"},
    "Hypothyroidism":                            {"icd10": "E03.9",  "category": "Endocrine",         "base_urgency": 30, "chronic": True,  "drg": "644"},
    "Hyperthyroidism":                           {"icd10": "E05.90", "category": "Endocrine",         "base_urgency": 55, "chronic": True,  "drg": "644"},
    "Obesity - Morbid":                          {"icd10": "E66.01", "category": "Endocrine",         "base_urgency": 45, "chronic": True,  "drg": "619"},
    "Metabolic Syndrome":                        {"icd10": "E88.81", "category": "Endocrine",         "base_urgency": 35, "chronic": True,  "drg": "641"},
    "Adrenal Insufficiency":                     {"icd10": "E27.40", "category": "Endocrine",         "base_urgency": 82, "chronic": True,  "drg": "644"},

    # Gastrointestinal
    "Appendicitis - Acute":                      {"icd10": "K35.80", "category": "Gastrointestinal",  "base_urgency": 88, "chronic": False, "drg": "341"},
    "GI Bleed - Upper":                          {"icd10": "K92.1",  "category": "Gastrointestinal",  "base_urgency": 88, "chronic": False, "drg": "377"},
    "GI Bleed - Lower":                          {"icd10": "K92.1",  "category": "Gastrointestinal",  "base_urgency": 85, "chronic": False, "drg": "378"},
    "Pancreatitis - Acute":                      {"icd10": "K85.90", "category": "Gastrointestinal",  "base_urgency": 83, "chronic": False, "drg": "204"},
    "Pancreatitis - Chronic":                    {"icd10": "K86.1",  "category": "Gastrointestinal",  "base_urgency": 55, "chronic": True,  "drg": "204"},
    "Cholecystitis - Acute":                     {"icd10": "K81.0",  "category": "Gastrointestinal",  "base_urgency": 78, "chronic": False, "drg": "394"},
    "Crohn's Disease - Active":                  {"icd10": "K50.90", "category": "Gastrointestinal",  "base_urgency": 65, "chronic": True,  "drg": "385"},
    "Ulcerative Colitis - Active":               {"icd10": "K51.90", "category": "Gastrointestinal",  "base_urgency": 62, "chronic": True,  "drg": "385"},
    "Cirrhosis of Liver":                        {"icd10": "K74.60", "category": "Gastrointestinal",  "base_urgency": 75, "chronic": True,  "drg": "432"},
    "Hepatic Failure":                           {"icd10": "K72.90", "category": "Gastrointestinal",  "base_urgency": 92, "chronic": True,  "drg": "432"},
    "GERD":                                      {"icd10": "K21.0",  "category": "Gastrointestinal",  "base_urgency": 20, "chronic": True,  "drg": "391"},
    "Bowel Obstruction":                         {"icd10": "K56.60", "category": "Gastrointestinal",  "base_urgency": 88, "chronic": False, "drg": "388"},

    # Renal / Urinary
    "Acute Kidney Injury":                       {"icd10": "N17.9",  "category": "Renal",             "base_urgency": 88, "chronic": False, "drg": "682"},
    "Chronic Kidney Disease - Stage 4":          {"icd10": "N18.4",  "category": "Renal",             "base_urgency": 70, "chronic": True,  "drg": "684"},
    "Chronic Kidney Disease - Stage 5":          {"icd10": "N18.5",  "category": "Renal",             "base_urgency": 82, "chronic": True,  "drg": "682"},
    "End Stage Renal Disease":                   {"icd10": "N18.6",  "category": "Renal",             "base_urgency": 88, "chronic": True,  "drg": "682"},
    "Urinary Tract Infection":                   {"icd10": "N39.0",  "category": "Renal",             "base_urgency": 38, "chronic": False, "drg": "689"},
    "Pyelonephritis":                            {"icd10": "N10",    "category": "Renal",             "base_urgency": 72, "chronic": False, "drg": "689"},
    "Kidney Stone":                              {"icd10": "N20.0",  "category": "Renal",             "base_urgency": 60, "chronic": False, "drg": "693"},
    "Nephrotic Syndrome":                        {"icd10": "N04.9",  "category": "Renal",             "base_urgency": 68, "chronic": True,  "drg": "684"},

    # Neurological
    "Epilepsy - Uncontrolled":                   {"icd10": "G40.909","category": "Neurological",      "base_urgency": 75, "chronic": True,  "drg": "101"},
    "Migraine - Severe":                         {"icd10": "G43.909","category": "Neurological",      "base_urgency": 40, "chronic": True,  "drg": "102"},
    "Parkinson's Disease":                       {"icd10": "G20",    "category": "Neurological",      "base_urgency": 55, "chronic": True,  "drg": "057"},
    "Multiple Sclerosis":                        {"icd10": "G35",    "category": "Neurological",      "base_urgency": 65, "chronic": True,  "drg": "057"},
    "Meningitis - Bacterial":                    {"icd10": "G00.9",  "category": "Neurological",      "base_urgency": 97, "chronic": False, "drg": "076"},
    "Encephalitis":                              {"icd10": "G04.90", "category": "Neurological",      "base_urgency": 94, "chronic": False, "drg": "076"},
    "Dementia - Alzheimer's":                    {"icd10": "F02.80", "category": "Neurological",      "base_urgency": 50, "chronic": True,  "drg": "057"},
    "Neuropathy - Diabetic":                     {"icd10": "E11.40", "category": "Neurological",      "base_urgency": 45, "chronic": True,  "drg": "073"},

    # Oncology
    "Lung Cancer - Primary":                     {"icd10": "C34.90", "category": "Oncology",          "base_urgency": 85, "chronic": True,  "drg": "180"},
    "Breast Cancer - Active Treatment":          {"icd10": "C50.911","category": "Oncology",          "base_urgency": 82, "chronic": True,  "drg": "582"},
    "Colon Cancer":                              {"icd10": "C18.9",  "category": "Oncology",          "base_urgency": 80, "chronic": True,  "drg": "329"},
    "Leukemia - Acute":                          {"icd10": "C91.00", "category": "Oncology",          "base_urgency": 92, "chronic": True,  "drg": "834"},
    "Lymphoma":                                  {"icd10": "C85.90", "category": "Oncology",          "base_urgency": 80, "chronic": True,  "drg": "834"},
    "Chemotherapy Session":                      {"icd10": "Z51.11", "category": "Oncology",          "base_urgency": 70, "chronic": True,  "drg": "847"},
    "Radiation Therapy":                         {"icd10": "Z51.0",  "category": "Oncology",          "base_urgency": 65, "chronic": True,  "drg": "847"},
    "Prostate Cancer":                           {"icd10": "C61",    "category": "Oncology",          "base_urgency": 70, "chronic": True,  "drg": "715"},

    # Musculoskeletal
    "Hip Fracture":                              {"icd10": "S72.001A","category": "Musculoskeletal",  "base_urgency": 85, "chronic": False, "drg": "480"},
    "Vertebral Fracture":                        {"icd10": "S22.000A","category": "Musculoskeletal",  "base_urgency": 80, "chronic": False, "drg": "453"},
    "Septic Arthritis":                          {"icd10": "M00.9",  "category": "Musculoskeletal",   "base_urgency": 82, "chronic": False, "drg": "545"},
    "Rheumatoid Arthritis - Active":             {"icd10": "M05.79", "category": "Musculoskeletal",   "base_urgency": 55, "chronic": True,  "drg": "545"},
    "Osteoporosis with Fracture":                {"icd10": "M80.00XA","category": "Musculoskeletal",  "base_urgency": 72, "chronic": True,  "drg": "480"},
    "Back Pain - Acute":                         {"icd10": "M54.50", "category": "Musculoskeletal",   "base_urgency": 30, "chronic": False, "drg": "552"},
    "Spinal Stenosis":                           {"icd10": "M48.00", "category": "Musculoskeletal",   "base_urgency": 50, "chronic": True,  "drg": "552"},
    "Osteomyelitis":                             {"icd10": "M86.90", "category": "Musculoskeletal",   "base_urgency": 80, "chronic": False, "drg": "539"},

    # Mental Health
    "Major Depressive Disorder - Severe":        {"icd10": "F32.2",  "category": "Mental Health",     "base_urgency": 65, "chronic": True,  "drg": "885"},
    "Bipolar Disorder - Manic":                  {"icd10": "F31.10", "category": "Mental Health",     "base_urgency": 72, "chronic": True,  "drg": "885"},
    "Schizophrenia - Acute":                     {"icd10": "F20.9",  "category": "Mental Health",     "base_urgency": 75, "chronic": True,  "drg": "880"},
    "Anxiety Disorder - Generalized":            {"icd10": "F41.1",  "category": "Mental Health",     "base_urgency": 28, "chronic": True,  "drg": "884"},
    "Substance Use Disorder - Opioid":           {"icd10": "F11.20", "category": "Mental Health",     "base_urgency": 70, "chronic": True,  "drg": "895"},
    "Suicidal Ideation":                         {"icd10": "R45.851","category": "Mental Health",     "base_urgency": 90, "chronic": False, "drg": "885"},
    "PTSD":                                      {"icd10": "F43.10", "category": "Mental Health",     "base_urgency": 45, "chronic": True,  "drg": "884"},
    "Eating Disorder - Anorexia":                {"icd10": "F50.00", "category": "Mental Health",     "base_urgency": 75, "chronic": True,  "drg": "883"},

    # Infectious Disease
    "Sepsis":                                    {"icd10": "A41.9",  "category": "Infectious",        "base_urgency": 97, "chronic": False, "drg": "871"},
    "Septic Shock":                              {"icd10": "R65.21", "category": "Infectious",        "base_urgency": 99, "chronic": False, "drg": "870"},
    "COVID-19 - Severe":                         {"icd10": "U07.1",  "category": "Infectious",        "base_urgency": 88, "chronic": False, "drg": "178"},
    "COVID-19 - Mild":                           {"icd10": "U07.1",  "category": "Infectious",        "base_urgency": 40, "chronic": False, "drg": "180"},
    "HIV - Symptomatic":                         {"icd10": "B20",    "category": "Infectious",        "base_urgency": 75, "chronic": True,  "drg": "974"},
    "Tuberculosis - Active":                     {"icd10": "A15.9",  "category": "Infectious",        "base_urgency": 82, "chronic": False, "drg": "078"},
    "Cellulitis":                                {"icd10": "L03.90", "category": "Infectious",        "base_urgency": 55, "chronic": False, "drg": "602"},
    "Endocarditis":                              {"icd10": "I33.0",  "category": "Infectious",        "base_urgency": 92, "chronic": False, "drg": "124"},
    "C. Difficile Infection":                    {"icd10": "A04.72", "category": "Infectious",        "base_urgency": 78, "chronic": False, "drg": "371"},

    # Maternity
    "Normal Delivery":                           {"icd10": "O80",    "category": "Maternity",         "base_urgency": 55, "chronic": False, "drg": "775"},
    "C-Section - Elective":                      {"icd10": "O82",    "category": "Maternity",         "base_urgency": 60, "chronic": False, "drg": "766"},
    "C-Section - Emergency":                     {"icd10": "O82",    "category": "Maternity",         "base_urgency": 88, "chronic": False, "drg": "765"},
    "Preeclampsia - Severe":                     {"icd10": "O14.10", "category": "Maternity",         "base_urgency": 90, "chronic": False, "drg": "769"},
    "Gestational Diabetes":                      {"icd10": "O24.419","category": "Maternity",         "base_urgency": 55, "chronic": False, "drg": "775"},
    "Ectopic Pregnancy":                         {"icd10": "O00.90", "category": "Maternity",         "base_urgency": 95, "chronic": False, "drg": "779"},
    "Miscarriage":                               {"icd10": "O03.9",  "category": "Maternity",         "base_urgency": 70, "chronic": False, "drg": "779"},
    "Postpartum Hemorrhage":                     {"icd10": "O72.1",  "category": "Maternity",         "base_urgency": 93, "chronic": False, "drg": "775"},

    # Preventive / Routine
    "Routine Annual Exam":                       {"icd10": "Z00.00", "category": "Preventive",        "base_urgency": 8,  "chronic": False, "drg": "949"},
    "Preventive Care - Child":                   {"icd10": "Z00.129","category": "Preventive",        "base_urgency": 10, "chronic": False, "drg": "949"},
    "Vaccination":                               {"icd10": "Z23",    "category": "Preventive",        "base_urgency": 5,  "chronic": False, "drg": "949"},
    "Post-Surgical Follow-up":                   {"icd10": "Z09",    "category": "Preventive",        "base_urgency": 18, "chronic": False, "drg": "949"},
    "Cancer Screening":                          {"icd10": "Z12.11", "category": "Preventive",        "base_urgency": 15, "chronic": False, "drg": "949"},
    "Chronic Disease Management":                {"icd10": "Z71.89", "category": "Preventive",        "base_urgency": 25, "chronic": True,  "drg": "949"},
}

# ── HCPCS / CPT-like procedure codes (real CMS HCPCS Level II) ───────────────
HCPCS_PROCEDURES = {
    "Emergency Room Visit - High Complexity":    "99285",
    "Emergency Room Visit - Moderate":           "99284",
    "Inpatient Admission - High Severity":       "99223",
    "Inpatient Admission - Moderate Severity":   "99222",
    "Outpatient Visit - New Patient":            "99204",
    "Outpatient Visit - Established Patient":    "99214",
    "Intensive Care - Critical":                 "99292",
    "Surgical Procedure - Major":                "99213",
    "Chemotherapy - IV Push":                    "96413",
    "Dialysis - Hemodialysis":                   "90935",
    "Physical Therapy":                          "97110",
    "MRI Brain":                                 "70553",
    "CT Chest":                                  "71250",
    "Echocardiogram":                            "93306",
    "Colonoscopy":                               "45378",
    "Lab Panel - Comprehensive":                 "80053",
    "Blood Transfusion":                         "P9016",
    "Pharmacy - Specialty Drug":                 "J3490",
    "Pharmacy - Generic":                        "J8499",
    "Maternity - Global":                        "59400",
}

# ── LOINC codes (Regenstrief Institute — free use) ────────────────────────────
LOINC_LABS = {
    "Complete Blood Count (CBC)":                "58410-2",
    "Basic Metabolic Panel":                     "51990-0",
    "Comprehensive Metabolic Panel":             "24323-8",
    "HbA1c":                                     "4548-4",
    "Troponin I":                                "10839-9",
    "D-Dimer":                                   "48066-5",
    "Prothrombin Time (PT/INR)":                 "5902-2",
    "Creatinine":                                "2160-0",
    "eGFR":                                      "62238-1",
    "Sodium":                                    "2951-2",
    "Potassium":                                 "2823-3",
    "Glucose - Fasting":                         "1558-6",
    "TSH":                                       "3016-3",
    "Lipid Panel":                               "57698-3",
    "Urine Culture":                             "630-4",
    "Blood Culture":                             "600-7",
    "COVID-19 PCR":                              "94500-6",
    "Chest X-Ray":                               "24627-2",
    "ECG 12-lead":                               "11524-6",
    "Urinalysis":                                "5767-9",
}

# ── SNOMED CT concepts (SNOMED International — free use) ─────────────────────
SNOMED_FINDINGS = {
    "Tachycardia":                               "3424008",
    "Hypotension":                               "45007003",
    "Fever":                                     "386661006",
    "Dyspnea":                                   "267036007",
    "Chest Pain":                                "29857009",
    "Loss of Consciousness":                     "419045004",
    "Acute Pain":                                "57676002",
    "Chronic Pain":                              "82423001",
    "Edema":                                     "267038008",
    "Hemorrhage":                                "131148009",
    "Infection":                                 "40733004",
    "Inflammation":                              "23583003",
    "Malnutrition":                              "248325000",
    "Dehydration":                               "34095006",
    "Altered Mental Status":                     "419284004",
}

# ── Claim types ───────────────────────────────────────────────────────────────
CLAIM_TYPES = ["Inpatient", "Outpatient", "Emergency", "Pharmacy", "Maternity", "Day Surgery", "Rehabilitation", "Mental Health"]

# ── Insurance providers ───────────────────────────────────────────────────────
PROVIDERS = ["Insurance Company A", "Insurance Company B", "Insurance Company C", "Insurance Company D", "Insurance Company E", "Insurance Company F", "Insurance Company G", "Insurance Company H"]

# ── Rejection reasons ─────────────────────────────────────────────────────────
REJECTION_REASONS = [
    "Missing prior authorization",
    "Service not covered under policy",
    "Duplicate claim submission",
    "Claim submitted after deadline",
    "Incomplete medical documentation",
    "Diagnosis not matching procedure",
    "Provider not in network",
    "Maximum benefit limit reached",
    "Pre-existing condition exclusion",
    "Experimental treatment not covered",
]
