import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from medical_codes import (ICD10_CONDITIONS, HCPCS_PROCEDURES, LOINC_LABS,
                            SNOMED_FINDINGS, CLAIM_TYPES, PROVIDERS, REJECTION_REASONS)

st.set_page_config(page_title="ClaimIQ Pro", page_icon="🏥", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
*, html, body, [class*="css"] { font-family:'Inter',sans-serif !important; }
[data-testid="stSidebar"] { background:#0d1b2a !important; border-right:1px solid #1e3a5f; }
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,[data-testid="stSidebar"] span { color:#94a3b8 !important; }
.main .block-container { padding-top:1.5rem; padding-bottom:2rem; max-width:1400px; }
.kpi { background:white; border-radius:14px; border:1px solid #e8edf3; padding:1.1rem 1.4rem; }
.kpi-val { font-size:1.9rem; font-weight:800; line-height:1.1; }
.kpi-label { font-size:0.69rem; text-transform:uppercase; letter-spacing:.07em; color:#64748b; margin-top:3px; }
.kpi-delta { font-size:0.76rem; margin-top:4px; }
.score-box { background:white; border-radius:16px; border:1px solid #e8edf3; padding:1.6rem; text-align:center; }
.score-num { font-size:4rem; font-weight:800; line-height:1; }
.badge { display:inline-block; padding:4px 13px; border-radius:99px; font-size:0.72rem; font-weight:700; letter-spacing:.05em; }
.b-high { background:#fee2e2; color:#991b1b; }
.b-medium { background:#fef3c7; color:#92400e; }
.b-low { background:#dcfce7; color:#166534; }
.b-escalate { background:#fce7f3; color:#9d174d; }
.b-review { background:#fef3c7; color:#92400e; }
.b-approve { background:#dcfce7; color:#166534; }
.reason { background:#f8fafc; border-left:3px solid #3b82f6; padding:.55rem 1rem; border-radius:0 8px 8px 0; margin:.3rem 0; font-size:.84rem; color:#1e293b; }
.cpill { background:#eff6ff; color:#1d4ed8; padding:3px 9px; border-radius:6px; font-size:.76rem; font-family:monospace; margin-right:4px; }
.gpill { background:#f0fdf4; color:#166534; padding:3px 9px; border-radius:6px; font-size:.76rem; font-family:monospace; margin-right:4px; }
.ppill { background:#fdf4ff; color:#7e22ce; padding:3px 9px; border-radius:6px; font-size:.76rem; font-family:monospace; margin-right:4px; }
.sh { font-size:1rem; font-weight:700; color:#0f172a; margin:1.3rem 0 .7rem; padding-bottom:.4rem; border-bottom:2px solid #f1f5f9; }
.ph h1 { font-size:1.7rem; font-weight:800; color:#0f172a; margin:0; }
.ph p { font-size:.88rem; color:#64748b; margin:.2rem 0 1rem; }
.verr { color:#ef4444; font-size:.76rem; margin-top:2px; }
div[data-testid="stMetricValue"] { font-size:1.5rem !important; font-weight:700 !important; }
.stButton > button { border-radius:9px !important; font-weight:600 !important; padding:.45rem 1.2rem !important; }
.info-box { background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:1rem 1.2rem; font-size:.85rem; color:#1e40af; margin:.5rem 0; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all():
    m   = joblib.load("triage_model.pkl")
    iso = joblib.load("anomaly_model.pkl")
    enc = {col: joblib.load(f"le_{col}.pkl")
           for col in ["claim_type","diagnosis","category","provider","patient_gender"]}
    with open("encoder_classes.json") as f:
        cls = json.load(f)
    return m, iso, enc, cls

@st.cache_data
def load_data():
    return pd.read_csv("claims_data.csv")

model, iso_model, encoders, classes = load_all()
df = load_data()
DIAGNOSES = sorted(ICD10_CONDITIONS.keys())
C = {"High":"#ef4444","Medium":"#f59e0b","Low":"#22c55e",
     "Escalate":"#dc2626","Review":"#d97706","Approve":"#16a34a"}

def fs(ax, fig):
    fig.patch.set_facecolor("white"); ax.set_facecolor("#f8fafc")
    ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=8)

def score_one(ct, diag, age, amount, days, auth, chronic, missing, readm, prev, comorbid, gender, provider):
    errors = []
    if not ct:      errors.append("Claim type is required")
    if not diag:    errors.append("Diagnosis is required")
    if age <= 0:    errors.append("Valid patient age is required")
    if amount <= 0: errors.append("Claim amount must be greater than 0")
    if errors: return None, errors

    info   = ICD10_CONDITIONS[diag]
    cat    = info["category"]
    ct_e   = encoders["claim_type"].transform([ct])[0]
    diag_e = encoders["diagnosis"].transform([diag])[0]
    cat_e  = encoders["category"].transform([cat])[0]
    prov_e = encoders["provider"].transform([provider])[0] if provider in classes["provider"] else 0
    gen_e  = encoders["patient_gender"].transform([gender])[0]

    _FEAT = ["claim_type_enc","diagnosis_enc","category_enc","patient_age",
             "claim_amount_usd","days_since_submission","has_prior_auth",
             "is_chronic","missing_documents","is_readmission","num_prev_claims","comorbidities"]
    X = pd.DataFrame([[ct_e, diag_e, cat_e, age, amount, days,
                       1 if auth=="Yes" else 0, int(chronic),
                       1 if missing=="Yes" else 0, int(readm), prev, comorbid]],
                     columns=_FEAT)

    score = float(np.clip(model.predict(X)[0], 1, 100))
    anom  = float(np.clip(-iso_model.score_samples(X)[0]*40, 0, 100))

    priority = "High" if score>=70 else ("Medium" if score>=40 else "Low")
    rec      = ("Escalate" if (score>=80 or (auth=="No" and amount>5000))
                else ("Review" if (score>=50 or missing=="Yes") else "Approve"))
    eta      = 1 if score>=70 else (3 if score>=40 else 7)

    reasons = []
    if score>=70:      reasons.append(f"🔴 High-acuity diagnosis: {diag} ({info['icd10']})")
    if ct=="Emergency":reasons.append("🚨 Emergency claim — automatic priority boost")
    if age>70:         reasons.append(f"👴 Age {age} — elevated clinical risk")
    if age<5:          reasons.append(f"👶 Paediatric patient — elevated risk")
    if amount>20000:   reasons.append(f"💰 High value ${amount:,.0f} — financial risk flag")
    if readm:          reasons.append("🔁 Readmission — complication signal")
    if auth=="No":     reasons.append("⚠️ No prior authorisation on record")
    if missing=="Yes": reasons.append("📄 Missing documents — processing risk")
    if days>30:        reasons.append(f"⏰ {days} days pending — SLA breach risk")
    if chronic:        reasons.append("🩺 Chronic condition — ongoing management")
    if comorbid>=3:    reasons.append(f"⚕️ {comorbid} comorbidities — complex case")
    if anom>50:        reasons.append(f"🚩 Anomaly score {anom:.0f} — flag for review")
    if not reasons:    reasons.append("✅ Standard claim — no high-risk flags")

    return {"score":round(score,1),"priority":priority,"rec":rec,"anom":round(anom,1),
            "eta":eta,"reasons":reasons[:6],"icd10":info["icd10"],
            "drg":info["drg"],"category":cat}, []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:.8rem 0 .4rem;'>
      <div style='font-size:1.35rem;font-weight:800;color:#f1f5f9;'>🏥 ClaimIQ Pro</div>
      <div style='font-size:.7rem;color:#475569;margin-top:2px;'>Intelligent Claims Platform v2.0</div>
    </div>
    <hr style='border-color:#1e3a5f;margin:.5rem 0 .8rem;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "📊  Dashboard",
        "📋  Score a Claim",
        "📤  Batch Upload",
        "👤  Patient Profile",
        "🛡️  Fraud & Anomaly",
        "📈  Analytics",
        "📖  Code Reference",
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:#1e3a5f;margin:.8rem 0;'>
    <div style='font-size:.67rem;color:#334155;line-height:1.9;'>
      <b style='color:#64748b;'>Dataset</b><br>
      14,000 claims · 109 diagnoses<br>
      1,800 patient profiles<br><br>
      <b style='color:#64748b;'>Medical Codes</b><br>
      ICD-10-CM (CMS/WHO · public domain)<br>
      MS-DRG (CMS · public domain)<br>
      HCPCS Level II (CMS · public domain)<br>
      LOINC (Regenstrief · free licence)<br>
      SNOMED CT (free tier)<br><br>
      <b style='color:#64748b;'>ML Models</b><br>
      Gradient Boosting · R²=0.98<br>
      Isolation Forest (anomaly)<br><br>
      <b style='color:#64748b;'>Built by</b><br>
      Rana Musab Bin Tariq<br>
      MSc Digital Health · DIT Germany
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Dashboard":
    st.markdown('<div class="ph"><h1>📊 Live Claims Dashboard</h1><p>Real-time portfolio overview · 14,000 claims · ICD-10-CM coded · ML-triaged</p></div>', unsafe_allow_html=True)

    total=len(df); high=int((df.priority=="High").sum()); med=int((df.priority=="Medium").sum())
    low=int((df.priority=="Low").sum()); flags=int((df.anomaly_score>50).sum())
    avg_sc=df.triage_score.mean(); tot_sp=df.claim_amount_usd.sum()

    cols = st.columns(7)
    data = [
        (f"{total:,}","Total Claims","#0f172a",""),
        (f"{high:,}","High Priority","#ef4444",f"🔴 {high/total*100:.1f}%"),
        (f"{med:,}","Medium Priority","#f59e0b",f"🟡 {med/total*100:.1f}%"),
        (f"{low:,}","Low Priority","#22c55e",f"🟢 {low/total*100:.1f}%"),
        (f"{avg_sc:.1f}","Avg Triage Score","#3b82f6",""),
        (f"${tot_sp/1e6:.1f}M","Total Spend","#7c3aed",""),
        (f"{flags:,}","Anomaly Flags","#dc2626","⚠️ Review queue"),
    ]
    for col,(val,label,color,delta) in zip(cols,data):
        col.markdown(f'<div class="kpi"><div class="kpi-val" style="color:{color}">{val}</div>'
                     f'<div class="kpi-label">{label}</div><div class="kpi-delta">{delta}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns([1.4,1])
    with cl:
        st.markdown('<p class="sh">Monthly claims — priority stack</p>', unsafe_allow_html=True)
        m = df.groupby(["year","month","priority"]).size().unstack(fill_value=0).reset_index()
        m["period"] = m["year"].astype(str)+"-"+m["month"].astype(str).str.zfill(2)
        m = m.sort_values("period")
        fig,ax = plt.subplots(figsize=(8,3.2)); fs(ax,fig)
        x = np.arange(len(m))
        lo_v = m.get("Low",pd.Series([0]*len(m))).values
        me_v = m.get("Medium",pd.Series([0]*len(m))).values
        hi_v = m.get("High",pd.Series([0]*len(m))).values
        ax.bar(x, lo_v, color="#22c55e", alpha=.85, width=.7, label="Low")
        ax.bar(x, me_v, color="#f59e0b", alpha=.85, width=.7, bottom=lo_v, label="Medium")
        ax.bar(x, hi_v, color="#ef4444", alpha=.85, width=.7, bottom=lo_v+me_v, label="High")
        step = max(1,len(m)//10)
        ax.set_xticks(x[::step]); ax.set_xticklabels(m["period"].iloc[::step], rotation=40, fontsize=7)
        ax.legend(fontsize=8,frameon=False); ax.set_ylabel("Claims",fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cr:
        st.markdown('<p class="sh">Recommendation split</p>', unsafe_allow_html=True)
        rc = df.recommendation.value_counts()
        fig,ax = plt.subplots(figsize=(4,3.2)); fig.patch.set_facecolor("white")
        w,t,at = ax.pie(rc.values, labels=rc.index, autopct="%1.1f%%",
                        colors=[C.get(r,"#94a3b8") for r in rc.index],
                        startangle=90, wedgeprops={"edgecolor":"white","linewidth":2})
        for tx in t:   tx.set_fontsize(9)
        for at_ in at: at_.set_fontsize(8); at_.set_color("white"); at_.set_fontweight("bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("<hr style='border-color:#f1f5f9;'>", unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<p class="sh">Top 12 diagnoses by volume</p>', unsafe_allow_html=True)
        td = df.diagnosis.value_counts().head(12)
        fig,ax = plt.subplots(figsize=(6,4)); fs(ax,fig)
        ax.barh(range(len(td)), td.values, color="#3b82f6", alpha=.82)
        ax.set_yticks(range(len(td)))
        ax.set_yticklabels([d[:36]+"…" if len(d)>36 else d for d in td.index], fontsize=7.5)
        for i,v in enumerate(td.values): ax.text(v+3,i,str(v),va="center",fontsize=7.5,color="#475569")
        ax.set_xlabel("Claims",fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with cb:
        st.markdown('<p class="sh">Avg claim amount by type</p>', unsafe_allow_html=True)
        ct_amt = df.groupby("claim_type")["claim_amount_usd"].mean().sort_values()
        fig,ax = plt.subplots(figsize=(6,4)); fs(ax,fig)
        bar_cs = ["#6366f1","#8b5cf6","#a855f7","#d946ef","#ec4899","#f43f5e","#f97316","#eab308"]
        ax.barh(ct_amt.index, ct_amt.values, color=bar_cs[:len(ct_amt)], alpha=.85)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        ax.set_xlabel("Avg Claim (USD)",fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<p class="sh">Most recent 25 claims</p>', unsafe_allow_html=True)
    recent = df.tail(25)[["claim_id","patient_id","diagnosis","icd10_code","drg_code",
                          "claim_type","claim_amount_usd","triage_score","priority",
                          "recommendation","anomaly_score"]].copy()
    recent["claim_amount_usd"] = recent["claim_amount_usd"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(recent, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# SCORE A CLAIM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋  Score a Claim":
    st.markdown('<div class="ph"><h1>📋 Score a Claim</h1><p>Enter claim details. Required fields marked ★. AI model scores urgency, recommends action, and detects anomalies.</p></div>', unsafe_allow_html=True)

    cf, cr = st.columns([1.05,1], gap="large")
    with cf:
        st.markdown('<p class="sh">★ Claim information</p>', unsafe_allow_html=True)
        claim_type = st.selectbox("Claim Type ★", [""]+CLAIM_TYPES)
        if not claim_type: st.markdown('<p class="verr">★ Required</p>', unsafe_allow_html=True)

        diagnosis = st.selectbox("Diagnosis ★", [""]+DIAGNOSES)
        if not diagnosis:
            st.markdown('<p class="verr">★ Required</p>', unsafe_allow_html=True)
        elif diagnosis in ICD10_CONDITIONS:
            info = ICD10_CONDITIONS[diagnosis]
            st.markdown(f'<span class="cpill">ICD-10: {info["icd10"]}</span>'
                        f'<span class="gpill">DRG: {info["drg"]}</span>'
                        f'<span class="ppill">{info["category"]}</span>'
                        +(' <span class="badge b-review">Chronic</span>' if info["chronic"] else ""),
                        unsafe_allow_html=True)

        st.markdown('<p class="sh">★ Patient details</p>', unsafe_allow_html=True)
        pa1,pa2 = st.columns(2)
        age    = pa1.number_input("Age ★", 0, 120, 0)
        gender = pa2.selectbox("Gender", ["Male","Female"])
        if age<=0: st.markdown('<p class="verr">★ Required — must be > 0</p>', unsafe_allow_html=True)

        st.markdown('<p class="sh">★ Financial & administrative</p>', unsafe_allow_html=True)
        fb1,fb2 = st.columns(2)
        amount = fb1.number_input("Claim Amount (USD) ★", 0, 2000000, 0, step=100)
        days   = fb2.number_input("Days Since Submission", 0, 730, 0)
        if amount<=0: st.markdown('<p class="verr">★ Required — must be > 0</p>', unsafe_allow_html=True)

        fc1,fc2 = st.columns(2)
        auth     = fc1.selectbox("Prior Authorisation ★", ["Yes","No"])
        provider = fc2.selectbox("Provider", PROVIDERS)

        fd1,fd2,fd3 = st.columns(3)
        readm    = fd1.selectbox("Readmission?", ["No","Yes"])
        missing  = fd2.selectbox("Missing Docs?", ["No","Yes"])
        comorbid = fd3.number_input("Comorbidities", 0, 20, 0)

        st.markdown('<p class="sh">Clinical history</p>', unsafe_allow_html=True)
        prev    = st.slider("Previous claims (this patient)", 0, 30, 0)
        chronic = st.checkbox("Mark as chronic condition",
                               value=bool(ICD10_CONDITIONS.get(diagnosis,{}).get("chronic",False)) if diagnosis else False)

        st.markdown('<p class="sh">Optional clinical codes</p>', unsafe_allow_html=True)
        oc1,oc2 = st.columns(2)
        loinc_s = oc1.selectbox("LOINC Lab", ["—"]+sorted(LOINC_LABS.keys()))
        hcpcs_s = oc2.selectbox("HCPCS Procedure", ["—"]+sorted(HCPCS_PROCEDURES.keys()))
        if loinc_s!="—": st.markdown(f'LOINC: <span class="cpill">{LOINC_LABS[loinc_s]}</span>', unsafe_allow_html=True)
        if hcpcs_s!="—": st.markdown(f'HCPCS: <span class="cpill">{HCPCS_PROCEDURES[hcpcs_s]}</span>', unsafe_allow_html=True)
        snomed_s = st.selectbox("SNOMED CT Finding", ["—"]+sorted(SNOMED_FINDINGS.keys()))
        if snomed_s!="—": st.markdown(f'SNOMED: <span class="cpill">{SNOMED_FINDINGS[snomed_s]}</span>', unsafe_allow_html=True)

        submitted = st.button("⚡ Calculate Triage Score", use_container_width=True, type="primary")

    with cr:
        st.markdown('<p class="sh">Result</p>', unsafe_allow_html=True)
        if submitted:
            result, errors = score_one(claim_type,diagnosis,age,amount,days,auth,chronic,
                                       missing,readm=="Yes",prev,comorbid,gender,provider)
            if errors:
                for e in errors: st.markdown(f'<p class="verr">❌ {e}</p>', unsafe_allow_html=True)
            else:
                sc=result["score"]; pr=result["priority"]; rc=result["rec"]
                sc_col=C[pr]
                st.markdown(f"""
                <div class="score-box">
                  <div class="score-num" style="color:{sc_col}">{sc}</div>
                  <div style="font-size:.76rem;color:#94a3b8;margin:5px 0 10px;">triage score / 100</div>
                  <span class="badge b-{pr.lower()}">{pr} Priority</span>
                  &nbsp;
                  <span class="badge b-{rc.lower()}">{rc}</span>
                </div>""", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                m1,m2,m3 = st.columns(3)
                m1.metric("ICD-10",result["icd10"])
                m2.metric("DRG",result["drg"])
                m3.metric("Turnaround",f"{result['eta']}d")
                m4,m5 = st.columns(2)
                m4.metric("Category",result["category"])
                m5.metric("Anomaly",f"{result['anom']:.0f}/100",
                          delta="⚠️ Flag" if result["anom"]>50 else "✅ Clean",
                          delta_color="inverse" if result["anom"]>50 else "normal")

                st.markdown('<p class="sh">Scoring factors</p>', unsafe_allow_html=True)
                for r in result["reasons"]: st.markdown(f'<div class="reason">{r}</div>', unsafe_allow_html=True)

                st.markdown('<p class="sh">Adjudicator action</p>', unsafe_allow_html=True)
                if rc=="Escalate": st.error("🚨 **Escalate immediately** — forward to senior adjudicator.")
                elif rc=="Review": st.warning("🔍 **Manual review required** — verify docs and coverage.")
                else:              st.success("✅ **Approve** — meets criteria. Standard workflow.")

                if result["anom"]>70:   st.error(f"🚩 High anomaly ({result['anom']:.0f}/100) — fraud investigation recommended.")
                elif result["anom"]>50: st.warning(f"⚠️ Moderate anomaly ({result['anom']:.0f}/100) — review for irregularities.")

                # Score gauge
                st.markdown('<p class="sh">Score gauge</p>', unsafe_allow_html=True)
                fig,ax = plt.subplots(figsize=(5,2.2),subplot_kw={"aspect":"equal"})
                fig.patch.set_facecolor("white")
                for col_g,lo_g,hi_g in [("#22c55e",0,40),("#f59e0b",40,70),("#ef4444",70,100)]:
                    t1=np.pi-(lo_g/100)*np.pi; t2=np.pi-(hi_g/100)*np.pi
                    th=np.linspace(t1,t2,50)
                    ax.fill_between(np.cos(th),np.sin(th)*.7,np.sin(th),color=col_g,alpha=.85)
                needle=np.pi-(sc/100)*np.pi
                ax.plot([0,np.cos(needle)*.82],[0,np.sin(needle)*.82],color="#0f172a",lw=2.5,solid_capstyle="round")
                ax.add_patch(plt.Circle((0,0),.07,color="#0f172a"))
                ax.set_xlim(-1.1,1.1); ax.set_ylim(-.15,1.1)
                ax.text(0,-.1,f"{sc}",ha="center",va="center",fontsize=18,fontweight="bold",color=sc_col)
                ax.axis("off"); plt.tight_layout(); st.pyplot(fig); plt.close()
        else:
            st.markdown('<div class="info-box">Fill in the required fields (★) and click <b>Calculate Triage Score</b>.</div>', unsafe_allow_html=True)
            with st.expander("How the scoring model works"):
                st.markdown("""
**Model:** Gradient Boosting Regressor · R² = 0.98 · MAE ≈ 2.64

| Score | Priority | Action | ETA |
|-------|----------|--------|-----|
| 70–100 | 🔴 High | Escalate | Same day |
| 40–69 | 🟡 Medium | Review | 3 days |
| 1–39 | 🟢 Low | Approve | 7 days |

**Anomaly detection:** Isolation Forest flags unusual claims for fraud review.
**Codes:** ICD-10-CM · MS-DRG · HCPCS · LOINC · SNOMED CT — all public domain / free licence.
                """)

# ══════════════════════════════════════════════════════════════════════════════
# BATCH UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📤  Batch Upload":
    st.markdown('<div class="ph"><h1>📤 Batch Claim Scoring</h1><p>Upload a CSV of claims — system validates, scores every row, and exports results.</p></div>', unsafe_allow_html=True)

    REQUIRED = ["claim_type","diagnosis","patient_age","claim_amount_usd"]

    with st.expander("📥 Download CSV template"):
        tmpl = pd.DataFrame([
            {"claim_type":"Emergency","diagnosis":"Sepsis","patient_age":68,"claim_amount_usd":18500,
             "days_since_submission":2,"has_prior_auth":1,"is_chronic":0,"missing_documents":0,
             "is_readmission":1,"num_prev_claims":3,"comorbidities":2,"patient_gender":"Male","provider":"AXA Health"},
            {"claim_type":"Inpatient","diagnosis":"Hip Fracture","patient_age":79,"claim_amount_usd":32000,
             "days_since_submission":1,"has_prior_auth":1,"is_chronic":0,"missing_documents":0,
             "is_readmission":0,"num_prev_claims":2,"comorbidities":3,"patient_gender":"Female","provider":"Bupa International"},
            {"claim_type":"Outpatient","diagnosis":"Routine Annual Exam","patient_age":35,"claim_amount_usd":180,
             "days_since_submission":5,"has_prior_auth":1,"is_chronic":0,"missing_documents":0,
             "is_readmission":0,"num_prev_claims":1,"comorbidities":0,"patient_gender":"Female","provider":"Cigna Global"},
        ])
        st.dataframe(tmpl, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Download template CSV", tmpl.to_csv(index=False).encode(),"claimiq_template.csv","text/csv")

    uploaded = st.file_uploader("Upload claims CSV", type=["csv"])
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            st.info(f"📂 Loaded {len(raw)} rows · {len(raw.columns)} columns")
            miss_cols = [c for c in REQUIRED if c not in raw.columns]
            if miss_cols: st.error(f"Missing columns: {', '.join(miss_cols)}"); st.stop()

            val_msgs = []
            for _,row in raw.iterrows():
                errs=[]
                if str(row.get("claim_type","")) not in CLAIM_TYPES: errs.append("invalid claim_type")
                if str(row.get("diagnosis","")) not in ICD10_CONDITIONS: errs.append("unknown diagnosis")
                if pd.isna(row.get("patient_age")) or float(row.get("patient_age",0))<=0: errs.append("invalid age")
                if pd.isna(row.get("claim_amount_usd")) or float(row.get("claim_amount_usd",0))<=0: errs.append("invalid amount")
                val_msgs.append(", ".join(errs) if errs else "✅ OK")
            raw["validation"]=val_msgs

            n_bad=int((raw.validation!="✅ OK").sum()); n_good=len(raw)-n_bad
            vc1,vc2,vc3=st.columns(3)
            vc1.metric("Total",len(raw)); vc2.metric("Valid",n_good)
            vc3.metric("Invalid",n_bad,delta=f"-{n_bad}" if n_bad else "None",delta_color="inverse" if n_bad else "off")
            if n_bad>0:
                st.warning(f"{n_bad} invalid rows will be skipped.")
                with st.expander("View invalid rows"):
                    st.dataframe(raw[raw.validation!="✅ OK"][REQUIRED+["validation"]],use_container_width=True,hide_index=True)

            valid=raw[raw.validation=="✅ OK"].copy()
            if len(valid)==0: st.error("No valid rows."); st.stop()

            prog=st.progress(0,text="Scoring…")
            scores,priorities,recs,anoms,icd10s,drgs=[],[],[],[],[],[]
            for i,(_,row) in enumerate(valid.iterrows()):
                diag=str(row["diagnosis"]); ct=str(row["claim_type"])
                info=ICD10_CONDITIONS.get(diag,{})
                res,_=score_one(ct,diag,int(row["patient_age"]),float(row["claim_amount_usd"]),
                    int(row.get("days_since_submission",0)),
                    "Yes" if int(row.get("has_prior_auth",1)) else "No",
                    bool(info.get("chronic",row.get("is_chronic",0))),
                    "Yes" if int(row.get("missing_documents",0)) else "No",
                    bool(int(row.get("is_readmission",0))),
                    int(row.get("num_prev_claims",0)),int(row.get("comorbidities",0)),
                    str(row.get("patient_gender","Male")),str(row.get("provider","AXA Health")))
                if res:
                    scores.append(res["score"]); priorities.append(res["priority"])
                    recs.append(res["rec"]);      anoms.append(res["anom"])
                    icd10s.append(res["icd10"]);  drgs.append(res["drg"])
                else:
                    scores.append(None); priorities.append("Error"); recs.append("Error")
                    anoms.append(None);  icd10s.append(""); drgs.append("")
                prog.progress((i+1)/len(valid),text=f"Scoring {i+1}/{len(valid)}…")
            prog.empty()

            valid=valid.copy()
            valid["triage_score"]=scores; valid["priority"]=priorities
            valid["recommendation"]=recs; valid["anomaly_score"]=anoms
            valid["icd10_code"]=icd10s;   valid["drg_code"]=drgs
            st.success(f"✅ Scored {len(valid)} claims")

            r1,r2,r3,r4,r5=st.columns(5)
            r1.metric("High",  (valid.priority=="High").sum())
            r2.metric("Medium",(valid.priority=="Medium").sum())
            r3.metric("Low",   (valid.priority=="Low").sum())
            r4.metric("Escalate",(valid.recommendation=="Escalate").sum())
            r5.metric("Anomaly flags",(valid.anomaly_score>50).sum())

            fig,axes=plt.subplots(1,2,figsize=(10,3)); fig.patch.set_facecolor("white")
            pc=valid.priority.value_counts()
            axes[0].pie(pc.values,labels=pc.index,autopct="%1.0f%%",
                        colors=[C.get(p,"#94a3b8") for p in pc.index],
                        wedgeprops={"edgecolor":"white","linewidth":2},startangle=90)
            axes[0].set_title("Priority split",fontsize=10,fontweight="bold")
            axes[1].hist(valid.triage_score.dropna(),bins=20,color="#3b82f6",alpha=.8,edgecolor="white")
            axes[1].set_xlabel("Triage Score",fontsize=9); axes[1].set_ylabel("Claims",fontsize=9)
            axes[1].set_title("Score distribution",fontsize=10,fontweight="bold")
            axes[1].set_facecolor("#f8fafc"); axes[1].spines[["top","right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            showcols=[c for c in ["claim_type","diagnosis","icd10_code","drg_code","patient_age",
                                  "claim_amount_usd","triage_score","priority","recommendation",
                                  "anomaly_score","validation"] if c in valid.columns]
            st.dataframe(valid[showcols].head(200),use_container_width=True,hide_index=True)
            st.download_button("⬇️ Download scored CSV",valid.to_csv(index=False).encode(),
                               "claimiq_scored.csv","text/csv",use_container_width=True)
        except Exception as e:
            st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PATIENT PROFILE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤  Patient Profile":
    st.markdown('<div class="ph"><h1>👤 Patient Risk Profile</h1><p>Full claim history, risk trajectory, and clinical summary for any patient.</p></div>', unsafe_allow_html=True)

    pid = st.selectbox("Select Patient ID", sorted(df.patient_id.unique()))
    if pid:
        pat=df[df.patient_id==pid].sort_values(["year","month"])
        avg_sc=pat.triage_score.mean(); max_sc=pat.triage_score.max()
        tot_sp=pat.claim_amount_usd.sum(); readmit=int(pat.is_readmission.sum())
        chronic=pat.is_chronic.any(); age_val=int(pat.patient_age.iloc[-1])
        gen_val=pat.patient_gender.iloc[-1]; anom_cl=int((pat.anomaly_score>50).sum())
        risk_col="#ef4444" if avg_sc>=70 else ("#f59e0b" if avg_sc>=40 else "#22c55e")

        st.markdown(f"""
        <div style="background:white;border-radius:14px;border:1px solid #e8edf3;
                    padding:1.2rem 1.5rem;display:flex;align-items:center;gap:1.5rem;margin-bottom:1rem;">
          <div style="width:52px;height:52px;border-radius:50%;background:#eff6ff;
                      display:flex;align-items:center;justify-content:center;
                      font-size:1.1rem;font-weight:800;color:#1d4ed8;">{pid[:2].upper()}</div>
          <div>
            <div style="font-size:1.15rem;font-weight:700;color:#0f172a;">{pid}</div>
            <div style="font-size:.82rem;color:#64748b;">{gen_val} · Age {age_val} ·
              {'<span style="color:#ef4444;font-weight:600;">Chronic on record</span>' if chronic else 'No chronic flags'}</div>
          </div>
          <div style="margin-left:auto;text-align:right;">
            <div style="font-size:1.9rem;font-weight:800;color:{risk_col};">{avg_sc:.1f}</div>
            <div style="font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em;">Avg Risk Score</div>
          </div>
        </div>""", unsafe_allow_html=True)

        m1,m2,m3,m4,m5=st.columns(5)
        m1.metric("Total Claims",len(pat)); m2.metric("Peak Score",f"{max_sc:.0f}")
        m3.metric("Total Spend",f"${tot_sp:,.0f}"); m4.metric("Readmissions",readmit)
        m5.metric("Anomaly Flags",anom_cl)

        cl2,cr2=st.columns([1.6,1])
        with cl2:
            st.markdown('<p class="sh">Triage score over time</p>', unsafe_allow_html=True)
            fig,ax=plt.subplots(figsize=(6,2.8)); fs(ax,fig)
            y=pat.triage_score.values
            ax.plot(range(len(pat)),y,color="#3b82f6",lw=2,marker="o",ms=4,zorder=3)
            ax.fill_between(range(len(pat)),y,alpha=.12,color="#3b82f6")
            ax.axhline(70,color="#ef4444",ls="--",alpha=.45,lw=1,label="High threshold")
            ax.axhline(40,color="#f59e0b",ls="--",alpha=.45,lw=1,label="Medium threshold")
            ax.set_ylabel("Score",fontsize=9); ax.set_xticks([]); ax.legend(fontsize=7,frameon=False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with cr2:
            st.markdown('<p class="sh">Claims by category</p>', unsafe_allow_html=True)
            cc=pat.category.value_counts()
            fig,ax=plt.subplots(figsize=(4,2.8)); fig.patch.set_facecolor("white")
            ax.barh(cc.index,cc.values,color="#6366f1",alpha=.82)
            ax.spines[["top","right"]].set_visible(False); ax.tick_params(labelsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown('<p class="sh">Full claim history</p>', unsafe_allow_html=True)
        show=pat[["year","month","claim_id","claim_type","diagnosis","icd10_code","drg_code",
                  "claim_amount_usd","triage_score","priority","recommendation","anomaly_score"]].copy()
        show["claim_amount_usd"]=show["claim_amount_usd"].apply(lambda x:f"${x:,.0f}")
        st.dataframe(show,use_container_width=True,hide_index=True)

        if readmit>0:  st.error(f"🔁 {readmit} readmission(s) — elevated complication risk.")
        if chronic:    st.warning("⚕️ Chronic condition — ongoing management protocol required.")
        if anom_cl>0:  st.warning(f"🚩 {anom_cl} anomalous claim(s) flagged.")
        if avg_sc>=70: st.error("🔴 High-risk patient — claims consistently require priority handling.")

# ══════════════════════════════════════════════════════════════════════════════
# FRAUD & ANOMALY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛡️  Fraud & Anomaly":
    st.markdown('<div class="ph"><h1>🛡️ Fraud & Anomaly Detection</h1><p>Isolation Forest ML model identifies statistically unusual claims. A flag is a signal for review — not an accusation.</p></div>', unsafe_allow_html=True)

    c1,c2=st.columns([2,1])
    threshold=c1.slider("Anomaly threshold — flag claims above this score",20,90,50)
    c2.markdown("<br>"); c2.info("Lower = more sensitive")

    flagged=df[df.anomaly_score>=threshold]; clean=df[df.anomaly_score<threshold]
    f1,f2,f3,f4,f5=st.columns(5)
    f1.metric("Total",f"{len(df):,}"); f2.metric("Flagged",f"{len(flagged):,}",
              delta=f"{len(flagged)/len(df)*100:.1f}%",delta_color="inverse")
    f3.metric("Clean",f"{len(clean):,}")
    f4.metric("Avg flagged amt",f"${flagged.claim_amount_usd.mean():,.0f}")
    f5.metric("Max anomaly score",f"{df.anomaly_score.max():.0f}")

    col_l,col_r=st.columns(2)
    with col_l:
        st.markdown('<p class="sh">Anomaly score distribution</p>', unsafe_allow_html=True)
        fig,ax=plt.subplots(figsize=(5,2.8)); fs(ax,fig)
        ax.hist(df.anomaly_score,bins=35,color="#6366f1",alpha=.75,edgecolor="white")
        ax.axvline(threshold,color="#ef4444",ls="--",lw=2,label=f"Threshold: {threshold}")
        ax.set_xlabel("Anomaly Score",fontsize=9); ax.set_ylabel("Claims",fontsize=9)
        ax.legend(fontsize=8,frameon=False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.markdown('<p class="sh">Flagged claims by type</p>', unsafe_allow_html=True)
        fbt=flagged.claim_type.value_counts()
        fig,ax=plt.subplots(figsize=(5,2.8)); fs(ax,fig)
        ax.bar(fbt.index,fbt.values,color="#ef4444",alpha=.75,edgecolor="white")
        ax.set_ylabel("Flagged",fontsize=9); ax.tick_params(axis="x",rotation=30,labelsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    col_a,col_b=st.columns(2)
    with col_a:
        st.markdown('<p class="sh">Flagged amount vs anomaly score</p>', unsafe_allow_html=True)
        samp=flagged.sample(min(500,len(flagged)),random_state=42)
        fig,ax=plt.subplots(figsize=(5,2.8)); fs(ax,fig)
        ax.scatter(samp.claim_amount_usd,samp.anomaly_score,color="#ef4444",alpha=.3,s=10)
        ax.set_xlabel("Claim Amount (USD)",fontsize=9); ax.set_ylabel("Anomaly Score",fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x/1e3:.0f}K"))
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        st.markdown('<p class="sh">Flags by diagnosis category</p>', unsafe_allow_html=True)
        fcat=flagged.category.value_counts().head(8)
        fig,ax=plt.subplots(figsize=(5,2.8)); fs(ax,fig)
        ax.barh(fcat.index,fcat.values,color="#f97316",alpha=.82)
        ax.tick_params(labelsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<p class="sh">Investigation queue — top 100 flagged claims</p>', unsafe_allow_html=True)
    queue=flagged.sort_values("anomaly_score",ascending=False).head(100)
    q_show=queue[["claim_id","patient_id","diagnosis","icd10_code","claim_type","provider",
                  "claim_amount_usd","days_since_submission","anomaly_score","triage_score","recommendation"]].copy()
    q_show["claim_amount_usd"]=q_show["claim_amount_usd"].apply(lambda x:f"${x:,.0f}")
    st.dataframe(q_show,use_container_width=True,hide_index=True)
    st.download_button("⬇️ Export investigation queue",queue.to_csv(index=False).encode(),
                       "investigation_queue.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Analytics":
    st.markdown('<div class="ph"><h1>📈 Analytics & Insights</h1><p>Deep-dive into cost drivers, population risk, provider patterns, and clinical trends.</p></div>', unsafe_allow_html=True)

    f1,f2,f3=st.columns(3)
    yr  =f1.multiselect("Year",    sorted(df.year.unique()),     default=sorted(df.year.unique()))
    cat =f2.multiselect("Category",sorted(df.category.unique()), default=sorted(df.category.unique()))
    prov=f3.multiselect("Provider",sorted(df.provider.unique()), default=sorted(df.provider.unique()))
    fdf=df[df.year.isin(yr)&df.category.isin(cat)&df.provider.isin(prov)]

    a1,a2,a3,a4,a5=st.columns(5)
    a1.metric("Claims",f"{len(fdf):,}"); a2.metric("Total spend",f"${fdf.claim_amount_usd.sum()/1e6:.2f}M")
    a3.metric("Avg claim",f"${fdf.claim_amount_usd.mean():,.0f}")
    a4.metric("Avg score",f"{fdf.triage_score.mean():.1f}")
    a5.metric("Approval rate",f"{(fdf.recommendation=='Approve').mean()*100:.1f}%")

    st.markdown("<hr style='border-color:#f1f5f9;'>")
    c1,c2=st.columns(2)
    with c1:
        st.markdown('<p class="sh">Total spend by category</p>', unsafe_allow_html=True)
        cs=fdf.groupby("category").claim_amount_usd.sum().sort_values()
        fig,ax=plt.subplots(figsize=(6,4)); fs(ax,fig)
        cm=plt.cm.get_cmap("Blues",len(cs)+3)
        ax.barh(cs.index,cs.values,color=[cm(i+3) for i in range(len(cs))])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x/1e6:.1f}M" if x>=1e6 else f"${x/1e3:.0f}K"))
        ax.tick_params(labelsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        st.markdown('<p class="sh">Risk score by age group</p>', unsafe_allow_html=True)
        fdf2=fdf.copy()
        fdf2["age_grp"]=pd.cut(fdf2.patient_age,[0,17,35,50,65,120],labels=["0–17","18–35","36–50","51–65","65+"])
        ag=fdf2.groupby("age_grp",observed=True).triage_score.mean()
        fig,ax=plt.subplots(figsize=(6,4)); fs(ax,fig)
        bar_c=["#22c55e","#22c55e","#f59e0b","#f59e0b","#ef4444"]
        ax.bar(ag.index.astype(str),ag.values,color=bar_c[:len(ag)],alpha=.85,edgecolor="white")
        ax.axhline(70,color="#ef4444",ls="--",alpha=.4,lw=1.5)
        ax.axhline(40,color="#f59e0b",ls="--",alpha=.4,lw=1.5)
        ax.set_ylabel("Avg Triage Score",fontsize=9)
        for i,(v,_) in enumerate(zip(ag.values,ag.index)):
            ax.text(i,v+.5,f"{v:.1f}",ha="center",fontsize=8,color="#475569")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    c3,c4=st.columns(2)
    with c3:
        st.markdown('<p class="sh">Claims volume by provider</p>', unsafe_allow_html=True)
        pc=fdf.provider.value_counts()
        fig,ax=plt.subplots(figsize=(6,3.5)); fs(ax,fig)
        ax.bar(pc.index,pc.values,color="#8b5cf6",alpha=.82,edgecolor="white")
        ax.tick_params(axis="x",rotation=35,labelsize=8); ax.set_ylabel("Claims",fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c4:
        st.markdown('<p class="sh">Triage score vs claim amount</p>', unsafe_allow_html=True)
        samp=fdf.sample(min(1200,len(fdf)),random_state=42)
        fig,ax=plt.subplots(figsize=(6,3.5)); fs(ax,fig)
        ax.scatter(samp.claim_amount_usd,samp.triage_score,
                   c=[C.get(p,"#94a3b8") for p in samp.priority],alpha=.3,s=10)
        patches=[mpatches.Patch(color=C[p],label=p) for p in ["High","Medium","Low"]]
        ax.legend(handles=patches,fontsize=8,frameon=False)
        ax.set_xlabel("Claim Amount (USD)",fontsize=9); ax.set_ylabel("Triage Score",fontsize=9)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x/1e3:.0f}K"))
        plt.tight_layout(); st.pyplot(fig); plt.close()

    c5,c6=st.columns(2)
    with c5:
        st.markdown('<p class="sh">Chronic vs acute — cost comparison</p>', unsafe_allow_html=True)
        ch=fdf.groupby("is_chronic").claim_amount_usd.agg(["mean","sum"]).reset_index()
        ch["is_chronic"]=ch["is_chronic"].map({0:"Acute",1:"Chronic"})
        fig,axes=plt.subplots(1,2,figsize=(6,3)); fig.patch.set_facecolor("white")
        for ax_ in axes:
            ax_.set_facecolor("#f8fafc"); ax_.spines[["top","right"]].set_visible(False); ax_.tick_params(labelsize=8)
        axes[0].bar(ch["is_chronic"],ch["mean"], color=["#3b82f6","#ef4444"],alpha=.82,edgecolor="white")
        axes[0].set_title("Mean Claim",fontsize=9)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x:,.0f}"))
        axes[1].bar(ch["is_chronic"],ch["sum"],  color=["#3b82f6","#ef4444"],alpha=.82,edgecolor="white")
        axes[1].set_title("Total Spend",fontsize=9)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_:f"${x/1e6:.1f}M"))
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c6:
        st.markdown('<p class="sh">Prior auth impact on triage score</p>', unsafe_allow_html=True)
        auth_sc=fdf.groupby("has_prior_auth").triage_score.mean().reset_index()
        auth_sc["has_prior_auth"]=auth_sc["has_prior_auth"].map({0:"No Auth",1:"Auth Obtained"})
        fig,ax=plt.subplots(figsize=(6,3)); fs(ax,fig)
        ax.bar(auth_sc["has_prior_auth"],auth_sc["triage_score"],
               color=["#ef4444","#22c55e"],alpha=.82,edgecolor="white",width=.5)
        ax.set_ylabel("Avg Triage Score",fontsize=9)
        for i,v in enumerate(auth_sc["triage_score"]):
            ax.text(i,v+.4,f"{v:.1f}",ha="center",fontsize=10,fontweight="bold",color="#475569")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<p class="sh">Top 10 highest-cost diagnoses</p>', unsafe_allow_html=True)
    tc=(fdf.groupby("diagnosis")["claim_amount_usd"].agg(["mean","count"]).reset_index()
        .rename(columns={"mean":"Avg Claim","count":"Volume"})
        .sort_values("Avg Claim",ascending=False).head(10))
    tc["Avg Claim"]=tc["Avg Claim"].apply(lambda x:f"${x:,.0f}")
    tc["ICD-10"]=tc["diagnosis"].apply(lambda d:ICD10_CONDITIONS.get(d,{}).get("icd10",""))
    st.dataframe(tc[["diagnosis","ICD-10","Avg Claim","Volume"]],use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# CODE REFERENCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📖  Code Reference":
    st.markdown('<div class="ph"><h1>📖 Medical Code Reference</h1><p>Searchable reference for all ICD-10-CM, DRG, HCPCS, LOINC, and SNOMED CT codes used in this platform.</p></div>', unsafe_allow_html=True)

    tab1,tab2,tab3,tab4=st.tabs(["ICD-10-CM + DRG","HCPCS Procedures","LOINC Labs","SNOMED CT"])

    with tab1:
        search=st.text_input("Search diagnosis, code, or category","")
        icd_df=pd.DataFrame([
            {"Diagnosis":k,"ICD-10-CM":v["icd10"],"Category":v["category"],
             "DRG":v["drg"],"Chronic":"Yes" if v["chronic"] else "No","Base Urgency":v["base_urgency"]}
            for k,v in ICD10_CONDITIONS.items()
        ])
        if search:
            mask=(icd_df["Diagnosis"].str.contains(search,case=False)|
                  icd_df["ICD-10-CM"].str.contains(search,case=False)|
                  icd_df["Category"].str.contains(search,case=False))
            icd_df=icd_df[mask]
        st.caption(f"{len(icd_df)} codes · Source: CMS ICD-10-CM (public domain)")
        st.dataframe(icd_df.sort_values("Category"),use_container_width=True,hide_index=True)

    with tab2:
        h_df=pd.DataFrame([{"Procedure":k,"HCPCS Code":v} for k,v in HCPCS_PROCEDURES.items()])
        st.caption("Source: CMS HCPCS Level II — public domain")
        st.dataframe(h_df,use_container_width=True,hide_index=True)

    with tab3:
        l_df=pd.DataFrame([{"Test / Observation":k,"LOINC Code":v} for k,v in LOINC_LABS.items()])
        st.caption("Source: Regenstrief Institute — free use under LOINC licence")
        st.dataframe(l_df,use_container_width=True,hide_index=True)

    with tab4:
        s_df=pd.DataFrame([{"Finding":k,"SNOMED CT Concept ID":v} for k,v in SNOMED_FINDINGS.items()])
        st.caption("Source: SNOMED International — free in most countries via National Release Centre")
        st.dataframe(s_df,use_container_width=True,hide_index=True)

    st.markdown("---")
    st.markdown("""
### Licence & attribution
All medical code sets used in ClaimIQ Pro are from open/public-domain sources:
- **ICD-10-CM** — U.S. CMS & WHO. Public domain.
- **MS-DRG** — CMS Medicare Severity DRGs. Public domain.
- **HCPCS Level II** — CMS. Public domain.
- **LOINC** — Regenstrief Institute. Free under [LOINC licence](https://loinc.org/license/).
- **SNOMED CT** — SNOMED International. Free via National Release Centre affiliation.

*This platform is for educational and demonstrative purposes only. Not for clinical decision-making.*
    """)
