
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path.cwd() / "job_market_ai_outputs"
DATA = ROOT / "data" / "processed_jobs.csv"
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"

st.set_page_config(page_title="AI Job Market Analytics", page_icon="📊", layout="wide")
st.title("AI-Based Job Market Analytics")
st.caption("Academic prototype. Salary estimates are not guarantees; results describe the selected dataset snapshot.")

if not DATA.exists():
    st.error("Processed data not found. Run the notebook through the data preparation and training sections first.")
    st.stop()

@st.cache_data
def read_data():
    return pd.read_csv(DATA)

df = read_data()
TARGET = "annual_salary_usd"
page = st.sidebar.selectbox("Module", ["Market overview", "Salary prediction", "Job recommendations", "Skill demand"])

if page == "Market overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Job postings", f"{len(df):,}")
    c2.metric("Usable USD salary rows", f"{df[TARGET].notna().sum():,}")
    c3.metric("Countries", f"{df['country'].nunique():,}")
    st.subheader("Top job titles")
    st.bar_chart(df["title"].replace("", "Unknown").value_counts().head(15))
    salary = df[TARGET].dropna()
    if not salary.empty:
        st.subheader("Annualized advertised salary distribution (USD)")
        st.bar_chart(salary.clip(upper=salary.quantile(0.99)).value_counts(bins=25).sort_index())
    st.caption("Salary analytics include supported USD salary ranges only.")

elif page == "Salary prediction":
    model_path = MODELS / "salary_best_pipeline.joblib"
    if not model_path.exists():
        st.error("Model not found. Run the notebook training section first.")
        st.stop()
    model = joblib.load(model_path)
    with st.form("salary_form"):
        title = st.text_input("Job title", "Data Scientist")
        department = st.text_input("Department", "Data")
        industry = st.text_input("Company industry", "Technology")
        country = st.text_input("Country", "United States")
        city = st.text_input("City", "Remote")
        employment_type = st.selectbox("Employment type", ["full_time", "contract", "part_time", "internship", "unknown"])
        remote = st.selectbox("Remote status", ["remote", "not_remote", "hybrid", "unknown"])
        skills = st.text_input("Skills (comma-separated)", "python, sql, machine learning")
        year = st.number_input("Posting year", min_value=2020, max_value=2035, value=2026)
        month = st.number_input("Posting month", min_value=1, max_value=12, value=1)
        go = st.form_submit_button("Estimate salary")
    if go:
        feature_row = pd.DataFrame([{
            "title": title, "department": department, "company_industry": industry,
            "country": country, "city": city, "employment_type": employment_type,
            "is_remote": remote, "salary_from_text": "unknown",
            "skill_count": len([x for x in skills.split(",") if x.strip()]),
            "posted_year": year, "posted_month": month,
        }])
        prediction = float(model.predict(feature_row)[0])
        st.metric("Estimated annual salary (USD)", f"${prediction:,.0f}")
        st.info("Educational estimate from this dataset snapshot. It is not an offer or compensation guarantee.")

elif page == "Job recommendations":
    recommender_path = MODELS / "job_recommender.joblib"
    if not recommender_path.exists():
        st.error("Recommender not found. Run the notebook recommendation section first.")
        st.stop()
    query = st.text_area("Preferred role, skills and location", "Data Scientist Python SQL machine learning remote")
    top_k = st.slider("Number of results", 3, 15, 5)
    if st.button("Find matching postings"):
        rec = joblib.load(recommender_path)
        query_vector = rec["vectorizer"].transform([query])
        scores = cosine_similarity(query_vector, rec["matrix"]).ravel()
        inds = np.argsort(scores)[::-1][:top_k]
        result = rec["jobs"].iloc[inds].copy()
        result.insert(0, "similarity", scores[inds])
        cols = [c for c in ["similarity", "title", "company_industry", "city", "country", "employment_type", "skills_text", "url"] if c in result]
        st.dataframe(result[cols], use_container_width=True)
        st.caption("Ranked by text similarity only; this does not establish candidate eligibility or hiring probability.")

else:
    skills_path = ARTIFACTS / "skill_demand.csv"
    if not skills_path.exists():
        st.error("Skill table not found. Run the notebook skill-analysis section first.")
        st.stop()
    skills_df = pd.read_csv(skills_path).head(30)
    st.subheader("Most frequently tagged skills")
    st.dataframe(skills_df, use_container_width=True)
    if not skills_df.empty:
        st.bar_chart(skills_df.set_index("skill")["posting_mentions"])
    st.caption("Counts describe skill tags in this dataset snapshot, not skill proficiency or importance.")
