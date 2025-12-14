import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="Microbiome Diet Tracker",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
   
    /* Main app background - Dark */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }
   
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
   
    /* Sidebar - Dark glass */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.95);
        backdrop-filter: blur(10px);
        border-right: none;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4);
    }
   
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
   
    /* Headings */
    h1 {
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
    }
   
    h2, h3, h4, h5, h6 {
        color: #cbd5e1 !important;
        font-weight: 700 !important;
    }
   
    /* General text */
    p, span, div, label, .stMarkdown {
        color: #cbd5e1 !important;
    }
   
    .subtitle {
        color: rgba(255, 255, 255, 0.85) !important;
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.4);
    }
   
    /* Glass cards - Dark version */
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
   
    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 0, 0, 0.6);
    }
   
    /* Metric containers */
    .metric-container {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 16px;
        color: white;
        margin-bottom: 1rem;
    }
   
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
   
    .metric-label {
        font-size: 1rem;
        font-weight: 500;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
   
    /* Status cards */
    .status-card {
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        border-left: 5px solid;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        background: rgba(51, 65, 85, 0.6);
        transition: all 0.3s ease;
    }
   
    .status-card:hover {
        transform: translateX(8px);
    }
   
    .status-card.success {
        border-left-color: #10b981;
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%);
    }
   
    .status-card.warning {
        border-left-color: #f59e0b;
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%);
    }
   
    .status-card.danger {
        border-left-color: #ef4444;
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(220, 38, 38, 0.1) 100%);
    }
   
    /* Diet badges */
    .plant-badge {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: #064e3b;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4);
    }
   
    .western-badge {
        background: linear-gradient(135deg, #f43f5e 0%, #fb923c 100%);
        padding: 1.5rem;
        border-radius: 20px;
        color: #7f1d1d;
        margin: 1rem 0;
        box-shadow: 0 8px 24px rgba(244, 63, 94, 0.4);
    }
   
    /* Insight cards */
    .insight-card {
        background: rgba(51, 65, 85, 0.6);
        padding: 1.8rem;
        border-radius: 20px;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 6px solid #a78bfa;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
   
    .insight-card:hover {
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        transform: translateY(-2px);
    }
   
    .insight-title {
        color: #e2e8f0;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
   
    .insight-reason {
        color: #94a3b8;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
   
    .insight-action {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2) 0%, rgba(167, 139, 250, 0.15) 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        display: flex;
        align-items: center;
        gap: 12px;
        border: 1px solid rgba(167, 139, 250, 0.3);
    }
   
    .action-text {
        color: #c4b5fd;
        font-weight: 600;
        flex: 1;
    }
   
    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 0.8rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.5);
        width: 100%;
    }
   
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.7);
    }
   
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(51, 65, 85, 0.5);
        padding: 0.5rem;
        border-radius: 12px;
    }
   
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        background: transparent;
        color: #94a3b8;
    }
   
    .stTabs [aria-selected="true"] {
        background: #6366f1;
        color: white;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
   
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff !important;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid rgba(99, 102, 241, 0.5);
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.4);
    }
   
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, rgba(255,255,255,0.1), transparent);
    }
   
    /* Selectbox and inputs */
    .stSelectbox > div > div {
        background: rgba(51, 65, 85, 0.6);
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
   
    /* Plotly charts - ensure dark compatibility */
    .js-plotly-plot .plotly .modebar {
        filter: invert(1) hue-rotate(180deg);
    }
</style>
""", unsafe_allow_html=True)

DATA_DIR = '/Users/hamzaelghonemy/Desktop/University/Senior/Bioinformatics/Project/Disease_Biomarker_Discovery/data'

PATHWAY_MAP = {
    'map00500': 'Starch and sucrose metabolism',
    'map00531': 'Glycosaminoglycan degradation',
    'map00790': 'Folate biosynthesis',
    'map00540': 'Lipopolysaccharide biosynthesis',
    'map00290': 'Valine, leucine and isoleucine biosynthesis',
    'map02030': 'Bacterial chemotaxis',
    'map00633': 'Nitrotoluene degradation',
    'map00350': 'Tyrosine metabolism',
    'map00440': 'Phosphonate and phosphinate metabolism',
    'map00791': 'Atrazine degradation',
    'map00670': 'One carbon pool by folate',
    'map02040': 'Flagellar assembly',
    'map01210': '2-Oxocarboxylic acid metabolism',
    'map00903': 'Limonene and pinene degradation'
}

@st.cache_data
def load_data():
    path = os.path.join(DATA_DIR, 'abundance_tables.xlsx')
    df_genus = pd.read_excel(path, sheet_name='genus')
    df_pathway = pd.read_excel(path, sheet_name='pathway')
   
    id_col = df_pathway.columns[0]
    df_pathway = df_pathway.set_index(id_col)
   
    return df_genus, df_pathway

@st.cache_data
def determine_diet_labels(df_genus):
    sample_cols = [c for c in df_genus.columns if isinstance(c, str) and (c.startswith('D') or c.startswith('H') or c.startswith('P')) and len(c) > 3]
   
    bact_row, prev_row = None, None
    for col in df_genus.select_dtypes(include=['object']).columns:
        b_match = df_genus[df_genus[col].astype(str).str.contains('Bacteroides', case=False, na=False)]
        p_match = df_genus[df_genus[col].astype(str).str.contains('Prevotella', case=False, na=False)]
       
        if not b_match.empty:
            bact_row = b_match.index[0]
            exact = df_genus[df_genus[col].astype(str).str.strip() == 'Bacteroides']
            if not exact.empty: bact_row = exact.index[0]
        if not p_match.empty:
            prev_row = p_match.index[0]
            exact = df_genus[df_genus[col].astype(str).str.strip() == 'Prevotella']
            if not exact.empty: prev_row = exact.index[0]
        if bact_row is not None and prev_row is not None: break
           
    if bact_row is None or prev_row is None:
        return pd.Series()
    bact_vals = df_genus.loc[bact_row, sample_cols].astype(float)
    prev_vals = df_genus.loc[prev_row, sample_cols].astype(float)
   
    epsilon = 1e-6
    ratios = bact_vals / (prev_vals + epsilon)
   
    labels = pd.Series(index=sample_cols, data=-1)
    labels[ratios >= 2.0] = 0
    labels[ratios <= 0.5] = 1
   
    return labels[labels != -1]

@st.cache_resource
def train_model(df_pathway, labels):
    common_samples = [s for s in labels.index if s in df_pathway.columns]
    y = labels[common_samples].values
    X = df_pathway[common_samples].T.values
    X = np.nan_to_num(X)
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
   
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
   
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_res, y_train_res)
   
    score = clf.score(X_test, y_test)
   
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
   
    return clf, score, indices, common_samples

def generate_ai_insight(diet_type, score, risks, top_pathways):
    models_to_try = ['gemini-flash-latest']
   
    genai.configure(api_key=GOOGLE_API_KEY)
   
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
           
            prompt = f"""
            You are a Microbiome Health Coach. Analyze this user's gut profile and give 3 actionable food recommendations.
            Make sure the recommendations are based on the user's gut profile and not on their diet pattern.
            Make the output in words a general person could understand, if you generate any complex words explain what it means briefly.
           
            USER PROFILE:
            - Dominant Diet Pattern: {diet_type} ({score:.1f}% confidence)
            - Health Flags: {', '.join(risks) if risks else 'None (Healthy)'}
            - Key Active Pathways: {', '.join(top_pathways)}
           
            OUTPUT FORMAT:
            Provide a strictly valid JSON list of objects. Do not wrap in markdown code blocks.
            [
                {{
                    "title": "Insight Name",
                    "reason": "Scientific reason why...",
                    "action": "Eat this specific food..."
                }},
                ...
            ]
            """
           
            with st.spinner(f'🤖 AI Coach is thinking (using {model_name})...'):
                response = model.generate_content(prompt)
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                return json.loads(text.strip())
               
        except Exception as e:
            continue
           
    return [{"title": "Connection Error", "reason": "Could not connect to AI services.", "action": "Please try again later."}]

def main():
    st.markdown('<h1>🧬 Microbiome Diet & Health Tracker</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover your diet pattern and health insights through advanced microbiome analysis</p>', unsafe_allow_html=True)
   
    with st.spinner('🔬 Loading microbiome data...'):
        df_genus, df_pathway = load_data()
        labels = determine_diet_labels(df_genus)
   
    if labels.empty:
        st.error("Could not determine enterotypes from the data. Please check data format.")
        return
    with st.spinner('🧠 Training AI Model...'):
        clf, accuracy, top_feature_indices, common_samples = train_model(df_pathway, labels)
   
    st.sidebar.markdown("### 🎯 Model Performance")
    st.sidebar.markdown(f"""
    <div class="metric-container">
        <div class="metric-value" style="color: white !important;">{accuracy:.1%}</div>
        <div class="metric-label" style="color: rgba(255,255,255,0.9) !important;">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
   
    st.sidebar.markdown('<p style="color: #cbd5e1 !important; font-weight: 600; font-size: 1.1rem; margin-bottom: 0.5rem;">👤 User Selection</p>', unsafe_allow_html=True)
   
    sample_id = st.sidebar.selectbox(
        "Select Sample ID to Analyze",
        sorted(common_samples),
        index=0
    )
   
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Currently Analyzing:** {sample_id}")
   
    if sample_id:
        x_sample = df_pathway[sample_id].values
        x_sample = np.nan_to_num(x_sample)
       
        probs = clf.predict_proba(x_sample.reshape(1, -1))[0]
        western_prob = probs[0]
        plant_prob = probs[1]
       
        diet_type = "Plant-Based" if plant_prob > 0.5 else "Western Diet"
        diet_score = max(plant_prob, western_prob) * 100
       
        lps_row = df_pathway.loc['map00540', common_samples].values
        bcaa_row = df_pathway.loc['map00290', common_samples].values
       
        lps_val = df_pathway.loc['map00540', sample_id]
        bcaa_val = df_pathway.loc['map00290', sample_id]
       
        lps_pct = (lps_row < lps_val).mean() * 100
        bcaa_pct = (bcaa_row < bcaa_val).mean() * 100
       
        risk_list = []
        if lps_pct > 75: risk_list.append(f"High Inflammation (LPS: Top {100-lps_pct:.0f}%)")
        if bcaa_pct > 75: risk_list.append(f"Insulin Resistance Risk (BCAA: Top {100-bcaa_pct:.0f}%)")
       
        col1, col2 = st.columns([2, 1])
       
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h2 style="margin-top: 0;">🥗 Diet Pattern Analysis</h2>', unsafe_allow_html=True)
           
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = plant_prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Plant-Based Score", 'font': {'size': 20, 'color': '#e2e8f0'}},
                number = {'suffix': "%", 'font': {'size': 50, 'color': '#e2e8f0'}},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#6366f1"},
                    'bar': {'color': "#10b981" if plant_prob > 0.5 else "#f43f5e", 'thickness': 0.8},
                    'bgcolor': "rgba(0,0,0,0)",
                    'borderwidth': 3,
                    'bordercolor': "#475569",
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                        {'range': [50, 100], 'color': 'rgba(16, 185, 129, 0.2)'}],
                    'threshold': {
                        'line': {'color': "#8b5cf6", 'width': 6},
                        'thickness': 0.8,
                        'value': 75}
                    }
            ))
            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=80, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Inter', 'color': '#e2e8f0'}
            )
            st.plotly_chart(fig, use_container_width=True)
           
            if plant_prob > 0.5:
                st.markdown(f"""
                <div class="plant-badge">
                    <h3 style="color: #064e3b; margin: 0; font-size: 1.5rem;">🌱 Plant-Based Dominant</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #064e3b; font-size: 1.1rem;">Your microbiome shows signatures of high fiber digestion and vitamin production (Folate).</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="western-badge">
                    <h3 style="color: #7f1d1d; margin: 0; font-size: 1.5rem;">🍖 Western Diet Dominant</h3>
                    <p style="margin: 0.5rem 0 0 0; color: #7f1d1d; font-size: 1.1rem;">Your microbiome shows signatures of mucin degradation and protein fermentation.</p>
                </div>
                """, unsafe_allow_html=True)
           
            st.markdown('</div>', unsafe_allow_html=True)
               
        with col2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<h2 style="margin-top: 0;">🏥 Health Risk Flags</h2>', unsafe_allow_html=True)
           
            st.markdown("#### Inflammation Risk")
            if lps_pct > 75:
                st.markdown(f"""
                <div class="status-card danger">
                    <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">⚠️ High Risk</div>
                    <div style="font-size: 0.95rem;">Top {100-lps_pct:.0f}% of population</div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.8;">Your LPS Biosynthesis is elevated</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-card success">
                    <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">✅ Low Risk</div>
                    <div style="font-size: 0.95rem;">Normal levels detected</div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.8;">Percentile: {lps_pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
           
            st.markdown("#### Insulin Resistance")
            if bcaa_pct > 75:
                st.markdown(f"""
                <div class="status-card danger">
                    <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">⚠️ High Risk</div>
                    <div style="font-size: 0.95rem;">Top {100-bcaa_pct:.0f}% of population</div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.8;">Your BCAA Biosynthesis is elevated</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="status-card success">
                    <div style="font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem;">✅ Low Risk</div>
                    <div style="font-size: 0.95rem;">Normal levels detected</div>
                    <div style="font-size: 0.85rem; margin-top: 0.5rem; opacity: 0.8;">Percentile: {bcaa_pct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)
           
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
       
        top_ids = df_pathway.index[top_feature_indices]
        top_names = [PATHWAY_MAP.get(k, k) for k in top_ids]
        top_scores = clf.feature_importances_[top_feature_indices]
       
        st.markdown('<p class="section-header">🔍 Analysis Deep Dive</p>', unsafe_allow_html=True)
       
        tab1, tab2 = st.tabs(["📊 Feature Analysis", "🤖 AI Health Coach"])
       
        with tab1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### Key Metabolic Pathways")
            st.markdown("These pathways have the strongest influence on your diet classification:")
           
            feat_df = pd.DataFrame({'Pathway': top_names[:10], 'Importance': top_scores[:10]})
            fig_feat = px.bar(
                feat_df,
                x='Importance',
                y='Pathway',
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis',
                text='Importance'
            )
            fig_feat.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_feat.update_layout(
                yaxis={'categoryorder':'total ascending'},
                height=500,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'family': 'Inter', 'size': 12, 'color': '#e2e8f0'},
                showlegend=False
            )
            st.plotly_chart(fig_feat, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
           
        with tab2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("### 🧬 Personalized Health Recommendations")
            st.caption("AI-powered insights based on your unique microbiome signature")
           
            if st.button("✨ Generate My Personalized Plan"):
                insights_data = generate_ai_insight(
                    diet_type,
                    diet_score,
                    risk_list,
                    top_names[:5]
                )
               
                if not insights_data or isinstance(insights_data, str):
                    st.warning("⚠️ Please provide a valid API Key" if not insights_data else insights_data)
                else:
                    for i, item in enumerate(insights_data, 1):
                        st.markdown(f"""
                        <div class="insight-card">
                            <div class="insight-title">💡 {item.get('title', 'Insight')}</div>
                            <div class="insight-reason"><strong>Why this matters:</strong> {item.get('reason', '')}</div>
                            <div class="insight-action">
                                <span class="action-icon">🍽️</span>
                                <span class="action-text">{item.get('action', '')}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
               
                    st.caption("💫 Generated by Google Gemini • Not medical advice • Consult healthcare professionals")
           
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()