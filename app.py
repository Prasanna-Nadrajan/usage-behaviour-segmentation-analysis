"""
Interactive Streamlit Web Application: Usage Behaviour Segmentation.
Provides visual exploration of segments, PCA cluster projections,
radar benchmark charts, business action playbooks, and a live user scorer.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Import class so pickle unpickling finds the definition in __main__ and src.modeling
from src.modeling import SegmentationPipeline

st.set_page_config(
    page_title="Usage Behaviour Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and pipeline
@st.cache_data
def load_data():
    df = pd.read_parquet("data/processed/segmented_users.parquet")
    eval_df = pd.read_csv("reports/cluster_evaluation_metrics.csv")
    return df, eval_df

@st.cache_resource
def load_pipeline():
    return joblib.load("models/segmentation_pipeline.joblib")

df, eval_df = load_data()
pipeline = load_pipeline()

# Business recommendations playbook
RECOMMENDATIONS = {
    "Power Champions": {
        "tagline": "High frequency, broad feature adoption & top revenue contributors.",
        "actions": [
            "Invite to an Exclusive Advisory Board / Early Beta Access program.",
            "Offer annual Enterprise tier upgrades with tailored volume discounts.",
            "Establish proactive VIP support to ensure zero SLA breaches."
        ],
        "kpi": "Net Revenue Retention (NRR) & Expansion ARR"
    },
    "Core Active Users": {
        "tagline": "Steady, reliable mid-tier engagement with consistent daily workflows.",
        "actions": [
            "Trigger automated in-app guides for unused high-value features.",
            "Target with collaborative multi-seat upgrade campaigns.",
            "Provide self-service mastery webinars and template galleries."
        ],
        "kpi": "Feature Adoption Rate & Seat Expansion Rate"
    },
    "Emerging / High-Growth": {
        "tagline": "Recent signups exhibiting accelerating momentum and fast adoption.",
        "actions": [
            "Deliver a high-touch 14-day onboarding cadence with milestone nudges.",
            "Offer a limited-time conversion discount to transition from Starter to Pro.",
            "Pair with customer success specialist for initial workflow setup."
        ],
        "kpi": "30-Day Conversion Rate & Day-60 Retention"
    },
    "Casual Explorers": {
        "tagline": "Infrequent sessions, short duration, primarily utilizing basic capabilities.",
        "actions": [
            "Deploy contextual behavioral re-engagement emails highlighting quick wins.",
            "Gamify product discovery with interactive checklists.",
            "Test low-friction micro-upgrades (e.g. pay-as-you-go add-ons)."
        ],
        "kpi": "Weekly Active Users (WAU) & Session Frequency"
    },
    "At-Risk Dormant": {
        "tagline": "High recency (dormant), sharp drop in momentum, elevated friction index.",
        "actions": [
            "Launch a 'We miss you / What went wrong?' win-back campaign with direct executive outreach.",
            "Prioritize pending support tickets with senior technical specialists.",
            "Offer a 1-month complimentary plan extension to eliminate immediate friction."
        ],
        "kpi": "Churn Reduction Rate & Win-Back Reactivation Rate"
    }
}

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview & Executive Summary", "Interactive Segment Explorer", "Cluster Evaluation & PCA", "Live User Segment Scorer"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project Context**: SaaS Behavioral Segmentation\n\n"
    f"**Total Tracked Users**: {len(df):,}\n\n"
    f"**Model**: K-Means ($k=5$) + Yeo-Johnson PowerTransformer"
)

# ----------------- PAGE 1: OVERVIEW -----------------
if page == "Overview & Executive Summary":
    st.title("🎯 SaaS Usage Behaviour Segmentation Dashboard")
    st.markdown("Segmenting users by **actual behavioral telemetry** rather than static demographics.")

    # Top KPI cards
    col1, col2, col3, col4 = st.columns(4)
    total_rev = df['monetary_revenue_90d'].sum()
    avg_sess = df['sessions_per_week'].mean()
    dormant_pct = (df['segment_name'] == 'At-Risk Dormant').mean() * 100
    power_pct = (df['segment_name'] == 'Power Champions').mean() * 100

    col1.metric("Total 90d Revenue", f"${total_rev:,.0f}")
    col2.metric("Avg Sessions / Week", f"{avg_sess:.1f}")
    col3.metric("At-Risk Users", f"{dormant_pct:.1f}%", delta="-Churn Risk", delta_color="inverse")
    col4.metric("Power Champions", f"{power_pct:.1f}%", delta="+High LTV")

    st.markdown("### 👥 Segment Breakdown & Summary Profiles")
    seg_summary = df.groupby('segment_name').agg({
        'user_id': 'count',
        'monetary_revenue_90d': 'mean',
        'recency_days': 'mean',
        'sessions_per_week': 'mean',
        'total_engagement_hours_90d': 'mean',
        'feature_adoption_rate': 'mean',
        'usage_momentum_30d': 'mean'
    }).reset_index()

    seg_summary.columns = [
        'Segment Name', 'User Count', 'Avg Revenue ($)', 'Avg Recency (Days)',
        'Sessions / Wk', 'Avg Engagement (Hours)', 'Feature Adoption Rate', 'Momentum Index'
    ]
    seg_summary['User Share %'] = (seg_summary['User Count'] / len(df) * 100).round(1)

    st.dataframe(seg_summary.sort_values(by='Avg Revenue ($)', ascending=False).style.format({
        'Avg Revenue ($)': '${:.2f}',
        'Avg Recency (Days)': '{:.1f}',
        'Sessions / Wk': '{:.1f}',
        'Avg Engagement (Hours)': '{:.1f} hrs',
        'Feature Adoption Rate': '{:.0%}',
        'Momentum Index': '{:.2f}',
        'User Share %': '{:.1f}%'
    }), use_container_width=True)

    # Segment population chart
    fig_bar = px.bar(
        seg_summary,
        x='Segment Name',
        y='User Count',
        color='Segment Name',
        text='User Count',
        title="User Distribution across Behavioral Segments"
    )
    fig_bar.update_layout(showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ----------------- PAGE 2: SEGMENT EXPLORER -----------------
elif page == "Interactive Segment Explorer":
    st.title("🔍 Interactive Segment Deep-Dive")

    selected_segment = st.selectbox("Select Segment to Inspect", list(RECOMMENDATIONS.keys()))
    seg_info = RECOMMENDATIONS[selected_segment]

    st.subheader(f"{selected_segment}")
    st.write(f"*{seg_info['tagline']}*")

    c1, c2 = st.columns([3, 2])
    with c1:
        # Radar Chart comparing selected segment vs average
        feat_cols = ['sessions_per_week', 'total_engagement_hours_90d', 'feature_adoption_rate', 'usage_momentum_30d', 'monetary_revenue_90d']
        norm_df = df[feat_cols].copy()
        for col in feat_cols:
            max_v = norm_df[col].max()
            if max_v > 0:
                norm_df[col] = norm_df[col] / max_v

        seg_means = norm_df[df['segment_name'] == selected_segment].mean()
        overall_means = norm_df.mean()

        categories = ['Sessions/Wk', 'Total Hours', 'Feature Breadth', 'Usage Momentum', 'Revenue']
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=seg_means.values.tolist() + [seg_means.values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=selected_segment,
            line=dict(color='#2a9d8f')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=overall_means.values.tolist() + [overall_means.values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='All Users Average',
            line=dict(color='#999999', dash='dot')
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Relative Behavioral Fingerprint (Normalized 0-1)"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with c2:
        st.markdown("### 💼 Recommended Business Playbook")
        for i, action in enumerate(seg_info['actions'], 1):
            st.markdown(f"**{i}.** {action}")
        
        st.markdown("---")
        st.markdown(f"**🎯 Primary KPI to Track:** `{seg_info['kpi']}`")

        # Plan type distribution in this segment
        seg_sub = df[df['segment_name'] == selected_segment]
        fig_pie = px.pie(seg_sub, names='plan_type', title="Plan Type Distribution in Segment")
        st.plotly_chart(fig_pie, use_container_width=True)

# ----------------- PAGE 3: CLUSTER EVALUATION & PCA -----------------
elif page == "Cluster Evaluation & PCA":
    st.title("🔬 Cluster Evaluation & Latent Space Projection")

    st.markdown("### 1. Optimal Cluster Diagnostics ($K=2$ to $K=7$)")
    c1, c2 = st.columns(2)
    with c1:
        fig_elbow = px.line(eval_df, x='k', y='inertia', markers=True, title="Elbow Method (Inertia vs. k)")
        st.plotly_chart(fig_elbow, use_container_width=True)
    with c2:
        fig_sil = px.line(eval_df, x='k', y='silhouette', markers=True, title="Silhouette Score vs. k (Peak indicates separation)")
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("### 2. 2D PCA Latent Space Projection")
    sample_df = df.sample(n=2500, random_state=42)
    fig_pca = px.scatter(
        sample_df,
        x='pca_x',
        y='pca_y',
        color='segment_name',
        hover_data=['user_id', 'plan_type', 'monetary_revenue_90d', 'sessions_per_week'],
        title="PCA 2D Cluster Separation (Sample of 2,500 Users)"
    )
    st.plotly_chart(fig_pca, use_container_width=True)

# ----------------- PAGE 4: LIVE SCORER -----------------
elif page == "Live User Segment Scorer":
    st.title("⚡ Live User Segment Assignment & Inference")
    st.markdown("Input behavioral telemetry parameters to assign a customer to their predicted segment in real time.")

    with st.form("user_score_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            recency = st.slider("Recency (Days since last active)", 0, 90, 5)
            sessions_pw = st.slider("Sessions per Week", 0.1, 40.0, 8.0)
            engagement_hrs = st.slider("Total 90d Hours", 0.5, 120.0, 25.0)
        with c2:
            feat_adopt = st.slider("Feature Adoption Rate (0-1)", 0.1, 1.0, 0.7)
            momentum = st.slider("Usage Momentum (30d vs baseline)", 0.0, 3.0, 1.2)
            revenue = st.number_input("90-Day Revenue ($)", min_value=0.0, max_value=2000.0, value=150.0)
        with c3:
            support_friction = st.slider("Support Ticket Friction (tickets/session)", 0.0, 2.0, 0.02)
            weekend_share = st.slider("Weekend Session Share", 0.0, 1.0, 0.20)
            submit = st.form_submit_button("Predict Segment", use_container_width=True)

    if submit:
        input_data = pd.DataFrame([{
            'recency_days': recency,
            'sessions_per_week': sessions_pw,
            'total_engagement_hours_90d': engagement_hrs,
            'feature_adoption_rate': feat_adopt,
            'usage_momentum_30d': momentum,
            'monetary_revenue_90d': revenue,
            'support_ticket_friction': support_friction,
            'weekend_session_share': weekend_share
        }])

        pred_cluster = pipeline.predict(input_data)[0]
        cluster_name = df[df['cluster'] == pred_cluster]['segment_name'].iloc[0]
        rec = RECOMMENDATIONS.get(cluster_name, {"tagline": "", "actions": [], "kpi": ""})

        st.success(f"### Predicted Segment: **{cluster_name}**")
        st.write(f"*{rec['tagline']}*")
        st.markdown(f"**Recommended Action**: {rec['actions'][0]}")
        st.markdown(f"**Target KPI**: `{rec['kpi']}`")
