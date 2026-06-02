"""
Streamlit dashboard showing AlignForge results.
"""
import streamlit as st
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

st.set_page_config(page_title="AlignForge — DPO Results", layout="wide", initial_sidebar_state="expanded")

# --- CSS Styling ---
st.markdown("""
<style>
    /* Dark mode terminal/research aesthetic */
    body {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Courier New', Courier, monospace;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.5);
    }
    .metric-title {
        font-size: 14px;
        color: #8b949e;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #58a6ff;
    }
    .neon-accent {
        color: #39d353;
    }
</style>
""", unsafe_allow_html=True)

st.title("🚀 AlignForge — DPO Fine-tuning Results")
st.markdown("A portfolio-grade DPO fine-tuning pipeline for Qwen2-0.5B.")

# --- Data Loading ---
def load_json(filename):
    path = Path("./results") / filename
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

metrics = load_json("final_metrics.json")
judgments = load_json("judgments.json")

# --- Sidebar / Navigation ---
with st.sidebar:
    st.markdown("## AlignForge")
    selected = option_menu(
        menu_title=None,
        options=["Overview", "Model Comparison", "Samples Explorer"],
        icons=["bar-chart-fill", "layout-split", "search"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#58a6ff", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#21262d"},
            "nav-link-selected": {"background-color": "#21262d", "border-left": "4px solid #58a6ff"},
        }
    )

if not metrics or not judgments:
    st.warning("No evaluation data found in `results/`. Please run the evaluation pipeline first.")
else:
    if selected == "Overview":
        st.markdown("### Evaluation Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Win Rate</div><div class="metric-value neon-accent">{metrics.get("win_rate", 0)*100:.1f}%</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Tie Rate</div><div class="metric-value" style="color:#d2a8ff;">{metrics.get("tie_rate", 0)*100:.1f}%</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Loss Rate</div><div class="metric-value" style="color:#ff7b72;">{metrics.get("loss_rate", 0)*100:.1f}%</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### Average Response Length")
        fig = go.Figure(data=[
            go.Bar(name='Baseline', x=['Baseline'], y=[metrics.get('avg_length_baseline', 0)], marker_color='#8b949e'),
            go.Bar(name='DPO', x=['DPO'], y=[metrics.get('avg_length_dpo', 0)], marker_color='#58a6ff')
        ])
        fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    elif selected == "Model Comparison":
        st.markdown("### Baseline vs DPO Comparisons")
        df = pd.DataFrame(judgments)
        
        st.dataframe(
            df[['prompt', 'baseline_response', 'dpo_response', 'winner']],
            use_container_width=True,
            column_config={
                "prompt": st.column_config.TextColumn("Prompt", width="medium"),
                "baseline_response": st.column_config.TextColumn("Baseline", width="large"),
                "dpo_response": st.column_config.TextColumn("DPO", width="large"),
                "winner": st.column_config.TextColumn("Decision", width="small"),
            },
            hide_index=True,
        )

    elif selected == "Samples Explorer":
        st.markdown("### Samples Explorer")
        
        df = pd.DataFrame(judgments)
        search_query = st.text_input("Search prompts:", "")
        
        if search_query:
            filtered_df = df[df['prompt'].str.contains(search_query, case=False)]
        else:
            filtered_df = df
            
        for i, row in filtered_df.head(10).iterrows():
            with st.expander(f"Prompt: {row['prompt'][:80]}..."):
                st.markdown(f"**Decision:** `{row['winner']}`")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("#### Baseline")
                    st.info(row['baseline_response'])
                with col2:
                    st.markdown("#### DPO")
                    st.success(row['dpo_response'])
