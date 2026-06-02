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

st.title("🚀 AlignForge — Local RLHF Results")
st.markdown("A 100% local, portfolio-grade 4-stage RLHF pipeline (SFT -> RM -> DPO -> Eval).")

# --- Data Loading ---
def load_json(filename):
    path = Path("./results") / filename
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

def load_trainer_state(checkpoint_dir):
    path = Path("./results") / checkpoint_dir / "trainer_state.json"
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return None

metrics = load_json("final_metrics.json")
judgments = load_json("reward_eval_outputs.json")

# --- Sidebar / Navigation ---
with st.sidebar:
    st.markdown("## AlignForge")
    selected = option_menu(
        menu_title=None,
        options=["Overview", "Reward Distributions", "Loss Curves", "Model Comparison", "Samples Explorer"],
        icons=["bar-chart-fill", "activity", "graph-up-arrow", "layout-split", "search"],
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
    st.warning("No evaluation data found in `results/`. Please run the full evaluation pipeline first.")
else:
    if selected == "Overview":
        st.markdown("### Evaluation Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Win Rate</div><div class="metric-value neon-accent">{metrics.get("win_rate", 0)*100:.1f}%</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Avg Reward Gain</div><div class="metric-value neon-accent">+{metrics.get("reward_gain", 0):.4f}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-title">Baseline Reward</div><div class="metric-value" style="color:#d2a8ff;">{metrics.get("avg_baseline_reward", 0):.4f}</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-title">DPO Reward</div><div class="metric-value" style="color:#ff7b72;">{metrics.get("avg_dpo_reward", 0):.4f}</div></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### Average Response Length")
        fig = go.Figure(data=[
            go.Bar(name='Baseline', x=['Baseline'], y=[metrics.get('avg_length_baseline', 0)], marker_color='#8b949e'),
            go.Bar(name='DPO', x=['DPO'], y=[metrics.get('avg_length_dpo', 0)], marker_color='#58a6ff')
        ])
        fig.update_layout(template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    elif selected == "Reward Distributions":
        st.markdown("### Reward Score Distributions")
        df = pd.DataFrame(judgments)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df['baseline_reward'], name='Baseline Reward', opacity=0.75, marker_color='#8b949e'))
        fig.add_trace(go.Histogram(x=df['dpo_reward'], name='DPO Reward', opacity=0.75, marker_color='#58a6ff'))
        
        fig.update_layout(
            barmode='overlay',
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Reward Score",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Reward Gain (DPO - Baseline)")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(x=df['dpo_reward'] - df['baseline_reward'], name='Reward Gain', marker_color='#39d353'))
        fig2.update_layout(
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Reward Gain",
            yaxis_title="Count"
        )
        st.plotly_chart(fig2, use_container_width=True)

    elif selected == "Loss Curves":
        st.markdown("### Training Loss Curves")
        
        def plot_loss(checkpoint_dir, title):
            state = load_trainer_state(checkpoint_dir)
            if state and 'log_history' in state:
                history = [h for h in state['log_history'] if 'loss' in h]
                if history:
                    steps = [h['step'] for h in history]
                    loss = [h['loss'] for h in history]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=steps, y=loss, mode='lines', name=title, line=dict(color='#58a6ff')))
                    fig.update_layout(
                        title=title,
                        template='plotly_dark',
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        xaxis_title="Step",
                        yaxis_title="Loss",
                        height=350
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info(f"No loss history found in {checkpoint_dir}.")
            else:
                st.info(f"Trainer state not found for {checkpoint_dir}. Run training first.")
                
        plot_loss("sft_checkpoint", "SFT Training Loss")
        plot_loss("rm_checkpoint", "Reward Model Training Loss")
        plot_loss("dpo_checkpoint", "DPO Training Loss")

    elif selected == "Model Comparison":
        st.markdown("### Baseline vs DPO Comparisons")
        df = pd.DataFrame(judgments)
        
        st.dataframe(
            df[['prompt', 'baseline_response', 'dpo_response', 'baseline_reward', 'dpo_reward', 'winner']],
            use_container_width=True,
            column_config={
                "prompt": st.column_config.TextColumn("Prompt", width="medium"),
                "baseline_response": st.column_config.TextColumn("Baseline", width="large"),
                "dpo_response": st.column_config.TextColumn("DPO", width="large"),
                "baseline_reward": st.column_config.NumberColumn("Base Reward", format="%.4f"),
                "dpo_reward": st.column_config.NumberColumn("DPO Reward", format="%.4f"),
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
                st.markdown(f"**Decision:** `{row['winner']}` | **Reward Gain:** `+{row['dpo_reward'] - row['baseline_reward']:.4f}`")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"#### Baseline (Score: {row['baseline_reward']:.4f})")
                    st.info(row['baseline_response'])
                with col2:
                    st.markdown(f"#### DPO (Score: {row['dpo_reward']:.4f})")
                    st.success(row['dpo_response'])
