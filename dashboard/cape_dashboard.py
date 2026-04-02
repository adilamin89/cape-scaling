"""
CAPE Dashboard — Streamlit Edition
Lying Is Just a Phase · It's Not a Phase
Interactive capability coupling analysis for frontier AI models
"""
import streamlit as st
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page Config ──
st.set_page_config(
    page_title="CAPE — Lying Is Just a Phase",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@400;500;700&display=swap');

.stApp { background-color: #0a0e17; color: #c8d6e5; }
[data-testid="stSidebar"] { background-color: #0f1420; border-right: 1px solid #1a2035; }
h1, h2, h3 { font-family: 'DM Sans', sans-serif !important; color: #e8ecf1 !important; }
.stMetric label { font-family: 'Space Mono', monospace !important; color: #4cc9f0 !important; font-size: 11px !important; }
.stMetric [data-testid="stMetricValue"] { font-family: 'Space Mono', monospace !important; color: #4cc9f0 !important; }

.cape-hero {
    background: linear-gradient(135deg, #0a0e17 0%, #1a1a2e 50%, #16213e 100%);
    border: 1px solid #4cc9f020;
    border-radius: 12px;
    padding: 2rem;
    margin-bottom: 2rem;
    text-align: center;
}
.cape-hero h1 { font-size: 2.5rem; margin: 0; color: #4cc9f0 !important; }
.cape-hero .subtitle { color: #7a9abf; font-size: 1.1rem; margin-top: 0.5rem; }
.cape-hero .thesis { color: #06d6a0; font-style: italic; margin-top: 1rem; font-size: 1rem; }

.stat-card {
    background: #0f1420;
    border: 1px solid #1a2035;
    border-left: 3px solid #4cc9f0;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.stat-card .val { font-family: 'Space Mono', monospace; font-size: 1.8rem; color: #4cc9f0; }
.stat-card .lbl { font-family: 'DM Sans', sans-serif; font-size: 0.75rem; color: #7a9abf; text-transform: uppercase; letter-spacing: 1px; }

.prediction-pass { color: #06d6a0; font-weight: bold; }
.prediction-fail { color: #ef476f; font-weight: bold; }
.prediction-pending { color: #ffd166; }
</style>
""", unsafe_allow_html=True)

# ── Load Data ──
DATA_DIR = Path(__file__).parent.parent / "data"

@st.cache_data
def load_frontier():
    try:
        with open(DATA_DIR / "frontier_final_consolidated.json") as f:
            return json.load(f)
    except:
        return {"models": [], "r_all": 0.729, "slope": 0.519, "intercept": 45.661}

@st.cache_data
def load_opt_nc2():
    results = {}
    for name in ["opt30b_internal_nc2", "opt66b_internal_nc2"]:
        try:
            with open(DATA_DIR / f"{name}.json") as f:
                results[name] = json.load(f)
        except:
            pass
    return results

@st.cache_data
def load_self_align():
    results = {}
    for name in ["self_aligning_demo_410m", "self_align_modal_2.8b"]:
        try:
            with open(DATA_DIR / f"{name}.json") as f:
                results[name] = json.load(f)
        except:
            pass
    return results

frontier = load_frontier()
opt_nc2 = load_opt_nc2()
self_align = load_self_align()

SLOPE, INTERCEPT = 0.52, 45.7

# ── Hero ──
st.markdown("""
<div class="cape-hero">
    <h1>⚗ Lying Is Just a Phase</h1>
    <div class="subtitle">The Hidden Alignment Transition in Language Model Scaling</div>
    <div class="thesis">"The alignment tax is not a law of nature — it is an engineerable bottleneck."</div>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ──
c1, c2, c3, c4, c5 = st.columns(5)
with c1: st.metric("Critical Scale Nc", "3.5B", help="[2.9B, 13.4B] 95% CI")
with c2: st.metric("Pre-transition r", "−0.989", help="p < 10⁻⁵")
with c3: st.metric("Base Models", "63", help="16 families")
with c4: st.metric("Frontier Models", "31", help="8 labs")
with c5: st.metric("Predictions", "7", help="Falsifiable, timestamped")

# ── Sidebar ──
st.sidebar.markdown("### Papers")
st.sidebar.markdown("[📄 Lying Is Just a Phase (Nature)](https://github.com/adilamin89/cape-scaling/blob/main/paper3a_nature.pdf)")
st.sidebar.markdown("[📄 It's Not a Phase (NeurIPS)](https://github.com/adilamin89/cape-scaling/blob/main/paper3b_neurips.pdf)")
st.sidebar.markdown("---")
st.sidebar.markdown("### Links")
st.sidebar.markdown("[🔗 GitHub](https://github.com/adilamin89/cape-scaling)")
st.sidebar.markdown("[📊 Static Dashboard](https://adilamin89.github.io/cape-scaling)")

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚗ h-field Calculator",
    "🔬 Frontier Coupling",
    "🔄 Nc Cascade",
    "🎯 Self-Aligning Demo",
    "📊 Predictions",
    "🧪 OPT Internal Ladder"
])

# ══════════════════════════════════════
# TAB 1: h-field Calculator
# ══════════════════════════════════════
with tab1:
    st.header("h-field Calculator")
    st.markdown("Enter SWE-bench and GPQA scores → get coupling diagnostic")

    col1, col2 = st.columns(2)
    with col1:
        swe = st.number_input("SWE-bench Verified (%)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    with col2:
        gpqa = st.number_input("GPQA Diamond (%)", min_value=0.0, max_value=100.0, value=88.0, step=0.1)

    if st.button("🔍 Compute h-field", type="primary"):
        h = gpqa - (SLOPE * swe + INTERCEPT)
        predicted_gpqa = SLOPE * swe + INTERCEPT

        if h > 10:
            phase = "Reasoning-specialist"
            color = "#4cc9f0"
        elif h < -10:
            phase = "Coding-specialist excursion"
            color = "#ef476f"
        else:
            phase = "Cooperative (on trend)"
            color = "#06d6a0"

        st.markdown(f"""
        ### Result
        | Metric | Value |
        |--------|-------|
        | **h-field** | **{h:+.1f} pp** |
        | **Phase** | {phase} |
        | **Predicted GPQA** | {predicted_gpqa:.1f}% |
        | **Actual GPQA** | {gpqa:.1f}% |
        | **Direction** | {"Reasoning-rich ↑" if h > 0 else "Coding-rich ↓"} |
        """)

        # Comparison to labs
        st.markdown("### How this compares to frontier labs")
        lab_h = {"Google": 5.7, "OpenAI": 3.4, "Alibaba": 3.1, "Meta": 2.6,
                 "DeepSeek": 2.2, "Moonshot": 2.1, "MiniMax": -2.1, "Anthropic": -6.7}

        fig = go.Figure()
        labs = list(lab_h.keys())
        vals = list(lab_h.values())
        colors = ['#06d6a0' if v > 0 else '#ef476f' for v in vals]
        fig.add_trace(go.Bar(x=vals, y=labs, orientation='h', marker_color=colors, name='Labs'))
        fig.add_trace(go.Scatter(x=[h], y=['Your Model'], mode='markers',
                                 marker=dict(size=16, color='#ffd166', symbol='diamond'), name='Your Model'))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#0a0e17', plot_bgcolor='#0f1420',
            height=350, showlegend=False,
            xaxis_title="h-field (pp)", title="Your model vs frontier labs"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Prescriptive
        if h < -5:
            st.info("💡 **Suggestion**: Your model is coding-heavy. Reasoning investment has the highest marginal coupling return.")
        elif h > 5:
            st.info("💡 **Suggestion**: Your model is reasoning-saturated. Coding investment may have higher marginal value.")
        else:
            st.success("✅ Your model is balanced on the cooperation trend. Either axis investment is efficient.")

# ══════════════════════════════════════
# TAB 2: Frontier Coupling
# ══════════════════════════════════════
with tab2:
    st.header("Frontier Coupling: 31 Models, 8 Labs")

    models = frontier.get("models", [])
    if models:
        lab_colors = {
            'Anthropic': '#ef476f', 'OpenAI': '#4cc9f0', 'Google': '#06d6a0',
            'DeepSeek': '#ffd166', 'Meta': '#9b5de5', 'Moonshot': '#8c564b',
            'Alibaba': '#e377c2', 'MiniMax': '#17becf'
        }

        fig = go.Figure()
        for lab in set(m['lab'] for m in models):
            lab_models = [m for m in models if m['lab'] == lab]
            fig.add_trace(go.Scatter(
                x=[m['swe'] for m in lab_models],
                y=[m['gpqa'] for m in lab_models],
                mode='markers+text',
                name=lab,
                marker=dict(size=10, color=lab_colors.get(lab, '#888')),
                text=[m['name'].replace('Claude ', 'C').replace('Gemini ', 'G')[:12] for m in lab_models],
                textposition='top center', textfont=dict(size=8)
            ))

        x_fit = np.linspace(15, 85, 100)
        fig.add_trace(go.Scatter(
            x=x_fit, y=SLOPE * x_fit + INTERCEPT,
            mode='lines', line=dict(dash='dash', color='rgba(76,201,240,0.3)'),
            name=f'GPQA = {SLOPE}·SWE + {INTERCEPT}'
        ))

        fig.update_layout(
            template='plotly_dark', paper_bgcolor='#0a0e17', plot_bgcolor='#0f1420',
            xaxis_title='SWE-bench Verified (%)', yaxis_title='GPQA Diamond (%)',
            height=500, title=f'Frontier Coupling (r = +{frontier.get("r_all", 0.729):.3f})'
        )
        st.plotly_chart(fig, use_container_width=True)

        # h-field table
        st.markdown("### Per-Lab h-field (core models)")
        lab_data = []
        for lab in ['Google', 'OpenAI', 'Alibaba', 'Meta', 'DeepSeek', 'Moonshot', 'MiniMax', 'Anthropic']:
            lab_models = [m for m in models if m['lab'] == lab]
            h_vals = [m['gpqa'] - (SLOPE * m['swe'] + INTERCEPT) for m in lab_models]
            lab_data.append({"Lab": lab, "n": len(lab_models), "h̄ (pp)": f"{np.mean(h_vals):+.1f}",
                           "Direction": "Reasoning ↑" if np.mean(h_vals) > 2 else ("Coding ↓" if np.mean(h_vals) < -2 else "Balanced")})
        st.dataframe(lab_data, use_container_width=True)

# ══════════════════════════════════════
# TAB 3: Nc Cascade
# ══════════════════════════════════════
with tab3:
    st.header("The Nc Cascade — It Doesn't Stop")
    st.markdown("Each transition follows the same pattern: **old axes lock together, new ones emerge.**")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("#### Nc1 ✅")
        st.markdown("**~1-7B params**")
        st.markdown("HS↔TQA sign flip")
        st.markdown("*Confirmed across 16 families*")
    with col2:
        st.markdown("#### Nc2 🔬")
        st.markdown("**~30-66B params**")
        st.markdown("OPT cooperation drops")
        st.markdown("*Internal evidence (this work)*")
    with col3:
        st.markdown("#### Nc3 📡")
        st.markdown("**~114B predicted**")
        st.markdown("SWE saturating, IFEval activating")
        st.markdown("*Preliminary (n=4)*")
    with col4:
        st.markdown("#### Nc4 🔮")
        st.markdown("**~200B+ predicted**")
        st.markdown("IFEval→HLE handoff")
        st.markdown("*Framework prediction*")

    # Coupling matrix
    st.markdown("### Multi-Benchmark Coupling Matrix")
    matrix_data = [
        {"Pair": "SWE ↔ GPQA", "r": "+0.848", "p": "<10⁻⁶", "n": "21", "Status": "✅ Cooperative"},
        {"Pair": "GPQA ↔ HLE", "r": "+0.715", "p": "0.02", "n": "10", "Status": "✅ Cooperative"},
        {"Pair": "SWE ↔ HLE", "r": "−0.251", "p": "0.52", "n": "9", "Status": "⚠️ Decoupled"},
    ]
    st.dataframe(matrix_data, use_container_width=True)
    st.caption("SWE decouples from HLE while GPQA and HLE cooperate — the Nc3 signature.")

# ══════════════════════════════════════
# TAB 4: Self-Aligning Demo
# ══════════════════════════════════════
with tab4:
    st.header("Self-Aligning: Steering at the Bottleneck")
    st.markdown("""
    The coupling structure is **exploitable**. Steering at the identified bottleneck layer
    corrects misaligned outputs. The tax, once located, can be surgically addressed.

    **Anyone can test this** with the provided scripts and an open-source model.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Pythia-410M (Tax Phase)")
        st.metric("Changed", "14/14", help="All tax-phase prompts corrected")
        st.metric("Probe Layer", "6", help="Quarter-depth, where coupling bottleneck lives")
        st.progress(1.0)
        st.caption("Steering highly effective where tax is active")

    with col2:
        st.markdown("### Pythia-2.8B (Bonus Phase)")
        st.metric("Changed", "6/14", help="Less misalignment to correct above Nc")
        st.metric("Probe Layer", "8", help="Quarter-depth")
        st.progress(6/14)
        st.caption("Less effective — cooperative regime has less to fix")

    st.markdown("### Try It Yourself")
    st.code("""
# Install
pip install transformer-lens torch

# Run self-aligning demo
python scripts/modal_self_aligning_v2.py --model pythia-410m

# Or via Modal (cloud GPU)
modal run scripts/modal_self_aligning_v2.py --model pythia-2.8b
    """, language="bash")

    st.markdown("### How It Works")
    st.markdown("""
    1. **Compute truth direction** from calibration prompts (true vs false statements)
    2. **Identify probe layer** at quarter-depth (where coupling bottleneck lives)
    3. **Add steering vector** (truth direction × strength) at probe layer during generation
    4. **Output changes** from misaligned to aligned without capability degradation
    """)

# ══════════════════════════════════════
# TAB 5: Predictions
# ══════════════════════════════════════
with tab5:
    st.header("7 Falsifiable Predictions")
    st.markdown("Each prediction has a **timestamped pass/fail criterion**. Check back as new models release.")

    predictions = [
        {"#": 1, "Prediction": "SWE saturation", "Deadline": "Dec 2026",
         "Pass": "Top-5 spread < 2pp", "Fail": "Spread > 5pp", "Status": "⏳ Pending"},
        {"#": 2, "Prediction": "IFEval activation", "Deadline": "Dec 2026",
         "Pass": "r(GPQA,IFEval) > +0.6, n≥8", "Fail": "r < 0.3", "Status": "⏳ Pending"},
        {"#": 3, "Prediction": "DeepSeek coding-first", "Deadline": "Next 2 releases",
         "Pass": "h < 0 both", "Fail": "h > +5 either", "Status": "⏳ Pending"},
        {"#": 4, "Prediction": "Google reasoning advantage", "Deadline": "Next 2 releases",
         "Pass": "h > +3 both", "Fail": "h < 0 either", "Status": "⏳ Pending"},
        {"#": 5, "Prediction": "Cooperative coupling persists", "Deadline": "May 2027",
         "Pass": "r(SWE,GPQA) > +0.5, n≥30", "Fail": "r < 0.3", "Status": "⏳ Pending"},
        {"#": 6, "Prediction": "IFEval→HLE handoff (Nc4)", "Deadline": "Dec 2027",
         "Pass": "IFEval spread < 3pp, HLE > 15pp", "Fail": "IFEval > 8pp", "Status": "⏳ Pending"},
        {"#": 7, "Prediction": "SWE-HLE decoupling", "Deadline": "Dec 2026",
         "Pass": "r(SWE,HLE) < 0, n≥10", "Fail": "r > +0.3", "Status": "⏳ Pending"},
    ]
    st.dataframe(predictions, use_container_width=True, hide_index=True)

    # Confirmed predictions
    st.markdown("### Already Confirmed (Base Scale)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("✅ OLMo at γ₁₂ = 0.000\nZero-parameter prediction confirmed by AI2")
    with c2:
        st.success("✅ Llama-2 holdout: 5.6% MAE\nCross-family, twice polynomial accuracy")
    with c3:
        st.success("✅ Qwen3 cooperative at all scales\nTax eliminated by data curation")

# ══════════════════════════════════════
# TAB 6: OPT Internal Ladder
# ══════════════════════════════════════
with tab6:
    st.header("OPT Internal Coupling: The Nc2 Cascade")
    st.markdown("Cooperation rises, peaks, drops, and begins recovering — the same cycle as Nc1.")

    opt_sizes = ['125M', '1.3B', '6.7B', '13B', '30B', '66B']
    opt_coupling = [0.514, 0.645, 0.741, 0.876, 0.356, 0.396]
    opt_competing = [0, 0, 0, 0, 75, 150]

    fig = go.Figure()
    colors = ['#ffd166', '#ffd166', '#06d6a0', '#06d6a0', '#ef476f', '#4cc9f0']
    fig.add_trace(go.Bar(
        x=opt_sizes, y=opt_coupling,
        marker_color=colors,
        text=[f'{c:.3f}' for c in opt_coupling],
        textposition='outside'
    ))

    fig.add_annotation(x='13B', y=0.876, text='PEAK', showarrow=True,
                       arrowhead=2, font=dict(color='#06d6a0', size=14))
    fig.add_annotation(x='30B', y=0.356, text='Nc2 DROP', showarrow=True,
                       arrowhead=2, font=dict(color='#ef476f', size=14))
    fig.add_annotation(x='66B', y=0.396, text='Recovery begins', showarrow=True,
                       arrowhead=2, font=dict(color='#4cc9f0', size=12))

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0a0e17', plot_bgcolor='#0f1420',
        yaxis_title='Net Internal Coupling', xaxis_title='Model Size',
        height=450, title='OPT Internal Coupling Trajectory (125M → 66B)',
        yaxis=dict(range=[0, 1.0])
    )
    st.plotly_chart(fig, use_container_width=True)

    # Competing units
    st.markdown("### Competing Units Emerge at Nc2")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=opt_sizes, y=opt_competing, marker_color='#ef476f'))
    fig2.update_layout(
        template='plotly_dark', paper_bgcolor='#0a0e17', plot_bgcolor='#0f1420',
        yaxis_title='Competing Units', height=300,
        title='Zero competing heads through 13B → 75 at 30B → 150 at 66B'
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    **Interpretation**: OPT cooperation increases monotonically from 125M to 13B (the Nc1 bonus phase),
    then **drops sharply at 30B** with 75 competing units appearing where there were none.
    At 66B, coupling partially recovers — the same rise→peak→drop→recovery pattern
    that governs Nc1 in Pythia, now repeating at Nc2 scale.
    """)

# ── Footer ──
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#7a9abf; font-size:0.85rem;">
    CAPE: Capability Coupling Analysis of Phase Emergence · Adil Amin (2026)<br>
    <a href="https://github.com/adilamin89/cape-scaling" style="color:#4cc9f0">GitHub</a> ·
    <a href="https://adilamin89.github.io/cape-scaling" style="color:#4cc9f0">Static Dashboard</a> ·
    Papers: <a href="https://github.com/adilamin89/cape-scaling/blob/main/paper3a_nature.pdf" style="color:#06d6a0">Nature</a> ·
    <a href="https://github.com/adilamin89/cape-scaling/blob/main/paper3b_neurips.pdf" style="color:#ffd166">NeurIPS</a>
</div>
""", unsafe_allow_html=True)
