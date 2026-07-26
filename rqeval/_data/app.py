"""
LLM Reasoning Quality Evaluation Framework
Streamlit Web Interface
Run: streamlit run app.py
"""

import streamlit as st
import json
import os
import sys
import yaml
import tempfile
import subprocess
import pandas as pd
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM Reasoning Quality Evaluator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.03em; }
.stButton > button {
    background: #4f46e5; color: white; border: none; border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace; font-size: 13px;
    padding: 10px 24px; transition: all 0.2s;
}
.stButton > button:hover { background: #6366f1; transform: translateY(-1px); }
.model-card {
    background: #1a1d27; border: 1px solid #2d3149;
    border-radius: 8px; padding: 12px 16px; margin: 6px 0;
}
.dim-badge {
    display: inline-block; background: #1e2235; border: 1px solid #3d4566;
    border-radius: 4px; padding: 3px 10px;
    font-family: 'IBM Plex Mono', monospace; font-size: 12px;
    color: #a5b4fc; margin: 2px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
METRICS = ["correctness", "consistency", "robustness",
           "logical_coherence", "efficiency", "stability"]

METRIC_LABELS = {
    "correctness":       "CQ - Correctness",
    "consistency":       "CS - Consistency",
    "robustness":        "RS - Robustness",
    "logical_coherence": "LS - Logical Coherence",
    "efficiency":        "ES - Efficiency",
    "stability":         "SS - Stability",
}

METRIC_DESC = {
    "correctness":       "Is the final answer correct?",
    "consistency":       "Same answer across K repeated runs?",
    "robustness":        "Stable under rephrased questions?",
    "logical_coherence": "No contradictions in reasoning steps?",
    "efficiency":        "Correct and concise?",
    "stability":         "Same reasoning process across runs?",
}

PRESET_STRATEGIES = {
    "Balanced":           [1/6]*6,
    "Clinical/Medical":   [0.40, 0.05, 0.30, 0.20, 0.03, 0.02],
    "Legal/Compliance":   [0.15, 0.25, 0.20, 0.35, 0.03, 0.02],
    "Accuracy Priority":  [0.40, 0.25, 0.15, 0.10, 0.05, 0.05],
    "Efficiency Priority":[0.20, 0.15, 0.15, 0.10, 0.30, 0.10],
    "Edge Device/IoT":    [0.30, 0.03, 0.10, 0.05, 0.50, 0.02],
    "Custom":             None,
}

MODEL_TYPES = ["openai", "anthropic", "gemini", "deepseek", "asu", "local", "mock"]

PROVIDER_DEFAULTS = {
    "openai":    {"base_url": "", "key_placeholder": "sk-..."},
    "anthropic": {"base_url": "", "key_placeholder": "sk-ant-..."},
    "gemini":    {"base_url": "", "key_placeholder": "AIza..."},
    "deepseek":  {"base_url": "https://api.deepseek.com", "key_placeholder": "sk-..."},
    "asu":       {"base_url": "", "key_placeholder": "CreateAI token — Model ID as provider:model_key, e.g. openai:gpt4o_mini"},
    "local":     {"base_url": "", "key_placeholder": "(no key needed)"},
    "mock":      {"base_url": "", "key_placeholder": "(no key needed)"},
}

# ── Session state init ────────────────────────────────────────────────────────
if "models" not in st.session_state:
    st.session_state.models = [
        {"name": "Mock-Model",    "type": "mock",      "model_id": "",                      "api_key_env": "",               "base_url": "", "key_val": "", "enabled": True,  "max_tokens": 256, "temperature": 0.7},
        {"name": "GPT-4o-mini",   "type": "openai",    "model_id": "gpt-4o-mini",           "api_key_env": "OPENAI_API_KEY", "base_url": "", "key_val": "", "enabled": False, "max_tokens": 256, "temperature": 0.7},
        {"name": "Claude Haiku",  "type": "anthropic", "model_id": "claude-haiku-4-5-20251001", "api_key_env": "ANTHROPIC_API_KEY", "base_url": "", "key_val": "", "enabled": False, "max_tokens": 256, "temperature": 0.7},
        {"name": "Gemini Flash",  "type": "gemini",    "model_id": "gemini-2.0-flash",      "api_key_env": "GOOGLE_API_KEY", "base_url": "", "key_val": "", "enabled": False, "max_tokens": 256, "temperature": 0.7},
        {"name": "DeepSeek-V3",   "type": "deepseek",  "model_id": "deepseek-chat",         "api_key_env": "DEEPSEEK_API_KEY", "base_url": "https://api.deepseek.com", "key_val": "", "enabled": False, "max_tokens": 256, "temperature": 0.7},
    ]

if "results" not in st.session_state:
    st.session_state.results = None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style="margin:0; font-size:28px; color:#e2e8f0;">
    🧠 LLM Reasoning Quality Evaluator
</h1>
<p style="margin:6px 0 16px; color:#6b7db3; font-family:'IBM Plex Mono',monospace; font-size:13px;">
    Multi-dimensional behavioral assessment beyond accuracy &nbsp;·&nbsp;
    <a href="https://arxiv.org/abs/2605.24661" style="color:#818cf8;">arXiv:2605.24661</a>
</p>
<hr style="border-color:#2d3149; margin-bottom:24px;"/>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")

    # ── STEP 1: Dataset ───────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Step 1 — Dataset")

    dataset_source = st.radio(
        "Source", ["Upload JSON/CSV", "Built-in benchmark"],
        label_visibility="collapsed",
    )

    uploaded_dataset = None
    builtin_dataset  = None
    n_samples        = 20

    if dataset_source == "Upload JSON/CSV":
        uploaded_file = st.file_uploader(
            "Upload dataset", type=["json","csv"],
            help="JSON: [{id, question, answer, perturbations}]\nCSV: question, answer columns",
        )
        if uploaded_file:
            uploaded_dataset = uploaded_file
            st.success(f"Loaded: {uploaded_file.name}")

        with st.expander("Required JSON format"):
            st.code('[{"id":"q1","question":"...","answer":"...","perturbations":["...","...","..."]}]')
    else:
        builtin_dataset = st.selectbox("Benchmark", ["synthetic","gsm8k","strategyqa","mmlu"])
        n_samples = st.slider("Items", 5, 250, 20, step=5)

    # ── STEP 2: Models ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Step 2 — Models")

    # Existing models
    for i, m in enumerate(st.session_state.models):
        with st.expander(f"{'[ON] ' if m['enabled'] else '[OFF]'} {m['name']}", expanded=False):
            col1, col2 = st.columns([3,1])
            with col1:
                m["enabled"] = st.checkbox("Enable", value=m["enabled"], key=f"en_{i}")
            with col2:
                if st.button("Del", key=f"del_{i}"):
                    st.session_state.models.pop(i)
                    st.rerun()

            if m["enabled"]:
                m["name"] = st.text_input("Display name", value=m["name"], key=f"nm_{i}")
                m["type"] = st.selectbox("Type", MODEL_TYPES,
                    index=MODEL_TYPES.index(m["type"]) if m["type"] in MODEL_TYPES else 0,
                    key=f"tp_{i}")

                if m["type"] not in ("mock",):
                    m["model_id"] = st.text_input("Model ID", value=m["model_id"], key=f"mid_{i}",
                        placeholder=f"e.g. gpt-4o, claude-3-5-sonnet-20241022")

                if m["type"] == "local":
                    m["model_id"] = st.text_input("HuggingFace model ID", value=m["model_id"],
                        key=f"hf_{i}", placeholder="e.g. Qwen/Qwen2.5-1.5B-Instruct")
                    device = st.selectbox("Device", ["cuda","cpu"], key=f"dev_{i}")
                    m["device"] = device
                    m["use_4bit"] = st.checkbox("4-bit quantization", value=True, key=f"4bit_{i}")
                    m["max_new_tokens"] = st.slider("Max new tokens", 32, 512, 128, key=f"mnt_{i}")
                elif m["type"] != "mock":
                    m["key_val"] = st.text_input("API Key", type="password",
                        value=m.get("key_val",""), key=f"kv_{i}",
                        placeholder=PROVIDER_DEFAULTS.get(m["type"],{}).get("key_placeholder",""))
                    default_url = PROVIDER_DEFAULTS.get(m["type"],{}).get("base_url","")
                    m["base_url"] = st.text_input("Base URL (optional)", key=f"bu_{i}",
                        value=m.get("base_url", default_url),
                        placeholder="Leave empty for default")

                col_t, col_k = st.columns(2)
                with col_t:
                    m["temperature"] = st.number_input("Temp", 0.0, 2.0,
                        value=m.get("temperature", 0.7), step=0.1, key=f"tmp_{i}")
                with col_k:
                    m["max_tokens"] = st.number_input("Max tokens", 64, 2048,
                        value=m.get("max_tokens", 256), step=64, key=f"mtk_{i}")

    # Add new model button
    st.markdown("---")
    if st.button("+ Add New Model", use_container_width=True):
        st.session_state.models.append({
            "name": f"Model-{len(st.session_state.models)+1}",
            "type": "openai",
            "model_id": "",
            "api_key_env": f"API_KEY_{len(st.session_state.models)+1}",
            "base_url": "",
            "key_val": "",
            "enabled": True,
            "max_tokens": 256,
            "temperature": 0.7,
        })
        st.rerun()

    # ── STEP 3: Aggregation ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Step 3 — Aggregation Strategy")

    preset = st.selectbox("Preset", list(PRESET_STRATEGIES.keys()))

    weights = {}
    if preset == "Custom":
        st.markdown("Set weights (auto-normalized to 1.0):")
        total = 0
        for m in METRICS:
            w = st.slider(METRIC_LABELS[m], 0.0, 1.0, 1/6, step=0.05,
                key=f"w_{m}", help=METRIC_DESC[m])
            weights[m] = w
            total += w
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        st.caption(f"Sum = {total:.2f} -> normalized to 1.0")
    else:
        vals = PRESET_STRATEGIES[preset]
        for i, m in enumerate(METRICS):
            weights[m] = vals[i]

    # Weight preview
    cols = st.columns(3)
    for i, m in enumerate(METRICS):
        short = METRIC_LABELS[m].split("-")[0].strip()
        cols[i%3].metric(short, f"{weights[m]:.2f}")

    # ── STEP 4: Run ───────────────────────────────────────────────────────────
    st.markdown("---")
    run_btn = st.button("Run Evaluation", use_container_width=True)

# ── Build config ──────────────────────────────────────────────────────────────
def build_config(ds_path=None) -> dict:
    models_cfg = []
    for m in st.session_state.models:
        if not m["enabled"]:
            continue

        if m["type"] == "mock":
            models_cfg.append({
                "name": m["name"], "type": "mock",
                "params": {"accuracy_level": 0.8, "seed": 42}
            })
        elif m["type"] == "local":
            models_cfg.append({
                "name": m["name"], "type": "local",
                "params": {
                    "model_id": m["model_id"],
                    "device": m.get("device","cuda"),
                    "use_4bit": m.get("use_4bit", True),
                    "max_new_tokens": m.get("max_new_tokens", 128),
                    "trust_remote_code": True,
                    "temperature": m["temperature"],
                }
            })
        elif m["type"] == "asu":
            # Model ID convention: "provider:model_key" (e.g. "aws:claude4_5_haiku").
            # Without a colon, provider defaults to "openai".
            key_env = m["api_key_env"] or "ASU_CREATEAI_TOKEN"
            if m.get("key_val"):
                os.environ[key_env] = m["key_val"]
            mid = m["model_id"]
            provider, _, mkey = mid.partition(":")
            if not mkey:
                provider, mkey = "openai", mid
            entry = {
                "name": m["name"], "type": "asu",
                "params": {
                    "model_name": mkey,
                    "model_provider": provider,
                    "api_key_env": key_env,
                    "max_tokens": m["max_tokens"],
                    "temperature": m["temperature"],
                    "max_retries": 5,
                    "timeout": 120,
                }
            }
            if m.get("base_url"):
                entry["params"]["base_url"] = m["base_url"]
            models_cfg.append(entry)
        else:
            key_env = m["api_key_env"] or f"KEY_{m['name'].upper().replace('-','_')}"
            if m.get("key_val"):
                os.environ[key_env] = m["key_val"]
            entry = {
                "name": m["name"], "type": m["type"],
                "params": {
                    "model_id": m["model_id"],
                    "api_key_env": key_env,
                    "max_tokens": m["max_tokens"],
                    "temperature": m["temperature"],
                    "max_retries": 3,
                    "timeout": 60,
                }
            }
            if m.get("base_url"):
                entry["params"]["base_url"] = m["base_url"]
            models_cfg.append(entry)

    datasets_cfg = []
    if ds_path:
        datasets_cfg.append({
            "name": "uploaded_dataset", "type": "json",
            "params": {"path": ds_path, "num_samples": 9999}
        })
    elif builtin_dataset:
        datasets_cfg.append({
            "name": builtin_dataset, "type": builtin_dataset,
            "params": {"num_samples": n_samples, "seed": 42}
        })

    strategy_name = preset.lower().replace(" ","_").replace("/","_")
    if strategy_name == "custom":
        strategy_name = "my_strategy"

    return {
        "experiment": {
            "name": "WebUI_Evaluation",
            "seed": 42, "deterministic": False,
            "output_dir": "outputs", "verbose": True,
        },
        "models": models_cfg,
        "datasets": datasets_cfg,
        "metrics": {
            "consistency_runs": 3,
            "robustness_perturbations": 3,
            "stability_runs": 3,
            "nli_model": "cross-encoder/nli-deberta-v3-small",
            "bertscore_model": "distilbert-base-uncased",
        },
        "aggregation": {
            "strategies": {
                "balanced": {m: round(1/6,4) for m in METRICS},
                strategy_name: weights,
            }
        }
    }

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Results", "Dataset Preview", "Config Preview"])

with tab2:
    st.markdown("#### Dataset Preview")
    if uploaded_dataset is not None:
        try:
            content = uploaded_dataset.read()
            uploaded_dataset.seek(0)
            if uploaded_dataset.name.endswith(".json"):
                data = json.loads(content)
                df = pd.DataFrame(data)
            else:
                import io
                df = pd.read_csv(io.BytesIO(content))
            st.info(f"{len(df)} items")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Parse error: {e}")
    elif builtin_dataset:
        st.info(f"Built-in: **{builtin_dataset}** — {n_samples} items")
    else:
        st.info("No dataset selected.")

with tab3:
    st.markdown("#### Generated Config YAML")
    st.code(yaml.dump(build_config(), default_flow_style=False, allow_unicode=True), language="yaml")

with tab1:
    if run_btn:
        # Validate
        enabled = [m for m in st.session_state.models if m["enabled"]]
        errors = []
        if not enabled:
            errors.append("Enable at least one model.")
        if uploaded_dataset is None and builtin_dataset is None:
            errors.append("Select a dataset.")
        for m in enabled:
            if m["type"] not in ("mock","local") and not m.get("key_val") and not os.environ.get(m.get("api_key_env","")):
                errors.append(f"API key missing for {m['name']}.")
            if m["type"] != "mock" and not m.get("model_id") and m["type"] != "local":
                errors.append(f"Model ID missing for {m['name']}.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            tmpdir = tempfile.mkdtemp()
            ds_path = None

            # Save uploaded dataset
            if uploaded_dataset is not None:
                content = uploaded_dataset.read()
                uploaded_dataset.seek(0)
                ds_path = os.path.join(tmpdir, "dataset.json")
                if uploaded_dataset.name.endswith(".csv"):
                    import io
                    df_up = pd.read_csv(io.BytesIO(content))
                    records = [
                        {"id": f"q{i:04d}", "question": str(row.get("question","")),
                         "answer": str(row.get("answer","")), "type": "reasoning"}
                        for i, row in df_up.iterrows()
                    ]
                    with open(ds_path, "w") as f:
                        json.dump(records, f)
                else:
                    with open(ds_path, "wb") as f:
                        f.write(content)

            cfg = build_config(ds_path)
            cfg_path = os.path.join(tmpdir, "config.yaml")
            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

            main_py = os.path.join(os.getcwd(), "main.py")
            if not os.path.exists(main_py):
                st.error("main.py not found. Run 'llm-eval setup' first.")
            else:
                progress = st.progress(0, text="Starting...")
                log_area = st.empty()
                status   = st.empty()

                env = os.environ.copy()
                proc = subprocess.Popen(
                    [sys.executable, main_py, "--config", cfg_path],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, env=env, cwd=os.getcwd(),
                )

                lines = []
                step  = 0
                for line in proc.stdout:
                    lines.append(line.rstrip())
                    step = min(step + 2, 95)
                    progress.progress(step, text=line.strip()[:80] or "Running...")
                    log_area.code("\n".join(lines[-20:]), language="bash")

                proc.wait()
                progress.progress(100, text="Done!")

                if proc.returncode == 0:
                    status.success("Evaluation complete!")
                    # Find latest Excel
                    outputs_dir = os.path.join(os.getcwd(), "outputs")
                    excel_files = sorted(
                        Path(outputs_dir).rglob("reasoning_quality_results.xlsx"),
                        key=lambda p: p.stat().st_mtime, reverse=True,
                    )
                    if excel_files:
                        xlsx = excel_files[0]
                        try:
                            df_res = pd.read_excel(xlsx, sheet_name="Overall Raw Metrics")
                            st.session_state.results = df_res
                            st.session_state.xlsx_path = str(xlsx)
                            # Also read aggregated
                            df_agg = pd.read_excel(xlsx, sheet_name="Aggregated Scores")
                            st.session_state.results_agg = df_agg
                        except Exception as e:
                            st.warning(f"Could not read Excel: {e}")
                else:
                    status.error("Evaluation failed. Check logs above.")

    # Show results
    if st.session_state.results is not None:
        df = st.session_state.results
        st.markdown("### Raw Scores")

        # Model score cards
        model_cols = st.columns(max(len(df), 1))
        for i, (_, row) in enumerate(df.iterrows()):
            with model_cols[i % len(model_cols)]:
                st.markdown(f"**{row.get('Model','')}**")
                for m in METRICS:
                    label = m.replace("_"," ").title()
                    val = row.get(label) or row.get(m.title()) or row.get(m) or 0
                    st.metric(label, f"{float(val):.3f}")

        st.markdown("---")
        st.markdown("### Full Results Table")
        st.dataframe(df, use_container_width=True)

        if hasattr(st.session_state, "results_agg") and st.session_state.results_agg is not None:
            st.markdown("### Aggregated Scores")
            st.dataframe(st.session_state.results_agg, use_container_width=True)

        with open(st.session_state.xlsx_path, "rb") as f:
            st.download_button(
                "Download Excel Results",
                f,
                file_name="reasoning_quality_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    else:
        if not run_btn:
            st.markdown("""
            <div style="text-align:center; padding:60px 20px;">
                <div style="font-size:48px; margin-bottom:16px;">🧠</div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:15px; color:#6b7db3;">
                    Configure your evaluation in the sidebar<br>and click Run Evaluation
                </div>
                <div style="margin-top:24px;">
                    <span class="dim-badge">CQ Correctness</span>
                    <span class="dim-badge">CS Consistency</span>
                    <span class="dim-badge">RS Robustness</span><br/>
                    <span class="dim-badge">LS Coherence</span>
                    <span class="dim-badge">ES Efficiency</span>
                    <span class="dim-badge">SS Stability</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
