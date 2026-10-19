import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))
from api_client import ApiClient, ApiClientError, cognito_login

DEMO_EXPERIMENTS = [{
    "name": "CartPole demo study",
    "experiment_id": "demo-experiment-001",
    "run_count": 3,
    "created_at": "2026-10-19T09:00:00Z",
}]
DEMO_RUNS = [
    {"run_id": "demo-run-001", "name": "PPO baseline", "status": "SUCCEEDED", "algo": "PPO",
     "env_id": "CartPole-v1", "current_step": 50000, "total_timesteps": 50000,
     "best_reward": 410.0, "final_reward": 410.0, "created_at": "2026-10-19T09:10:00Z"},
    {"run_id": "demo-run-002", "name": "PPO tuned", "status": "RUNNING", "algo": "PPO",
     "env_id": "CartPole-v1", "current_step": 40000, "total_timesteps": 50000,
     "best_reward": 480.0, "final_reward": 450.0, "created_at": "2026-10-19T09:20:00Z"},
    {"run_id": "demo-run-003", "name": "A2C comparison", "status": "FAILED", "algo": "A2C",
     "env_id": "CartPole-v1", "current_step": 20000, "total_timesteps": 50000,
     "best_reward": 35.0, "final_reward": 33.0, "created_at": "2026-10-19T09:30:00Z"},
]
DEMO_STEPS = [0, 10000, 20000, 30000, 40000, 50000]
DEMO_METRICS = {
    "ep_rew_mean": [[20, 70, 160, 250, 335, 410], [20, 100, 210, 330, 430, 480],
                    [20, 32, 35, 33, 34, 33]],
    "ep_len_mean": [[15, 40, 90, 140, 180, 205], [15, 55, 115, 170, 205, 230],
                    [15, 20, 22, 21, 22, 21]],
    "loss": [[1.0, 0.8, 0.6, 0.4, 0.3, 0.2], [1.0, 0.7, 0.5, 0.3, 0.2, 0.1],
             [1.0, 0.9, 0.8, 0.9, 0.8, 0.9]],
    "value_loss": [[0.8, 0.6, 0.5, 0.3, 0.2, 0.15], [0.8, 0.5, 0.4, 0.25, 0.15, 0.1],
                   [0.8, 0.75, 0.8, 0.78, 0.82, 0.8]],
    "policy_loss": [[-0.02, -0.015, -0.012, -0.01, -0.008, -0.006],
                    [-0.02, -0.018, -0.014, -0.012, -0.009, -0.007],
                    [-0.02, -0.01, -0.005, -0.004, -0.003, -0.002]],
    "fps": [[1200, 1250, 1230, 1280, 1260, 1300], [1180, 1220, 1270, 1290, 1320, 1310],
            [1150, 1170, 1130, 1160, 1140, 1120]],
}


def demo_comparison(run_ids, metric):
    runs_by_id = {run["run_id"]: run for run in DEMO_RUNS}
    run_indexes = {run["run_id"]: index for index, run in enumerate(DEMO_RUNS)}
    series = {
        run_id: DEMO_METRICS[metric][run_indexes[run_id]]
        for run_id in run_ids
    }
    summary = []
    for run_id, values in series.items():
        summary.append({
            "run_id": run_id,
            "name": runs_by_id[run_id]["name"],
            "status": runs_by_id[run_id]["status"],
            "best": max(values),
            "final": values[-1],
            "auc": sum(
                (x1 - x0) * (y0 + y1) / 2
                for x0, x1, y0, y1 in zip(DEMO_STEPS, DEMO_STEPS[1:], values, values[1:])
            ),
            "steps_to_threshold": None,
        })
    ranking = [
        row["run_id"]
        for row in sorted(summary, key=lambda row: (row["final"] is None, -(row["final"] or 0)))
    ]
    return {"steps": DEMO_STEPS, "series": series, "summary": summary, "ranking": ranking}


st.set_page_config(page_title="RL Platform", layout="wide")
st.title("RL Training Platform")
with st.sidebar:
    st.header("Connection")
    demo_mode = st.toggle(
        "Demo mode (sample data)",
        help="Preview the dashboard without AWS. Actions that need real runs or artifacts are disabled.",
    )
    if demo_mode:
        st.caption("Showing sample data. No AWS credentials or deployment required.")
    else:
        api = st.text_input("API URL", st.session_state.get("api", ""))
        client_id = st.text_input("User pool client id", st.session_state.get("client_id", ""))
        region = st.text_input("Region", st.session_state.get("region", "us-east-1"))
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.button("Log in"):
            try:
                st.session_state.update(token=cognito_login(client_id, user, pw, region), api=api,
                                        client_id=client_id, region=region)
                st.success("logged in")
            except Exception as e:
                st.error(str(e))
if not demo_mode and "token" not in st.session_state:
    st.info("Log in from the sidebar.")
    st.stop()
c = None if demo_mode else ApiClient(st.session_state["api"], st.session_state["token"])

# Calls an API method and caches the result for a short time
@st.cache_data(ttl=15, show_spinner=False)
def _call(kind, token, *args):
    cl = ApiClient(st.session_state["api"], token)
    return getattr(cl, kind)(*args)

if demo_mode:
    exps = DEMO_EXPERIMENTS
else:
    try:
        exps = _call("list_experiments", st.session_state["token"])["items"]
    except ApiClientError as e:
        st.error(f"{e} (token expired? log in again)")
        st.stop()
if not exps:
    st.warning("No experiments yet. Create one with `rlctl experiments create`.")
    st.stop()
st.subheader("Experiments")
st.dataframe(pd.DataFrame(exps)[["name", "experiment_id", "run_count", "created_at"]], use_container_width=True)
exp = st.selectbox("Experiment", exps, format_func=lambda e: f"{e['name']} ({e['experiment_id'][:8]})")
status = st.selectbox("Status filter", ["", "RUNNING", "QUEUED", "SUCCEEDED", "FAILED", "CANCELLED"])
if demo_mode:
    runs = [run for run in DEMO_RUNS if not status or run["status"] == status]
else:
    runs = _call("list_runs", st.session_state["token"], exp["experiment_id"], status or None)["items"]
if not runs:
    st.info("No runs.")
    st.stop()
df = pd.DataFrame(runs)
cols = [x for x in ["run_id", "name", "status", "algo", "env_id", "current_step", "total_timesteps",
                    "best_reward", "final_reward", "created_at"] if x in df]
st.subheader("Runs")
st.dataframe(df[cols], use_container_width=True)
st.subheader("Reward curves")
picked = st.multiselect("Runs to overlay", df["run_id"].tolist(), default=df["run_id"].tolist()[:3],
                        format_func=lambda r: f"{df.set_index('run_id').loc[r, 'name']} ({r[:8]})")
metric = st.selectbox("Metric", ["ep_rew_mean", "ep_len_mean", "loss", "value_loss", "policy_loss", "fps"])
if picked:
    cmp = demo_comparison(picked, metric) if demo_mode else c.compare(exp["experiment_id"], picked, metric)
    curves = pd.DataFrame(cmp["series"], index=cmp["steps"])
    curves.columns = [f"{df.set_index('run_id').loc[r, 'name']} ({r[:6]})" for r in curves.columns]
    st.line_chart(curves)
    st.dataframe(pd.DataFrame(cmp["summary"]), use_container_width=True)
st.subheader("Artifacts")
rid = st.selectbox("Run", df["run_id"].tolist())
cols_ = st.columns(4)
for col, kind in zip(cols_, ["model", "checkpoint", "log", "config"]):
    with col:
        if st.button(f"Get {kind} link", key=kind, disabled=demo_mode):
            try:
                st.link_button(f"Download {kind}", c.artifact_url(rid, kind)["url"])
            except ApiClientError as e:
                st.warning(e.message)
if demo_mode:
    st.caption("Artifact downloads and run cancellation require a deployed platform.")
if st.button("Cancel selected run", disabled=demo_mode):
    try:
        st.write(c.cancel_run(rid))
    except ApiClientError as e:
        st.error(e.message)
