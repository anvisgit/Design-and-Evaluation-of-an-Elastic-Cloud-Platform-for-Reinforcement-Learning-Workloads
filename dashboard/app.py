import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "cli"))
from api_client import ApiClient, ApiClientError, cognito_login

st.set_page_config(page_title="RL Platform", layout="wide")
st.title("RL Training Platform")
with st.sidebar:
    st.header("Connection")
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
if "token" not in st.session_state:
    st.info("Log in from the sidebar.")
    st.stop()
c = ApiClient(st.session_state["api"], st.session_state["token"])

# Calls an API method and caches the result for a short time
@st.cache_data(ttl=15, show_spinner=False)
def _call(kind, token, *args):
    cl = ApiClient(st.session_state["api"], token)
    return getattr(cl, kind)(*args)

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
    cmp = c.compare(exp["experiment_id"], picked, metric)
    curves = pd.DataFrame(cmp["series"], index=cmp["steps"])
    curves.columns = [f"{df.set_index('run_id').loc[r, 'name']} ({r[:6]})" for r in curves.columns]
    st.line_chart(curves)
    st.dataframe(pd.DataFrame(cmp["summary"]), use_container_width=True)
st.subheader("Artifacts")
rid = st.selectbox("Run", df["run_id"].tolist())
cols_ = st.columns(4)
for col, kind in zip(cols_, ["model", "checkpoint", "log", "config"]):
    with col:
        if st.button(f"Get {kind} link", key=kind):
            try:
                st.link_button(f"Download {kind}", c.artifact_url(rid, kind)["url"])
            except ApiClientError as e:
                st.warning(e.message)
if st.button("Cancel selected run"):
    try:
        st.write(c.cancel_run(rid))
    except ApiClientError as e:
        st.error(e.message)
