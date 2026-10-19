from __future__ import annotations
from typing import Optional
from boto3.dynamodb.conditions import Key
from common import db
from common.config_schema import CompareIn
from common.responses import ApiError, api_handler, ok, owner_of, parse_body, path_param

POINTS = 200

# Finds the value of a series at x, or None outside it
def interpolate(series: list[tuple[float, float]], x: float) -> Optional[float]:
    if not series or x < series[0][0] or x > series[-1][0]:
        return None
    lo, hi = 0, len(series) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if series[mid][0] <= x:
            lo = mid
        else:
            hi = mid
    (x0, y0), (x1, y1) = series[lo], series[hi]
    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

# Returns the area under a series
def auc(series: list[tuple[float, float]]) -> float:
    return sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(series, series[1:]))

# Returns best, final, area and steps to the threshold
def summarize(series, threshold):
    if not series:
        return {"best": None, "final": None, "auc": None, "steps_to_threshold": None}
    steps_to = None
    if threshold is not None:
        steps_to = next((int(x) for x, y in series if y >= threshold), None)
    return {"best": max(y for _, y in series), "final": series[-1][1], "auc": auc(series),
            "steps_to_threshold": steps_to}

# Loads the sorted points of a metric for a run
def fetch_series(run_id: str, metric: str) -> list[tuple[float, float]]:
    rows = db.query_all(db.metrics_table(), KeyConditionExpression=Key("run_id").eq(run_id),
                        ProjectionExpression="step, #m", ExpressionAttributeNames={"#m": metric})
    pts = [(float(r["step"]), float(r[metric])) for r in rows if metric in r]
    return sorted(pts)

# Compares runs on one shared step grid
@api_handler
def handler(event, context):
    owner = owner_of(event)
    exp_id = path_param(event, "experiment_id")
    body = CompareIn.model_validate(parse_body(event))
    all_series, rows = {}, []
    for rid in body.run_ids:
        run = db.get_run(rid)
        if not run or run["owner"] != owner or run["experiment_id"] != exp_id:
            raise ApiError("RUN_NOT_FOUND", f"run {rid} not found in this experiment", 404)
        series = fetch_series(rid, body.metric)
        all_series[rid] = series
        rows.append({"run_id": rid, "name": run.get("name"), "status": run["status"],
                     **summarize(series, body.threshold)})
    xs = [x for s in all_series.values() for x, _ in s]
    if xs:
        lo, hi = min(xs), max(xs)
        grid = [lo + (hi - lo) * i / (POINTS - 1) for i in range(POINTS)] if hi > lo else [lo]
    else:
        grid = []
    aligned = {rid: [interpolate(s, x) for x in grid] for rid, s in all_series.items()}
    ranking = [r["run_id"] for r in sorted(rows, key=lambda r: (r["final"] is None, -(r["final"] or 0)))]
    return ok({"metric": body.metric, "steps": [round(x) for x in grid], "series": aligned,
               "summary": rows, "ranking": ranking})
