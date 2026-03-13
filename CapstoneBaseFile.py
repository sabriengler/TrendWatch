from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from statsmethods import shewhart, ewma, mc_ewma, cusum, farrington, glm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy.stats import norm
import pandas as pd
from scipy.optimize import minimize_scalar
import threading
import uuid
import json
import os
import tempfile

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# ─────────────────────────────────────────────
# File-based job + results store (works across gunicorn workers)
# ─────────────────────────────────────────────

JOB_DIR = tempfile.gettempdir()
RESULTS_FILE = os.path.join(JOB_DIR, "previous_results.json")


def save_job(job_id, data):
    path = os.path.join(JOB_DIR, f"job_{job_id}.json")
    with open(path, 'w') as f:
        json.dump(data, f)


def load_job(job_id):
    path = os.path.join(JOB_DIR, f"job_{job_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def delete_job(job_id):
    path = os.path.join(JOB_DIR, f"job_{job_id}.json")
    if os.path.exists(path):
        os.remove(path)


def load_results():
    if not os.path.exists(RESULTS_FILE):
        return []
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)


def save_results(results):
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f)


# ─────────────────────────────────────────────
# Simulation functions
# ─────────────────────────────────────────────

def generate_behavior_data_sim(behavior, params, n_baseline):
    x = np.arange(n_baseline)
    if behavior == 'stable':
        dt = params.get('distribution_type', 'normal')
        if dt == 'normal':
            data = np.random.normal(loc=params['mean'], scale=params['std'], size=n_baseline)
        elif dt == 'lognormal':
            m, s = params['mean'], params['std']
            std_log = np.sqrt(np.log(1 + (s**2) / (m**2)))
            mu_log = np.log(m) - 0.5 * std_log**2
            data = np.random.lognormal(mean=mu_log, sigma=std_log, size=n_baseline)
        else:
            raise ValueError("Unsupported distribution type!")
    elif behavior == 'trending':
        noise = params.get('noise', 1.0)
        data = params['start'] + params['slope'] * x + np.random.normal(scale=noise, size=n_baseline)
    elif behavior == 'periodic':
        noise = params.get('noise', 1.0)
        data = params['mean'] + params['amplitude'] * np.sin(2 * np.pi * x / params['period']) + np.random.normal(scale=noise, size=n_baseline)
    else:
        raise ValueError("Unsupported behavior type!")
    return list(data)


def calculate_limits_sim(data, sigma_multiplier, analysis_method="shewhart", lambda_val=0.3):
    mean = np.mean(data)
    if analysis_method == "mc-ewma":
        result = mc_ewma(data, mean, 1, sigma_multiplier, lambda_val, baseline_period=0)
        mc_series = result['series']
        residuals = np.array(data) - mc_series
        MR = np.abs(np.diff(residuals))
    else:
        MR = np.abs(np.diff(data))
    MR_bar = np.mean(MR) if len(MR) > 0 else 0
    sigma = MR_bar / 1.128
    return (mean - sigma_multiplier * sigma, mean + sigma_multiplier * sigma), \
           (mean - (sigma_multiplier - 1) * sigma, mean + (sigma_multiplier - 1) * sigma), mean, sigma


def apply_change_sim(data, change, change_day, params, original_behavior, baseline_mean, sigma,
                     analysis_method, sigma_multiplier, baseline_period, lambda_val, max_days, alpha_val,
                     k_val=None, h_val=None):
    noise_val = params.get('noise', 1.0)
    std = params.get('std', None)
    if original_behavior == 'periodic':
        period = params.get('period', 50)
        amplitude = params.get('amplitude', 10)
    elif original_behavior == 'trending':
        slope = params.get('slope', 0.1)
        start = params['start']

    while len(data) < baseline_period:
        idx = len(data)
        if original_behavior == 'stable':
            new_value = np.random.normal(loc=baseline_mean, scale=std)
        elif original_behavior == 'periodic':
            cycle = idx % period
            new_value = baseline_mean + amplitude * np.sin(2*np.pi*cycle/period) + np.random.normal(scale=noise_val)
        elif original_behavior == 'trending':
            new_value = np.random.normal(loc=start + slope * idx, scale=noise_val)
        data.append(new_value)

    starting_value = None
    step_change_done = False
    new_intercept = None
    out_of_control_index = None

    while len(data) < max_days:
        idx = len(data)
        if change and idx >= (change_day if change_day is not None else baseline_period):
            start_idx = change_day if change_day is not None else baseline_period
            if change['type'] == 'step':
                factor = change.get('factor', 1.0)
                if original_behavior == 'stable':
                    new_value = np.random.normal(loc=baseline_mean * factor, scale=std)
                elif original_behavior == 'periodic':
                    cycle = idx % period
                    new_value = np.random.normal(loc=baseline_mean * factor, scale=noise_val) + amplitude * np.sin(2*np.pi*cycle/period)
                elif original_behavior == 'trending':
                    if not step_change_done:
                        window = min(5, len(data))
                        new_intercept = np.mean(data[-window:]) * factor
                        step_change_done = True
                    new_value = np.random.normal(loc=new_intercept + slope * (idx - start_idx), scale=noise_val)
            elif change['type'] == 'trending':
                added_slope = change['slope']
                duration = change['duration']
                if original_behavior == 'trending' and starting_value is None:
                    starting_value = data[change_day - 1] if (change_day - 1) < len(data) else start + slope * (start_idx - 1)
                trend_index = idx - start_idx
                if trend_index < duration:
                    if original_behavior == 'stable':
                        new_value = np.random.normal(loc=baseline_mean + added_slope * trend_index, scale=std)
                    elif original_behavior == 'periodic':
                        cycle = idx % period
                        new_value = np.random.normal(loc=baseline_mean + added_slope * trend_index, scale=noise_val) + amplitude * np.sin(2*np.pi*cycle/period)
                    elif original_behavior == 'trending':
                        new_value = np.random.normal(loc=starting_value + added_slope * trend_index, scale=noise_val)
                else:
                    if original_behavior == 'stable':
                        new_value = np.random.normal(loc=baseline_mean + added_slope * duration, scale=std)
                    elif original_behavior == 'periodic':
                        cycle = idx % period
                        new_value = np.random.normal(loc=baseline_mean + added_slope * duration, scale=noise_val) + amplitude * np.sin(2*np.pi*cycle/period)
                    elif original_behavior == 'trending':
                        new_value = np.random.normal(loc=starting_value + added_slope * duration + slope * (idx - (change_day + duration)), scale=noise_val)
        else:
            if original_behavior == 'stable':
                new_value = np.random.normal(loc=baseline_mean, scale=std)
            elif original_behavior == 'periodic':
                cycle = idx % period
                new_value = baseline_mean + amplitude * np.sin(2*np.pi*cycle/period) + np.random.normal(scale=noise_val)
            elif original_behavior == 'trending':
                if change and change['type'] == 'step' and step_change_done:
                    new_value = np.random.normal(loc=new_intercept + slope * (idx - change_day), scale=noise_val)
                else:
                    new_value = np.random.normal(loc=start + slope * idx, scale=noise_val)
        data.append(new_value)

        if len(data) > baseline_period:
            if analysis_method == 'shewhart':
                result = shewhart(data, baseline_mean, sigma, sigma_multiplier, baseline_period)
            elif analysis_method == 'ewma':
                result = ewma(data, baseline_mean, sigma, sigma_multiplier, lambda_val, baseline_period)
            elif analysis_method == 'mc-ewma':
                result = mc_ewma(data, baseline_mean, sigma, sigma_multiplier, lambda_val, baseline_period)
            elif analysis_method == 'cusum':
                result = cusum(data, baseline_mean, sigma, sigma_multiplier, baseline_period, k_val=k_val, h_val=h_val)
            elif analysis_method == 'farrington':
                data_farr = np.clip(np.round(data), 0, None)
                result = farrington(data_farr, baseline_period, alpha=alpha_val)
            elif analysis_method == 'glm':
                result = glm(data, baseline_period, alpha=alpha_val)
            out_of_control_index = result["out_of_control_index"]
            if out_of_control_index is not None:
                data = data[:out_of_control_index + 1]
                break

    return data, out_of_control_index


def plot_replicates_and_histogram(replications, run_lengths, change_day, analysis_method, sigma_multiplier,
                                  baseline_period, n_replications, arl_value, metric_label, avg_sigma,
                                  avg_change_day, limit_stopped_percentage, lambda_val, ucl_ci, lcl_ci,
                                  late_threshold, arl_moe, alpha_val, k_val=None, h_val=None):
    fig = plt.figure(figsize=(16, 5.5))
    fig.patch.set_facecolor('#f8fafc')
    gs = fig.add_gridspec(nrows=2, ncols=3, width_ratios=[0.8, 1.8, 2.5])

    legend_ax = fig.add_subplot(gs[:, 0])
    legend_ax.axis("off")

    run_arr = np.array(run_lengths)
    rl_percentiles = np.percentile(run_arr, [10, 25, 50, 75, 90])
    p10, p25, p50, p75, p90 = rl_percentiles

    x = float(late_threshold) if late_threshold is not None else None
    detections_within_x = np.sum(run_arr <= x) if x is not None else 0
    percent_within_x = 100 * detections_within_x / len(run_arr) if x is not None and len(run_arr) else 0

    handle_map = {
        "shewhart": [
            Line2D([0],[0], color="blue", lw=2, label="Simulated Data"),
            Line2D([0],[0], color="green", lw=2, label="Center Line (X̄)"),
            Line2D([0],[0], color="red", lw=2, linestyle="dashed", label="Upper/Lower CL"),
            Line2D([0],[0], color="orange", lw=2, linestyle="dashed", label="Upper/Lower Warning"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Out-of-Control"),
        ],
        "ewma": [
            Line2D([0],[0], color="blue", lw=2, label="Simulated Data"),
            Line2D([0],[0], color="green", lw=2, label="EWMA"),
            Line2D([0],[0], color="red", lw=2, linestyle="dashed", label="Upper/Lower EWMA CL"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Out-of-Control"),
        ],
        "mc-ewma": [
            Line2D([0],[0], color="blue", lw=2, label="Simulated Data"),
            Line2D([0],[0], color="green", lw=2, label="MC-EWMA"),
            Line2D([0],[0], color="red", lw=2, linestyle="dashed", label="Upper/Lower MC-EWMA CL"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Out-of-Control"),
        ],
        "cusum": [
            Line2D([0],[0], color="blue", lw=2, label="Simulated Data"),
            Line2D([0],[0], color="green", lw=2, label="CUSUM+"),
            Line2D([0],[0], color="orange", lw=2, label="CUSUM-"),
            Line2D([0],[0], color="red", lw=2, linestyle="dashed", label="+h / -h"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Out-of-Control"),
        ],
        "farrington": [
            Line2D([0],[0], color="blue", lw=2, label="Simulated Data"),
            Line2D([0],[0], color="red", lw=2, linestyle="dashed", label="Upper Farrington CL"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Out-of-Control"),
        ],
        "glm": [
            Line2D([0],[0], color="blue", lw=2, label="Simulated Data"),
            Line2D([0],[0], color="red", lw=2, linestyle="dashed", label="Upper/Lower GLM CL"),
            Line2D([0],[0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Out-of-Control"),
        ],
    }
    handles = handle_map.get(analysis_method, [])
    if x is not None:
        handles.append(Line2D([0],[0], color="black", lw=2, linestyle="dashed", label=f"RL Threshold ({x:.0f} days)"))
    legend_ax.legend(handles=handles, loc="center")

    rep_positions = [(0, 1), (1, 1)]
    for idx, pos in enumerate(rep_positions):
        if idx < len(replications):
            ax = fig.add_subplot(gs[pos[0], pos[1]])
            ax.set_facecolor('#ffffff')
            ax.grid(color='#f1f5f9', linestyle='-', linewidth=1, zorder=0)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            data, out_idx, _, _, baseline_mean, sigma = replications[idx]
            ax.plot(data, color="#3b82f6", zorder=1)
            if change_day is not None:
                ax.axvline(change_day, color="purple", linestyle="dotted", zorder=2)
            rl = (out_idx - baseline_period + 1) if out_idx is not None else "∞"
            ax.set_title(f"Replication {idx+1} (RL: {rl})")

            if analysis_method == "shewhart":
                result = shewhart(data, baseline_mean, sigma, sigma_multiplier, baseline_period)
                ax.axhline(baseline_mean, color="green", zorder=2)
                ax.axhline(baseline_mean + sigma_multiplier * sigma, color="red", linestyle="dashed", zorder=2)
                ax.axhline(baseline_mean - sigma_multiplier * sigma, color="red", linestyle="dashed", zorder=2)
                ax.axhline(baseline_mean + (sigma_multiplier-1) * sigma, color="orange", linestyle="dashed", zorder=2)
                ax.axhline(baseline_mean - (sigma_multiplier-1) * sigma, color="orange", linestyle="dashed", zorder=2)
                if out_idx is not None:
                    ax.scatter(out_idx, data[out_idx], color="red", s=100, zorder=3)
            elif analysis_method == "ewma":
                result = ewma(data, baseline_mean, sigma, sigma_multiplier, lambda_val, baseline_period)
                ax.plot(result["series"], color="green", zorder=2)
                ax.plot(result["ucl"], color="red", linestyle="dashed", zorder=2)
                ax.plot(result["lcl"], color="red", linestyle="dashed", zorder=2)
                if out_idx is not None:
                    ax.scatter(out_idx, result["series"][out_idx], color="red", s=100, zorder=3)
            elif analysis_method == "mc-ewma":
                result = mc_ewma(data, baseline_mean, sigma, sigma_multiplier, lambda_val, baseline_period)
                ax.plot(result["series"], color="green", zorder=2)
                ax.plot(result["ucl"], color="red", linestyle="dashed", zorder=2)
                ax.plot(result["lcl"], color="red", linestyle="dashed", zorder=2)
                if out_idx is not None:
                    ax.scatter(out_idx, data[out_idx], color="red", s=100, zorder=3)
            elif analysis_method == "cusum":
                result = cusum(data, baseline_mean, sigma, sigma_multiplier, baseline_period, k_val=k_val, h_val=h_val)
                c_plus, c_minus = result["series"]
                ax.plot(c_plus, color="green", lw=2)
                ax.plot(c_minus, color="orange", lw=2)
                ax.axhline(result["ucl"], color="red", linestyle="dashed")
                ax.axhline(result["lcl"], color="red", linestyle="dashed")
                if out_idx is not None:
                    marker_value = c_plus[out_idx] if c_plus[out_idx] > abs(c_minus[out_idx]) else c_minus[out_idx]
                    ax.scatter(out_idx, marker_value, color="red", s=100, zorder=3)
            elif analysis_method == "farrington":
                result = farrington(data, baseline_period, alpha=alpha_val)
                ax.plot(result["series"], color="green", zorder=2)
                ax.plot(result["ucl"], color="red", linestyle="dashed", zorder=2)
                if result["out_of_control_index"] is not None:
                    ax.scatter(result["out_of_control_index"], result["series"][result["out_of_control_index"]], color="red", s=100, zorder=3)
            elif analysis_method == "glm":
                result = glm(data, baseline_period, alpha=alpha_val)
                ax.plot(result["series"], color="green", zorder=2)
                ax.plot(result["ucl"], color="red", linestyle="dashed", zorder=2)
                ax.plot(result["lcl"], color="red", linestyle="dashed", zorder=2)
                if result["out_of_control_index"] is not None:
                    ax.scatter(result["out_of_control_index"], result["series"][result["out_of_control_index"]], color="red", s=100, zorder=3)

    hist_ax = fig.add_subplot(gs[:, 2])
    hist_ax.set_facecolor('#ffffff')
    hist_ax.grid(axis='y', color='#f1f5f9', linestyle='-', linewidth=1, zorder=0)
    hist_ax.spines['top'].set_visible(False)
    hist_ax.spines['right'].set_visible(False)

    unique_vals = np.unique(run_arr)
    bins = np.concatenate(([unique_vals[0]-0.5], unique_vals+0.5)) if len(unique_vals) <= 10 else int(np.ceil(np.sqrt(n_replications)))
    hist_ax.hist(run_arr, bins=bins, color='#3b82f6', edgecolor="white", alpha=0.85, zorder=2)
    hist_ax.set_title("Distribution of Run Lengths", fontsize=13, fontweight='bold', color='#1e293b')
    hist_ax.set_xlabel("Run Length (Days)", fontsize=11, fontweight='bold', color='#475569')
    hist_ax.set_ylabel("Frequency", fontsize=11, fontweight='bold', color='#475569')
    if x is not None:
        hist_ax.axvline(x, color="#ef4444", linestyle="--", linewidth=2.5, label=f"Threshold ({x:.0f} days)", zorder=3)
        hist_ax.legend(loc="upper right")

    plt.tight_layout()
    return {
        "rl_percentiles": {"P10": float(p10), "P25": float(p25), "P50": float(p50), "P75": float(p75), "P90": float(p90)},
        "detections_within_x": int(detections_within_x),
        "percent_within_x": float(percent_within_x),
        "x_days": float(x) if x is not None else None,
        "arl_moe": float(arl_moe),
        "ucl_ci": [float(ucl_ci[0]), float(ucl_ci[1])],
        "lcl_ci": [float(lcl_ci[0]), float(lcl_ci[1])],
        "limit_pct": float(limit_stopped_percentage)
    }


def run_simulation(behavior, params, n_baseline, change, change_day, analysis_method, n_replications,
                   sigma_multiplier, max_days, lambda_val, late_threshold, alpha_val,
                   custom_data=None, k_val=None, h_val=None):
    run_lengths = []
    replications = []
    sigmas = []
    change_days = []
    iterations = n_replications
    baseline_period = change_day if (change and change_day is not None) else n_baseline

    for _ in range(iterations):
        if custom_data is not None:
            data = list(custom_data)
            baseline_subset = data[:n_baseline]
            _, _, baseline_mean, sigma = calculate_limits_sim(baseline_subset, sigma_multiplier, analysis_method, lambda_val)
            if analysis_method == 'shewhart':
                res = shewhart(data, baseline_mean, sigma, sigma_multiplier, n_baseline)
            elif analysis_method == 'ewma':
                res = ewma(data, baseline_mean, sigma, sigma_multiplier, lambda_val, n_baseline)
            elif analysis_method == 'mc-ewma':
                res = mc_ewma(data, baseline_mean, sigma, sigma_multiplier, lambda_val, n_baseline)
            elif analysis_method == 'cusum':
                res = cusum(data, baseline_mean, sigma, sigma_multiplier, n_baseline, k_val=k_val, h_val=h_val)
            elif analysis_method == 'farrington':
                data_farr = np.clip(np.round(data), 0, None)
                res = farrington(data_farr, n_baseline, alpha=alpha_val)
            elif analysis_method == 'glm':
                res = glm(data, n_baseline, alpha=alpha_val)
            out_idx = res["out_of_control_index"]
        else:
            data = generate_behavior_data_sim(behavior, params, n_baseline)
            _, _, baseline_mean, sigma = calculate_limits_sim(data, sigma_multiplier, analysis_method, lambda_val)
            data, out_idx = apply_change_sim(data, change, change_day, params, behavior, baseline_mean, sigma,
                                             analysis_method, sigma_multiplier, baseline_period, lambda_val,
                                             max_days, alpha_val, k_val=k_val, h_val=h_val)

        sigmas.append(sigma)
        run_length = (out_idx - baseline_period + 1) if out_idx is not None else (len(data) - baseline_period + 1)
        run_lengths.append(run_length)
        replications.append((data, out_idx, None, None, baseline_mean, sigma))
        if change_day is not None:
            change_days.append(change_day)

    arl_value = np.mean(run_lengths) if run_lengths else float('inf')
    arl_moe = 0.0
    if len(run_lengths) > 1:
        arl_std = np.std(run_lengths, ddof=1)
        arl_moe = 1.96 * (arl_std / np.sqrt(iterations))

    ucl_list = [rep[4] + sigma_multiplier * rep[5] for rep in replications]
    lcl_list = [rep[4] - sigma_multiplier * rep[5] for rep in replications]
    ucl_ci = (np.percentile(ucl_list, 2.5), np.percentile(ucl_list, 97.5)) if ucl_list else (0, 0)
    lcl_ci = (np.percentile(lcl_list, 2.5), np.percentile(lcl_list, 97.5)) if lcl_list else (0, 0)

    avg_sigma = np.mean(sigmas) if sigmas else 0
    avg_change_day = np.mean(change_days) if change_days else None
    limit_pct = (sum(1 for r in replications if r[1] is None) / iterations) * 100
    metric_label = "FAR" if change is None else "ARL"

    formatted_lambda = format(lambda_val, '.3f').rstrip('0').rstrip('.')
    if analysis_method == "shewhart":
        chart_title = f"Shewhart Chart ({sigma_multiplier}σ)"
    elif analysis_method == "ewma":
        chart_title = f"EWMA Chart (λ={formatted_lambda}, {sigma_multiplier}σ)"
    elif analysis_method == "mc-ewma":
        chart_title = f"MC-EWMA Chart (λ={formatted_lambda}, {sigma_multiplier}σ)"
    elif analysis_method == "cusum":
        k_display = k_val if k_val is not None else 0.5 * avg_sigma
        h_display = h_val if h_val is not None else sigma_multiplier * avg_sigma
        chart_title = f"CUSUM Chart (k={k_display:.3f}, h={h_display:.3f})"
    else:
        chart_title = analysis_method.upper()

    buf = io.BytesIO()
    extra_stats = plot_replicates_and_histogram(
        replications, run_lengths, change_day, analysis_method, sigma_multiplier,
        baseline_period, iterations, arl_value, metric_label, avg_sigma, avg_change_day,
        limit_pct, lambda_val, ucl_ci, lcl_ci, late_threshold, arl_moe, alpha_val,
        k_val=k_val, h_val=h_val)
    plt.savefig(buf, format="png", dpi=80)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close()
    return img_base64, arl_value, chart_title, extra_stats


# ─────────────────────────────────────────────
# Background job runner — ONE definition only
# ─────────────────────────────────────────────

def run_job_in_background(job_id, sim_kwargs):
    try:
        save_job(job_id, {"status": "running"})
        kwargs = {k: v for k, v in sim_kwargs.items() if not k.startswith('_')}
        img, arl_value, chart_title, extra_stats = run_simulation(**kwargs)
        save_job(job_id, {
            "status": "done",
            "result": {
                "image": img,
                "title": chart_title,
                "arl": round(arl_value, 2),
                "extra_stats": extra_stats
            },
            "full_params": sim_kwargs.get("_full_params")
        })
    except Exception as e:
        save_job(job_id, {"status": "error", "message": str(e)})


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/poll/<job_id>")
def poll(job_id):
    job = load_job(job_id)
    if job is None:
        return jsonify({"status": "error", "message": "Job not found"})
    if job["status"] == "running":
        return jsonify({"status": "running"})
    if job["status"] == "error":
        delete_job(job_id)
        return jsonify({"status": "error", "message": job.get("message", "Unknown error")})
    if job["status"] == "done":
        result = job["result"]
        delete_job(job_id)  # delete first to prevent double-append
        results = load_results()
        results.append(result)
        save_results(results)
        return jsonify({"status": "done"})


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if request.form.get("clear"):
            save_results([])
            session.pop('full_params', None)
            return redirect(url_for('index'))

        fd = request.form.to_dict()
        data_source = fd.get("data_source")
        custom_data = None

        try:
            n_baseline = int(fd.get("n_baseline", "50"))
            n_replications = min(int(fd.get("n_replications", "25")), 100)
            sigma_multiplier = float(fd.get("sigma_multiplier", "3.0"))
            late_threshold = float(fd.get("late_threshold", 120))
        except ValueError:
            n_baseline, n_replications, sigma_multiplier, late_threshold = 50, 100, 3.0, 120

        if data_source == "upload":
            file = request.files.get('csv_file')
            if not file or file.filename == '':
                return render_template("home.html", error="No file uploaded", results=load_results())
            try:
                df = pd.read_csv(file)
                raw_values = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
                if len(raw_values) < 10:
                    raise ValueError("Dataset too small (<10 points).")
                behavior = "stable"
                params = {"mean": float(np.mean(raw_values)), "std": float(np.std(raw_values)), "distribution_type": "normal"}
            except Exception as e:
                return render_template("home.html", error=f"Invalid File: {str(e)}", results=load_results())
        else:
            behavior = fd.get("behavior")
            params = {}
            if behavior == "stable":
                params['mean'] = float(fd.get("mean", "100"))
                params['std'] = float(fd.get("std", "10"))
                params['distribution_type'] = fd.get("dist_type", "normal")
            elif behavior == "trending":
                params['start'] = float(fd.get("start", "100"))
                params['slope'] = float(fd.get("slope", "0.5"))
                params['noise'] = float(fd.get("noise", "5"))
            elif behavior == "periodic":
                params['mean'] = float(fd.get("p_mean", "100"))
                params['amplitude'] = float(fd.get("amplitude", "20"))
                params['period'] = float(fd.get("period", "50"))
                params['noise'] = float(fd.get("p_noise", "5"))

        induce = fd.get("induce_change", "no").lower() == "yes"
        if induce:
            change_day = int(fd.get("change_day", "50"))
            ct = fd.get("change_type", "step")
            change = {"type": "step", "factor": float(fd.get("factor", "1.5"))} if ct == "step" else \
                     {"type": "trending", "slope": float(fd.get("change_slope", "0.2")), "duration": int(fd.get("trend_duration", "50"))}
        else:
            change, change_day = None, None

        analysis_method = fd.get("analysis_method")
        max_days = 10000
        lambda_val = float(fd.get("lambda_val", "0.3")) if fd.get("lambda_val") else 0.3
        alpha_val = float(fd.get("alpha_val", "0.05")) if fd.get("alpha_val") else 0.05
        k_val = float(fd.get("k_val")) if fd.get("k_val") else None
        h_val = float(fd.get("h_val")) if fd.get("h_val") else None

        full_params = {
            "behavior": behavior, "params": params, "n_baseline": n_baseline,
            "n_replications": n_replications, "sigma_multiplier": sigma_multiplier,
            "change": change, "change_day": change_day, "max_days": max_days,
            "late_threshold": late_threshold, "alpha_val": alpha_val,
            "k_val": k_val, "h_val": h_val
        }
        session['full_params'] = full_params

        job_id = str(uuid.uuid4())
        sim_kwargs = {
            "behavior": behavior, "params": params, "n_baseline": n_baseline,
            "change": change, "change_day": change_day, "analysis_method": analysis_method,
            "n_replications": n_replications, "sigma_multiplier": sigma_multiplier,
            "max_days": max_days, "lambda_val": lambda_val,
            "late_threshold": late_threshold, "alpha_val": alpha_val,
            "custom_data": custom_data, "k_val": k_val, "h_val": h_val,
            "_full_params": full_params
        }
        t = threading.Thread(target=run_job_in_background, args=(job_id, sim_kwargs), daemon=True)
        t.start()

        return render_template("home.html", results=load_results(),
                               full_params_exists=('full_params' in session),
                               pending_job_id=job_id)

    return render_template("home.html", results=load_results(),
                           full_params_exists=('full_params' in session))


@app.route("/instructions")
def instructions():
    return render_template("instructions.html")


@app.route("/clear", methods=["POST"])
def clear():
    save_results([])
    session.pop('full_params', None)
    return redirect(url_for('index'))


@app.route("/reanalyze", methods=["GET", "POST"])
def reanalyze():
    if 'full_params' not in session or not load_results():
        return render_template("home.html",
                               error="No simulation data found. Please perform an initial run first.",
                               results=load_results())
    if request.method == "POST":
        fp = session['full_params']
        analysis_method = request.form.get("analysis_method")
        lambda_val = float(request.form.get("lambda_val", 0.3)) if analysis_method in ["ewma", "mc-ewma"] else 0.3
        sigma_multiplier = float(request.form.get("sigma_multiplier_re") or fp["sigma_multiplier"])
        k_val = float(request.form.get("k_val")) if request.form.get("k_val") else fp.get("k_val")
        h_val = float(request.form.get("h_val")) if request.form.get("h_val") else fp.get("h_val")
        alpha_val = float(request.form.get("alpha_val", 0.05)) if request.form.get("alpha_val") else fp.get("alpha_val", 0.05)

        job_id = str(uuid.uuid4())
        sim_kwargs = {
            "behavior": fp["behavior"], "params": fp["params"], "n_baseline": fp["n_baseline"],
            "change": fp["change"], "change_day": fp["change_day"],
            "analysis_method": analysis_method,
            "n_replications": fp["n_replications"], "sigma_multiplier": sigma_multiplier,
            "max_days": fp["max_days"], "lambda_val": lambda_val,
            "late_threshold": fp.get("late_threshold", 120),
            "alpha_val": alpha_val, "custom_data": None,
            "k_val": k_val, "h_val": h_val,
            "_full_params": fp
        }
        t = threading.Thread(target=run_job_in_background, args=(job_id, sim_kwargs), daemon=True)
        t.start()

        return render_template("home.html", results=load_results(),
                               full_params_exists=True,
                               pending_job_id=job_id)

    return render_template("analysis.html")


@app.route("/finalize")
def finalize():
    results = load_results()
    valid_results = [r for r in results if r.get('arl') is not None]
    ranked_data = sorted(valid_results, key=lambda x: x['arl'])
    return render_template("home.html", results=results, rankings=ranked_data)


if __name__ == "__main__":
    app.run(debug=True)
