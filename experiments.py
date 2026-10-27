"""Calibration, 30-seed runs and hypothesis checks (proposal 5.5-5.9).

    python experiments.py calibrate   # calibration seeds 1000-1009 only
    python experiments.py run         # test seeds 0-29: OD pairs x scenarios x noise levels x scenario sets
    python experiments.py analyze     # Table 6 decision rules -> results/hypotheses.json
    python experiments.py sensitivity # Adaptation Time under smaller epsilon, calibration seeds only

Two scenario sets are evaluated:
  doc         Table 4 exactly (length weight 0.3)
  calibrated  the length weight chosen by the rule in config.yaml (environment only)
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

import sim

CFG = sim.load_config()
RES = sim.ROOT / CFG["evaluation"]["results_dir"]
CURVE_STEP = 10   # keep every 10th round of the cumulative-regret curves


def load_routes():
    G, D, places, meta = sim.load_campus()
    base = sim.edge_base_density(D, places, CFG["context"]["hotspot_decay_m"])
    return sim.load_pair_routes(D, base)


def env_args(noise_sd):
    c = CFG["context"]
    return dict(T=CFG["experiment"]["T"], noise_sd=noise_sd, edge_noise_sd=c["edge_noise_sd"],
                rounds_per_day=c["rounds_per_day"], peaks=tuple(c["daily_peaks_hour"]))


def build_env(routes, sc, seed, noise_sd):
    a = env_args(noise_sd)
    return sim.build_environment(routes, sc, a["T"], seed, noise_sd, a["edge_noise_sd"], a["rounds_per_day"], a["peaks"])


# --------------------------------------------------------------------------- calibration
def theta_rule(routes):
    """Pick the scenario length weight from the environment only (no policy is run)."""
    cal = CFG["calibration"]
    noise = CFG["reward"]["noise_sd"]
    table = []
    for a in cal["theta_length_weight_candidates"]:
        scs = sim.scenario_set(CFG, a)
        ok_pairs = 0
        detail = {}
        for i, R in routes.items():
            s1_share, s2_ok = [], []
            for seed in cal["seeds"]:
                e1 = build_env(R, scs["S1"], seed, noise)
                shortest = int(np.argmin(e1.X[0, :, 0]))
                s1_share.append(float(np.mean(e1.mu.argmax(1) != shortest)))
                e2 = build_env(R, scs["S2"], seed, noise)
                o = e2.mu.argmax(1)
                tc = scs["S2"]["change_at"]
                s2_ok.append(abs(float(np.mean(o[:tc] != shortest)) - float(np.mean(o[tc:] != shortest))))
            share, change = float(np.mean(s1_share)), float(np.mean(s2_ok))
            changed = change >= cal["theta_rule_min_share"]
            ok = share >= cal["theta_rule_min_share"] and changed
            ok_pairs += ok
            detail[int(i)] = {"s1_share_best_not_shortest": round(share, 3), "s2_share_change": round(change, 3), "ok": bool(ok)}
        table.append({"length_weight": a, "pairs_ok": int(ok_pairs), "detail": detail})
        print("weight", a, "pairs ok", ok_pairs, flush=True)
    passing = [r["length_weight"] for r in table if r["pairs_ok"] >= cal["theta_rule_min_pairs"]]
    return (max(passing) if passing else None), table


def _calib_job(args):
    routes, sc, seed, name, params = args
    p = CFG["policies"]
    if name == "LinUCB":
        pol = sim.LinUCB(params["alpha"], p["lambda"], np.random.default_rng(seed + 1))
    else:
        pol = sim.DLinUCB(params["alpha"], p["lambda"], p["dlinucb_tau"], params["delta1"], params["delta1_tilde"],
                          params["delta2"], p["dlinucb_noise_sd_assumed"], np.random.default_rng(seed + 2))
    out = sim.run(routes, sc, seed=seed, policies=[pol], **env_args(CFG["reward"]["noise_sd"]))
    return float(out["policies"][pol.name]["cum_regret"][-1])


def tune_policies(routes, scs, ex):
    cal = CFG["calibration"]
    chosen, report = {}, {}
    for name, grid in (("LinUCB", cal["linucb_grid"]), ("dLinUCB", cal["dlinucb_grid"])):
        assert len(grid) == cal["budget_per_policy"]
        scores = []
        for params in grid:
            jobs = [(routes[i], scs[s], seed, name, params) for i in routes for s in cal["scenarios"] for seed in cal["seeds"]]
            m = float(np.mean(list(ex.map(_calib_job, jobs))))
            scores.append({"params": params, "mean_cum_regret": round(m, 4)})
            print("  ", name, params, round(m, 3), flush=True)
        chosen[name] = min(scores, key=lambda s: s["mean_cum_regret"])["params"]
        report[name] = scores
    return chosen, report


def calibrate():
    pairs, routes = load_routes()
    a_cal, table = theta_rule(routes)
    if a_cal is None:
        raise SystemExit("no candidate weight satisfies the rule; report this instead of choosing by hand")
    sets = {"doc": CFG["experiment"]["theta_length_weight_doc"], "calibrated": a_cal}
    out = {"theta_rule": table, "sets": {}, "note": "selected on calibration seeds 1000-1009 only; test seeds 0-29 unused"}
    with ProcessPoolExecutor() as ex:
        for set_name, a in sets.items():
            print("set", set_name, "length weight", a, flush=True)
            chosen, report = tune_policies(routes, sim.scenario_set(CFG, a), ex)
            out["sets"][set_name] = {"length_weight": a, "chosen": chosen, "grid_results": report}
    (sim.DATA / "calibrated.yaml").write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(yaml.safe_dump({k: {"length_weight": v["length_weight"], "chosen": v["chosen"]} for k, v in out["sets"].items()}))


def load_calibration():
    return yaml.safe_load((sim.DATA / "calibrated.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 30-seed runs
def _run_job(args):
    set_name, a, params, pair_id, routes, sc_key, noise_sd, seed = args
    sc = sim.scenario_set(CFG, a)[sc_key]
    out = sim.run(routes, sc, seed=seed, policies=sim.make_policies(CFG, params, seed), **env_args(noise_sd))
    metrics = sim.run_metrics(out, sc, CFG, routes)
    rows = [{"set": set_name, "pair": pair_id, "scenario": sc_key, "noise_sd": noise_sd, "seed": seed, "policy": n, **m}
            for n, m in metrics.items()]
    curves = {n: out["policies"][n]["cum_regret"][::CURVE_STEP].astype(np.float32) for n in metrics}
    return rows, (set_name, pair_id, sc_key, noise_sd, curves)


def run_all():
    pairs, routes = load_routes()
    cal = load_calibration()
    n = CFG["evaluation"]["test_seeds"]
    jobs = [(sn, s["length_weight"], s["chosen"], i, routes[i], sc, nz, seed)
            for sn, s in cal["sets"].items() for i in routes for sc in CFG["experiment"]["scenarios"]
            for nz in CFG["reward"]["extra_noise_levels"] for seed in range(n)]
    RES.mkdir(exist_ok=True)
    rows, curves = [], {}
    with ProcessPoolExecutor() as ex:
        for k, (r, (sn, pid, sc, nz, cv)) in enumerate(ex.map(_run_job, jobs, chunksize=4)):
            rows.extend(r)
            for name, arr in cv.items():
                curves.setdefault(f"{sn}|{pid}|{sc}|{nz}|{name}", []).append(arr)
            if k % 300 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    pd.DataFrame(rows).to_csv(RES / "summary.csv", index=False, encoding="utf-8-sig")
    packed = {}
    for key, arrs in curves.items():
        a = np.stack(arrs)
        packed[key + "|mean"] = a.mean(axis=0)
        packed[key + "|lo"] = np.percentile(a, 2.5, axis=0)
        packed[key + "|hi"] = np.percentile(a, 97.5, axis=0)
    np.savez_compressed(RES / "curves.npz", **packed)
    print("saved", len(rows), "rows")


# --------------------------------------------------------------------------- Table 6
def analyze():
    df = pd.read_csv(RES / "summary.csv")
    ev = CFG["evaluation"]
    nboot = ev["bootstrap_resamples"]
    main_noise = CFG["reward"]["noise_sd"]
    curves = np.load(RES / "curves.npz")
    med = lambda x: float(np.median(np.where(np.isnan(x), np.inf, x)))  # "ไม่ฟื้นตัว" counts as infinite
    res = {}
    for set_name in df["set"].unique():
        res[set_name] = {}
        for pid in sorted(df["pair"].unique()):
            d = df[(df["set"] == set_name) & (df["pair"] == pid) & (df["noise_sd"] == main_noise)]

            def col(sc, pol, c):
                return d[(d["scenario"] == sc) & (d["policy"] == pol)].sort_values("seed")[c].to_numpy(dtype=float)

            R = "dynamic_cumulative_regret"
            rl, rd = col("S1", "LinUCB", R), col("S1", "dLinUCB", R)
            m, lo, hi = sim.paired_bootstrap_ci(rl - rd, nboot)
            thr = ev["h1_margin"] * float(rl.mean())
            h1 = {"mean_d": m, "ci": [lo, hi], "threshold": thr, "supported": bool(hi <= thr),
                  "false_alarm_rate_mean": float(np.nanmean(col("S1", "dLinUCB", "false_alarm_rate")))}
            pl, pdl = col("S2", "LinUCB", "regret_after_change"), col("S2", "dLinUCB", "regret_after_change")
            m2, lo2, hi2 = sim.paired_bootstrap_ci(pl - pdl, nboot)
            tl, td = col("S2", "LinUCB", "adaptation_time"), col("S2", "dLinUCB", "adaptation_time")
            dd = col("S2", "dLinUCB", "detection_delay")
            h2 = {"mean_diff_after_change": m2, "ci": [lo2, hi2],
                  "median_tau_linucb": med(tl), "median_tau_dlinucb": med(td),
                  "not_recovered_linucb": int(np.isnan(tl).sum()), "not_recovered_dlinucb": int(np.isnan(td).sum()),
                  "detected_runs": int((~np.isnan(dd)).sum()),
                  "median_detection_delay": float(np.nanmedian(dd)) if (~np.isnan(dd)).any() else None,
                  "supported": bool(lo2 > 0 and med(td) < med(tl))}
            h3 = {}
            for sc in ["S1", "S2", "S3", "CTRL"]:
                rf, rl3 = col(sc, "Fixed Context-Weighted Score", R), col(sc, "LinUCB", R)
                m3, lo3, hi3 = sim.paired_bootstrap_ci(rf - rl3, nboot)
                mf = curves[f"{set_name}|{pid}|{sc}|{main_noise}|Fixed Context-Weighted Score|mean"]
                ml = curves[f"{set_name}|{pid}|{sc}|{main_noise}|LinUCB|mean"]
                below = ml < mf
                cross = next((int(i * CURVE_STEP) for i in range(len(below)) if below[i:].all()), None)
                h3[sc] = {"mean_diff": m3, "ci": [lo3, hi3], "learning_better": bool(lo3 > 0),
                          "fixed_better": bool(hi3 < 0), "linucb_wins_from_round": cross}
            res[set_name][int(pid)] = {"H1": h1, "H2": h2,
                                       "H3": {"by_scenario": h3,
                                              "supported_first_part": bool(all(h3[s]["learning_better"] for s in ["S1", "S2", "S3"]))}}
    (RES / "hypotheses.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for sn, per in res.items():
        for pid, h in per.items():
            print(sn, pid, "H1", h["H1"]["supported"], "H2", h["H2"]["supported"], "H3", h["H3"]["supported_first_part"])


# --------------------------------------------------------------------------- sensitivity of epsilon
def _sens_job(args):
    set_name, a, params, pair_id, routes, sc_key, seed = args
    sc = sim.scenario_set(CFG, a)[sc_key]
    out = sim.run(routes, sc, seed=seed, policies=sim.make_policies(CFG, params, seed), **env_args(CFG["reward"]["noise_sd"]))
    tc = sim.change_point(sc)
    ev = CFG["evaluation"]
    rows = []
    for name, p in out["policies"].items():
        for eps in ev["sensitivity_eps"]:
            rows.append({"set": set_name, "pair": pair_id, "scenario": sc_key, "seed": seed, "policy": name, "eps": eps,
                         "adaptation_time": sim.adaptation_time(p["regret"], tc, eps, ev["adaptation_window"])})
    return rows


def sensitivity():
    """Adaptation Time under smaller thresholds, on calibration seeds only (not the main result)."""
    pairs, routes = load_routes()
    cal = load_calibration()
    jobs = [(sn, s["length_weight"], s["chosen"], i, routes[i], sc, seed)
            for sn, s in cal["sets"].items() for i in routes for sc in ["S2", "S3"] for seed in CFG["calibration"]["seeds"]]
    rows = []
    with ProcessPoolExecutor() as ex:
        for r in ex.map(_sens_job, jobs, chunksize=2):
            rows.extend(r)
    RES.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "eps_sensitivity.csv", index=False, encoding="utf-8-sig")
    summ = df.groupby(["set", "scenario", "policy", "eps"])["adaptation_time"].agg(
        mean="mean", not_recovered=lambda x: float(x.isna().mean())).round(2)
    print(summ.to_string())


if __name__ == "__main__":
    {"calibrate": calibrate, "run": run_all, "analyze": analyze, "sensitivity": sensitivity}[sys.argv[1]]()
