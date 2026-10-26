"""Simulation core for the Dynamic GIS Recommender demo.

Each part follows a section of the proposal (เค้าโครงโครงงานฉบับสมบูรณ์):
  5.2  walking graph and candidate routes (K-shortest loopless paths, Yen 1971)
  5.3  context: rho_e(t) = rho_e_base * g(t) + eps_e ; features (min-max length, crowd exposure)
  5.4  reward mu_t(a) = x_{t,a} . theta_t, feedback r = mu + noise, oracle for evaluation only
  5.5  policies: Shortest Path, Fixed Context-Weighted Score, LinUCB, dLinUCB
  5.6  scenarios S1-S3 and the control case (Table 4)
  5.8  metrics (Table 5)
Crowd densities and user preferences are simulated values, not measurements.
"""
from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

import networkx as nx
import numpy as np
import yaml

ROOT = Path(__file__).parent
DATA = ROOT / "data"

# how strongly each kind of place attracts people (author choice, simulated assumption)
HOTSPOT_WEIGHT = {
    "amenity=food_court": 1.0, "amenity=restaurant": 0.9, "amenity=fast_food": 0.9,
    "amenity=marketplace": 1.0, "amenity=canteen": 1.0, "amenity=library": 0.8,
    "amenity=cafe": 0.5, "amenity=hospital": 0.7, "building=university": 0.4,
    "building=dormitory": 0.4, "leisure=stadium": 0.3, "leisure=sports_centre": 0.3,
}


def load_config(path=ROOT / "config.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 5.2 data and routes
def great_circle(lat1, lon1, lat2, lon2):
    """Metres between a point and arrays of points (haversine)."""
    r = 6371009.0
    p1, p2 = np.radians(lat1), np.radians(np.asarray(lat2))
    dphi = p2 - p1
    dl = np.radians(np.asarray(lon2) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def load_campus():
    """Return (multigraph G, simple digraph D, places, meta) from the cached OSM data."""
    import osmnx as ox
    G = ox.load_graphml(DATA / "kku_walk.graphml")
    D = ox.convert.to_digraph(G, weight="length")
    places = json.loads((DATA / "places.json").read_text(encoding="utf-8"))
    places = [p for p in places if p["node"] in D]
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    return G, D, places, meta


def edge_base_density(D, places, decay_m):
    """rho_e_base in [0, 1]: decays with distance from places where people gather (5.3)."""
    hl = np.array([[p["lat"], p["lon"]] for p in places])
    hw = np.array([HOTSPOT_WEIGHT.get(p["kind"], 0.2) for p in places])
    base = {}
    for u, v in D.edges():
        lat = (D.nodes[u]["y"] + D.nodes[v]["y"]) / 2
        lon = (D.nodes[u]["x"] + D.nodes[v]["x"]) / 2
        d = great_circle(lat, lon, hl[:, 0], hl[:, 1])
        base[(u, v)] = 0.05 + float(np.sum(hw * np.exp(-d / decay_m)))
    m = max(base.values())
    return {e: min(1.0, val / m) for e, val in base.items()}


@dataclass
class Route:
    nodes: list
    length_m: float
    edge_len: list          # length of every edge on the route (m)
    edge_base: list         # rho_e_base of every edge on the route
    edges: list = field(default_factory=list)

    @property
    def base_crowd(self):
        return float(np.dot(self.edge_len, self.edge_base) / self.length_m)


def candidate_routes(D, base, origin, dest, k=5, max_overlap=0.8, max_paths=300):
    """5.2.2: K-shortest loopless paths in increasing length (Yen's method as implemented by
    networkx.shortest_simple_paths). A path is dropped when its length shared with an
    already kept route exceeds `max_overlap` of the shorter of the two."""
    kept: list[Route] = []
    for path in itertools.islice(nx.shortest_simple_paths(D, origin, dest, weight="length"), max_paths):
        edges = list(zip(path[:-1], path[1:]))
        lens = [D.edges[e]["length"] for e in edges]
        total = float(sum(lens))
        es = set(edges)
        too_close = False
        for r in kept:
            shared = sum(l for e, l in zip(edges, lens) if e in set(r.edges))
            if shared / min(total, r.length_m) > max_overlap:
                too_close = True
                break
        if not too_close:
            kept.append(Route(path, total, lens, [base[e] for e in edges], edges))
        if len(kept) == k:
            break
    return kept


def routes_from_nodes(D, base, node_lists):
    """Rebuild Route objects from saved node lists (data/od_pairs.json)."""
    out = []
    for path in node_lists:
        edges = list(zip(path[:-1], path[1:]))
        lens = [D.edges[e]["length"] for e in edges]
        out.append(Route(path, float(sum(lens)), lens, [base[e] for e in edges], edges))
    return out


def load_pair_routes(D, base):
    """OD pairs and their candidate routes exactly as recorded before the experiments (5.2.2)."""
    pairs = json.loads((DATA / "od_pairs.json").read_text(encoding="utf-8"))
    return pairs, {i: routes_from_nodes(D, base, p["routes"]) for i, p in enumerate(pairs)}


def route_latlon(G, nodes):
    pts = []
    for u, v in zip(nodes[:-1], nodes[1:]):
        data = min(G.get_edge_data(u, v).values(), key=lambda d: d.get("length", 0))
        if "geometry" in data:
            seg = [(y, x) for x, y in data["geometry"].coords]
        else:
            seg = [(G.nodes[u]["y"], G.nodes[u]["x"]), (G.nodes[v]["y"], G.nodes[v]["x"])]
        pts.extend(seg if not pts else seg[1:])
    return pts


# --------------------------------------------------------------------------- 5.3-5.4 environment
def daily_pattern(t, rounds_per_day=96, peaks=(8, 12, 17)):
    """g(t): crowd multiplier over a simulated day with morning, noon and evening peaks."""
    h = (t % rounds_per_day) * 24.0 / rounds_per_day
    return 0.1 + 0.9 * max(math.exp(-((h - p) ** 2) / (2 * 0.75 ** 2)) for p in peaks)


def hour_label(t, rounds_per_day=96):
    m = int((t % rounds_per_day) * 24 * 60 / rounds_per_day)
    return f"{m // 60:02d}:{m % 60:02d}"


def theta_schedule(sc: dict, T: int):
    """Latent preference theta_t (Table 4)."""
    a = np.array(sc["theta_start"], dtype=float)
    th = np.tile(a, (T, 1))
    if sc["mode"] == "abrupt":
        th[sc["change_at"]:] = np.array(sc["theta_end"], dtype=float)
    elif sc["mode"] == "gradual":
        b = np.array(sc["theta_end"], dtype=float)
        s, e = sc["change_at"], sc["change_end"]
        w = np.linspace(0, 1, e - s)[:, None]
        th[s:e] = (1 - w) * a + w * b
        th[e:] = b
    return th


def scenario_set(cfg, length_weight):
    """Table 4 scenarios with the length weight a: S1-S3 start at theta = (-a, -(1-a)) and
    S2/S3 end at (-(1-a), -a). a = 0.3 reproduces Table 4 exactly. The control case keeps theta = w."""
    out = {}
    for key, sc in cfg["experiment"]["scenarios"].items():
        sc = dict(sc)
        if key != "CTRL":
            a = float(length_weight)
            sc["theta_start"] = [-a, -(1 - a)]
            if "theta_end" in sc:
                sc["theta_end"] = [-(1 - a), -a]
        out[key] = sc
    return out


def change_point(sc: dict):
    """t_c for Adaptation Time (Table 5): S2 change round, S3 end of the change."""
    if sc["mode"] == "abrupt":
        return sc["change_at"]
    if sc["mode"] == "gradual":
        return sc["change_end"]
    return None


def no_change_rounds(sc: dict, T: int):
    """Rounds with no real change, used for the False Alarm Rate."""
    if sc["mode"] == "constant":
        return np.arange(T)
    return np.arange(sc["change_at"])


@dataclass
class Env:
    X: np.ndarray        # (T, K, 2) features seen by the policies
    theta: np.ndarray    # (T, 2) latent preference, simulator only
    mu: np.ndarray       # (T, K) expected reward, simulator only
    noise: np.ndarray    # (T, K) reward noise
    g: np.ndarray        # (T,) daily crowd multiplier


def build_environment(routes, sc, T, seed, noise_sd, edge_noise_sd, rounds_per_day=96, peaks=(8, 12, 17)):
    rng = np.random.default_rng(seed)
    K = len(routes)
    # 5.3: rho_e(t) = rho_e_base * g(t) + eps_e on every edge used by any route
    g = np.array([daily_pattern(t, rounds_per_day, peaks) for t in range(T)])
    c = np.zeros((T, K))
    for k, r in enumerate(routes):
        base = np.asarray(r.edge_base)
        rho = np.clip(base[None, :] * g[:, None] + rng.normal(0, edge_noise_sd, (T, len(base))), 0, 1)
        c[:, k] = rho @ np.asarray(r.edge_len) / r.length_m       # length-weighted mean
    L = np.array([r.length_m for r in routes])
    ln = (L - L.min()) / (L.max() - L.min()) if L.max() > L.min() else np.zeros(K)  # min-max within the set
    X = np.stack([np.broadcast_to(ln, (T, K)), np.clip(c, 0, 1)], axis=2)
    theta = theta_schedule(sc, T)
    mu = np.einsum("tkd,td->tk", X, theta)
    noise = rng.normal(0, noise_sd, (T, K))
    return Env(X, theta, mu, noise, g)


# --------------------------------------------------------------------------- 5.5 policies
class Policy:
    """A policy sees only the features of the candidate routes and the reward of its own pick."""
    name = "base"
    learns = False
    last = None

    def select(self, t, X_t):
        raise NotImplementedError

    def update(self, t, x_chosen, reward):
        pass


class ShortestPath(Policy):
    name = "Shortest Path"

    def select(self, t, X_t):
        return int(np.argmin(X_t[:, 0]))


class FixedWeighted(Policy):
    name = "Fixed Context-Weighted Score"

    def __init__(self, w):
        self.w = np.asarray(w, dtype=float)

    def select(self, t, X_t):
        return int(np.argmax(X_t @ self.w))


class _LinModel:
    def __init__(self, d, lam, born=0):
        self.A = lam * np.eye(d)
        self.b = np.zeros(d)
        self.born = born
        self.errors: list[int] = []     # e_i(m) since creation (dLinUCB only)

    def theta(self):
        return np.linalg.solve(self.A, self.b)

    def width(self, X):
        Ainv = np.linalg.inv(self.A)
        return np.sqrt(np.einsum("kd,de,ke->k", X, Ainv, X))

    def update(self, x, r):
        self.A += np.outer(x, x)
        self.b += r * x


class LinUCB(Policy):
    name = "LinUCB"
    learns = True

    def __init__(self, alpha, lam, rng):
        self.alpha, self.rng = alpha, rng
        self.m = _LinModel(2, lam)

    def select(self, t, X_t):
        est = X_t @ self.m.theta()
        bonus = self.alpha * self.m.width(X_t)
        ucb = est + bonus
        self.last = (est, bonus)
        return int(self.rng.choice(np.flatnonzero(np.isclose(ucb, ucb.max()))))

    def update(self, t, x, r):
        self.m.update(x, r)


class DLinUCB(Policy):
    """Dynamic Linear UCB following Algorithm 1 of Wu, Iyer, & Wang (2018).

    A master keeps a set of LinUCB slave models. After each round, slave m records
    e_t(m) = 1{|x.theta_m - r| > B_t(m, a) + eps}, with eps = sigma * Phi^-1(1 - delta1/2)
    (Eq. 4) and B_t from the paper's confidence bound (see _alpha). Only slaves with e_t(m) = 0 are updated. Badness e_hat(m) is the mean of e over
    the last tau_tilde = min(t - t_m, tau) rounds and d(m) = sqrt(ln(1/delta2) / (2 tau_tilde)).
    A slave is discarded when e_hat >= delta1 + d; a new slave is created when no slave has
    e_hat < delta1_tilde + d. The acting slave minimises e_hat - sqrt(ln tau) * d (LCB of badness).
    Arm selection uses the exploration weight alpha tuned like LinUCB (proposal Table 3 lists alpha
    for dLinUCB), while the error test keeps the paper's bound. Parameter values come from
    calibration (proposal 5.5)."""
    name = "dLinUCB"
    learns = True

    def __init__(self, alpha, lam, tau, delta1, delta1_tilde, delta2, noise_sd, rng, theta_norm_bound=1.0):
        self.alpha, self.lam, self.tau = alpha, lam, tau
        self.delta1, self.delta1_tilde, self.delta2 = delta1, delta1_tilde, delta2
        self.sigma, self.S = noise_sd, theta_norm_bound
        self.eps = noise_sd * NormalDist().inv_cdf(1 - delta1 / 2)
        self.rng = rng
        self.models = [_LinModel(2, lam, born=0)]
        self.detections: list[int] = []
        self.active_count: list[int] = []
        self.chosen = None

    def _alpha(self, m):
        """Confidence width multiplier of a LinUCB slave, as in the paper's bound B_t(m, a)
        (Abbasi-Yadkori style): sigma * sqrt(d ln(1 + n/(d lambda)) + 2 ln(1/delta1)) + sqrt(lambda) S."""
        d, n = 2, len(m.errors)
        return (self.sigma * math.sqrt(d * math.log(1 + n / (d * self.lam)) + 2 * math.log(1 / self.delta1))
                + math.sqrt(self.lam) * self.S)

    def _stats(self, m):
        n = min(len(m.errors), self.tau)
        if n == 0:
            return 0.0, 0.0
        e_hat = float(np.mean(m.errors[-n:]))
        d = math.sqrt(math.log(1 / self.delta2) / (2 * n))
        return e_hat, d

    def select(self, t, X_t):
        def lcb(m):
            e_hat, d = self._stats(m)
            return e_hat - math.sqrt(math.log(self.tau)) * d
        self.chosen = min(self.models, key=lcb)
        est = X_t @ self.chosen.theta()
        bonus = self.alpha * self.chosen.width(X_t)   # exploration, tuned like LinUCB (Table 3: alpha)
        ucb = est + bonus
        self.last = (est, bonus)
        return int(self.rng.choice(np.flatnonzero(np.isclose(ucb, ucb.max()))))

    def update(self, t, x, r):
        create_new = True
        keep = []
        for m in self.models:
            bound = self._alpha(m) * m.width(x[None, :])[0] + self.eps
            e = int(abs(float(x @ m.theta()) - r) > bound)
            if e == 0:
                m.update(x, r)
            m.errors.append(e)
            e_hat, d = self._stats(m)
            if e_hat < self.delta1_tilde + d:
                create_new = False
            if e_hat >= self.delta1 + d:
                continue                       # discard this slave
            keep.append(m)
        self.models = keep
        if create_new or not self.models:
            self.models.append(_LinModel(2, self.lam, born=t + 1))
            self.detections.append(t + 1)
        self.active_count.append(len(self.models))


def make_policies(cfg, params, seed):
    p = cfg["policies"]
    lp, dp = params["LinUCB"], params["dLinUCB"]
    return [ShortestPath(), FixedWeighted(p["fixed_w"]),
            LinUCB(lp["alpha"], p["lambda"], np.random.default_rng(seed + 1)),
            DLinUCB(dp["alpha"], p["lambda"], p["dlinucb_tau"], dp["delta1"], dp["delta1_tilde"],
                    dp["delta2"], p["dlinucb_noise_sd_assumed"], np.random.default_rng(seed + 2))]


# --------------------------------------------------------------------------- run
def run(routes, sc, T, seed, noise_sd, edge_noise_sd, policies, rounds_per_day=96, peaks=(8, 12, 17)):
    """Run every policy on the same environment: same contexts and same noise (5.7)."""
    env = build_environment(routes, sc, T, seed, noise_sd, edge_noise_sd, rounds_per_day, peaks)
    Tn, K = env.mu.shape
    oracle = env.mu.argmax(axis=1)
    out = {"env": env, "oracle": oracle, "policies": {}}
    for pol in policies:
        choice = np.zeros(Tn, dtype=int)
        reward = np.zeros(Tn)
        regret = np.zeros(Tn)
        est = np.full((Tn, K), np.nan)
        bonus = np.full((Tn, K), np.nan)
        elapsed = 0.0
        for t in range(Tn):
            t0 = time.perf_counter()
            a = pol.select(t, env.X[t])
            r = env.mu[t, a] + env.noise[t, a]          # only the chosen route's reward is revealed
            pol.update(t, env.X[t, a], r)
            elapsed += time.perf_counter() - t0
            choice[t], reward[t] = a, r
            regret[t] = env.mu[t, oracle[t]] - env.mu[t, a]
            if pol.learns and pol.last is not None:
                est[t], bonus[t] = pol.last
        out["policies"][pol.name] = {
            "choice": choice, "reward": reward, "regret": regret, "cum_regret": np.cumsum(regret),
            "est": est, "bonus": bonus, "ms_per_round": 1000 * elapsed / Tn,
            "detections": [d for d in getattr(pol, "detections", []) if d < Tn],
            "active_models": np.array(getattr(pol, "active_count", [])),
        }
    return out


# --------------------------------------------------------------------------- 5.8 metrics
def adaptation_time(regret, change_at, eps=0.05, window=50):
    """Rounds after the change point until the moving average (window W) of per-round regret
    is <= eps for W consecutive positions; None means "ไม่ฟื้นตัวภายในช่วงทดลอง"."""
    r = np.asarray(regret[change_at:], dtype=float)
    if len(r) < 2 * window:
        return None
    ma = np.convolve(r, np.ones(window) / window, mode="valid")
    run_len = 0
    for i, ok in enumerate(ma <= eps):
        run_len = run_len + 1 if ok else 0
        if run_len >= window:
            return (i - window + 1) + window
    return None


def detection_delay(detections, change_at):
    after = [d for d in detections if d >= change_at]
    return after[0] - change_at if after else None


def false_alarm_rate(detections, quiet_rounds):
    """Detections in rounds without a real change, per 1,000 rounds."""
    if len(quiet_rounds) == 0:
        return None
    q = set(int(x) for x in quiet_rounds)
    return 1000 * sum(1 for d in detections if d in q) / len(q)


def run_metrics(out, sc, cfg, routes):
    ev = cfg["evaluation"]
    T = len(out["oracle"])
    tc = change_point(sc)
    rows = {}
    for name, p in out["policies"].items():
        ch = p["choice"]
        row = {
            "dynamic_cumulative_regret": float(p["cum_regret"][-1]),
            "average_reward": float(p["reward"].mean()),
            "crowd_exposure": float(out["env"].X[np.arange(T), ch, 1].mean()),
            "route_length_m": float(np.mean([routes[i].length_m for i in ch])),
            "ms_per_round": float(p["ms_per_round"]),
        }
        if tc is not None:
            row["adaptation_time"] = adaptation_time(p["regret"], tc, ev["adaptation_eps"], ev["adaptation_window"])
            row["regret_after_change"] = float(p["regret"][tc:].sum()) if sc["mode"] == "abrupt" else None
        if name == "dLinUCB":
            row["false_alarm_rate"] = false_alarm_rate(p["detections"], no_change_rounds(sc, T))
            row["detection_delay"] = detection_delay(p["detections"], sc["change_at"]) if sc["mode"] == "abrupt" else None
        rows[name] = row
    return rows


def paired_bootstrap_ci(diff, n=10000, seed=0, level=0.95):
    """95% CI of the mean of paired per-seed differences (5.9)."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    means = diff[rng.integers(0, len(diff), (n, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(diff.mean()), float(lo), float(hi)
