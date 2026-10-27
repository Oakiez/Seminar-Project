"""Checks promised in proposal objective 3.1, plus the metric and scenario definitions,
on a small synthetic route set (no OpenStreetMap data needed)."""
import numpy as np

import sim

CFG = sim.load_config()


def toy_routes():
    # (length m, base crowd on every edge): short and crowded ... long and quiet
    spec = [(600, 0.9), (650, 0.6), (720, 0.4), (800, 0.2)]
    return [sim.Route(nodes=[i], length_m=L, edge_len=[L / 2, L / 2], edge_base=[c, c]) for i, (L, c) in enumerate(spec)]


def run(policies, sc, T=60, seed=3):
    return sim.run(toy_routes(), sc, T=T, seed=seed, noise_sd=0.1, edge_noise_sd=0.05, policies=policies)


def env(sc, T=60, seed=3):
    return sim.build_environment(toy_routes(), sc, T, seed, 0.1, 0.05)


SC = sim.scenario_set(CFG, 0.3)


class Spy(sim.Policy):
    """Records exactly what the runner hands to a policy."""
    name = "spy"

    def __init__(self):
        self.seen_X, self.updates = [], []

    def select(self, t, X_t):
        self.seen_X.append(np.array(X_t))
        return t % X_t.shape[0]

    def update(self, t, x, r):
        self.updates.append((t, np.array(x), r))


def test_policy_gets_feedback_only_for_the_chosen_route():
    spy = Spy()
    out = run([spy], SC["S1"])
    e = env(SC["S1"])
    assert len(spy.updates) == 60
    for t, x, r in spy.updates:
        a = out["policies"]["spy"]["choice"][t]
        assert np.allclose(x, e.X[t, a])
        assert np.isclose(r, e.mu[t, a] + e.noise[t, a])


def test_policy_never_receives_oracle_or_latent_preference():
    sc = dict(SC["S2"], change_at=20)
    spy = Spy()
    run([spy], sc, T=40)
    e = env(sc, T=40)
    for t, X_t in enumerate(spy.seen_X):
        assert X_t.shape == (4, 2)                       # only (length, crowd exposure) per route
        assert np.allclose(X_t, e.X[t])                  # exactly the features, nothing else
        assert not np.allclose(X_t[:, 0], e.mu[t])


def test_oracle_has_zero_regret():
    sc = dict(SC["S3"], change_at=10, change_end=30)
    e = env(sc, T=50)

    class Oracle(sim.Policy):  # test-only: allowed to peek at mu
        name = "oracle"

        def select(self, t, X_t):
            return int(np.argmax(e.mu[t]))

    out = run([Oracle()], sc, T=50)
    assert np.allclose(out["policies"]["oracle"]["regret"], 0.0)


def test_same_seed_same_results():
    params = {"LinUCB": {"alpha": 0.1}, "dLinUCB": {"alpha": 0.1, "delta1": 0.1, "delta1_tilde": 0.05, "delta2": 0.05}}
    sc = dict(SC["S2"], change_at=100)
    a = run(sim.make_policies(CFG, params, 11), sc, T=200, seed=11)
    b = run(sim.make_policies(CFG, params, 11), sc, T=200, seed=11)
    for name in a["policies"]:
        assert np.array_equal(a["policies"][name]["choice"], b["policies"][name]["choice"])


def test_doc_scenarios_match_table_4():
    s = sim.scenario_set(CFG, 0.3)
    assert s["S1"]["theta_start"] == [-0.3, -0.7]
    assert s["S2"]["theta_end"] == [-0.7, -0.3] and s["S2"]["change_at"] == 1000
    assert s["S3"]["change_at"] == 750 and s["S3"]["change_end"] == 1250
    assert s["CTRL"]["theta_start"] == [-0.5, -0.5]
    th = sim.theta_schedule(s["S3"], 2000)
    assert np.allclose(th[0], [-0.3, -0.7]) and np.allclose(th[-1], [-0.7, -0.3])


def test_adaptation_time_definition():
    regret = np.r_[np.zeros(100), np.full(30, 0.2), np.zeros(200)]
    tau = sim.adaptation_time(regret, change_at=100, eps=0.05, window=20)
    assert tau is not None and 30 <= tau <= 60
    assert sim.adaptation_time(np.full(300, 0.2), change_at=100, eps=0.05, window=20) is None


def test_false_alarm_rate_per_thousand_rounds():
    assert sim.false_alarm_rate([10, 500, 1500], np.arange(1000)) == 2.0


def test_same_seed_same_regret_and_metrics():
    """Objective 3.1 (1): with the same configuration and seed, the choices, the regret and every
    metric repeat exactly. Computation Time is measured wall-clock time and is excluded."""
    params = {"LinUCB": {"alpha": 0.1}, "dLinUCB": {"alpha": 0.1, "delta1": 0.1, "delta1_tilde": 0.05, "delta2": 0.05}}
    sc = dict(SC["S2"], change_at=150)
    routes = toy_routes()

    def once():
        out = sim.run(routes, sc, T=300, seed=21, noise_sd=0.1, edge_noise_sd=0.05,
                      policies=sim.make_policies(CFG, params, 21))
        return out, sim.run_metrics(out, sc, CFG, routes)

    (a, ma), (b, mb) = once(), once()
    for name in a["policies"]:
        assert np.array_equal(a["policies"][name]["regret"], b["policies"][name]["regret"])
        assert a["policies"][name]["detections"] == b["policies"][name]["detections"]
        for k, v in ma[name].items():
            if k != "ms_per_round":
                assert v == mb[name][k], (name, k)
