"""Prepare, before the demo runs, the candidate routes for every ordered pair of the listed
campus places (proposal 5.1.2: the prototype takes its origin and destination from a list
prepared in advance; 5.2.2: K-shortest loopless paths, K = 5, drop routes that overlap more
than 80%, keep the pair only if at least 3 routes remain).

Places come from OpenStreetMap (data/places.json) plus data/extra_places.json, which holds
places OpenStreetMap does not name, with coordinates supplied by the project authors.

Run:  .venv/Scripts/python build_route_catalog.py
"""
import json
from concurrent.futures import ProcessPoolExecutor

import sim
from select_od_pairs import PLACES

CFG = sim.load_config()
EXTRA = sim.DATA / "extra_places.json"


def catalog_places(D, places):
    by = {p["name"]: p for p in places}
    chosen = [dict(by[n], source="OpenStreetMap") for n in PLACES]
    if EXTRA.exists():
        for e in json.loads(EXTRA.read_text(encoding="utf-8")):
            node = sim_nearest(D, e["lat"], e["lon"])
            chosen.append({"name": e["name"], "name_en": e.get("name_en", ""), "kind": "extra",
                           "lat": e["lat"], "lon": e["lon"], "node": node, "source": e.get("source", "authors")})
    return chosen


def sim_nearest(D, lat, lon):
    ids = list(D.nodes)
    d = sim.great_circle(lat, lon, [D.nodes[n]["y"] for n in ids], [D.nodes[n]["x"] for n in ids])
    return int(ids[int(d.argmin())])


_STATE = {}


def _init():
    G, D, places, meta = sim.load_campus()
    _STATE["D"] = D
    _STATE["base"] = sim.edge_base_density(D, places, CFG["context"]["hotspot_decay_m"])


def _job(args):
    o, d = args
    rc = CFG["routes"]
    rs = sim.candidate_routes(_STATE["D"], _STATE["base"], o["node"], d["node"], rc["k"], rc["max_overlap"], rc["max_paths_enumerated"])
    return {"origin": o["name"], "destination": d["name"], "routes_after_filter": len(rs),
            "ok": len(rs) >= rc["min_routes"], "route_lengths_m": [round(r.length_m, 1) for r in rs],
            "routes": [[int(n) for n in r.nodes] for r in rs]}


def main():
    G, D, places, meta = sim.load_campus()
    plist = catalog_places(D, places)
    jobs = [(o, d) for o in plist for d in plist if o["name"] != d["name"] and o["node"] != d["node"]]
    with ProcessPoolExecutor(initializer=_init) as ex:
        pairs = list(ex.map(_job, jobs, chunksize=2))
    out = {"places": plist, "pairs": pairs, "rule": CFG["routes"]}
    (sim.DATA / "route_catalog.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    ok = sum(p["ok"] for p in pairs)
    print(f"{len(plist)} places, {len(pairs)} ordered pairs, {ok} pass the 3-route rule")
    for p in pairs:
        if not p["ok"]:
            print("  rejected:", p["origin"][:30], "->", p["destination"][:30], p["routes_after_filter"])


if __name__ == "__main__":
    main()
