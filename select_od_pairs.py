"""Choose the origin-destination pairs once and save them to data/od_pairs.json (proposal 5.2.2:
"เริ่มจาก 1 คู่เป็นกรณีหลัก และเพิ่มเป็น 3–5 คู่ ... กำหนดและบันทึกในไฟล์กำหนดค่าก่อนเริ่มการทดลองหลัก").

Selection uses geometry only, never algorithm results:
  * main pair: EN04 (near the College of Computing, which OpenStreetMap does not name) -> Central Library
  * other pairs: from a fixed list of well-known places, pairs 500-1500 m apart (straight line)
    whose Yen candidate set keeps at least `min_routes` routes after the overlap filter;
    chosen greedily so their midpoints are as far as possible from pairs already chosen.
"""
import json

import numpy as np

import sim

PLACES = [
    "สาขาวิชาวิศวกรรมคอมพิวเตอร์, สาขาวิชาวิศวกรรมไฟฟ้า, EN04",
    "หอสมุดกลาง มหาวิทยาลัยขอนแก่น",
    "ศูนย์อาหารและบริการ 1 (Complex)",
    "คณะศึกษาศาสตร์",
    "คณะเศรษฐศาสตร์",
    "วิทยาลัยนานาชาติ",
    "อาคารเรียนรวมและปฏิบัติการ SC. 09",
    "หอพักนักศึกษาที่ 7",
    "สนามกีฬากลาง มหาวิทยาลัยขอนแก่น",
    "ตลาดหอแปดหลัง",
    "ไปรษณีย์ มหาวิทยาลัยขอนแก่น",
    "คณะเภสัขศาสตร์",
]
MAIN = (PLACES[0], PLACES[1])
N_PAIRS = 5


def main():
    cfg = sim.load_config()
    rc = cfg["routes"]
    G, D, places, meta = sim.load_campus()
    base = sim.edge_base_density(D, places, cfg["context"]["hotspot_decay_m"])
    by = {p["name"]: p for p in places}
    missing = [n for n in PLACES if n not in by]
    assert not missing, missing

    def routes_ok(o, d):
        rs = sim.candidate_routes(D, base, by[o]["node"], by[d]["node"], rc["k"], rc["max_overlap"], rc["max_paths_enumerated"])
        return rs if len(rs) >= rc["min_routes"] else None

    def mid(o, d):
        return ((by[o]["lat"] + by[d]["lat"]) / 2, (by[o]["lon"] + by[d]["lon"]) / 2)

    chosen = []
    rs = routes_ok(*MAIN)
    assert rs, "main pair fails the 3-route rule"
    chosen.append((MAIN, rs))

    cands = []
    for o, d in [(a, b) for i, a in enumerate(PLACES) for b in PLACES[i + 1:]]:
        dist = float(sim.great_circle(by[o]["lat"], by[o]["lon"], [by[d]["lat"]], [by[d]["lon"]])[0])
        if 500 <= dist <= 1500 and (o, d) != MAIN:
            cands.append((o, d))
    while len(chosen) < N_PAIRS and cands:
        def spread(pair):
            m = mid(*pair)
            return min(float(sim.great_circle(m[0], m[1], [mid(*c[0])[0]], [mid(*c[0])[1]])[0]) for c in chosen)
        cands.sort(key=spread, reverse=True)
        pair = cands.pop(0)
        rs = routes_ok(*pair)
        if rs:
            chosen.append((pair, rs))
        print("checked", pair, "->", "ok" if rs else "rejected")

    out = []
    for (o, d), rs in chosen:
        out.append({"origin": o, "destination": d, "origin_node": by[o]["node"], "dest_node": by[d]["node"],
                    "routes_after_filter": len(rs),
                    "route_lengths_m": [round(r.length_m, 1) for r in rs],
                    "route_base_crowd": [round(r.base_crowd, 3) for r in rs],
                    "routes": [[int(n) for n in r.nodes] for r in rs]})
    (sim.DATA / "od_pairs.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for x in out:
        print(x["origin"][:25], "->", x["destination"][:25], x["route_lengths_m"], x["route_base_crowd"])


if __name__ == "__main__":
    main()
