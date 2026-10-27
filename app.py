"""Prototype demo for the proposal "The Dynamic GIS Recommender" (KKU walking routes).

Inputs follow proposal 5.1.2 only: OD pair from a prepared list, scenario S1-S3 (and the control
case), random seed, and the range of rounds to view. The noise levels of 5.6 are offered as the
"additional" comparison. Outputs follow 5.1.3: per-round choices, CSV log, metrics of Table 5,
95% CIs over 30 seeds, the Table 6 decisions and a map of the routes each policy picked.
Run:  .venv/Scripts/streamlit run app.py
"""
import io
import json
import os
from pathlib import Path

import folium
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from matplotlib import font_manager as fm
from streamlit_folium import st_folium

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sim  # noqa: E402

st.set_page_config(page_title="Dynamic GIS Recommender – Demo", layout="wide")

# --------------------------------------------------------------------------- fonts (Thai in Matplotlib)
for d in [Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts", Path("C:/Windows/Fonts")]:
    for f in ["THSarabunNew.ttf", "THSarabunNew Bold.ttf"]:
        if (d / f).exists():
            fm.fontManager.addfont(str(d / f))
plt.rcParams.update({"font.family": ["TH Sarabun New", "Tahoma", "DejaVu Sans"], "font.size": 15,
                     "axes.titlesize": 17, "axes.spines.top": False, "axes.spines.right": False})

CFG = sim.load_config()
CAL = yaml.safe_load((sim.DATA / "calibrated.yaml").read_text(encoding="utf-8"))
RES = sim.ROOT / CFG["evaluation"]["results_dir"]
T = CFG["experiment"]["T"]
ROUTE_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
ROUTE_NAMES = ["A", "B", "C", "D", "E"]
POLICY_COLORS = {"Shortest Path": "#7f7f7f", "Fixed Context-Weighted Score": "#bcbd22",
                 "LinUCB": "#1f77b4", "dLinUCB": "#d62728"}
POLICY_TH = {"Shortest Path": "Shortest Path (ไม่เรียนรู้)",
             "Fixed Context-Weighted Score": "Fixed Context-Weighted Score (ไม่เรียนรู้)",
             "LinUCB": "LinUCB (เรียนรู้)", "dLinUCB": "การปรับใช้ตามแนวคิด dLinUCB (เรียนรู้ + ตรวจจับการเปลี่ยนแปลง)"}
DL = "การปรับใช้ตามแนวคิด dLinUCB"   # proposal 5.5: name used when dLinUCB is adapted
SHORT = {"LinUCB": "LinUCB", "dLinUCB": DL}
SET_TH = {"doc": "ค่าตามเอกสาร (ตารางที่ 4)", "calibrated": "ค่าหลังสอบเทียบ (ข้อ 5.6)"}


# --------------------------------------------------------------------------- cached data
@st.cache_resource(show_spinner="กำลังโหลดโครงข่ายทางเดินของ มข. ...")
def campus():
    """Walking graph plus the places and routes prepared in advance (5.1.2, 5.2.2)."""
    G, D, places, meta = sim.load_campus()
    base = sim.edge_base_density(D, places, CFG["context"]["hotspot_decay_m"])
    exp_pairs, exp_routes = sim.load_pair_routes(D, base)            # the 5 pairs of the main experiment
    cat = json.loads((sim.DATA / "route_catalog.json").read_text(encoding="utf-8"))
    cat_places = {p["name"]: p for p in cat["places"]}
    cat_pairs = {(p["origin"], p["destination"]): p for p in cat["pairs"] if p["ok"]}
    exp_index = {(p["origin"], p["destination"]): i for i, p in enumerate(exp_pairs)}
    background = [[(D.nodes[u]["y"], D.nodes[u]["x"]), (D.nodes[v]["y"], D.nodes[v]["x"])]
                  for u, v in D.edges() if u < v]
    return G, D, base, cat_places, cat_pairs, exp_pairs, exp_routes, exp_index, background, meta


@st.cache_resource(show_spinner="กำลังเตรียมเส้นทาง ...")
def routes_for(o_name, d_name):
    G, D, base, cat_places, cat_pairs, exp_pairs, exp_routes, exp_index, background, meta = campus()
    if (o_name, d_name) in exp_index:
        rs = exp_routes[exp_index[(o_name, d_name)]]
    else:
        rs = sim.routes_from_nodes(D, base, cat_pairs[(o_name, d_name)]["routes"])
    return rs, [sim.route_latlon(G, r.nodes) for r in rs]


@st.cache_data(show_spinner="กำลังจำลอง ...")
def simulate(set_name, o_name, d_name, sc_key, seed, noise_sd):
    rs, _ = routes_for(o_name, d_name)
    s = CAL["sets"][set_name]
    sc = sim.scenario_set(CFG, s["length_weight"])[sc_key]
    c = CFG["context"]
    out = sim.run(rs, sc, T=T, seed=seed, noise_sd=noise_sd, edge_noise_sd=c["edge_noise_sd"],
                  policies=sim.make_policies(CFG, s["chosen"], seed), rounds_per_day=c["rounds_per_day"],
                  peaks=tuple(c["daily_peaks_hour"]))
    metrics = sim.run_metrics(out, sc, CFG, rs)
    return out, metrics, sc


@st.cache_data
def results():
    if not (RES / "summary.csv").exists():
        return None, None, None
    df = pd.read_csv(RES / "summary.csv")
    curves = dict(np.load(RES / "curves.npz"))
    hyp = json.loads((RES / "hypotheses.json").read_text(encoding="utf-8"))
    return df, curves, hyp


# --------------------------------------------------------------------------- helpers
def base_map(background, show_tiles):
    m = folium.Map(location=[16.4746, 102.823], zoom_start=16, tiles="OpenStreetMap" if show_tiles else None, control_scale=True)
    folium.PolyLine(background, color="#c8c8c8", weight=1, opacity=0.8).add_to(m)
    return m


def add_od(m, o, d):
    # dashed link from each pin to the walking-graph node the routes start or end at
    for p in (o, d):
        nd = D.nodes[p["node"]]
        folium.PolyLine([(p["lat"], p["lon"]), (nd["y"], nd["x"])], color="#555", weight=2, dash_array="4,4",
                        tooltip="ระยะจากหมุดถึงทางเดินที่ใกล้ที่สุด").add_to(m)
    folium.Marker([o["lat"], o["lon"]], tooltip="ต้นทาง: " + o["name"], icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([d["lat"], d["lon"]], tooltip="ปลายทาง: " + d["name"], icon=folium.Icon(color="red", icon="flag")).add_to(m)


def fit(m, lines):
    pts = [p for line in lines for p in line]
    m.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)], [max(p[0] for p in pts), max(p[1] for p in pts)]], padding=(20, 20))


def mark_change(ax, sc):
    if sc["mode"] == "abrupt":
        ax.axvline(sc["change_at"], color="black", ls="--", lw=1)
        ax.text(sc["change_at"], ax.get_ylim()[1], " ความชอบเปลี่ยน", va="top", fontsize=13)
    elif sc["mode"] == "gradual":
        ax.axvspan(sc["change_at"], sc["change_end"], color="gray", alpha=0.15)
        ax.text(sc["change_at"], ax.get_ylim()[1], " ช่วงความชอบค่อยๆ เปลี่ยน", va="top", fontsize=13)


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


# --------------------------------------------------------------------------- sidebar: proposal 5.1.2
G, D, base, cat_places, cat_pairs, exp_pairs, exp_routes, exp_index, background, meta = campus()
st.sidebar.title("ข้อมูลนำเข้าของต้นแบบ")
st.sidebar.caption("ตามเอกสารข้อ 5.1.2: เลือกจากรายการที่เตรียมไว้ล่วงหน้า ไม่รับพิกัดอิสระ")
set_name = st.sidebar.radio("ชุดค่าสถานการณ์", list(CAL["sets"]), index=list(CAL["sets"]).index("calibrated"), format_func=lambda k: SET_TH[k],
                            help="ค่าตามเอกสารทำให้ทางสั้นที่สุดดีที่สุดเกือบทุกรอบ จึงมีชุดที่สอบเทียบตามข้อ 5.6 เพิ่ม")
names = list(cat_places)
EXP_MARK = " ★"
o_name = st.sidebar.selectbox("(ก) ต้นทาง", names, index=names.index(exp_pairs[0]["origin"]))
dest_ok = [n for n in names if (o_name, n) in cat_pairs]
d_default = exp_pairs[0]["destination"] if exp_pairs[0]["destination"] in dest_ok else dest_ok[0]
d_name = st.sidebar.selectbox("(ก) ปลายทาง", dest_ok, index=dest_ok.index(d_default),
                              format_func=lambda n: n + (EXP_MARK if (o_name, n) in exp_index else ""),
                              help="แสดงเฉพาะปลายทางที่มีเส้นทางทางเลือกอย่างน้อย 3 เส้นหลังการกรอง (ข้อ 5.2.2) · "
                                   "★ = คู่ที่บันทึกไว้สำหรับการทดลองหลัก มีผล 30 Seed")
exp_labels = [f"{p['origin']} → {p['destination']}" for p in exp_pairs]
pair_label = f"{o_name} → {d_name}"
pair_id = exp_index.get((o_name, d_name))
sc_key = st.sidebar.radio("(ข) สถานการณ์", list(CFG["experiment"]["scenarios"]), index=list(CFG["experiment"]["scenarios"]).index("S2"),
                          format_func=lambda k: CFG["experiment"]["scenarios"][k]["label"])
seed = st.sidebar.selectbox("(ค) Random Seed (ชุดทดลองจริง 0–29)", list(range(CFG["evaluation"]["test_seeds"])), index=0)
r0, r1 = st.sidebar.slider("(ง) ช่วงของรอบที่ต้องการดู", 0, T - 1, (0, T - 1), 10)
st.sidebar.divider()
noise_sd = st.sidebar.select_slider("ส่วนเพิ่มเติม (ข้อ 5.6): ระดับ Noise σ", CFG["reward"]["extra_noise_levels"],
                                    value=CFG["reward"]["noise_sd"])
show_tiles = st.sidebar.checkbox("แสดงแผนที่พื้นหลัง (ต้องใช้อินเทอร์เน็ต)", True)

routes_p, lines_p = routes_for(o_name, d_name)
o_place, d_place = cat_places[o_name], cat_places[d_name]
K = len(routes_p)
key = f"{names.index(o_name)}_{names.index(d_name)}"
out, metrics, sc = simulate(set_name, o_name, d_name, sc_key, seed, noise_sd)
pol = out["policies"]
env = out["env"]
a_w = CAL["sets"][set_name]["length_weight"]

st.title("The Dynamic GIS Recommender: Demo")
st.caption("ต้นแบบประกอบการเสนอเค้าโครง · ความหนาแน่นของผู้คนและความชอบของผู้ใช้เป็น**ค่าจำลอง** · "
           "ตัวเลขเป็น**ผลนำร่องด้วยค่าเบื้องต้น** ไม่ใช่ผลการทดลองฉบับสมบูรณ์ · " + meta["attribution"])

if set_name == "calibrated":
    st.info(f"กำลังดูชุดค่าหลังสอบเทียบ (ข้อ 5.6): น้ำหนัก θ ในตารางที่ 4 เปลี่ยนจาก 0.3 เป็น {CAL['sets']['calibrated']['length_weight']} "
            "เพราะเมื่อใช้ค่าในตารางที่ 4 ตรงตัว เส้นทางทุกเส้นยาวใกล้กันมาก ทางที่สั้นที่สุดจึงดีที่สุดเกือบทุกรอบ "
            "และไม่มีอะไรให้อัลกอริทึมเรียนรู้ ค่าตามตารางที่ 4 กดดูได้ที่แถบซ้าย")
STEPS = ["① ระบบและเส้นทางทางเลือก", "② การตัดสินใจทีละรอบ", "③ ผลของรันนี้", "④ ผล 30 Seed และสมมติฐาน", "⑤ ตรวจเทียบกับเอกสาร"]
step = st.radio("ขั้นตอน", STEPS, horizontal=True, label_visibility="collapsed")

# --------------------------------------------------------------------------- ① system and routes
if step == STEPS[0]:
    show_route = st.radio("แสดงเส้นทาง", ["ทั้งหมด"] + ROUTE_NAMES[:K], horizontal=True,
                          format_func=lambda x: x if x == "ทั้งหมด" else f"เส้นทาง {x}")
    c1, c2 = st.columns([3, 2])
    with c1:
        m = base_map(background, show_tiles)
        for i, line in enumerate(lines_p):
            if show_route != "ทั้งหมด" and ROUTE_NAMES[i] != show_route:
                continue
            folium.PolyLine(line, color=ROUTE_COLORS[i], weight=6, opacity=0.85,
                            tooltip=f"เส้นทาง {ROUTE_NAMES[i]}: {routes_p[i].length_m:.0f} ม.").add_to(m)
        add_od(m, o_place, d_place)
        fit(m, lines_p)
        st_folium(m, height=520, returned_objects=[], use_container_width=True, key=f"m1_{key}_{show_route}")
    with c2:
        st.markdown(f"**{pair_label}**")
        st.markdown(f"เส้นทางทางเลือก {K} เส้น จากวิธี K-Shortest Loopless Paths ของ Yen (1971) "
                    f"K = {CFG['routes']['k']} ตัดเส้นทางที่ซ้อนทับเกินร้อยละ {int(CFG['routes']['max_overlap'] * 100)} (ข้อ 5.2.2) "
                    "แต่ละเส้นคือทางเลือก (Arm) หนึ่งตัว นโยบายเห็นเพียง 2 ค่าต่อเส้นทาง (ตารางที่ 2)")
        rows = []
        for i, r in enumerate(routes_p):
            rows.append({"เส้นทาง": ROUTE_NAMES[i], "ความยาว (ม.)": round(r.length_m, 1),
                         "ความยาวหลังปรับ Min–Max": round(float(env.X[0, i, 0]), 2),
                         "ความแออัด 08:00": round(r.base_crowd * sim.daily_pattern(32), 2),
                         "ความแออัด 14:00": round(r.base_crowd * sim.daily_pattern(56), 2),
                         "ความแออัด 21:00": round(r.base_crowd * sim.daily_pattern(84), 2)})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("ความแออัดเป็นค่าจำลองตามข้อ 5.3: ρ = ค่าฐานตามตำแหน่ง × รูปแบบรายวัน + Noise รายขอบ")
    fig1 = sim.ROOT.parent / "Figure1_process.png"
    if fig1.exists():
        with st.expander("ภาพที่ 1 ในเอกสาร: กระบวนการวิจัยและโครงสร้างของระบบทดลอง"):
            st.image(str(fig1), width=720)

# --------------------------------------------------------------------------- ② decision per round
if step == STEPS[1]:
    if "t" not in st.session_state or not (r0 <= st.session_state.t <= r1):
        st.session_state.t = r0
    b = st.columns([1, 1, 1, 1, 6])
    for col, (lab, dv) in zip(b, [("◀◀ −50", -50), ("◀ −1", -1), ("+1 ▶", 1), ("+50 ▶▶", 50)]):
        if col.button(lab):
            st.session_state.t = int(np.clip(st.session_state.t + dv, r0, r1))
    t = st.slider("รอบที่", r0, r1, key="t")
    cc = st.columns([2, 2, 2])
    learner = cc[0].radio("ดูการตัดสินใจของ", ["LinUCB", "dLinUCB"], horizontal=True, format_func=lambda x: SHORT[x])
    show_all2 = cc[1].checkbox("แสดงเส้นทางทางเลือกอื่นแบบจาง", False)
    show_best2 = cc[2].checkbox("แสดงเส้นทางที่ดีที่สุดจริง (Oracle)", True)
    P = pol[learner]
    a, best, mu_t = int(P["choice"][t]), int(out["oracle"][t]), env.mu[t]
    st.markdown(f"**รอบ {t}** · เวลา {sim.hour_label(t)} น. · g(t) = {env.g[t]:.2f} · "
                f"ความชอบแฝง θ = ({env.theta[t][0]:.2f}, {env.theta[t][1]:.2f}) ซ่อนจากนโยบาย")
    c1, c2 = st.columns([3, 2])
    with c2:
        est, bonus = P["est"][t], P["bonus"][t]
        fig, ax = plt.subplots(figsize=(6, 4.4))
        xs = np.arange(K)
        for i in range(K):
            ax.plot([i, i], [est[i], est[i] + bonus[i]], color=ROUTE_COLORS[i], lw=10 if i == a else 6, alpha=0.35, solid_capstyle="butt")
        ax.scatter(xs, est, s=[140 if i == a else 70 for i in range(K)], c=ROUTE_COLORS[:K], zorder=3, label="ค่าที่นโยบายประมาณ")
        ax.scatter(xs, est + bonus, marker="_", s=500, c="#444", zorder=3, label="ขอบบน (UCB)")
        ax.scatter(xs, mu_t, marker="D", s=70, facecolors="none", edgecolors="black", zorder=4, label="ค่าจริง μ (ผู้ประเมินเห็นเท่านั้น)")
        ax.annotate("เลือก", (a, est[a] + bonus[a]), xytext=(0, 18), textcoords="offset points", ha="center", arrowprops=dict(arrowstyle="->"))
        ax.set_xticks(xs, [f"เส้นทาง {ROUTE_NAMES[i]}" for i in range(K)])
        ax.set_ylabel("Reward (ยิ่งสูงยิ่งดี)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=12, frameon=False)
        st.pyplot(fig, clear_figure=True)
        st.caption("นโยบายเลือกเส้นทางที่ขอบบนสูงที่สุด (Upper Confidence Bound) เส้นแนวตั้งยาวแปลว่ายังไม่แน่ใจ จึงคุ้มที่จะลอง")
    with c1:
        m = base_map(background, show_tiles)
        if show_all2:
            for i, line in enumerate(lines_p):
                folium.PolyLine(line, color=ROUTE_COLORS[i], weight=3, opacity=0.35).add_to(m)
        if show_best2:
            folium.PolyLine(lines_p[best], color="black", weight=9, opacity=0.35, tooltip="เส้นทางที่ดีที่สุดจริง (Oracle)").add_to(m)
        folium.PolyLine(lines_p[a], color=ROUTE_COLORS[a], weight=6, opacity=1, tooltip=f"{SHORT[learner]} เลือกเส้นทาง {ROUTE_NAMES[a]}").add_to(m)
        add_od(m, o_place, d_place)
        fit(m, lines_p)
        st_folium(m, height=440, returned_objects=[], use_container_width=True, key=f"m2_{key}_{learner}_{t}_{show_all2}_{show_best2}")
        st.caption(f"เส้นสี = เส้นทางที่ {SHORT[learner]} เลือก · เงาสีดำ = เส้นทางที่ดีที่สุดจริง (Oracle ใช้ประเมินผลเท่านั้น)")
    verdict = "ตรงกับเส้นทางที่ดีที่สุด" if a == best else f"เส้นทางที่ดีที่สุดคือ {ROUTE_NAMES[best]}"
    st.info(f"{SHORT[learner]} เลือกเส้นทาง {ROUTE_NAMES[a]} → ได้ Feedback = {P['reward'][t]:.3f} (เห็นเฉพาะเส้นทางที่เลือก) · "
            f"{verdict} · Regret รอบนี้ = {P['regret'][t]:.3f}")
    if learner == "dLinUCB":
        det = [x for x in P["detections"] if x <= t]
        st.write(f"{DL} สร้างแบบจำลองใหม่ไปแล้ว {len(det)} ครั้ง" + (f" (รอบ {', '.join(map(str, det))})" if det else ""))
    st.dataframe(pd.DataFrame([{"นโยบาย": POLICY_TH[n], "เลือกเส้นทาง": ROUTE_NAMES[int(p['choice'][t])],
                                "Feedback": round(float(p["reward"][t]), 3), "Regret รอบนี้": round(float(p["regret"][t]), 3),
                                "Regret สะสม": round(float(p["cum_regret"][t]), 2)} for n, p in pol.items()]),
                 hide_index=True, width="stretch")

# --------------------------------------------------------------------------- ③ this run
if step == STEPS[2]:
    st.subheader(f"ผลของรันนี้: {SET_TH[set_name]} · {sc['label']} · Seed {seed} · σ = {noise_sd}")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(r0, r1 + 1)
    for n, p in pol.items():
        ax.plot(x, p["cum_regret"][r0:r1 + 1], color=POLICY_COLORS[n], lw=2.2, label=POLICY_TH[n])
    for d in pol["dLinUCB"]["detections"]:
        if r0 <= d <= r1:
            ax.axvline(d, color=POLICY_COLORS["dLinUCB"], ls=":", lw=1)
    mark_change(ax, sc)
    ax.set_xlabel("รอบ")
    ax.set_ylabel("Dynamic Cumulative Regret")
    ax.set_title("Dynamic Cumulative Regret (ยิ่งต่ำยิ่งดี) · เส้นประสีแดงบาง = รอบที่สร้างแบบจำลองใหม่")
    ax.legend(fontsize=12, frameon=False)
    st.pyplot(fig, clear_figure=True)

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot(x, env.theta[r0:r1 + 1, 0], label="θ ความยาว", color="black")
        ax.plot(x, env.theta[r0:r1 + 1, 1], label="θ ความแออัด", color="black", ls="--")
        ax.plot(x, env.g[r0:r1 + 1], label="g(t) ความแออัดตามเวลา", color="#ff7f0e", lw=0.8)
        ax.set_title("สิ่งที่เปลี่ยนในตัวจำลอง (นโยบายไม่เห็น)")
        ax.set_xlabel("รอบ")
        ax.legend(fontsize=11, frameon=False)
        st.pyplot(fig, clear_figure=True)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        w = CFG["evaluation"]["adaptation_window"]
        for n, p in pol.items():
            ma = np.convolve(p["reward"], np.ones(w) / w, mode="valid")
            xx = np.arange(w - 1, T)
            keep = (xx >= r0) & (xx <= r1)
            ax.plot(xx[keep], ma[keep], color=POLICY_COLORS[n], lw=1.4)
        ax.set_title(f"Average Reward (ค่าเฉลี่ยเคลื่อนที่ {w} รอบ)")
        ax.set_xlabel("รอบ")
        st.pyplot(fig, clear_figure=True)

    rows = []
    for n, mtr in metrics.items():
        row = {"นโยบาย": POLICY_TH[n], "Dynamic Cumulative Regret": fmt(mtr["dynamic_cumulative_regret"]),
               "Average Reward": fmt(mtr["average_reward"], 3), "Crowd Exposure": fmt(mtr["crowd_exposure"], 3),
               "Route Length (ม.)": fmt(mtr["route_length_m"], 1), "Computation Time (มิลลิวินาที/รอบ)": fmt(mtr["ms_per_round"], 3)}
        if "adaptation_time" in mtr:
            row["Adaptation Time (รอบ)"] = "ไม่ฟื้นตัว" if mtr["adaptation_time"] is None else str(mtr["adaptation_time"])
        if n == "dLinUCB":
            row["False Alarm Rate (ต่อ 1,000 รอบ)"] = fmt(mtr.get("false_alarm_rate"))
            if sc["mode"] == "abrupt":
                dd = mtr.get("detection_delay")
                row["Detection Delay (รอบ)"] = "ตรวจไม่พบ" if dd is None else str(dd)
        rows.append(row)
    st.markdown("**ตัวชี้วัดตามตารางที่ 5**")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Adaptation Time ใช้ ε = {CFG['evaluation']['adaptation_eps']} และ W = {CFG['evaluation']['adaptation_window']} ตามตารางที่ 5 · "
               "เมื่อใช้ Seed เดิม ทุกค่าในตารางจะเท่าเดิมทุกครั้ง (ข้อ 3.1) ยกเว้น Computation Time ที่วัดจากเวลาจริงของเครื่อง")

    # 5.1.3: per-run CSV of every round, and the run configuration in YAML
    log = []
    for n, p in pol.items():
        log.append(pd.DataFrame({"round": np.arange(T), "policy": n, "chosen_route": [ROUTE_NAMES[i] for i in p["choice"]],
                                 "reward": p["reward"], "regret": p["regret"], "cum_regret": p["cum_regret"],
                                 "oracle_route": [ROUTE_NAMES[i] for i in out["oracle"]]}))
    csv = pd.concat(log).to_csv(index=False).encode("utf-8-sig")
    run_cfg = {"set": set_name, "length_weight": a_w, "pair": pair_label, "scenario": sc_key, "scenario_definition": sc,
               "seed": seed, "noise_sd": noise_sd, "T": T, "policy_params": CAL["sets"][set_name]["chosen"],
               "routes": CFG["routes"], "context": CFG["context"], "evaluation": CFG["evaluation"]}
    d1, d2 = st.columns(2)
    d1.download_button("ดาวน์โหลดผลทุกรอบ (CSV)", csv, file_name=f"run_{set_name}_{key}_{sc_key}_s{seed}.csv", mime="text/csv")
    d2.download_button("ดาวน์โหลดไฟล์กำหนดค่าของรันนี้ (YAML)", yaml.safe_dump(run_cfg, allow_unicode=True, sort_keys=False),
                       file_name=f"run_{set_name}_{key}_{sc_key}_s{seed}.yaml", mime="text/yaml")

    st.subheader("เส้นทางที่แต่ละนโยบายเลือกในรอบสุดท้ายของช่วงที่ดู")
    show_pol = st.radio("แสดงเส้นทางของ", ["ทั้งหมด"] + list(pol) + ["Oracle"], horizontal=True,
                        format_func=lambda x: {"ทั้งหมด": "ทั้งหมด", "Oracle": "ที่ดีที่สุดจริง (Oracle)"}.get(x, x))
    m = base_map(background, show_tiles)
    last = r1
    if show_pol in ("ทั้งหมด", "Oracle"):
        folium.PolyLine(lines_p[int(out["oracle"][last])], color="black", weight=12, opacity=0.25 if show_pol == "ทั้งหมด" else 0.6,
                        tooltip="ดีที่สุดจริง (Oracle)").add_to(m)
    for j, (n, p) in enumerate(pol.items()):
        if show_pol not in ("ทั้งหมด", n):
            continue
        i = int(p["choice"][last])
        folium.PolyLine(lines_p[i], color=POLICY_COLORS[n], weight=5, opacity=0.9, dash_array=None if j % 2 == 0 else "8,6",
                        tooltip=f"{POLICY_TH[n]}: เส้นทาง {ROUTE_NAMES[i]}").add_to(m)
    add_od(m, o_place, d_place)
    fit(m, lines_p)
    cm, cl = st.columns([3, 2])
    with cm:
        st_folium(m, height=420, returned_objects=[], use_container_width=True, key=f"m3_{key}_{last}_{show_pol}")
    with cl:
        for n, p in pol.items():
            i = int(p["choice"][last])
            st.markdown(f"<span style='color:{POLICY_COLORS[n]}'>■</span> {POLICY_TH[n]}: เส้นทาง {ROUTE_NAMES[i]} "
                        f"({routes_p[i].length_m:.0f} ม.) · Regret สะสม {p['cum_regret'][last]:.2f}", unsafe_allow_html=True)
        st.markdown(f"เงาสีดำ: เส้นทางที่ดีที่สุดจริงในรอบ {last} = เส้นทาง {ROUTE_NAMES[int(out['oracle'][last])]}")

# --------------------------------------------------------------------------- ④ 30 seeds and Table 6
if step == STEPS[3]:
    df, curves, hyp = results()
    if df is None:
        st.warning("ยังไม่มีผล 30 Seed ให้รัน `python experiments.py run` และ `python experiments.py analyze` ก่อน")
        st.stop()
    if pair_id is None:
        st.info("คู่นี้ไม่อยู่ใน 5 คู่ที่บันทึกไว้สำหรับการทดลองหลัก (ข้อ 5.2.2) จึงไม่มีผล 30 Seed "
                "ดูผลรายรันได้ในหน้า ② และ ③ หรือเลือกคู่ที่มีเครื่องหมาย ★")
        for lab in exp_labels:
            st.markdown(f"- {lab}")
        st.stop()
    st.subheader(f"ผล 30 Seed: {SET_TH[set_name]} · {sc['label']} · σ = {noise_sd}")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for n in POLICY_COLORS:
        ck = f"{set_name}|{pair_id}|{sc_key}|{noise_sd}|{n}"
        if ck + "|mean" not in curves:
            continue
        xx = np.arange(len(curves[ck + "|mean"])) * 10
        keep = (xx >= r0) & (xx <= r1)
        ax.plot(xx[keep], curves[ck + "|mean"][keep], color=POLICY_COLORS[n], lw=2.2, label=POLICY_TH[n])
        ax.fill_between(xx[keep], curves[ck + "|lo"][keep], curves[ck + "|hi"][keep], color=POLICY_COLORS[n], alpha=0.15)
    mark_change(ax, sc)
    ax.set_xlabel("รอบ")
    ax.set_ylabel("Dynamic Cumulative Regret")
    ax.set_title("ค่าเฉลี่ยจาก 30 Seed (แถบสี = ช่วง 2.5–97.5 เปอร์เซ็นไทล์ระหว่าง Seed)")
    ax.legend(fontsize=12, frameon=False)
    st.pyplot(fig, clear_figure=True)

    d = df[(df["set"] == set_name) & (df["pair"] == pair_id) & (df["scenario"] == sc_key) & (df["noise_sd"] == noise_sd)]
    rows, tau_rows = [], []
    for n in POLICY_COLORS:
        g = d[d["policy"] == n]
        lo, hi = sim.paired_bootstrap_ci(g["dynamic_cumulative_regret"], 2000)[1:]
        row = {"นโยบาย": POLICY_TH[n], "Regret สะสม เฉลี่ย [95% CI]": f"{g['dynamic_cumulative_regret'].mean():.2f} [{lo:.2f}, {hi:.2f}]",
               "Average Reward": f"{g['average_reward'].mean():.3f}", "Crowd Exposure": f"{g['crowd_exposure'].mean():.3f}",
               "Route Length (ม.)": f"{g['route_length_m'].mean():.1f}", "เวลา (มิลลิวินาที/รอบ)": f"{g['ms_per_round'].mean():.3f}"}
        if sc["mode"] != "constant":
            tau = g["adaptation_time"].dropna().to_numpy(dtype=float)
            nr = int(g["adaptation_time"].isna().sum())
            if len(tau) >= 2:
                mt, lo_t, hi_t = sim.paired_bootstrap_ci(tau, 2000)
                row["Adaptation Time เฉลี่ย [95% CI] (รอบ)"] = f"{mt:.0f} [{lo_t:.0f}, {hi_t:.0f}]"
                tau_rows.append({"n": n, "mean": mt, "lo": lo_t, "hi": hi_t, "nr": nr, "k": len(g)})
            else:
                row["Adaptation Time เฉลี่ย [95% CI] (รอบ)"] = "–"
                tau_rows.append({"n": n, "mean": np.nan, "lo": np.nan, "hi": np.nan, "nr": nr, "k": len(g)})
            row["ไม่ฟื้นตัวภายในช่วงทดลอง"] = f"{nr}/{len(g)} ({100 * nr / len(g):.0f}%)"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if sc["mode"] != "constant":
        st.caption("Adaptation Time เฉลี่ยจากเฉพาะรันที่ฟื้นตัว (CI 95% ด้วย Bootstrap) และรายงานสัดส่วนที่ไม่ฟื้นตัวแยก "
                   "ตามข้อ 3.2 และตารางที่ 5 · ε = 0.05, W = 50 ตามเล่ม")
        # 5.1.3: comparison chart of Adaptation Time between policies
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ys = np.arange(len(tau_rows))
        for y, r in zip(ys, tau_rows):
            col = POLICY_COLORS[r["n"]]
            if not np.isnan(r["mean"]):
                ax.barh(y, r["mean"], color=col, alpha=0.8)
                ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], color="black", capsize=4)
            ax.text((0 if np.isnan(r["mean"]) else r["hi"]) + 3, y, f" ไม่ฟื้นตัว {r['nr']}/{r['k']}", va="center", fontsize=13)
        ax.set_yticks(ys, [POLICY_TH[r["n"]] for r in tau_rows])
        ax.invert_yaxis()
        ax.set_xlabel("Adaptation Time (รอบหลังจุดเปลี่ยน, ยิ่งน้อยยิ่งปรับตัวเร็ว)")
        ax.set_title("การเปรียบเทียบ Adaptation Time ของแต่ละนโยบาย (ค่าเฉลี่ย 30 Seed พร้อม 95% CI)")
        st.pyplot(fig, clear_figure=True)

    sens_file = RES / "eps_sensitivity.csv"
    if sc["mode"] != "constant" and sens_file.exists():
        with st.expander("การวิเคราะห์ความไว (sensitivity) ของเกณฑ์ ε ใน Adaptation Time · ไม่ใช่ผลหลัก"):
            sdf = pd.read_csv(sens_file)
            sdf = sdf[(sdf["set"] == set_name) & (sdf["scenario"] == sc_key)]
            piv = []
            for n in POLICY_COLORS:
                row = {"นโยบาย": POLICY_TH[n]}
                for eps in CFG["evaluation"]["sensitivity_eps"]:
                    g = sdf[(sdf["policy"] == n) & (sdf["eps"] == eps)]["adaptation_time"]
                    row[f"ε = {eps}"] = f"{g.mean():.0f} (ไม่ฟื้นตัว {100 * g.isna().mean():.0f}%)" if g.notna().any() else f"ไม่ฟื้นตัว {100 * g.isna().mean():.0f}%"
                piv.append(row)
            st.dataframe(pd.DataFrame(piv), hide_index=True, width="stretch")
            st.caption("รันบน Seed สอบเทียบ 1000–1009 ทั้ง 5 คู่ (ข้อ 5.7) แยกจากผลหลักที่ใช้ ε = 0.05 ตามตารางที่ 5 · "
                       f"ค่า {CFG['evaluation']['adaptation_window']} รอบคือค่าต่ำสุดที่วัดได้ (เท่ากับ W) จึงแปลว่าเกณฑ์นั้นหลวมจนแยกนโยบายไม่ออก")

    st.subheader("ผลตัดสินสมมติฐานตามตารางที่ 6 (σ = 0.1, Bootstrap แบบจับคู่ 10,000 ครั้ง)")
    h = hyp[set_name][str(pair_id)]
    yes = lambda b: "สนับสนุน" if b else "ไม่สนับสนุน"
    h3 = h["H3"]["by_scenario"]
    table = [
        {"สมมติฐาน": "H1 (S1)", "เกณฑ์": f"ขอบบน CI ของ R_LinUCB − R_dLinUCB(ฉบับปรับใช้) ≤ {h['H1']['threshold']:.2f} (ร้อยละ 5 ของ R_LinUCB)",
         "ผลที่ได้": f"ผลต่างเฉลี่ย {h['H1']['mean_d']:.2f} CI [{h['H1']['ci'][0]:.2f}, {h['H1']['ci'][1]:.2f}] · False Alarm Rate {h['H1']['false_alarm_rate_mean']:.2f}/1,000 รอบ",
         "ผลตัดสิน": yes(h["H1"]["supported"])},
        {"สมมติฐาน": "H2 (S2)", "เกณฑ์": "CI ของ Regret หลังจุดเปลี่ยน (LinUCB − dLinUCB ฉบับปรับใช้) > 0 ทั้งช่วง และมัธยฐาน τ ของ dLinUCB ฉบับปรับใช้น้อยกว่า",
         "ผลที่ได้": f"ผลต่างเฉลี่ย {h['H2']['mean_diff_after_change']:.2f} CI [{h['H2']['ci'][0]:.2f}, {h['H2']['ci'][1]:.2f}] · "
                     f"มัธยฐาน τ LinUCB {h['H2']['median_tau_linucb']} / dLinUCB ฉบับปรับใช้ {h['H2']['median_tau_dlinucb']} · ตรวจพบ {h['H2']['detected_runs']}/30 รัน",
         "ผลตัดสิน": yes(h["H2"]["supported"])},
        {"สมมติฐาน": "H3 ส่วนแรก (S1–S3)", "เกณฑ์": "CI ของ R_Fixed − R_LinUCB > 0 ทั้งช่วง ในทุกสถานการณ์",
         "ผลที่ได้": " · ".join(f"{s}: {h3[s]['mean_diff']:.2f} [{h3[s]['ci'][0]:.2f}, {h3[s]['ci'][1]:.2f}]" for s in ["S1", "S2", "S3"]),
         "ผลตัดสิน": yes(h["H3"]["supported_first_part"])},
        {"สมมติฐาน": "H3 ส่วนหลัง (กรณีควบคุม)", "เกณฑ์": "รายงานว่า Fixed ชนะหรือไม่ เมื่อ θ = w และรอบที่ LinUCB เริ่มชนะ",
         "ผลที่ได้": f"ผลต่าง {h3['CTRL']['mean_diff']:.2f} CI [{h3['CTRL']['ci'][0]:.2f}, {h3['CTRL']['ci'][1]:.2f}] · "
                     f"LinUCB ชนะตั้งแต่รอบ {h3['CTRL']['linucb_wins_from_round'] if h3['CTRL']['linucb_wins_from_round'] is not None else 'ไม่ชนะ'}",
         "ผลตัดสิน": "Fixed ชนะ" if h3["CTRL"]["fixed_better"] else ("LinUCB ชนะ" if h3["CTRL"]["learning_better"] else "ไม่ต่างกัน")},
    ]
    st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    st.caption("ตารางที่ 6 ระบุให้วิเคราะห์ H2 ตามขนาดการเปลี่ยนของ θ ด้วย แต่ตารางที่ 4 มีขนาดการเปลี่ยนเดียว จึงยังไม่ได้วิเคราะห์ส่วนนี้ "
               "ผลที่ไม่สนับสนุนสมมติฐานถือเป็นข้อค้นพบตามข้อ 5.9")

    st.markdown("**ภาพรวมทุกคู่ (σ = 0.1)**")
    over = []
    for pid, hh in hyp[set_name].items():
        over.append({"คู่": exp_labels[int(pid)], "H1": yes(hh["H1"]["supported"]), "H2": yes(hh["H2"]["supported"]),
                     "H3 ส่วนแรก": yes(hh["H3"]["supported_first_part"])})
    st.dataframe(pd.DataFrame(over), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- ⑤ conformance
if step == STEPS[4]:
    st.subheader("demo กับเอกสารเค้าโครง")
    cal_rule = CAL["theta_rule"]
    st.markdown(f"""
| เอกสาร | ใน demo |
|---|---|
| 3.1 ทำซ้ำได้ และทดสอบอัตโนมัติ 4 รายการ | Seed เดิมได้ลำดับทางเลือก Regret และตัวชี้วัดเท่าเดิมทุกค่า (ยกเว้น Computation Time ที่เป็นเวลาจริงของเครื่อง) · pytest 8 รายการ (4 รายการตามข้อ 3.1 + การทำซ้ำของตัวชี้วัด + ตารางที่ 4 + Adaptation Time + False Alarm Rate) |
| 3.2 4 นโยบาย × 3 สถานการณ์ × 30 Seed | ครบ และเพิ่มกรณีควบคุมตามตารางที่ 4 · รายงานค่าเฉลี่ยพร้อม 95% CI |
| 3.3 ตัดสิน H1–H3 ตามเกณฑ์ล่วงหน้า | หน้า ④ ใช้เกณฑ์ตามตารางที่ 6 และ Bootstrap จับคู่ 10,000 ครั้ง |
| 3.4, 5.1.3 แผนที่เส้นทางของ 4 นโยบาย | หน้า ③ จากผลการรันจริง |
| 5.1.2 ข้อมูลนำเข้าของต้นแบบ | แถบซ้าย: (ก) ต้นทางและปลายทางจากรายการสถานที่ที่เตรียมไว้ (เฉพาะคู่ที่ผ่านเกณฑ์ 3 เส้นของข้อ 5.2.2, เตรียมเส้นทางล่วงหน้าใน `data/route_catalog.json`) (ข) สถานการณ์ (ค) Seed (ง) ช่วงรอบ · ไม่รับพิกัดอิสระ · ผล 30 Seed มีเฉพาะ 5 คู่ที่บันทึกไว้สำหรับการทดลองหลัก (★) |
| 5.1.3–5.1.4 CSV, YAML, Matplotlib, Folium, pytest, Python 3.12+ | ดาวน์โหลด CSV/YAML ในหน้า ③ · กราฟ Matplotlib · แผนที่ Folium · Python 3.13 |
| 5.2.1 กราฟทางเดินเท้า ส่วนเชื่อมต่อใหญ่สุด เก็บเมตาดาตา (ขอบเขตพื้นที่ วันที่ดึง เวอร์ชัน) | `data/kku_walk.graphml` และ `data/meta.json` (มี bbox วันที่ และเวอร์ชัน OSMnx/NetworkX) |
| 5.2.2 Yen, K = 5, ตัดซ้อนทับ > 80%, ≥ 3 เส้น, บันทึกคู่ล่วงหน้า | `data/od_pairs.json` 5 คู่ (คู่หลัก + 4 คู่) เลือกด้วยระยะทางและการกระจายพื้นที่ ไม่ใช้ผลของนโยบาย |
| 5.3 ρ = ฐาน × g(t) + ε รายขอบ, Min–Max, 96 รอบ/วัน | ตามสูตร |
| 5.4 μ = xᵀθ, σ = 0.1, Feedback เฉพาะที่เลือก, Oracle ประเมินผลเท่านั้น | ตามสูตร และมีการทดสอบรองรับ |
| 5.5, ตารางที่ 3 นโยบาย 4 แบบ | Shortest Path · Fixed w = (−0.5, −0.5) · LinUCB · การปรับใช้ตามแนวคิด dLinUCB (Algorithm 1 ของ Wu et al., 2018) ใช้ชื่อนี้ตามข้อ 5.5 เพราะมีการดัดแปลง |
| 5.5, 5.7 สอบเทียบด้วย Seed แยก งบเท่ากัน | Seed 1000–1009 · 4 ชุดค่าต่อนโยบาย · ทดลองจริงใช้ Seed 0–29 |
| 5.6, ตารางที่ 4 | T = 2,000 · S1, S2 (รอบ 1,000), S3 (รอบ 750–1,250), กรณีควบคุม · Noise 0.05/0.1/0.2 เป็นส่วนเพิ่มเติม |
| 5.8, ตารางที่ 5 | ε = 0.05, W = 50 (ผลหลัก) · ความไวของ ε วิเคราะห์แยกบน Seed สอบเทียบ (หน้า ④) · Detection Delay · False Alarm Rate ต่อ 1,000 รอบ · Average Reward · Crowd Exposure · Route Length · Computation Time |
""")
    st.subheader("สิ่งที่ต้องอธิบายกรรมการ")
    st.markdown(f"""
- **ชุดค่าหลังสอบเทียบ:** เมื่อใช้ค่าตามตารางที่ 4 ทางที่สั้นที่สุดดีที่สุดเกือบทุกรอบ เพราะเส้นทางจากวิธีของ Yen ยาวใกล้กันมาก
  (คู่หลักต่างกันราว 12 เมตร) และการปรับ Min–Max ทำให้ความต่างนี้เต็มสเกล 0–1 จึงสอบเทียบน้ำหนักในตารางที่ 4 ตามข้อ 5.6
  ด้วยเกณฑ์ที่ดูเฉพาะตัวจำลอง ได้น้ำหนักความยาว **{CAL['sets']['calibrated']['length_weight']}** (เอกสาร 0.3)
- **เกณฑ์เลือกน้ำหนัก:** ค่าที่ใกล้ 0.3 ที่สุด ซึ่งใน 3 จาก 5 คู่ (1) ใน S1 ทางที่ดีที่สุดไม่ใช่ทางสั้นสุดอย่างน้อยร้อยละ 10 ของรอบ และ
  (2) ใน S2 สัดส่วนนี้ก่อนกับหลังจุดเปลี่ยนต่างกันอย่างน้อย 10 จุด เกณฑ์ข้อ (2) ถูกแก้หนึ่งครั้งก่อนดูผลของนโยบาย (บันทึกใน `config.yaml`)
- **การปรับใช้ตามแนวคิด dLinUCB:** ทำตาม Algorithm 1 (ตัวแบบย่อยหลายชุด, badness ในหน้าต่าง τ = 200 ตามงานต้นฉบับ, δ₁ δ₂ δ̃₁, เลือกด้วย LCB ของ badness)
  ส่วนที่ปรับใช้คือ การเลือกเส้นทางใช้ α ที่สอบเทียบเหมือน LinUCB ตามตารางที่ 3 ส่วนการทดสอบความผิดพลาดใช้ขอบความเชื่อมั่นตามสูตรต้นฉบับ
- **α ที่เลือกได้อยู่ที่ขอบล่างของช่วงที่ลอง (0.05)** ควรขยายช่วงในการทดลองจริง
- **ไม่ได้ทำ:** กรณีแนะนำสถานที่ (ทางเลือกเป็นคู่สถานที่–เส้นทาง) และการวิเคราะห์ H2 ตามขนาดการเปลี่ยนของ θ
""")
    with st.expander("ผลการสอบเทียบน้ำหนัก (เฉพาะตัวจำลอง ไม่ได้รันนโยบาย)"):
        st.dataframe(pd.DataFrame([{"น้ำหนักความยาว": r["length_weight"], "จำนวนคู่ที่ผ่าน": r["pairs_ok"]} for r in cal_rule]),
                     hide_index=True)
    with st.expander("ผลการสอบเทียบพารามิเตอร์ของนโยบาย (Seed 1000–1009)"):
        for sn, s in CAL["sets"].items():
            st.markdown(f"**{SET_TH[sn]}** (น้ำหนักความยาว {s['length_weight']})")
            for pn, rows in s["grid_results"].items():
                st.dataframe(pd.DataFrame([{"นโยบาย": pn, **r["params"], "Regret สะสมเฉลี่ย": r["mean_cum_regret"]} for r in rows]),
                             hide_index=True)
    bb = meta.get("bbox", {})
    st.caption(f"ข้อมูลกราฟดึงเมื่อ {meta['fetched_at']} · {meta['nodes']} จุด {meta['edges']} เส้น · OSMnx {meta['osmnx_version']} · "
               f"ขอบเขตพื้นที่ (ข้อ 5.2.1): ละติจูด {bb.get('south')}–{bb.get('north')}, ลองจิจูด {bb.get('west')}–{bb.get('east')} · {meta['attribution']}")
