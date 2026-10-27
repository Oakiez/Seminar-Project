"""Prototype demo for the proposal "The Dynamic GIS Recommender" (KKU walking routes).

One page, read top to bottom: (1) pick a start and an end, (2) pick what happens to the simulated
user, (3) watch how each method chooses over 2,000 trips, (4) read a plain summary. The full
numbers for the committee (Table 5 metrics, 30-seed results, Table 6, conformance with the
proposal) sit in a section at the bottom.
Inputs follow proposal 5.1.2: OD pair from a prepared list, scenario S1-S3 (and the control case),
random seed and the range of rounds. The noise levels of 5.6 are the "additional" comparison.
Outputs follow 5.1.3.
Run:  .venv/Scripts/streamlit run app.py
"""
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

st.set_page_config(page_title="ระบบแนะนำเส้นทางเดินใน มข. ที่เรียนรู้ได้", layout="wide")

# --------------------------------------------------------------------------- fonts (Thai in Matplotlib)
for d in [Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/Windows/Fonts", Path("C:/Windows/Fonts")]:
    for f in ["THSarabunNew.ttf", "THSarabunNew Bold.ttf"]:
        if (d / f).exists():
            fm.fontManager.addfont(str(d / f))
plt.rcParams.update({"font.family": ["TH Sarabun New", "Tahoma", "DejaVu Sans"], "font.size": 16,
                     "axes.titlesize": 18, "axes.spines.top": False, "axes.spines.right": False})

CFG = sim.load_config()
CAL = yaml.safe_load((sim.DATA / "calibrated.yaml").read_text(encoding="utf-8"))
RES = sim.ROOT / CFG["evaluation"]["results_dir"]
T = CFG["experiment"]["T"]
RPD = CFG["context"]["rounds_per_day"]
ROUTE_COLORS = ["#009E73", "#E69F00", "#CC79A7", "#8C564B", "#17becf"]
POLICY_COLORS = {"Shortest Path": "#7f7f7f", "Fixed Context-Weighted Score": "#a6a61c",
                 "LinUCB": "#1f4e9c", "dLinUCB": "#d62728"}
# plain names first, the names used in the proposal (Table 3, 5.5) in brackets
POLICY_TH = {"Shortest Path": "ทางสั้นที่สุดเสมอ",
             "Fixed Context-Weighted Score": "สูตรคงที่",
             "LinUCB": "ระบบเรียนรู้ (LinUCB)",
             "dLinUCB": "ระบบเรียนรู้ที่คอยจับการเปลี่ยนใจ (การปรับใช้ตามแนวคิด dLinUCB)"}
POLICY_SHORT = {"Shortest Path": "ทางสั้นที่สุดเสมอ", "Fixed Context-Weighted Score": "สูตรคงที่",
                "LinUCB": "ระบบเรียนรู้", "dLinUCB": "ระบบเรียนรู้ที่คอยจับการเปลี่ยนใจ"}
POLICY_HOW = {"Shortest Path": "เลือกทางที่สั้นที่สุดทุกครั้ง ไม่สนใจว่าคนเยอะแค่ไหน",
              "Fixed Context-Weighted Score": "ให้คะแนนทุกทางด้วยสูตรที่ตั้งไว้ก่อน (ระยะทางกับความแออัดสำคัญเท่ากัน) แล้วเลือกทางที่คะแนนดีที่สุด ไม่เรียนรู้",
              "LinUCB": "เดาจากคะแนนที่ผู้ใช้ให้ว่าผู้ใช้ชอบทางแบบไหน และบางครั้งลองทางที่ยังไม่แน่ใจ",
              "dLinUCB": "เรียนรู้แบบเดียวกัน แต่คอยดูว่าคะแนนที่ได้ผิดจากที่เดาไว้บ่อยไหม ถ้าผิดบ่อย จะเริ่มเรียนรู้ใหม่"}
DL = "การปรับใช้ตามแนวคิด dLinUCB"   # proposal 5.5: name used when dLinUCB is adapted
SET_TH = {"doc": "ค่าตามตารางที่ 4 ในเอกสาร", "calibrated": "ค่าหลังสอบเทียบ (ข้อ 5.6)"}
SC_TH = {"S1": "ผู้ใช้ชอบแบบเดิมตลอด", "S2": "ผู้ใช้เปลี่ยนใจกะทันหัน", "S3": "ผู้ใช้ค่อยๆ เปลี่ยนใจ",
         "CTRL": "กรณีควบคุม: ผู้ใช้ชอบตรงกับสูตรคงที่พอดี"}
SC_DOC = {"S1": "S1", "S2": "S2", "S3": "S3", "CTRL": "กรณีควบคุม"}


# --------------------------------------------------------------------------- cached data
@st.cache_resource(show_spinner="กำลังโหลดทางเดินของ มข. ...")
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


@st.cache_data(show_spinner="กำลังจำลองการเดิน 2,000 เที่ยว ...")
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
    folium.Marker([o["lat"], o["lon"]], tooltip="เริ่มต้น: " + o["name"], icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([d["lat"], d["lon"]], tooltip="จุดหมาย: " + d["name"], icon=folium.Icon(color="red", icon="flag")).add_to(m)


def fit(m, lines):
    pts = [p for line in lines for p in line]
    m.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)], [max(p[0] for p in pts), max(p[1] for p in pts)]], padding=(20, 20))


def mark_change(ax, sc, label=True):
    if sc["mode"] == "abrupt":
        ax.axvline(sc["change_at"], color="black", ls="--", lw=1.2)
        if label:
            ax.text(sc["change_at"], ax.get_ylim()[1], " ผู้ใช้เปลี่ยนใจตรงนี้", va="top", fontsize=15)
    elif sc["mode"] == "gradual":
        ax.axvspan(sc["change_at"], sc["change_end"], color="gray", alpha=0.15)
        if label:
            ax.text(sc["change_at"], ax.get_ylim()[1], " ช่วงที่ผู้ใช้ค่อยๆ เปลี่ยนใจ", va="top", fontsize=15)


def fmt(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def route_names(routes):
    """Name each route by what it is like: length, and whether it is the quietest or busiest."""
    L = [r.length_m for r in routes]
    C = [r.base_crowd for r in routes]
    s, q, b = int(np.argmin(L)), int(np.argmin(C)), int(np.argmax(C))
    names = []
    for i, r in enumerate(routes):
        tags = []
        if i == s:
            tags.append("สั้นที่สุด")
        if i == q:
            tags.append("คนน้อยที่สุด")
        elif i == b:
            tags.append("คนเยอะที่สุด")
        names.append(f"ทางที่ {i + 1} ({r.length_m:.0f} ม.{' · ' + ' · '.join(tags) if tags else ''})")
    return names


def pref_text(theta):
    wl, wc = abs(theta[0]), abs(theta[1])
    pc = 100 * wc / (wl + wc)
    return f"สนใจเลี่ยงคนเยอะ {pc:.0f}% · สนใจทางสั้น {100 - pc:.0f}%"


# --------------------------------------------------------------------------- data and settings
G, D, base, cat_places, cat_pairs, exp_pairs, exp_routes, exp_index, background, meta = campus()
names = list(cat_places)
exp_labels = [f"{p['origin']} → {p['destination']}" for p in exp_pairs]
COLLEGE = "วิทยาลัยการคอมพิวเตอร์"
MAIN_D = exp_pairs[0]["destination"]

st.title("ระบบแนะนำเส้นทางเดินใน มข. ที่เรียนรู้จากผู้ใช้ได้")
st.markdown("#### ถ้าแอปนำทางไม่รู้ว่าคุณชอบทางแบบไหน มันจะเรียนรู้จากคะแนนที่คุณให้ได้ไหม และถ้าคุณเปลี่ยนใจ มันจะตามทันไหม")
st.markdown(
    "หน้านี้จำลองนักศึกษา 1 คนที่เดินทางเดิมซ้ำ 2,000 เที่ยว ทุกเที่ยวระบบแนะนำทางให้ 1 ทาง "
    "ผู้ใช้เดินแล้วให้คะแนนความพอใจกลับมา ระบบเห็นคะแนนของทางที่แนะนำไปเท่านั้น "
    "และไม่รู้ล่วงหน้าว่าผู้ใช้ชอบทางสั้นหรือทางที่คนน้อย เราเทียบ 4 วิธีแนะนำทางว่าวิธีไหนแนะนำได้ตรงใจกว่ากัน")
st.caption("จำนวนคนบนทางเดินและความชอบของผู้ใช้เป็นค่าจำลองทั้งหมด ตัวเลขเป็นผลนำร่อง ยังไม่ใช่ผลการทดลองฉบับสมบูรณ์ · "
           + meta["attribution"])

with st.expander("4 วิธีที่นำมาเทียบกัน"):
    for n in POLICY_COLORS:
        st.markdown(f"<span style='color:{POLICY_COLORS[n]};font-size:1.3em'>■</span> **{POLICY_TH[n]}**: {POLICY_HOW[n]}",
                    unsafe_allow_html=True)

# --------------------------------------------------------------------------- step 1: start and end
st.divider()
st.header("ขั้นที่ 1 · เลือกจุดเริ่มต้นและจุดหมาย")
c1, c2 = st.columns(2)
o_default = COLLEGE if COLLEGE in names else exp_pairs[0]["origin"]
o_name = c1.selectbox("เดินจาก", names, index=names.index(o_default))
dest_ok = [n for n in names if (o_name, n) in cat_pairs]
d_default = MAIN_D if MAIN_D in dest_ok else dest_ok[0]
d_name = c2.selectbox("ไปที่", dest_ok, index=dest_ok.index(d_default),
                      help="แสดงเฉพาะจุดหมายที่มีทางให้เลือกอย่างน้อย 3 ทางที่ไม่ทับกันเกินไป (ข้อ 5.2.2)")
pair_label = f"{o_name} → {d_name}"
pair_id = exp_index.get((o_name, d_name))
routes_p, lines_p = routes_for(o_name, d_name)
o_place, d_place = cat_places[o_name], cat_places[d_name]
K = len(routes_p)
RN = route_names(routes_p)
key = f"{names.index(o_name)}_{names.index(d_name)}"

# advanced inputs (5.1.2 (c)-(d), 5.6 noise): kept in an expander so they keep their values when closed
with st.expander("ตั้งค่าเพิ่มเติม (สำหรับกรรมการ ไม่ต้องแตะก็ได้)"):
    a1, a2, a3 = st.columns(3)
    set_name = a1.radio("ค่าความชอบของผู้ใช้", list(CAL["sets"]), index=list(CAL["sets"]).index("calibrated"),
                        format_func=lambda k: SET_TH[k],
                        help="ถ้าใช้ค่าตามตารางที่ 4 ตรงตัว ทางที่สั้นที่สุดจะดีที่สุดเกือบทุกเที่ยว เพราะทุกทางยาวใกล้กันมาก "
                             "จึงสอบเทียบค่าใหม่ตามข้อ 5.6 เหตุผลอยู่ในส่วนรายละเอียดด้านล่าง")
    seed = a2.selectbox("ชุดเลขสุ่ม (Seed)", list(range(CFG["evaluation"]["test_seeds"])), index=0,
                        help="Seed คือชุดตัวเลขสุ่ม ถ้าใช้ Seed เดิม ผลจะออกมาเหมือนเดิมทุกครั้ง เปลี่ยน Seed เพื่อดูว่าผลยังคล้ายเดิมไหม")
    noise_sd = a3.select_slider("คะแนนที่ผู้ใช้ให้แกว่งแค่ไหน (Noise σ)", CFG["reward"]["extra_noise_levels"],
                                value=CFG["reward"]["noise_sd"],
                                help="ผู้ใช้ให้คะแนนไม่ตรงเป๊ะทุกครั้ง ค่านี้ยิ่งมาก คะแนนยิ่งแกว่ง ระบบยิ่งเรียนรู้ยาก (ส่วนเพิ่มเติมของข้อ 5.6)")
    r0, r1 = st.slider("ช่วงเที่ยวที่ใช้ในตารางตัวเลขด้านล่าง (ข้อ 5.1.2 ง)", 0, T - 1, (0, T - 1), 10)
    show_tiles = st.checkbox("แสดงภาพแผนที่พื้นหลัง (ต้องใช้อินเทอร์เน็ต)", True)

a_w = CAL["sets"][set_name]["length_weight"]
if set_name == "doc":
    st.warning("ตอนนี้ใช้ค่าความชอบตามตารางที่ 4 ตรงตัว ทางที่สั้นที่สุดจะดีที่สุดเกือบทุกเที่ยว จึงแทบไม่มีอะไรให้ระบบเรียนรู้")

c1, c2 = st.columns([3, 2])
with c2:
    st.markdown(f"มี **{K} ทาง** ให้เลือก ทุกทางพาไปถึงจุดหมายเดียวกัน แต่ระยะทางและจำนวนคนต่างกัน")
    show_route = st.radio("ดูบนแผนที่", ["ทุกทาง"] + RN, label_visibility="collapsed")
    rows = []
    for i, r in enumerate(routes_p):
        rows.append({"ทาง": f"ทางที่ {i + 1}", "ระยะทาง (ม.)": round(r.length_m), "ยาวกว่าทางที่สั้นที่สุด (ม.)": round(r.length_m - routes_p[0].length_m),
                     "คนเยอะแค่ไหน ตอน 8 โมงเช้า": round(r.base_crowd * sim.daily_pattern(32), 2),
                     "ตอนบ่าย 2": round(r.base_crowd * sim.daily_pattern(56), 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("ความแออัด 0 = ว่าง 1 = แน่นที่สุดในมหาวิทยาลัย คนจะเยอะตอน 8 โมง เที่ยง และ 5 โมงเย็น (ค่าจำลองตามข้อ 5.3) · "
               "ทางทั้งหมดหามาด้วยวิธีหาทางที่สั้นที่สุด 5 อันดับแรกของ Yen แล้วตัดทางที่ทับกันเกิน 80% (ข้อ 5.2.2)")
with c1:
    m = base_map(background, show_tiles)
    for i, line in enumerate(lines_p):
        if show_route != "ทุกทาง" and RN[i] != show_route:
            continue
        folium.PolyLine(line, color=ROUTE_COLORS[i], weight=6, opacity=0.9, tooltip=RN[i]).add_to(m)
    add_od(m, o_place, d_place)
    fit(m, lines_p)
    st_folium(m, height=460, returned_objects=[], use_container_width=True, key=f"m1_{key}_{show_route}")
    st.markdown(" ".join(f"<span style='color:{ROUTE_COLORS[i]};font-size:1.3em'>━</span> {RN[i]}" for i in range(K)),
                unsafe_allow_html=True)

# --------------------------------------------------------------------------- step 2: what happens to the user
st.divider()
st.header("ขั้นที่ 2 · เลือกว่าผู้ใช้จะเป็นแบบไหน")
st.markdown("ระบบไม่รู้คำตอบของขั้นนี้ ต้องเดาเอาจากคะแนนที่ผู้ใช้ให้")
sc_keys = list(CFG["experiment"]["scenarios"])
sc_key = st.radio("ผู้ใช้", sc_keys, index=sc_keys.index("S1"), horizontal=True, label_visibility="collapsed",
                  format_func=lambda k: SC_TH[k])
out, metrics, sc = simulate(set_name, o_name, d_name, sc_key, seed, noise_sd)
pol, env, oracle = out["policies"], out["env"], out["oracle"]

t_start, t_end = sc["theta_start"], sc.get("theta_end", sc["theta_start"])
if sc["mode"] == "constant":
    st.info(f"ตลอด 2,000 เที่ยว ผู้ใช้ {pref_text(t_start)} ({SC_DOC[sc_key]} ในตารางที่ 4)")
elif sc["mode"] == "abrupt":
    st.info(f"เที่ยวที่ 1 ถึง {sc['change_at']:,}: ผู้ใช้ {pref_text(t_start)}  \n"
            f"ตั้งแต่เที่ยวที่ {sc['change_at']:,} ผู้ใช้เปลี่ยนใจทันที: {pref_text(t_end)} ({SC_DOC[sc_key]} ในตารางที่ 4)")
else:
    st.info(f"ช่วงแรก: ผู้ใช้ {pref_text(t_start)}  \n"
            f"ระหว่างเที่ยวที่ {sc['change_at']:,} ถึง {sc['change_end']:,} ผู้ใช้ค่อยๆ เปลี่ยนใจ จนเป็น: {pref_text(t_end)} ({SC_DOC[sc_key]} ในตารางที่ 4)")
fig, ax = plt.subplots(figsize=(11, 2.4))
th = np.abs(env.theta)
share = 100 * th[:, 1] / th.sum(axis=1)
ax.plot(np.arange(T), share, color="black", lw=2.2, label="สนใจเลี่ยงคนเยอะ")
ax.plot(np.arange(T), 100 - share, color="black", lw=1.4, ls="--", label="สนใจทางสั้น")
ax.set_ylim(0, 100)
ax.set_ylabel("%")
ax.set_xlabel("เที่ยวที่")
ax.set_title("สิ่งที่ผู้ใช้สนใจ ในแต่ละเที่ยว (ระบบมองไม่เห็นกราฟนี้)")
ax.legend(loc="center right", frameon=False, fontsize=14)
mark_change(ax, sc, label=False)
st.pyplot(fig, clear_figure=True)
st.caption(f"ทุกแบบ จำนวนคนบนทางเปลี่ยนไปตามเวลาของวันเหมือนกัน 1 เที่ยว = 15 นาทีในวันจำลอง ({RPD} เที่ยว = 1 วัน)")

# --------------------------------------------------------------------------- step 3: watch the run
st.divider()
st.header("ขั้นที่ 3 · ดูว่าแต่ละวิธีเลือกทางเป็นอย่างไร")
st.markdown("ทุกเที่ยว เราเทียบทางที่ระบบเลือก กับทางที่ผู้ใช้จะพอใจที่สุดในเที่ยวนั้น "
            "ถ้าเลือกไม่ตรง ผู้ใช้เสียคะแนนไปเท่ากับส่วนต่าง เรียกว่า **ความเสียดาย (Regret)** "
            "กราฟนี้รวมความเสียดายตั้งแต่เที่ยวแรก **เส้นยิ่งต่ำ ยิ่งเลือกทางได้ตรงใจผู้ใช้** "
            "ถ้าเส้นเริ่มแบน แปลว่าวิธีนั้นเลือกถูกเกือบทุกเที่ยวแล้ว")
trip = st.slider("เลื่อนเพื่อดูเที่ยวที่ต้องการ", 1, T, 1 if sc["mode"] == "constant" else sc["change_at"] + 50, 1)
t = trip - 1
fig, ax = plt.subplots(figsize=(11, 4.2))
x = np.arange(1, T + 1)
for n, p in pol.items():
    ax.plot(x, p["cum_regret"], color=POLICY_COLORS[n], lw=2.4, label=POLICY_TH[n])
mark_change(ax, sc)
ax.axvline(trip, color="#444", lw=1, alpha=0.6)
ax.set_xlabel("เที่ยวที่")
ax.set_ylabel("ความเสียดายรวม (คะแนน)")
ax.set_title("ความเสียดายรวม: เส้นยิ่งต่ำยิ่งดี · เส้นตั้งสีเทา = เที่ยวที่กำลังดู")
ax.legend(fontsize=14, frameon=False, loc="upper left")
st.pyplot(fig, clear_figure=True)
if np.array_equal(pol["Shortest Path"]["choice"], pol["Fixed Context-Weighted Score"]["choice"]):
    st.caption("มองไม่เห็นเส้นสีเทา เพราะซ้อนอยู่ใต้เส้นสูตรคงที่ สองวิธีนี้เลือกทางเดียวกันทุกเที่ยว")

best = int(oracle[t])
c1, c2 = st.columns([3, 2])
with c2:
    st.markdown(f"#### เที่ยวที่ {trip:,} · เวลา {sim.hour_label(t)} น.")
    st.markdown(f"ทางที่ผู้ใช้จะพอใจที่สุดในเที่ยวนี้ คือ **{RN[best]}**")
    rows = []
    for n, p in pol.items():
        a = int(p["choice"][t])
        rows.append({"วิธี": POLICY_SHORT[n], "เลือก": f"ทางที่ {a + 1}", "ตรงใจไหม": "ตรง ✓" if a == best else "ไม่ตรง",
                     "เสียดายเที่ยวนี้": round(float(p["regret"][t]), 3), "เสียดายรวมถึงตอนนี้": round(float(p["cum_regret"][t]), 2)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    show_pol = st.radio("ดูบนแผนที่ว่าวิธีไหนเลือกทางอะไร", list(pol), index=3, format_func=lambda n: POLICY_SHORT[n])
    det = [d for d in pol["dLinUCB"]["detections"] if d <= t]
    if det:
        st.caption("ระบบที่คอยจับการเปลี่ยนใจ เริ่มเรียนรู้ใหม่ไปแล้ว " + f"{len(det)} ครั้ง (เที่ยวที่ " + ", ".join(f"{d + 1:,}" for d in det) + ")")
with c1:
    a = int(pol[show_pol]["choice"][t])
    m = base_map(background, show_tiles)
    folium.PolyLine(lines_p[best], color="black", weight=12, opacity=0.3, tooltip="ทางที่ผู้ใช้จะพอใจที่สุด").add_to(m)
    folium.PolyLine(lines_p[a], color=ROUTE_COLORS[a], weight=6, opacity=1, tooltip=f"{POLICY_SHORT[show_pol]} เลือก {RN[a]}").add_to(m)
    add_od(m, o_place, d_place)
    fit(m, lines_p)
    st_folium(m, height=420, returned_objects=[], use_container_width=True, key=f"m3_{key}_{show_pol}_{t}")
    st.markdown(f"<span style='color:{ROUTE_COLORS[a]};font-size:1.3em'>━</span> เส้นสี = {POLICY_SHORT[show_pol]} เลือก {RN[a]} · "
                "<span style='color:#999;font-size:1.3em'>━</span> เงาสีเทาหนา = ทางที่ผู้ใช้จะพอใจที่สุด", unsafe_allow_html=True)

# --------------------------------------------------------------------------- step 4: summary
st.divider()
st.header("ขั้นที่ 4 · สรุปผลของการจำลองรอบนี้")
final = {n: float(p["cum_regret"][-1]) for n, p in pol.items()}
order = sorted(final, key=final.get)
hit_from = T - 500
hit = {n: 100 * float(np.mean(p["choice"][hit_from:] == oracle[hit_from:])) for n, p in pol.items()}
k1, k2, k3 = st.columns(3)
with k1, st.container(border=True):
    st.markdown("**วิธีไหนเสียดายน้อยที่สุด**")
    st.markdown(f"### {POLICY_SHORT[order[0]]}")
    st.markdown("ความเสียดายรวมทั้ง 2,000 เที่ยว (น้อยไปมาก)  \n" + "  \n".join(f"{i + 1}. {POLICY_SHORT[n]}: {final[n]:.2f}" for i, n in enumerate(order)))
with k2, st.container(border=True):
    st.markdown("**500 เที่ยวสุดท้าย เลือกตรงใจกี่ %**")
    top = max(hit, key=hit.get)
    st.markdown(f"### {POLICY_SHORT[top]} {hit[top]:.0f}%")
    st.markdown("  \n".join(f"{POLICY_SHORT[n]}: {hit[n]:.0f}%" for n in pol))
with k3, st.container(border=True):
    md = metrics["dLinUCB"]
    if sc["mode"] == "constant":
        fa = len(pol["dLinUCB"]["detections"])
        st.markdown("**ผู้ใช้ไม่เปลี่ยนใจ ระบบเตือนผิดไหม**")
        st.markdown(f"### เริ่มเรียนรู้ใหม่ {fa} ครั้ง")
        st.markdown("ผู้ใช้ไม่ได้เปลี่ยนใจเลย ถ้าเป็น 0 แปลว่าระบบไม่ตกใจเกินเหตุ ถ้ามากกว่า 0 คือเตือนผิด")
    else:
        st.markdown("**หลังผู้ใช้เปลี่ยนใจ ระบบตามทันไหม**")
        dd = md.get("detection_delay")
        tau_d, tau_l = md.get("adaptation_time"), metrics["LinUCB"].get("adaptation_time")
        if sc["mode"] == "abrupt":
            st.markdown("### " + ("ไม่พบการเปลี่ยนใจ" if dd is None else f"จับได้หลังเปลี่ยนใจ {dd} เที่ยว"))
        else:
            st.markdown(f"### เริ่มเรียนรู้ใหม่ {len(pol['dLinUCB']['detections'])} ครั้ง")
        tt = lambda v: "ยังไม่กลับมาดีภายใน 2,000 เที่ยว" if v is None else f"{v} เที่ยว"
        st.markdown(f"กลับมาเลือกได้ดีหลังเปลี่ยนใจ (Adaptation Time)  \n{POLICY_SHORT['dLinUCB']}: {tt(tau_d)}  \n{POLICY_SHORT['LinUCB']}: {tt(tau_l)}")
learn_best = min(final["LinUCB"], final["dLinUCB"]) < min(final["Shortest Path"], final["Fixed Context-Weighted Score"])
if learn_best:
    st.success("รอบนี้ วิธีที่เรียนรู้จากคะแนนของผู้ใช้ เลือกทางได้ตรงใจกว่าวิธีที่ไม่เรียนรู้")
else:
    st.warning("รอบนี้ วิธีที่ไม่เรียนรู้ยังทำได้ดีกว่า ผลแบบนี้เราก็รายงานตามจริง (ข้อ 5.9) "
               "เพราะเป็นข้อค้นพบว่าระบบเรียนรู้ยังมีข้อจำกัดในสถานการณ์นี้")
st.caption("นี่คือผลจากชุดเลขสุ่มชุดเดียว ผลเฉลี่ยจาก 30 ชุดและการตัดสินสมมติฐาน อยู่ในส่วนรายละเอียดด้านล่าง")

# --------------------------------------------------------------------------- committee details
st.divider()
st.header("รายละเอียดสำหรับกรรมการ")
PAGES = ["ตัวเลขทั้งหมดของรอบนี้", "ผลเฉลี่ย 30 ชุดเลขสุ่ม และสมมติฐาน H1–H3", "ระบบเรียนรู้ตัดสินใจอย่างไรในแต่ละเที่ยว",
         "demo นี้ตรงกับเอกสารข้อไหน"]
page = st.radio("เลือกหัวข้อ", ["ซ่อน"] + PAGES, horizontal=True, label_visibility="collapsed")
fig1 = sim.ROOT.parent / "Figure1_process.png"

# ---- metrics of this run (Table 5) and downloads (5.1.3)
if page == PAGES[0]:
    st.subheader(f"ตัวชี้วัดตามตารางที่ 5 · {SET_TH[set_name]} · {SC_DOC[sc_key]} · Seed {seed} · σ = {noise_sd}")
    if (r0, r1) != (0, T - 1):
        st.caption(f"ตารางนี้ใช้ทั้ง 2,000 เที่ยวตามตารางที่ 5 ช่วงเที่ยว {r0}–{r1} ที่ตั้งไว้ ใช้กับกราฟด้านล่าง")
    rows = []
    for n, mtr in metrics.items():
        row = {"วิธี": POLICY_TH[n], "Dynamic Cumulative Regret": fmt(mtr["dynamic_cumulative_regret"]),
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
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"Adaptation Time ใช้ ε = {CFG['evaluation']['adaptation_eps']} และ W = {CFG['evaluation']['adaptation_window']} ตามตารางที่ 5 · "
               "ใช้ Seed เดิม ทุกค่าจะเท่าเดิม (ข้อ 3.1) ยกเว้น Computation Time ที่วัดจากเวลาจริงของเครื่อง")

    c1, c2 = st.columns(2)
    xr = np.arange(r0, r1 + 1)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.plot(xr, env.theta[r0:r1 + 1, 0], label="θ ความยาว", color="black")
        ax.plot(xr, env.theta[r0:r1 + 1, 1], label="θ ความแออัด", color="black", ls="--")
        ax.plot(xr, env.g[r0:r1 + 1], label="g(t) ความแออัดตามเวลา", color="#ff7f0e", lw=0.8)
        ax.set_title("θ และ g(t) ในตัวจำลอง (นโยบายไม่เห็น)")
        ax.set_xlabel("รอบ")
        ax.legend(fontsize=12, frameon=False)
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

    log = []
    for n, p in pol.items():
        log.append(pd.DataFrame({"round": np.arange(T), "policy": n, "chosen_route": [i + 1 for i in p["choice"]],
                                 "reward": p["reward"], "regret": p["regret"], "cum_regret": p["cum_regret"],
                                 "oracle_route": [i + 1 for i in oracle]}))
    csv = pd.concat(log).to_csv(index=False).encode("utf-8-sig")
    run_cfg = {"set": set_name, "length_weight": a_w, "pair": pair_label, "scenario": sc_key, "scenario_definition": sc,
               "seed": seed, "noise_sd": noise_sd, "T": T, "policy_params": CAL["sets"][set_name]["chosen"],
               "routes": CFG["routes"], "context": CFG["context"], "evaluation": CFG["evaluation"]}
    d1, d2 = st.columns(2)
    d1.download_button("ดาวน์โหลดผลทุกรอบ (CSV)", csv, file_name=f"run_{set_name}_{key}_{sc_key}_s{seed}.csv", mime="text/csv")
    d2.download_button("ดาวน์โหลดค่าที่ใช้ในรอบนี้ (YAML)", yaml.safe_dump(run_cfg, allow_unicode=True, sort_keys=False),
                       file_name=f"run_{set_name}_{key}_{sc_key}_s{seed}.yaml", mime="text/yaml")

# ---- 30 seeds and Table 6
if page == PAGES[1]:
    df, curves, hyp = results()
    if df is None:
        st.warning("ยังไม่มีผล 30 Seed ให้รัน `python experiments.py run` และ `python experiments.py analyze` ก่อน")
        st.stop()
    st.markdown("ผล 30 Seed มีเฉพาะ 5 คู่ที่บันทึกไว้ล่วงหน้าสำหรับการทดลองหลัก (ข้อ 5.2.2)")
    pid = st.selectbox("คู่ต้นทาง–ปลายทาง", list(range(len(exp_pairs))), index=pair_id if pair_id is not None else 0,
                       format_func=lambda i: exp_labels[i] + (" (คู่หลัก)" if i == 0 else ""))
    st.subheader(f"ผล 30 Seed: {SET_TH[set_name]} · {SC_DOC[sc_key]} ({SC_TH[sc_key]}) · σ = {noise_sd}")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for n in POLICY_COLORS:
        ck = f"{set_name}|{pid}|{sc_key}|{noise_sd}|{n}"
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
    ax.legend(fontsize=13, frameon=False)
    st.pyplot(fig, clear_figure=True)

    d = df[(df["set"] == set_name) & (df["pair"] == pid) & (df["scenario"] == sc_key) & (df["noise_sd"] == noise_sd)]
    rows, tau_rows = [], []
    for n in POLICY_COLORS:
        g = d[d["policy"] == n]
        lo, hi = sim.paired_bootstrap_ci(g["dynamic_cumulative_regret"], 2000)[1:]
        row = {"วิธี": POLICY_TH[n], "Regret สะสม เฉลี่ย [95% CI]": f"{g['dynamic_cumulative_regret'].mean():.2f} [{lo:.2f}, {hi:.2f}]",
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
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ys = np.arange(len(tau_rows))
        for y, r in zip(ys, tau_rows):
            col = POLICY_COLORS[r["n"]]
            if not np.isnan(r["mean"]):
                ax.barh(y, r["mean"], color=col, alpha=0.8)
                ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], color="black", capsize=4)
            ax.text((0 if np.isnan(r["mean"]) else r["hi"]) + 3, y, f" ไม่ฟื้นตัว {r['nr']}/{r['k']}", va="center", fontsize=14)
        ax.set_yticks(ys, [POLICY_SHORT[r["n"]] for r in tau_rows])
        ax.invert_yaxis()
        ax.set_xlabel("Adaptation Time (รอบหลังจุดเปลี่ยน ยิ่งน้อยยิ่งปรับตัวเร็ว)")
        ax.set_title("เทียบ Adaptation Time ของแต่ละวิธี (ค่าเฉลี่ย 30 Seed พร้อม 95% CI)")
        st.pyplot(fig, clear_figure=True)

    sens_file = RES / "eps_sensitivity.csv"
    if sc["mode"] != "constant" and sens_file.exists():
        with st.expander("ความไวของเกณฑ์ ε ใน Adaptation Time (ไม่ใช่ผลหลัก)"):
            sdf = pd.read_csv(sens_file)
            sdf = sdf[(sdf["set"] == set_name) & (sdf["scenario"] == sc_key)]
            piv = []
            for n in POLICY_COLORS:
                row = {"วิธี": POLICY_TH[n]}
                for eps in CFG["evaluation"]["sensitivity_eps"]:
                    g = sdf[(sdf["policy"] == n) & (sdf["eps"] == eps)]["adaptation_time"]
                    row[f"ε = {eps}"] = f"{g.mean():.0f} (ไม่ฟื้นตัว {100 * g.isna().mean():.0f}%)" if g.notna().any() else f"ไม่ฟื้นตัว {100 * g.isna().mean():.0f}%"
                piv.append(row)
            st.dataframe(pd.DataFrame(piv), hide_index=True, width="stretch")
            st.caption("รันบน Seed สอบเทียบ 1000–1009 ทั้ง 5 คู่ (ข้อ 5.7) แยกจากผลหลักที่ใช้ ε = 0.05 ตามตารางที่ 5 · "
                       f"ค่า {CFG['evaluation']['adaptation_window']} รอบคือค่าต่ำสุดที่วัดได้ (เท่ากับ W) ถ้าได้ค่านี้ แปลว่าเกณฑ์หลวมจนแยกวิธีไม่ออก")

    st.subheader("ผลตัดสินสมมติฐานตามตารางที่ 6 (σ = 0.1, Bootstrap แบบจับคู่ 10,000 ครั้ง)")
    h = hyp[set_name][str(pid)]
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
    st.caption("ตารางที่ 6 ให้วิเคราะห์ H2 ตามขนาดการเปลี่ยนของ θ ด้วย แต่ตารางที่ 4 มีขนาดการเปลี่ยนแบบเดียว จึงยังไม่ได้วิเคราะห์ส่วนนี้ "
               "ผลที่ไม่สนับสนุนสมมติฐานถือเป็นข้อค้นพบตามข้อ 5.9")
    st.markdown("**ภาพรวมทุกคู่ (σ = 0.1)**")
    over = []
    for k_, hh in hyp[set_name].items():
        over.append({"คู่": exp_labels[int(k_)], "H1": yes(hh["H1"]["supported"]), "H2": yes(hh["H2"]["supported"]),
                     "H3 ส่วนแรก": yes(hh["H3"]["supported_first_part"])})
    st.dataframe(pd.DataFrame(over), hide_index=True, width="stretch")

# ---- one decision of a learning policy, in detail
if page == PAGES[2]:
    learner = st.radio("ดูการตัดสินใจของ", ["LinUCB", "dLinUCB"], horizontal=True, format_func=lambda x: POLICY_TH[x])
    P = pol[learner]
    a, est, bonus, mu_t = int(P["choice"][t]), P["est"][t], P["bonus"][t], env.mu[t]
    st.markdown(f"เที่ยวที่ {trip:,} (เลือกได้จากแถบเลื่อนในขั้นที่ 3) · เวลา {sim.hour_label(t)} น. · "
                f"ความชอบจริงของผู้ใช้ θ = ({env.theta[t][0]:.2f}, {env.theta[t][1]:.2f}) ระบบไม่เห็นค่านี้")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(8, 4.6))
        xs = np.arange(K)
        for i in range(K):
            ax.plot([i, i], [est[i], est[i] + bonus[i]], color=ROUTE_COLORS[i], lw=10 if i == a else 6, alpha=0.4, solid_capstyle="butt")
        ax.scatter(xs, est, s=[150 if i == a else 70 for i in range(K)], c=ROUTE_COLORS[:K], zorder=3, label="คะแนนที่ระบบเดาไว้")
        ax.scatter(xs, est + bonus, marker="_", s=500, c="#444", zorder=3, label="คะแนนที่เป็นไปได้สูงสุด (UCB)")
        ax.scatter(xs, mu_t, marker="D", s=70, facecolors="none", edgecolors="black", zorder=4, label="คะแนนจริง (ระบบไม่เห็น)")
        ax.annotate("เลือก", (a, est[a] + bonus[a]), xytext=(0, 18), textcoords="offset points", ha="center", arrowprops=dict(arrowstyle="->"))
        ax.set_xticks(xs, [f"ทางที่ {i + 1}" for i in range(K)])
        ax.set_ylabel("คะแนนความพอใจ (ยิ่งสูงยิ่งดี)")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.4), ncol=3, fontsize=13, frameon=False)
        st.pyplot(fig, clear_figure=True)
    with c2:
        st.markdown("ระบบเลือกทางที่ **คะแนนที่เป็นไปได้สูงสุด** มากที่สุด (Upper Confidence Bound) "
                    "แท่งยาวแปลว่าระบบยังไม่ค่อยรู้จักทางนั้น เลยยอมลองดู แท่งจะสั้นลงเมื่อลองทางนั้นบ่อยขึ้น")
        st.markdown(f"เที่ยวนี้เลือก **{RN[a]}** ได้คะแนนกลับมา {P['reward'][t]:.3f} · ความเสียดาย {P['regret'][t]:.3f}")
        st.caption("ระบบเห็นข้อมูลแค่ 2 อย่างต่อทาง: ความยาวที่ปรับเป็น 0–1 ในชุดทางของเที่ยวนั้น และความแออัด (ตารางที่ 2)")
        if learner == "dLinUCB":
            det = [x for x in P["detections"] if x <= t]
            st.write(f"{DL} สร้างแบบจำลองใหม่ไปแล้ว {len(det)} ครั้ง" + (f" (รอบ {', '.join(map(str, det))})" if det else ""))

# ---- conformance with the proposal
if page == PAGES[3]:
    cal_rule = CAL["theta_rule"]
    if fig1.exists():
        st.image(str(fig1), width=720, caption="ภาพที่ 1 ในเอกสาร: กระบวนการวิจัยและโครงสร้างของระบบทดลอง")
    st.markdown(f"""
| เอกสาร | ใน demo |
|---|---|
| 3.1 ทำซ้ำได้ และทดสอบอัตโนมัติ 4 รายการ | Seed เดิมได้ลำดับทางเลือก Regret และตัวชี้วัดเท่าเดิมทุกค่า (ยกเว้น Computation Time ที่เป็นเวลาจริงของเครื่อง) · pytest 8 รายการ (4 รายการตามข้อ 3.1 + การทำซ้ำของตัวชี้วัด + ตารางที่ 4 + Adaptation Time + False Alarm Rate) |
| 3.2 4 นโยบาย × 3 สถานการณ์ × 30 Seed | ครบ และเพิ่มกรณีควบคุมตามตารางที่ 4 · รายงานค่าเฉลี่ยพร้อม 95% CI |
| 3.3 ตัดสิน H1–H3 ตามเกณฑ์ล่วงหน้า | หัวข้อ "ผลเฉลี่ย 30 ชุดเลขสุ่ม" ใช้เกณฑ์ตามตารางที่ 6 และ Bootstrap จับคู่ 10,000 ครั้ง |
| 3.4, 5.1.3 แผนที่เส้นทางของ 4 นโยบาย | ขั้นที่ 3 เลือกดูทางที่แต่ละวิธีเลือกในเที่ยวใดก็ได้ จากผลการรันจริง |
| 5.1.2 ข้อมูลนำเข้าของต้นแบบ | (ก) ต้นทางและปลายทางจากรายการที่เตรียมไว้ในขั้นที่ 1 (เฉพาะคู่ที่ผ่านเกณฑ์ 3 เส้นของข้อ 5.2.2 เตรียมเส้นทางล่วงหน้าใน `data/route_catalog.json`) (ข) สถานการณ์ในขั้นที่ 2 (ค) Seed และ (ง) ช่วงรอบ ใน "ตั้งค่าเพิ่มเติม" · ไม่รับพิกัดอิสระ · ผล 30 Seed มีเฉพาะ 5 คู่ที่บันทึกไว้สำหรับการทดลองหลัก |
| 5.1.3–5.1.4 CSV, YAML, Matplotlib, Folium, pytest, Python 3.12+ | ดาวน์โหลด CSV/YAML ในหัวข้อ "ตัวเลขทั้งหมดของรอบนี้" · กราฟ Matplotlib · แผนที่ Folium · Python 3.13 |
| 5.2.1 กราฟทางเดินเท้า ส่วนเชื่อมต่อใหญ่สุด เก็บเมตาดาตา (ขอบเขตพื้นที่ วันที่ดึง เวอร์ชัน) | `data/kku_walk.graphml` และ `data/meta.json` (มี bbox วันที่ และเวอร์ชัน OSMnx/NetworkX) |
| 5.2.2 Yen, K = 5, ตัดซ้อนทับ > 80%, ≥ 3 เส้น, บันทึกคู่ล่วงหน้า | `data/od_pairs.json` 5 คู่ (คู่หลัก + 4 คู่) เลือกด้วยระยะทางและการกระจายพื้นที่ ไม่ใช้ผลของนโยบาย |
| 5.3, ตารางที่ 2 ρ = ฐาน × g(t) + ε รายขอบ, 96 รอบ/วัน | ตามสูตร · ความยาว: Min–Max ภายในชุดเส้นทางของรอบนั้น · ความแออัด: ค่าฐานหารด้วยค่าสูงสุดของทั้งกราฟ แล้วตัดให้อยู่ในช่วง 0–1 (ตารางที่ 2 เขียนว่า "ปรับให้อยู่ในช่วง 0–1") |
| 5.4 μ = xᵀθ, σ = 0.1, Feedback เฉพาะที่เลือก, Oracle ประเมินผลเท่านั้น | ตามสูตร และมีการทดสอบรองรับ · ในหน้าจอเรียก Feedback ว่า "คะแนนความพอใจ" และเรียกทางของ Oracle ว่า "ทางที่ผู้ใช้จะพอใจที่สุด" |
| 5.5, ตารางที่ 3 นโยบาย 4 แบบ | Shortest Path ("ทางสั้นที่สุดเสมอ") · Fixed w = (−0.5, −0.5) ("สูตรคงที่") · LinUCB · การปรับใช้ตามแนวคิด dLinUCB (Algorithm 1 ของ Wu et al., 2018) ใช้ชื่อนี้ตามข้อ 5.5 เพราะมีการดัดแปลง |
| 5.5, 5.7 สอบเทียบด้วย Seed แยก งบเท่ากัน | Seed 1000–1009 · 4 ชุดค่าต่อนโยบาย · ทดลองจริงใช้ Seed 0–29 |
| 5.6, ตารางที่ 4 | T = 2,000 · S1, S2 (รอบ 1,000), S3 (รอบ 750–1,250), กรณีควบคุม · Noise 0.05/0.1/0.2 เป็นส่วนเพิ่มเติม |
| 5.8, ตารางที่ 5 | ε = 0.05, W = 50 (ผลหลัก) · ความไวของ ε วิเคราะห์แยกบน Seed สอบเทียบ · Detection Delay · False Alarm Rate ต่อ 1,000 รอบ · Average Reward · Crowd Exposure · Route Length · Computation Time · Dynamic Cumulative Regret เรียกในหน้าจอว่า "ความเสียดายรวม" |
""")
    st.subheader("สิ่งที่ต้องอธิบายกรรมการ")
    st.markdown(f"""
- **ชุดค่าหลังสอบเทียบ:** เมื่อใช้ค่าตามตารางที่ 4 ทางที่สั้นที่สุดดีที่สุดเกือบทุกรอบ เพราะเส้นทางจากวิธีของ Yen ยาวใกล้กันมาก
  (คู่หลักต่างกันราว 12 เมตร) และการปรับ Min–Max ทำให้ความต่างนี้เต็มสเกล 0–1 จึงสอบเทียบน้ำหนักในตารางที่ 4 ตามข้อ 5.6
  ด้วยเกณฑ์ที่ดูเฉพาะตัวจำลอง ได้น้ำหนักความยาว **{CAL['sets']['calibrated']['length_weight']}** (เอกสาร 0.3)
- **เกณฑ์เลือกน้ำหนัก:** ค่าที่ใกล้ 0.3 ที่สุด ซึ่งใน 3 จาก 5 คู่ (1) ใน S1 ทางที่ดีที่สุดไม่ใช่ทางสั้นสุดอย่างน้อยร้อยละ 10 ของรอบ และ
  (2) ใน S2 สัดส่วนนี้ก่อนกับหลังจุดเปลี่ยนต่างกันอย่างน้อย 10 จุด เกณฑ์ข้อ (2) ถูกแก้หนึ่งครั้ง โดยดูจากสถิติของตัวจำลองบน Seed สอบเทียบเท่านั้น (บันทึกใน `config.yaml`)
- **การรันเบื้องต้นก่อนสอบเทียบ:** ก่อนกำหนดเกณฑ์และก่อนสอบเทียบ ผู้จัดทำได้รันทดลองแบบสำรวจด้วย Seed 0–2 ซึ่งอยู่ในชุด Seed ทดลองจริง
  และเห็นผลของนโยบายในคู่หลักกับคู่คณะเศรษฐศาสตร์ → SC. 09 ทั้งด้วย θ ตามตารางที่ 4 และ θ = (−0.1, −0.9) การแก้เกณฑ์จึงเกิดหลังเห็นผลชุดนี้
  ผลนำร่องนี้จึงไม่ได้ปลอดจากการเห็นผลล่วงหน้าโดยสมบูรณ์ ในการทดลองจริงควรกำหนดเกณฑ์และค่าทั้งหมดก่อนรันนโยบายใดๆ
  และ Git ของโปรเจกต์ไม่ได้บันทึกลำดับนี้ไว้ (การสอบเทียบกับผล 30 Seed อยู่ใน commit เดียวกัน)
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
