"""Download the Khon Kaen University walking network and named campus places from
OpenStreetMap once, and cache them under data/ so the demo runs without internet.

Run:  .venv/Scripts/python prep_data.py
"""
import json
import datetime as dt
from pathlib import Path

import osmnx as ox
import networkx as nx

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

KKU_CENTER = (16.4746, 102.8230)  # fallback only, used if the campus polygon cannot be geocoded
PLACE_TAGS = {
    "amenity": ["library", "cafe", "restaurant", "food_court", "canteen", "hospital",
                "bank", "post_office", "fast_food", "marketplace", "theatre"],
    "building": ["university", "dormitory", "stadium"],
    "leisure": ["stadium", "sports_centre", "park"],
}


def nearest_node(G, lat, lon):
    """Nearest graph node by great-circle distance (avoids the scikit-learn dependency)."""
    ids = list(G.nodes)
    ys = [G.nodes[n]["y"] for n in ids]
    xs = [G.nodes[n]["x"] for n in ids]
    d = ox.distance.great_circle(lat, lon, ys, xs)
    return ids[int(d.argmin())]


def campus_polygon():
    for q in ("Khon Kaen University", "มหาวิทยาลัยขอนแก่น"):
        try:
            gdf = ox.geocode_to_gdf(q)
            geom = gdf.geometry.iloc[0]
            if geom.geom_type in ("Polygon", "MultiPolygon") and geom.area > 1e-5:
                return geom, q
        except Exception as e:  # noqa: BLE001 - report and try the next query
            print("geocode failed for", q, "->", e)
    return None, None


def main():
    poly, source = campus_polygon()
    if poly is not None:
        print("campus polygon from geocoder query:", source)
        G = ox.graph_from_polygon(poly, network_type="walk", retain_all=False)
    else:
        print("using fallback point + 1800 m radius")
        G = ox.graph_from_point(KKU_CENTER, dist=1800, network_type="walk", retain_all=False)
    G = ox.truncate.largest_component(G, strongly=True)
    print("graph nodes", G.number_of_nodes(), "edges", G.number_of_edges())
    ox.save_graphml(G, DATA / "kku_walk.graphml")

    # named places inside the same area
    feats = ox.features_from_polygon(poly, PLACE_TAGS) if poly is not None else \
        ox.features_from_point(KKU_CENTER, PLACE_TAGS, dist=1800)
    feats = feats[feats["name"].notna()] if "name" in feats.columns else feats.iloc[0:0]
    places, seen = [], set()
    for _, row in feats.iterrows():
        name = str(row["name"]).strip()
        if not name or name in seen:
            continue
        c = row.geometry.centroid
        node = nearest_node(G, c.y, c.x)
        kind = next((f"{k}={row[k]}" for k in PLACE_TAGS if k in row and isinstance(row[k], str)), "")
        name_en = row["name:en"] if "name:en" in row and isinstance(row["name:en"], str) else ""
        places.append({"name": name, "name_en": name_en, "kind": kind,
                       "lat": c.y, "lon": c.x, "node": int(node)})
        seen.add(name)
    print("named places", len(places))

    (DATA / "places.json").write_text(json.dumps(places, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "fetched_at": dt.datetime.now().isoformat(timespec="seconds"),
        "area_source": source or f"point {KKU_CENTER} radius 1800 m",
        "osmnx_version": ox.__version__,
        "networkx_version": nx.__version__,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "attribution": "Map data © OpenStreetMap contributors (ODbL)",
        # proposal 5.2.1: record the area (ขอบเขตพื้นที่) with the fetch date and tool versions
        "bbox": {"south": round(min(d["y"] for _, d in G.nodes(data=True)), 6),
                 "west": round(min(d["x"] for _, d in G.nodes(data=True)), 6),
                 "north": round(max(d["y"] for _, d in G.nodes(data=True)), 6),
                 "east": round(max(d["x"] for _, d in G.nodes(data=True)), 6),
                 "note": "extent of the walking-graph nodes (WGS84); area queried = OpenStreetMap polygon of \"Khon Kaen University\""},
    }
    (DATA / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
