import pyxdf
import numpy as np

xdf_path = "data/raw/session_01.xdf"
streams, header = pyxdf.load_xdf(xdf_path)

print("Loaded XDF OK")
print("Num streams:", len(streams))

def safe_get(info, key, default=""):
    try:
        return info.get(key, [default])[0]
    except Exception:
        return default

def show_markers(s):
    ts = s.get("time_stamps", [])
    data = s.get("time_series", [])
    if len(ts) == 0:
        return

    labels = []
    for row in data:
        if isinstance(row, (list, tuple, np.ndarray)) and len(row) > 0:
            labels.append(str(row[0]))
        else:
            labels.append(str(row))

    uniq = sorted(set(labels))
    print("  unique markers:", uniq[:50])
    if len(uniq) > 50:
        print("  ...", len(uniq), "total unique markers")

    print("  first 10 events:")
    for t, lab in list(zip(ts, labels))[:10]:
        print(f"    {float(t):.3f}  {lab}")

# (Optional) print stream names/types first
for s in streams:
    print(s["info"]["name"][0], "|", s["info"]["type"][0])

# Marker scan goes here (after load_xdf)
for s in streams:
    info = s["info"]
    name = safe_get(info, "name")
    stype = safe_get(info, "type")

    if ("marker" in name.lower()) or ("marker" in stype.lower()) or ("event" in name.lower()) or ("event" in stype.lower()):
        print("\nMARKER CANDIDATE:", name, "|", stype)
        show_markers(s)