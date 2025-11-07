import os, requests, datetime as dt

WORLDTIDES_KEY = os.getenv("WORLDTIDES_API_KEY")
LAT, LON = -12.9488, -38.3411  # Stella Maris

# Ajustes locais (afinados com sua comparação SurfGuru/Marinha)
OFFSETS_MIN = {"High": 32, "Low": 27}   # minutos
OFFSETS_M   = {"High": 1.40, "Low": 1.00}  # metros
ALTURA_OFFSET_SERIE = 1.20  # m (aplicado na série hora a hora)

def _parse_iso(s: str) -> dt.datetime:
    return dt.datetime.fromisoformat(s)

def _ajustar_extremo(e):
    t = _parse_iso(e["date"])
    t_corr = t + dt.timedelta(minutes=OFFSETS_MIN.get(e["type"], 30))
    h_corr = (e.get("height") or 0.0) + OFFSETS_M.get(e["type"], 1.20)
    return {"date": t_corr.isoformat(), "height": round(h_corr, 2), "type": e["type"]}

def _ajustar_heights(heights):
    return [
        {"date": h["date"], "height": round((h.get("height") or 0.0) + ALTURA_OFFSET_SERIE, 2)}
        for h in heights
    ]

def _proximo_extremo(extremes_adj, ref_time=None):
    if not ref_time:
        ref_time = dt.datetime.now(dt.timezone(dt.timedelta(hours=-3)))
    for e in sorted(extremes_adj, key=lambda x: x["date"]):
        if _parse_iso(e["date"]) >= ref_time:
            return e
    return extremes_adj[-1] if extremes_adj else None

def _altura_agora(heights_adj, ref_time=None):
    if not heights_adj:
        return None
    if ref_time is None:
        ref_time = dt.datetime.now(dt.timezone(dt.timedelta(hours=-3)))

    heights_sorted = sorted(heights_adj, key=lambda h: h["date"])
    before, after = None, None
    for h in heights_sorted:
        t = _parse_iso(h["date"])
        if t <= ref_time:
            before = h
        elif t > ref_time and not after:
            after = h
            break

    if not before or not after:
        nearest = min(heights_sorted, key=lambda h: abs(_parse_iso(h["date"]) - ref_time))
        return {"time": nearest["date"], "height_m": nearest["height"]}

    t1, h1 = _parse_iso(before["date"]), before["height"]
    t2, h2 = _parse_iso(after["date"]), after["height"]

    ratio = (ref_time - t1).total_seconds() / (t2 - t1).total_seconds()
    ratio = max(0.0, min(1.0, ratio))

    import math
    curve = (1 - math.cos(math.pi * ratio)) / 2
    altura_atual = h1 + (h2 - h1) * curve

    return {"time": ref_time.isoformat(), "height_m": round(altura_atual, 2)}

def get_tide_adjusted(day_offset: int = 0):
    """Obtém maré ajustada para o dia atual + offset (0 = hoje, 1 = amanhã, 2 = depois)."""
    if not WORLDTIDES_KEY:
        return {"error": "missing_worldtides_key"}

    base_date = dt.datetime.now(dt.timezone(dt.timedelta(hours=-3)))
    target_date = base_date + dt.timedelta(days=day_offset)

    # Dias futuros (WorldTides aceita até 7 dias)
    url = (
        f"https://www.worldtides.info/api/v3?"
        f"lat={LAT}&lon={LON}&heights&extremes&days=3"
        f"&start={int(target_date.timestamp())}&localtime&key={WORLDTIDES_KEY}"
    )

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"error": f"worldtides_request_failed: {e}"}

    extremes = data.get("extremes", [])
    heights  = data.get("heights", [])
    extremes_adj = [_ajustar_extremo(e) for e in extremes]
    heights_adj  = _ajustar_heights(heights)

    ref_time = target_date.replace(hour=12, minute=0, second=0)
    return {
        "provider": "worldtides+offset",
        "day_offset": day_offset,
        "extremes": extremes_adj,
        "heights": heights_adj,
        "next_extreme": _proximo_extremo(extremes_adj, ref_time),
        "now": _altura_agora(heights_adj, ref_time),
        "note": f"Ajuste local (Stella Maris) — maré simulada para {target_date.date().isoformat()}",
    }
