from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os, time, datetime, httpx
from typing import Any, Dict, Optional, List, Any as TypingAny
from services.tide_worldtides import get_tide_adjusted

# ============== Config & App ==============
load_dotenv()

app = Flask(__name__)
CORS(app)

LAT = float(os.getenv("LAT", "-12.9437"))
LON = float(os.getenv("LON", "-38.3539"))

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") or ""
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or ""
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

print("DEBUG - GEMINI_MODEL carregado do .env:", GEMINI_MODEL)
print("DEBUG - OPENWEATHER_API_KEY set:", bool(OPENWEATHER_API_KEY))
print("DEBUG - GEMINI_API_KEY set:", bool(GEMINI_API_KEY))

# ============== Cache simples (TTL) ==============
class TTLCache:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl = ttl_seconds
        self.store: Dict[str, Any] = {}

    def get(self, key: str) -> Optional[TypingAny]:
        item = self.store.get(key)
        if not item:
            return None
        ts, value = item
        if time.time() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: TypingAny) -> None:
        self.store[key] = (time.time(), value)

cache = TTLCache(ttl_seconds=180)

# ============== Helpers ==============
def parse_iso_list(iso_list: List[str]) -> List[datetime.datetime]:
    return [datetime.datetime.fromisoformat(t) for t in iso_list]

def nearest_index(times: List[datetime.datetime], now: Optional[datetime.datetime] = None) -> int:
    if not times:
        return 0
    if now is None:
        now = datetime.datetime.now()
    return min(range(len(times)), key=lambda i: abs(times[i] - now))

def safe_avg(values):
    valid = [v for v in values if isinstance(v, (int, float)) and v is not None]
    return sum(valid) / len(valid) if valid else 0

# 🟩 ============== Ajustes de percepção ==============
def classify_perceived_size(height_m: float, period_s: float) -> Dict[str, str]:
    """Estima o tamanho percebido com leve viés para baixo (em metros)."""
    h = float(height_m or 0)
    p = float(period_s or 0)
    energia = h * p

    if energia < 6:
        h_eff = h * 0.7
    elif energia < 10:
        h_eff = h * 0.8
    elif energia < 14:
        h_eff = h * 0.9
    else:
        h_eff = h * 0.95

    return {
        "height_effective_m": f"{h_eff:.2f}",
        "energy": f"{energia:.1f}",
    }

def format_next_tide_peak(next_extreme: Optional[Dict[str, Any]]) -> str:
    if not next_extreme:
        return ""
    tipo = (next_extreme.get("type") or "").lower()
    data = next_extreme.get("date") or ""
    try:
        hora = datetime.datetime.fromisoformat(data).strftime("%H:%M")
    except Exception:
        hora = data[11:16] if len(data) >= 16 else ""
    if tipo == "high":
        return f"Maré toda cheia às {hora}."
    elif tipo == "low":
        return f"Maré toda seca às {hora}."
    return ""

def wind_trend_summary(series: List[Dict[str, Any]]) -> str:
    if not series:
        return ""
    now = datetime.datetime.now()
    curr = min(series, key=lambda p: abs(datetime.datetime.fromisoformat(p["time"]) - now))
    target = now + datetime.timedelta(hours=6)
    fut = min(series, key=lambda p: abs(datetime.datetime.fromisoformat(p["time"]) - target))

    ws_now = curr.get("wind_speed_kmh")
    ws_fut = fut.get("wind_speed_kmh")
    wd_now = curr.get("wind_dir_deg") or curr.get("wind_wave_direction_deg")
    wd_fut = fut.get("wind_dir_deg") or fut.get("wind_wave_direction_deg")

    def dir_text(deg):
        if deg is None:
            return "indef."
        deg = float(deg) % 360
        nomes = ["N","NE","E","SE","S","SW","W","NW"]
        return nomes[int((deg + 22.5)//45) % 8]

    partes = []
    if isinstance(ws_now, (int, float)) and isinstance(ws_fut, (int, float)):
        delta = ws_fut - ws_now
        if abs(delta) >= 6:
            partes.append(("aumenta" if delta > 0 else "diminui") + f" cerca de {abs(delta):.0f} km/h")
    if wd_now and wd_fut and dir_text(wd_now) != dir_text(wd_fut):
        partes.append(f"vira de {dir_text(wd_now)} para {dir_text(wd_fut)}")
    return "Vento nas próximas horas: " + ", ".join(partes) + "." if partes else ""

# ============== APIs externas ==============
def fetch_open_meteo(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    key = f"openmeteo_marine:{lat:.4f},{lon:.4f}"
    cached = cache.get(key)
    if cached:
        return cached
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height,wave_period,wave_direction,wind_wave_height,wind_wave_period,wind_wave_direction",
        "length_unit": "metric",
        "timezone": "auto",
    }
    try:
        r = httpx.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        cache.set(key, data)
        return data
    except Exception as e:
        print("DEBUG Open-Meteo Marine error:", e)
        return None

def fetch_open_meteo_wind(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    key = f"openmeteo_wind:{lat:.4f},{lon:.4f}"
    cached = cache.get(key)
    if cached:
        return cached
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "windspeed_unit": "kmh",
        "timezone": "auto",
    }
    try:
        r = httpx.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        cache.set(key, data)
        return data
    except Exception as e:
        print("DEBUG Open-Meteo Wind error:", e)
        return None

def pick_open_meteo_point(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        times_iso = data["hourly"]["time"]
        times = parse_iso_list(times_iso)
        idx = nearest_index(times)
        return {
            "time": times_iso[idx],
            "wave_height_m": data["hourly"]["wave_height"][idx],
            "wave_period_s": data["hourly"]["wave_period"][idx],
            "wave_direction_deg": data["hourly"]["wave_direction"][idx],
        }
    except Exception:
        return None

def fetch_openweather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    if not OPENWEATHER_API_KEY:
        print("DEBUG OpenWeather: nenhuma API_KEY configurada")
        return None
    key = f"openweather:{lat:.4f},{lon:.4f}"
    cached = cache.get(key)
    if cached:
        return cached
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric"}
    try:
        r = httpx.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        cache.set(key, data)
        return data
    except Exception as e:
        print("DEBUG OpenWeather error:", e)
        return None

def pick_openweather_now(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        wind_ms = data.get("wind", {}).get("speed")
        wind_deg = data.get("wind", {}).get("deg")
        return {
            "wind_speed_kmh": round(float(wind_ms) * 3.6, 1) if isinstance(wind_ms, (int, float)) else None,
            "wind_direction_deg": wind_deg,
            "source": "openweather",
        }
    except Exception as e:
        print("DEBUG pick_openweather_now error:", e)
        return None

# ============== Forecast builder ==============
def build_forecast_series(om_raw, wind_raw=None):
    try:
        h = om_raw.get("hourly", {})
        times = h.get("time", [])
        heights = h.get("wave_height", [])
        periods = h.get("wave_period", [])
        dirs = h.get("wave_direction", [])
        wind_map = {}
        if wind_raw:
            hw = wind_raw.get("hourly", {})
            wt = hw.get("time", [])
            wsp = hw.get("wind_speed_10m", [])
            wdg = hw.get("wind_direction_10m", [])
            for i in range(min(len(wt), len(wsp), len(wdg))):
                wind_map[wt[i]] = (wsp[i], wdg[i])

        series = []
        for i, t in enumerate(times):
            spd, dir10 = wind_map.get(t, (None, None))
            altura = heights[i] if i < len(heights) else None
            periodo = periods[i] if i < len(periods) else None
            energia = altura * periodo if altura and periodo else None
            energia_level = "Baixa" if energia and energia <= 5 else "Média" if energia and energia <= 12 else "Alta" if energia else None
            series.append({
                "time": t,
                "wave_height_m": altura,
                "wave_period_s": periodo,
                "wave_direction_deg": dirs[i] if i < len(dirs) else None,
                "wind_speed_kmh": spd,
                "wind_dir_deg": dir10,
                "energy": round(energia, 1) if energia else None,
                "energy_level": energia_level,
            })
        return series
    except Exception as e:
        print("DEBUG build_forecast_series error:", e)
        return []

# ============== Gemini prompt ==============
def call_gemini_http(prompt: str, model_name: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    r = httpx.post(url, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["candidates"][0]["content"]["parts"][0]["text"]

def explain_with_gemini(
    level: str,
    merged: Dict[str, Any],
    name: str = "Surfista",
    stance: str = "",
    experience_months: int = 0,
    day_label: str = "hoje",
) -> str:
    # 🧮 Converter meses em formato legível (anos + meses)
    years = experience_months // 12
    months = experience_months % 12

    if years > 0 and months > 0:
        experience_text = f"{years} ano(s) e {months} mês(es)"
    elif years > 0:
        experience_text = f"{years} ano(s)"
    elif months > 0:
        experience_text = f"{months} mês(es)"
    else:
        experience_text = "menos de um mês"

    # 🦶 Texto da base
    if stance == "goofy":
        stance_text = (
            "base goofy (pé direito na frente), "
            "ficando de frente (frontside) nas esquerdas e de costas (backside) nas direitas."
        )
    elif stance == "regular":
        stance_text = (
            "base regular (pé esquerdo na frente), "
            "ficando de frente (frontside) nas direitas e de costas (backside) nas esquerdas."
        )
    else:
        stance_text = "base de surf não informada."

    # 💬 Prompt principal 
    prompt = f"""
IMPORTANTE: escreva texto puro em português do Brasil, sem usar markdown, asteriscos ou símbolos especiais.
Seja curto e direto (2 a 3 parágrafos). Adapte a linguagem para o nível indicado: {level}.
Evite ordens absolutas; prefira linguagem de orientação e avaliação de risco.
Se precisar usar termos técnicos para iniciantes, explique rapidamente o significado e siga simples.

CONTEXTO DO DIA:
- Esta explicação deve se referir às condições de {day_label}. Se {day_label} não for hoje, descreva de forma prospectiva (ex.: “amanhã tende a...”).

📌 Instruções de personalização:
- Sempre comece a explicação citando o surfista de forma natural, por exemplo:
  "Olá {name}, como surfista {stance_text}"
- Use essas informações para ajustar o tom e o foco das recomendações.
- Para iniciantes, explique de forma acessível; para avançados, use termos mais técnicos e objetivos.
- Não repita o nome em todas as frases — apenas no início.
- Sempre mencione se favorece as direitas ou esquerdas com base na direção do swell.
- Saiba sempre  que se a pessoa for goofy a onda para esquerda vai ser frontside e para direita backside, e vice-versa para regular( direita frontside e esquerda backside). Mas isso serve de conhecimento para melhorar a explicação e não necessáriamente precisa explicar explicitamente isso na resposta.
DADOS DO SURFISTA:
- Nome: {name}
- Experiência: {experience_text}
- Base (stance): {stance_text}

REGRAS CRÍTICAS DE LEITURA:
- A leitura do tamanho DEVE ser conservadora ("down-bias"). Em dia comum as ondas aparentam menores; só passa de "um metrão" com energia alta.
- Use esta escala textual para tamanho percebido, quando aplicável: meio metrinho, meio metro, meio metrão, um metrinho, um metro, um metrão.
- Se houver 'Resumo pico de maré' use a frase exatamente como foi fornecida.
- Se houver 'Tendência do vento (6h)' inclua-a ao final da Análise geral em uma frase.

Conhecimento local (Stella Maris, Salvador/BA):
- Fundos: areia com alguns corais; picos conhecidos como Padang/Loro e pico da Corrente.
- Com swell de sul ou sudeste, a maioria das ondas tende a ser para a direita; a corrente costuma puxar da direita para a esquerda (olhando do areal).
- Com swell de leste, há mais esquerdas; a corrente costuma puxar da esquerda para a direita.
- Vento terral (offshore) deixa o mar mais liso; vento lateral pode ajudar ou atrapalhar; maral (onshore) deixa o mar mexido se estiver forte.
- Maré: cheia costuma deixar ondas gordas/fechando mais na beira; seca deixa ondas rápidas/fechando mais no fundo; meia-tide costuma funcionar melhor em Stella.
- A interação entre vento, maré e swell muda a leitura.

Maré e sua influência:
- Maré atual (m): {merged.get("tide", {}).get("now", {}).get("height_m", "não informada")}
- Próximo pico de maré: {merged.get("tide", {}).get("next_extreme", {}).get("type", "não informado")} em {merged.get("tide", {}).get("next_extreme", {}).get("date", "sem dados")}

Estrutura da saída (sem símbolos; use exatamente estes subtítulos seguidos de dois pontos):

Análise geral:
Descreva como está o mar agora, considerando altura, período, energia percebida e vento (fraco, moderado, forte), se está liso ou mexido, e a tendência do swell. Relacione também o efeito da maré atual. Se houver tendência de vento para 6h, inclua nesta seção ao final.

Impacto para surfistas do nível {level}:
Explique o que esse cenário significa para esse nível: facilidade/dificuldade, se é bom para treinar, se a direção do swell favorece direitas ou esquerdas, e como a maré influencia para esse nível.  
Use também a base ({stance_text}) para comentar se as ondas estarão de frente (frontside) ou de costas (backside).

Recomendação final:
Faça um resumo curto e útil para a decisão de entrar ou não, considerando também se a maré e o vento podem melhorar ou piorar nas próximas horas.

Segurança e observações:
Para iniciante, sempre inclua cuidados práticos (correnteza lateral, séries maiores do que parecem, fundo, atenção ao cansaço). Para intermediário/avançado, inclua apenas se houver riscos relevantes (vento muito forte, correnteza intensa, energia alta, coral exposto).

Dados atuais (use e considere o tamanho percebido):
Altura prevista (m): {merged.get("wave_height_m")}
Período (s): {merged.get("wave_period_s")}
Energia estimada (altura x período): {(merged.get("wave_height_m") or 0) * (merged.get("wave_period_s") or 0)}
Energia (nível): {merged.get("energy_level")}
Tamanho percebido (texto): {merged.get("perceived", {}).get("label", "")}
Altura percebida (m): {merged.get("perceived", {}).get("height_effective_m", "")}
Vento (km/h) e direção (graus): {merged.get("wind_speed_kmh")} / {merged.get("wind_direction_deg")}
Direção do swell (graus): {merged.get("wave_direction_deg")}
Resumo pico de maré: {merged.get("tide_peak_text","")}
Tendência do vento (6h): {merged.get("wind_trend_text","")}

IMPORTANTE:
Na seção final da análise (após "Recomendação final"), inclua um pequeno trecho que diga se o mar deve subir ou cair ao longo do dia, se o vento tende a entrar ou não, e se existe uma boa janela de maré/vento para surfar.
"""
    try:
        return call_gemini_http(prompt, GEMINI_MODEL)
    except Exception as e:
        return f"Erro ao usar Gemini: {e}"


# ============== API principal ==============
@app.get("/api/explain")
def api_explain():
    level = (request.args.get("level") or "iniciante").lower()
    ai_mode = (request.args.get("ai") or "off").lower()
    day_offset = int(request.args.get("day", 0))
    # depois de ler day_offset:
    day_label = "hoje" if day_offset == 0 else "amanhã" if day_offset == 1 else "depois de amanhã"


    # 🧩 Novos dados do perfil do surfista (vindos da URL)
    name = request.args.get("name", "Surfista")  # nome do surfista
    stance = request.args.get("stance", "").lower()  # goofy ou regular
    experience_months = int(request.args.get("experience_months", 0))  # experiência em meses

    # 🔹 Dados brutos
    om_raw = fetch_open_meteo(LAT, LON)
    wind_raw = fetch_open_meteo_wind(LAT, LON)
    forecast_series = build_forecast_series(om_raw, wind_raw) if om_raw else []

    # 🔹 Ponto atual (para o cartão de condições)
    om_point = pick_open_meteo_point(om_raw) if om_raw else None
    ow_raw = fetch_openweather(LAT, LON)
    ow_now = pick_openweather_now(ow_raw) if ow_raw else None
    merged_now = {**(om_point or {}), **(ow_now or {})}

    # aliases compatíveis com seu OceanDataCard
    if "wave_direction_deg" in merged_now and "wave_dir_deg" not in merged_now:
        merged_now["wave_dir_deg"] = merged_now["wave_direction_deg"]
    if "wind_direction_deg" in merged_now and "wind_dir_deg" not in merged_now:
        merged_now["wind_dir_deg"] = merged_now["wind_direction_deg"]

    # 🔹 Dia selecionado (para gráficos e IA)
    selected_point = None
    if forecast_series:
        today = datetime.datetime.now().date()
        target_day = today + datetime.timedelta(days=day_offset)
        same_day_points = [
            p for p in forecast_series
            if datetime.datetime.fromisoformat(p["time"]).date() == target_day
        ]
        if same_day_points:
            avg_altura = safe_avg([p.get("wave_height_m") for p in same_day_points])
            avg_periodo = safe_avg([p.get("wave_period_s") for p in same_day_points])
            avg_vento = safe_avg([p.get("wind_speed_kmh") for p in same_day_points])
            avg_dir = safe_avg([p.get("wind_dir_deg") for p in same_day_points])
            selected_point = {
                "wave_height_m": round(avg_altura, 2),
                "wave_period_s": round(avg_periodo, 1),
                "wave_direction_deg": avg_dir,
                "wind_speed_kmh": round(avg_vento, 1),
                "wind_direction_deg": avg_dir,
            }

    # 🔹 Maré e tendências
    tide_raw = get_tide_adjusted(day_offset)
    tide_processed = {
        "extremes": tide_raw.get("extremes"),
        "heights": tide_raw.get("heights"),
        "now": tide_raw.get("now"),
        "next_extreme": tide_raw.get("next_extreme"),
    }
    tide_peak_text = format_next_tide_peak(tide_processed.get("next_extreme"))
    wind_trend = wind_trend_summary(forecast_series)

        # 🔹 Se o ponto atual estiver incompleto, usa o ponto médio do dia
    fonte_principal = merged_now
    if not merged_now.get("wave_height_m") or not merged_now.get("wave_period_s"):
        fonte_principal = selected_point or merged_now

    altura = fonte_principal.get("wave_height_m")
    periodo = fonte_principal.get("wave_period_s")
    direcao = fonte_principal.get("wave_direction_deg")

    # 🔹 Calcula energia (altura x período)
    if fonte_principal.get("wave_height_m") and fonte_principal.get("wave_period_s"):
        energia = fonte_principal["wave_height_m"] * fonte_principal["wave_period_s"]
        merged_now["energy"] = round(energia, 1)
        merged_now["energy_level"] = (
        "Baixa" if energia <= 5 else
        "Média" if energia <= 12 else
        "Alta"
    )
    else:
        merged_now["energy"] = None
        merged_now["energy_level"] = None
    
    # 🔹 Garante que a direção do swell nunca venha vazia
    if direcao:
        merged_now["wave_direction_deg"] = direcao
        merged_now["wave_dir_deg"] = direcao

    # 🔹 Mantém a parte da maré normalmente
    merged_now["tide"] = {
        "now": tide_processed.get("now", {}),
        "next_extreme": tide_processed.get("next_extreme", {}),
    }


    # 🔹 Percepção e pacote do dia (para IA)
    perceived = classify_perceived_size(
        (selected_point or merged_now).get("wave_height_m"),
        (selected_point or merged_now).get("wave_period_s")
    )

    merged_day = {
        **(selected_point or {}),
        "tide": tide_processed,
        "tide_peak_text": tide_peak_text,
        "wind_trend_text": wind_trend,
        "perceived": perceived,
    }
    # 🧾 DEBUG — mostra o perfil recebido do frontend
    print("=== DEBUG PERFIL RECEBIDO ===")
    print("Nome:", name)
    print("Base (stance):", stance)
    print("Experiência (meses):", experience_months)
    print("=============================")

    # 🧠 IA (Gemini) agora usa nome, base e experiência do surfista
    explanation_pt = ""
    if ai_mode == "on" and GEMINI_API_KEY:
        explanation_pt = explain_with_gemini(
            level,
            merged_day,
            name=name,
            stance=stance,
            experience_months=experience_months,
            day_label=day_label   
        )
    return jsonify({
        "spot": "Stella Maris, Salvador-BA",
        "level": level,
        "day": day_offset,
        "forecast_now": merged_now,         # agora -> OceanDataCard
        "forecast_series": forecast_series, # séries -> gráficos
        "forecast_day": merged_day,         # resumo do dia -> IA
        "explanation_pt": explanation_pt,
    })
# --- Mantém a instância acordada (ping leve) ---
@app.get("/health")
def health():
    # nada externo; só indica que o serviço está de pé
    return {"ok": True, "service": "ExplicaSurf Backend"}

# --- Pré-aquecimento de cache (opcional) ---
@app.get("/warmup")
def warmup():
    """
    Chama fontes baratas (Open-Meteo + vento + maré) para encher o cache.
    NÃO chama OpenWeather por padrão para não gastar cota.
    Use este endpoint só quando quiser “esquentar” o backend.
    """
    t0 = time.time()

    # 1) Previsão marinha e vento (Open-Meteo) -> já tem cache TTL
    om = fetch_open_meteo(LAT, LON)
    wind = fetch_open_meteo_wind(LAT, LON)

    # 2) Constrói a série (apenas processamento local)
    series_len = 0
    if om:
        fseries = build_forecast_series(om, wind)
        series_len = len(fseries)

    # 3) Maré (sua função local)
    tide = get_tide_adjusted(0)

    took_ms = round((time.time() - t0) * 1000)
    return {
        "ok": True,
        "cached": {
            "open_meteo": bool(om),
            "open_meteo_wind": bool(wind),
            "forecast_series_len": series_len,
            "tide": bool(tide),
        },
        "took_ms": took_ms,
    }
if __name__ == "__main__":
     app.run(host="0.0.0.0", port=8000, debug=True)

