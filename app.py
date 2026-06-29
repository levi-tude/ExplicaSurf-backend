from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import os, time, datetime, httpx
from typing import Any, Dict, Optional, List, Tuple
import google.generativeai as genai
from services.tide_worldtides import get_tide_adjusted

# ============== Config & App ==============
load_dotenv()

app = Flask(__name__)
CORS(app)

# ============== Redis Cache ==============
from flask_caching import Cache

cache = Cache(app, config={
    "CACHE_TYPE": "RedisCache",
    "CACHE_REDIS_URL": os.getenv("REDIS_URL"),
    "CACHE_DEFAULT_TIMEOUT": 300  # 5 minutos
})

LAT = float(os.getenv("LAT", "-12.9437"))
LON = float(os.getenv("LON", "-38.3539"))

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY") or ""
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or ""
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_MODELS_FALLBACK = list(dict.fromkeys([
    GEMINI_MODEL,
    "gemini-2.0-flash-001",
    "gemini-1.5-flash",
    "gemini-2.5-flash",
]))

TIMEZONE = "America/Bahia"
LOCAL_TZ = datetime.timezone(datetime.timedelta(hours=-3))
CACHE_VERSION = "v2"

print("DEBUG - GEMINI_MODEL carregado do .env:", GEMINI_MODEL)
print("DEBUG - OPENWEATHER_API_KEY set:", bool(OPENWEATHER_API_KEY))
print("DEBUG - GEMINI_API_KEY set:", bool(GEMINI_API_KEY))

# ============== Helpers ==============
def cache_get(key: str):
    try:
        return cache.get(key)
    except Exception as e:
        print("DEBUG cache get error:", e)
        return None

def cache_set(key: str, value: Any, timeout: int = 300):
    try:
        cache.set(key, value, timeout=timeout)
    except Exception as e:
        print("DEBUG cache set error:", e)

def hourly_time_key(iso_str: str) -> str:
    if not iso_str:
        return ""
    return iso_str[:16]

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
    """
    🔁 usa a Marine API para dados de ONDAS.
    """
    key = f"{CACHE_VERSION}:openmeteo_marine:{lat:.4f},{lon:.4f}"
    cached = cache_get(key)
    if cached:
        return cached

    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wave_height,wave_period,wave_direction",
        "length_unit": "metric",
        "timezone": TIMEZONE,
        "forecast_days": 7,
    }
    try:
        r = httpx.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        cache_set(key, data)
        return data
    except Exception as e:
        print("DEBUG Open-Meteo Marine error:", e)
        return None


def fetch_open_meteo_wind(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """vento de 10m da Open-Meteo"""
    key = f"{CACHE_VERSION}:openmeteo_wind:{lat:.4f},{lon:.4f}"
    cached = cache_get(key)
    if cached:
        return cached

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "wind_speed_10m,wind_direction_10m",
        "windspeed_unit": "kmh",
        "timezone": TIMEZONE,
        "forecast_days": 7,
    }
    try:
        r = httpx.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        cache_set(key, data)
        return data
    except Exception as e:
        print("DEBUG Open-Meteo Wind error:", e)
        return None


def pick_open_meteo_point(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Escolhe o ponto atual mais próximo do horário."""
    try:
        hourly = data.get("hourly", {})
        times_iso = hourly.get("time", [])

        if not times_iso:
            return None

        times = parse_iso_list(times_iso)
        idx = nearest_index(times)

        return {
            "time": times_iso[idx],
            "wave_height_m": hourly.get("wave_height", [None] * len(times_iso))[idx],
            "wave_period_s": hourly.get("wave_period", [None] * len(times_iso))[idx],
            "wave_direction_deg": hourly.get("wave_direction", [None] * len(times_iso))[idx],
        }

    except Exception as e:
        print("DEBUG pick_open_meteo_point error:", e)
        return None


def fetch_openweather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """Clima atual (vento + condições locais) via OpenWeather."""
    if not OPENWEATHER_API_KEY:
        print("DEBUG OpenWeather: nenhuma API_KEY configurada")
        return None

    key = f"{CACHE_VERSION}:openweather:{lat:.4f},{lon:.4f}"
    cached = cache_get(key)
    if cached:
        return cached

    # 🔹 Usa o endpoint "forecast" (retorna lista com dados de várias horas)
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric",
        "lang": "pt_br",
    }

    try:
        r = httpx.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        cache_set(key, data)
        return data
    except Exception as e:
        print("DEBUG OpenWeather error:", e)
        return None


def pick_openweather_now(data: dict) -> dict:
    """Extrai condições atuais (ou próximas) do OpenWeather para exibir no card."""
    try:
        if not data or "list" not in data or not data["list"]:
            return {}

        now = data["list"][0]  # primeiro horário da previsão (~agora)
        main = now.get("main", {})
        weather = now.get("weather", [{}])[0]
        clouds = now.get("clouds", {}).get("all")
        wind = now.get("wind", {})
        rain = now.get("rain", {}).get("3h", 0)
        pop = now.get("pop", 0) * 100  # probabilidade (%)

        return {
            "temp_c": main.get("temp"),
            "clouds": clouds,
            "precip_mm": rain,
            "precip_probability": round(pop, 1),
            "wind_speed_kmh": round(wind.get("speed", 0) * 3.6, 1),
            "wind_dir_deg": wind.get("deg"),
            "weather_main": weather.get("main"),
            "weather_desc": weather.get("description"),
        }

    except Exception as e:
        print("DEBUG pick_openweather_now error:", e)
        return {}

def fetch_weather(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    key = f"{CACHE_VERSION}:openmeteo_weather:{lat:.4f},{lon:.4f}"
    cached = cache_get(key)
    if cached:
        return cached

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation,precipitation_probability,cloudcover,temperature_2m",
        "timezone": TIMEZONE,
        "forecast_days": 7,
    }

    try:
        r = httpx.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        cache_set(key, data)
        return data
    except Exception as e:
        print("DEBUG fetch_weather error:", e)
        return None


def pick_openweather_for_day(ow_raw: dict, day_offset: int) -> dict:
    """Fallback de clima/vento por dia via OpenWeather (slots de 3h)."""
    try:
        if not ow_raw or "list" not in ow_raw:
            return {}

        target_date = datetime.datetime.now(LOCAL_TZ).date() + datetime.timedelta(days=day_offset)
        items = [
            it for it in ow_raw["list"]
            if datetime.datetime.fromtimestamp(it["dt"], tz=LOCAL_TZ).date() == target_date
        ]
        if not items:
            return {}

        temps, clouds, precip, pops, winds, dirs = [], [], [], [], [], []
        for it in items:
            main = it.get("main", {})
            wind = it.get("wind", {})
            temps.append(main.get("temp"))
            clouds.append(it.get("clouds", {}).get("all"))
            precip.append(it.get("rain", {}).get("3h", 0))
            pops.append((it.get("pop") or 0) * 100)
            winds.append(round((wind.get("speed") or 0) * 3.6, 1))
            dirs.append(wind.get("deg"))

        return {
            "temp_c": safe_avg(temps) or None,
            "clouds": round(safe_avg(clouds)) if clouds else None,
            "precip_mm": round(safe_avg(precip), 2) if precip else 0,
            "precip_probability": round(safe_avg(pops), 1) if pops else 0,
            "wind_speed_kmh": round(safe_avg(winds), 1) if winds else None,
            "wind_dir_deg": round(safe_avg(dirs)) if dirs else None,
        }
    except Exception as e:
        print("DEBUG pick_openweather_for_day error:", e)
        return {}


def build_openweather_lookup(ow_raw: dict) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    if not ow_raw or "list" not in ow_raw:
        return lookup

    for item in ow_raw["list"]:
        dt_unix = item.get("dt")
        if not dt_unix:
            continue
        local_dt = datetime.datetime.fromtimestamp(dt_unix, tz=LOCAL_TZ)
        key = local_dt.strftime("%Y-%m-%dT%H:%M")
        main = item.get("main", {})
        wind = item.get("wind", {})
        lookup[key] = {
            "wind_speed_kmh": round((wind.get("speed") or 0) * 3.6, 1),
            "wind_dir_deg": wind.get("deg"),
            "temp_c": main.get("temp"),
            "clouds": item.get("clouds", {}).get("all"),
            "precip_mm": item.get("rain", {}).get("3h", 0),
            "precip_probability": round((item.get("pop") or 0) * 100, 1),
        }
    return lookup


def enrich_series_from_openweather(
    series: List[Dict[str, Any]],
    ow_raw: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Preenche vento/clima na série horária quando Open-Meteo falhar."""
    lookup = build_openweather_lookup(ow_raw or {})
    if not lookup or not series:
        return series

    parsed_lookup: Dict[str, datetime.datetime] = {}
    for key in lookup:
        try:
            parsed_lookup[key] = datetime.datetime.fromisoformat(key)
        except ValueError:
            continue

    for point in series:
        t_key = hourly_time_key(point.get("time", ""))
        ow = lookup.get(t_key)
        if not ow and t_key and parsed_lookup:
            try:
                pt = datetime.datetime.fromisoformat(t_key)
                nearest_key = min(
                    parsed_lookup,
                    key=lambda k: abs((parsed_lookup[k] - pt).total_seconds()),
                )
                if abs((parsed_lookup[nearest_key] - pt).total_seconds()) <= 7200:
                    ow = lookup[nearest_key]
            except ValueError:
                ow = None

        if not ow:
            continue

        if point.get("wind_speed_kmh") is None:
            point["wind_speed_kmh"] = ow.get("wind_speed_kmh")
        if point.get("wind_dir_deg") is None:
            point["wind_dir_deg"] = ow.get("wind_dir_deg")
        if point.get("temp_c") is None:
            point["temp_c"] = ow.get("temp_c")
        if point.get("clouds") in (None, 0):
            point["clouds"] = ow.get("clouds")
        if not point.get("precip_mm"):
            point["precip_mm"] = ow.get("precip_mm")
        if not point.get("precip_probability"):
            point["precip_probability"] = ow.get("precip_probability")

    return series

def pick_weather_for_day(weather_raw: dict, day_offset: int) -> dict:
    """
    Retorna temperatura, nuvens e precipitação do dia selecionado (0 = hoje, 1 = amanhã, 2 = depois).
    Sempre tenta pegar o horário de 12:00 do dia alvo.
    """
    try:
        if not weather_raw or "hourly" not in weather_raw:
            return {}

        hourly = weather_raw["hourly"]
        times = hourly.get("time", [])

        if not times:
            return {}

        # Dia desejado
        target_date = (datetime.datetime.now().date() + datetime.timedelta(days=day_offset))

        # Procurar índice do horário das 12:00
        idx_match = None
        for i, t in enumerate(times):
            dt = datetime.datetime.fromisoformat(t)
            if dt.date() == target_date and dt.hour == 12:  # 12h = horário padrão estável
                idx_match = i
                break

        # Se não achou 12h, pega o horário mais próximo daquele dia
        if idx_match is None:
            candidates = [
                (i, abs((datetime.datetime.fromisoformat(t).date() - target_date).days))
                for i, t in enumerate(times)
            ]
            idx_match = min(candidates, key=lambda x: x[1])[0]

        # Extrair valores
        temp = hourly.get("temperature_2m", [None])[idx_match]
        clouds = hourly.get("cloudcover", [None])[idx_match]
        precip = hourly.get("precipitation", [None])[idx_match]
        pop = hourly.get("precipitation_probability", [None])[idx_match]

        return {
            "temp_c": temp,
            "clouds": clouds,
            "precip_mm": precip,
            "precip_probability": pop,
        }

    except Exception as e:
        print("DEBUG pick_weather_for_day error:", e)
        return {}

# ============== Forecast builder ==============
def build_hourly_field_map(
    raw: Optional[Dict[str, Any]],
    field_names: List[str],
) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """Monta mapa {hora -> {campo: valor}} a partir de hourly Open-Meteo."""
    field_map: Dict[str, Dict[str, Any]] = {}
    if not raw:
        return [], field_map

    hourly = raw.get("hourly", {})
    times = hourly.get("time", [])
    for field in field_names:
        values = hourly.get(field, [])
        for i, t in enumerate(times):
            key = hourly_time_key(t)
            field_map.setdefault(key, {})[field] = values[i] if i < len(values) else None

    return times, field_map


def build_forecast_series(om_raw, wind_raw=None, weather_raw=None):
    """
    Une: ondas (marine), vento 10m e clima (precipitação, nuvens, temperatura).
    """
    try:
        h = om_raw.get("hourly", {}) if om_raw else {}
        times = h.get("time", [])

        heights = h.get("wave_height", [])
        periods = h.get("wave_period", [])
        dirs = h.get("wave_direction", [])

        _, wind_map = build_hourly_field_map(
            wind_raw,
            ["wind_speed_10m", "wind_direction_10m"],
        )
        _, weather_map = build_hourly_field_map(
            weather_raw,
            ["precipitation", "precipitation_probability", "cloudcover", "temperature_2m"],
        )

        series = []
        for i, t in enumerate(times):
            t_key = hourly_time_key(t)
            wind_point = wind_map.get(t_key, {})
            weather_point = weather_map.get(t_key, {})

            spd = wind_point.get("wind_speed_10m")
            dir10 = wind_point.get("wind_direction_10m")

            # fallback por índice quando timestamps divergem levemente
            if spd is None and wind_raw:
                hw = wind_raw.get("hourly", {})
                if i < len(hw.get("wind_speed_10m", [])):
                    spd = hw["wind_speed_10m"][i]
                    dir10 = hw.get("wind_direction_10m", [None])[i]

            altura = heights[i] if i < len(heights) else None
            periodo = periods[i] if i < len(periods) else None
            energia = altura * periodo if altura and periodo else None
            energia_level = (
                "Baixa" if energia and energia <= 5 else
                "Média" if energia and energia <= 12 else
                "Alta" if energia else None
            )

            precip = weather_point.get("precipitation")
            precip_prob = weather_point.get("precipitation_probability")
            clouds = weather_point.get("cloudcover")
            temp = weather_point.get("temperature_2m")

            if temp is None and weather_raw:
                hwx = weather_raw.get("hourly", {})
                if i < len(hwx.get("temperature_2m", [])):
                    temp = hwx["temperature_2m"][i]
                    clouds = hwx.get("cloudcover", [None])[i]
                    precip = hwx.get("precipitation", [None])[i]
                    precip_prob = hwx.get("precipitation_probability", [None])[i]

            series.append({
                "time": t,
                "wave_height_m": altura,
                "wave_period_s": periodo,
                "wave_direction_deg": dirs[i] if i < len(dirs) else None,
                "wind_speed_kmh": spd,
                "wind_dir_deg": dir10,
                "energy": round(energia, 1) if energia else None,
                "energy_level": energia_level,
                "precip_mm": precip if precip is not None else 0,
                "precip_probability": precip_prob if precip_prob is not None else 0,
                "clouds": clouds if clouds is not None else 0,
                "temp_c": temp,
            })
        return series
    except Exception as e:
        print("DEBUG build_forecast_series error:", e)
        return []

# ============== Gemini prompt ==============
def call_gemini_http(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY não configurada")

    genai.configure(api_key=GEMINI_API_KEY)
    last_error: Optional[Exception] = None

    for model_name in GEMINI_MODELS_FALLBACK:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            text = (response.text or "").strip()
            if text:
                print(f"DEBUG Gemini ok model={model_name}")
                return text
        except Exception as e:
            last_error = e
            print(f"DEBUG Gemini fail model={model_name}:", e)
            continue

    raise last_error or RuntimeError("Nenhum modelo Gemini disponível")

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
- Esta explicação deve se referir às condições de {day_label}. Se {day_label} não for hoje, descreva de forma prospectiva (ex.: “amanhã tende a...”)

📌 Instruções de personalização:
- Sempre comece a explicação citando o surfista de forma natural, por exemplo:
  "Olá {name}, como surfista {stance_text}"
- Use essas informações para ajustar o tom e o foco das recomendações.
- Para iniciantes, explique de forma acessível; para avançados, use termos mais técnicos e objetivos.
- Não repita o nome em todas as frases — apenas no início.
- Sempre mencione se favorece as direitas ou esquerdas com base na direção do swell.
- Saiba sempre que se a pessoa for goofy a onda para esquerda vai ser frontside e para direita backside, e vice-versa para regular (direita frontside e esquerda backside). Mas isso serve de conhecimento para melhorar a explicação e não necessáriamente precisa explicar explicitamente isso na resposta.
- Inclua também uma breve descrição das condições do tempo previstas para o horário do surf (temperatura, cobertura de nuvens e chance de chuva), explicando como isso pode influenciar a experiência na água (visibilidade, conforto térmico e possíveis pancadas rápidas).

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
Descreva como está o mar agora, considerando altura, período, energia percebida e vento (fraco, moderado, forte), se está liso ou mexido, e a tendência do swell. Relacione também o efeito da maré atual. Se houver tendência do vento para 6h, inclua nesta seção ao final. Inclua também um comentário curto sobre o tempo (temperatura, nuvens e chance de chuva) e como isso pode afetar o surf.

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
        return call_gemini_http(prompt)
    except Exception as e:
        err = str(e)
        if "API key" in err or "API_KEY" in err:
            return (
                "Erro ao usar Gemini: chave de API inválida ou sem permissão. "
                "Verifique GEMINI_API_KEY no painel do Render e se a API Generative Language está ativa."
            )
        return f"Erro ao usar Gemini: {err.split('key=')[0].strip()}"

# ============== API principal ==============
@app.get("/api/explain")
def api_explain():
    try:
        level = (request.args.get("level") or "iniciante").lower()
        ai_mode = (request.args.get("ai") or "off").lower()
        day_offset = int(request.args.get("day", 0))
        day_label = "hoje" if day_offset == 0 else "amanhã" if day_offset == 1 else "depois de amanhã"

        # perfil
        name = request.args.get("name", "Surfista")
        stance = (request.args.get("stance") or "").lower()
        experience_months = int(request.args.get("experience_months", 0))

       
        # ------------------- dados externos -------------------
        om_raw = fetch_open_meteo(LAT, LON)
        if not om_raw or "hourly" not in om_raw:
            print("DEBUG /api/explain: om_raw vazio ou sem 'hourly'", om_raw)
            return jsonify({"error": "open-meteo indisponível"}), 502

        wind_raw = fetch_open_meteo_wind(LAT, LON)
        weather_raw = fetch_weather(LAT, LON)
        ow_raw = fetch_openweather(LAT, LON)

        forecast_series = build_forecast_series(om_raw, wind_raw, weather_raw)
        forecast_series = enrich_series_from_openweather(forecast_series, ow_raw)

# ========= 1. CLIMA DO DIA =========
        weather_point = pick_weather_for_day(weather_raw, day_offset)
        if not weather_point and ow_raw:
            weather_point = pick_openweather_for_day(ow_raw, day_offset)

# ========= 2. PONTO ATUAL (ONDA + VENTO + CLIMA ATUAL) =========
        om_point = pick_open_meteo_point(om_raw) or {}
        ow_now = pick_openweather_now(ow_raw) if ow_raw else {}
        merged_now = {**om_point, **ow_now}   # ponto AGORA real

# HOJE recebe clima do dia
        if day_offset == 0:
            merged_now.update(weather_point)

# ========= 3. PONTO DO DIA (AMANHÃ/DEPOIS = MÉDIA) =========
        selected_point = None
        if forecast_series:
            today = datetime.datetime.now().date()
            target_day = today + datetime.timedelta(days=day_offset)

            same_day_points = [
                p for p in forecast_series
                if "time" in p and datetime.datetime.fromisoformat(p["time"]).date() == target_day
    ]

            if same_day_points:
                avg_altura  = safe_avg([p.get("wave_height_m") for p in same_day_points])
                avg_periodo = safe_avg([p.get("wave_period_s") for p in same_day_points])
                avg_vento   = safe_avg([p.get("wind_speed_kmh") for p in same_day_points])
                avg_wind_dir = safe_avg([p.get("wind_dir_deg") for p in same_day_points])
                avg_wave_dir = safe_avg([p.get("wave_direction_deg") for p in same_day_points])

                selected_point = {
                    "wave_height_m": round(avg_altura, 2) if avg_altura else None,
                    "wave_period_s": round(avg_periodo, 1) if avg_periodo else None,
                    "wave_direction_deg": round(avg_wave_dir) if avg_wave_dir else None,
                    "wind_speed_kmh": round(avg_vento, 1) if avg_vento else None,
                    "wind_direction_deg": round(avg_wind_dir) if avg_wind_dir else None,
                }

# Todos os dias recebem clima do dia quando disponível
        if selected_point:
            selected_point.update({k: v for k, v in weather_point.items() if v is not None})
        elif day_offset != 0:
            selected_point = dict(weather_point)

# ========= 4. MARÉ =========
        tide_raw = get_tide_adjusted(day_offset) or {}
        tide_processed = {
            "extremes":      tide_raw.get("extremes"),
            "heights":       tide_raw.get("heights"),
            "now":           tide_raw.get("now"),
            "next_extreme":  tide_raw.get("next_extreme"),
}

        tide_peak_text = format_next_tide_peak(tide_processed.get("next_extreme"))
        wind_trend = wind_trend_summary(forecast_series) if forecast_series else ""

# ========= 5. ONDAS FINAL PARA O CARD =========
        fonte_principal = merged_now if day_offset == 0 else selected_point

        if not fonte_principal:
            fonte_principal = merged_now  # fallback

        altura  = fonte_principal.get("wave_height_m")
        periodo = fonte_principal.get("wave_period_s")
        direcao = fonte_principal.get("wave_direction_deg")

# Energia
        if isinstance(altura, (int, float)) and isinstance(periodo, (int, float)):
            energia = altura * periodo
            fonte_principal["energy"] = round(energia, 1)
            fonte_principal["energy_level"] = (
                "Baixa" if energia <= 5 else
                "Média" if energia <= 12 else
                "Alta"
    )

# Aliases
        if direcao is not None:
            fonte_principal["wave_direction_deg"] = direcao
            fonte_principal["wave_dir_deg"] = direcao
        wind_dir = fonte_principal.get("wind_dir_deg") or fonte_principal.get("wind_direction_deg")
        if wind_dir is not None:
            fonte_principal["wind_dir_deg"] = wind_dir
            fonte_principal["wind_direction_deg"] = wind_dir

# Maré dentro do forecast_now (hoje)
        merged_now["tide"] = {
            "now": tide_processed.get("now", {}),
            "next_extreme": tide_processed.get("next_extreme", {}),
}

# ========= 6. CARD FINAL PARA O FRONT =========
        perceived = classify_perceived_size(
            fonte_principal.get("wave_height_m"),
            fonte_principal.get("wave_period_s"),
)

        merged_day = {
            **(fonte_principal or {}),
            "tide": tide_processed,
            "tide_peak_text": tide_peak_text,
            "wind_trend_text": wind_trend,
            "perceived": perceived,
}

        # debug leve
        print("[/api/explain] ok",
              {"have_om": bool(om_raw), "have_wind": bool(wind_raw),
               "have_weather": bool(weather_raw), "have_ow": bool(ow_raw),
               "pts_series": len(forecast_series),
               "series_wind_pts": sum(1 for p in forecast_series if p.get("wind_speed_kmh") is not None),
               "have_now": bool(merged_now), "ai": ai_mode})

        explanation_pt = ""
        if ai_mode == "on" and GEMINI_API_KEY:
            explanation_pt = explain_with_gemini(
                level, merged_day, name=name, stance=stance,
                experience_months=experience_months, day_label=day_label
            )

        return jsonify({
            "spot": "Stella Maris, Salvador-BA",
            "level": level,
            "day": day_offset,
            "forecast_now": merged_now,
            "forecast_series": forecast_series,
            "forecast_day": merged_day,
            "explanation_pt": explanation_pt,
        })

    except Exception as e:
        # loga e devolve JSON (facilita ver a causa nos logs do Render)
        import traceback
        print("ERROR /api/explain ->", e)
        traceback.print_exc()
        return jsonify({"error": "internal", "detail": str(e)}), 500

# ============== Teste do Redis Cache ==============
@app.get("/teste-cache")
@cache.cached(timeout=20)
def teste_cache():
    import time
    agora = time.time()
    time.sleep(3)  # simula operação lenta
    return {"ts": agora}

