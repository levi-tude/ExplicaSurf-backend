# ExplicaSurf — Backend (TCC)

API Flask do **ExplicaSurf**: agrega previsões oceânicas/climáticas, aplica heurísticas de surf e gera explicações com **Google Gemini**, personalizadas ao perfil do surfista e ao conhecimento local de **Stella Maris** (Salvador/BA).

> **Protótipo acadêmico (TCC)** — Ciência da Computação, UNIJORGE.  
> Site: [explicasurfstella.com.br](https://explicasurfstella.com.br) · API em produção: [explicasurf-backend.onrender.com](https://explicasurf-backend.onrender.com)

Frontend: [levi-tude/ExplicaSurf-frontend](https://github.com/levi-tude/ExplicaSurf-frontend).

---

## Papel no sistema

O backend é o “cérebro” do TCC. O endpoint principal **`GET /api/explain`**:

1. Recebe parâmetros do frontend (nível, dia, nome, stance, experiência, flag de IA)
2. Busca dados externos (ondas, vento, clima, maré)
3. Une tudo em uma estrutura coerente (`forecast_day`)
4. Calcula métricas (energia, tamanho percebido “down-bias”, tendência de vento 6h, texto de pico/vale de maré)
5. Monta o prompt com **conhecimento local de Stella Maris** e chama o Gemini
6. Devolve JSON pronto para cards, gráficos e texto explicativo

Isso responde à proposta do artigo: transformar dados técnicos em linguagem acessível (*ocean literacy*), com segurança e inclusão para diferentes níveis de surf.

---

## Stack

| Tecnologia | Uso |
|------------|-----|
| Python 3 + Flask | API HTTP |
| Flask-CORS | Acesso do frontend |
| Flask-Caching + Redis | Cache (~5 min) |
| httpx / requests | APIs externas |
| google-generativeai | Gemini (com fallback de modelos) |
| python-dotenv | Variáveis de ambiente |
| gunicorn | Produção (Render) |

### Integrações de dados

| Fonte | Conteúdo |
|-------|----------|
| **Open-Meteo Marine** | Altura, período, direção de onda |
| **Open-Meteo Wind / Forecast** | Vento 10 m, clima (nuvens, chuva, temperatura) |
| **OpenWeather** | Complemento climático (quando configurado) |
| **WorldTides** | Extremos e série de maré + **offsets calibrados Stella** |

Coordenadas padrão: Stella Maris (~`-12.94`, `-38.35`), fuso `America/Bahia`.

---

## Calibração de maré (diferencial local)

Em `services/tide_worldtides.py`, offsets afinados com comparação Surfguru / Marinha:

- Ajuste de horário (minutos) e altura (metros) para High/Low
- Offset na série horária
- Interpolação por curva cosseno para altura “agora”
- Próximo extremo (cheia/seca) para o texto da IA

---

## Heurísticas de domínio (resumo)

Definidas principalmente em `app.py`:

- **Energia** ≈ altura × período (níveis baixa / média / alta)
- **Tamanho percebido** com leitura conservadora (“down-bias”)
- **Tendência de vento** (agora vs ~6 h)
- **Prompt Gemini** estruturado: Análise geral → Impacto por nível → Recomendação → Segurança
- **Stance:** regular/goofy → frontside/backside conforme direção do swell
- **Conhecimento local:** fundos, picos (Padang/Loro, Corrente), swell S/SE vs L, vento terral/maral, efeito de maré

---

## Estrutura do repositório

```
.
├── app.py                      # Flask app, fetchers, merge, prompt, /api/explain
├── requirements.txt
├── services/
│   └── tide_worldtides.py      # WorldTides + offsets Stella
└── README.md
```

---

## API

### `GET /api/explain`

Query params (principais):

| Param | Exemplo | Descrição |
|-------|---------|-----------|
| `level` | `iniciante` / `intermediario` / `avancado` | Nível do surfista |
| `day` | `0` / `1` / `2` | Hoje / amanhã / depois |
| `ai` | `on` / `off` | Gera texto Gemini ou só dados |
| `name` | string | Nome (personalização) |
| `stance` | `regular` / `goofy` | Base |
| `experience_months` | int | Experiência |

Resposta (resumo): JSON com série horária, métricas do dia, maré, vento, clima e, se `ai=on`, campo de explicação em texto.

### `GET /teste-cache`

Utilitário para validar cache Redis.

---

## Pré-requisitos

- Python 3.10+ recomendado
- Redis (local ou gerenciado) — opcional em dev se ajustar cache; em produção use `REDIS_URL`
- Chaves: Gemini, WorldTides; OpenWeather opcional

---

## Setup local

```bash
git clone https://github.com/levi-tude/ExplicaSurf-backend.git
cd ExplicaSurf-backend

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

Crie um arquivo `.env` na raiz do backend (não versionar):

```env
GEMINI_API_KEY=sua_chave
GEMINI_MODEL=gemini-2.5-flash-lite
WORLDTIDES_API_KEY=sua_chave
OPENWEATHER_API_KEY=                 # opcional
REDIS_URL=redis://localhost:6379/0
LAT=-12.9437
LON=-38.3539
```

Subir a API:

```bash
flask --app app run --host 0.0.0.0 --port 5000
# ou:
python app.py
```

Teste rápido:

```bash
curl "http://localhost:5000/api/explain?level=iniciante&day=0&ai=off"
```

Aponte o frontend (`VITE_API_BASE_URL`) para `http://localhost:5000`.

---

## Produção (Render)

- Serviço web com **gunicorn** (`requirements.txt` inclui `gunicorn`)
- Free tier **hiberna** após inatividade — o frontend faz warmup para reduzir cold start
- Configure as mesmas variáveis de ambiente no painel do Render

URL de referência: `https://explicasurf-backend.onrender.com`

---

## Dependências (`requirements.txt`)

```
flask==3.0.3
flask-cors==4.0.0
flask-caching==2.3.0
redis==5.0.8
httpx==0.27.0
python-dotenv==1.0.1
google-generativeai>=0.8.0
gunicorn==21.2.0
```

(`services/tide_worldtides.py` usa `requests` — instale se ainda não estiver no ambiente: `pip install requests`.)

---

## Relação com o produto comercial

| | Este repo (TCC) | Produto comercial |
|--|-----------------|-------------------|
| Pasta local | `ExplicaSurf/backend` | `ExplicaSurf-tio` |
| Runtime | Flask + Render | Next.js (lógica portada; **sem Flask**) |
| Escopo | Stella Maris | Multi-praia Salvador |

O comercial usa este código como **referência** (maré, prompt, heurísticas), não como dependência em produção.

---

## Autor

**Levi Davi Tude Silva** — TCC, UNIJORGE  
Contato: levidavitudesilva@gmail.com

Orientação (artigo): Jailson Santos · Marcos Santos Leite

---

## Licença

Projeto acadêmico / protótipo. Consulte o autor para uso além do TCC.
