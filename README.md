# transitpulse-lite
My CAIE Final Project. Build an AI-Powered  System That Improves Your World

# TransitPulse Lite 🚍📍  
**Live Vehicles + Headway Anomaly Detection (Traditional ML) + LLM Rider Updates (GenAI)**

**Live Demo (streamlit.app):** `https://transitpulse-lite-d3cd2rrut76pyotn4z5hke.streamlit.app/`

TransitPulse Lite is a lightweight, deployable public web app that turns **Malaysia Open Data GTFS feeds** into actionable transit service insights. It shows **live vehicle locations**, estimates **headway spacing along route shapes**, detects potential **bus bunching / service gaps** using **unsupervised ML (IsolationForest)**, and (optionally) generates **BM/EN rider updates** using an LLM.

> 🕒 All timestamps shown in the app are **Asia/Kuala_Lumpur (GMT+8)**.

---

## Why this project exists (Problem)
Commuters often face **irregular public transit service** (long waits followed by multiple buses arriving together) without clear, real-time explanations. Existing map views show vehicle dots, but not whether service is becoming **uneven**. This project bridges that gap by translating realtime vehicle positions into **route health** indicators and rider-friendly messaging.

---

## Key Capabilities (Course Requirements)
### 1) GenAI Capability
- Optional LLM-generated **BM/EN** rider updates grounded strictly on computed metrics (route, timestamp, bunched/gap counts).
- JSON-only output enforced to ensure reliable UI rendering.

### 2) Traditional ML Capability
- **IsolationForest** (unsupervised anomaly detection) applied to headway features:
  - `headway_m` (spacing along route shape)
  - `log(headway_m)` (stabilizes scale)

---

## Data Sources
- **Malaysia Official Open API (data.gov.my)**  
  - GTFS-Realtime: vehicle positions (protobuf)
  - GTFS-Static: routes, trips, shapes (ZIP)

Default (demo) feed:
- GTFS-Static: `prasarana?category=rapid-bus-kl`
- GTFS-RT vehicle positions: `prasarana?category=rapid-bus-kl`

> You may change the feed URLs in `app.py` to test other operators/categories.

---

## How it Works (High-Level)
1. **Ingest live GTFS-RT** vehicle positions (lat/lon/time).
2. **Join GTFS-static** to map vehicles to `route_id`, `route_name`, `shape_id` (and optionally `direction_id`).
3. **Snap vehicles to route shape** and compute `progress_m` (distance along route).
4. Compute **headway proxy** `headway_m = min(gap_ahead, gap_behind)` along the route shape.
5. Use **IsolationForest** to score anomalies + flag unusual headways.
6. Apply explainable thresholds vs route median to label:
   - **BUNCHED** (too close)
   - **GAP** (too far)
   - **NORMAL**
7. Present results on:
   - Map (color-coded vehicles)
   - Route Health table + trend
   - Optional LLM rider update button

---

## Tech Stack
- **Python**, **Streamlit**, **pydeck**
- **pandas**, **NumPy**
- **scikit-learn** (IsolationForest)
- **gtfs-realtime-bindings** (protobuf parsing)
- Optional LLM API: **Groq (OpenAI-compatible)**

---

## Project Structure
```text
.
├── app.py
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── gtfs_rt.py        # fetch + parse GTFS-RT vehicle positions
│   ├── gtfs_static.py    # load GTFS static (routes, trips, shapes)
│   ├── features.py       # joins + shape cache + progress/headway computation
│   ├── anomaly.py        # IsolationForest scoring + BUNCHED/GAP labels
│   ├── llm.py            # optional LLM client (secrets/env safe)
│   └── prompts.py        # system/user prompts for rider updates
└── report/               # optional: report assets/notes
```
---



## Setup & Run (Local / Codespaces)

### 1) Install dependencies
```bash
pip install -r requirements.txt


streamlit run app.py --server.address 0.0.0.0 --server.port 8501

```
