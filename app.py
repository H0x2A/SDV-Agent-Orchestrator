# ─────────────────────────────────────────────────────────────────────────────
#  SDV Orchestrator  ·  LangGraph Multi-Agent + MCP Tool Layer
#  pip install langgraph langchain-google-genai langchain-core streamlit
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import json
import uuid
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SDV Orchestrator Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CSS  
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #080c14;
    color: #c8dae8;
}
section[data-testid="stSidebar"] { background: #0a0f1c; }

/* ── typography ── */
.hdr-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.9rem; font-weight: 900;
    letter-spacing: 3px;
    background: linear-gradient(100deg, #00e5ff 0%, #00ffa3 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.hdr-sub {
    font-size: 0.68rem; color: #3a6a7a;
    letter-spacing: 4px; text-transform: uppercase; margin-top: 4px;
}
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem; font-weight: 600;
    color: #1a7a8a; letter-spacing: 3px; text-transform: uppercase;
    border-bottom: 1px solid rgba(0,229,255,0.1);
    padding-bottom: 6px; margin-bottom: 14px;
}

/* ── cards ── */
.card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(0,229,255,0.1);
    border-radius: 10px; padding: 18px 20px; margin-bottom: 12px;
}
.card-accent {
    border-left: 3px solid #00e5ff;
    background: rgba(0,229,255,0.04);
    border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin-bottom: 8px;
}

/* ── seat map ── */
.seat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px; margin-top: 8px;
}
.seat-btn {
    background: rgba(0,229,255,0.06);
    border: 1px solid rgba(0,229,255,0.18);
    border-radius: 8px; padding: 10px;
    text-align: center; cursor: pointer;
    font-size: 0.75rem; color: #7ac0d0;
    transition: all 0.2s;
}
.seat-occupied-child {
    background: rgba(160,80,255,0.12);
    border: 1px solid rgba(160,80,255,0.4);
    color: #c084fc; border-radius: 8px; padding: 10px;
    text-align: center; font-size: 0.75rem;
}
.seat-occupied-adult {
    background: rgba(0,255,150,0.08);
    border: 1px solid rgba(0,255,150,0.3);
    color: #4ade80; border-radius: 8px; padding: 10px;
    text-align: center; font-size: 0.75rem;
}
.seat-driver {
    background: rgba(0,229,255,0.08);
    border: 1px solid rgba(0,229,255,0.3);
    color: #00e5ff; border-radius: 8px; padding: 10px;
    text-align: center; font-size: 0.75rem;
    grid-column: span 2;
}

/* ── passenger offer cards ── */
.offer-child {
    background: rgba(160,80,255,0.07);
    border: 1px solid rgba(160,80,255,0.3);
    border-radius: 12px; padding: 18px;
}
.offer-adult {
    background: rgba(0,255,150,0.06);
    border: 1px solid rgba(0,255,150,0.25);
    border-radius: 12px; padding: 18px;
}
.offer-driver {
    background: rgba(0,229,255,0.05);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 12px; padding: 18px;
}
.voucher-code {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem; color: #00e5ff;
    background: rgba(0,229,255,0.06);
    border: 1px solid rgba(0,229,255,0.15);
    padding: 6px 12px; border-radius: 6px;
    letter-spacing: 2px; display: inline-block;
    margin-top: 8px;
}

/* ── score ring ── */
.score-big {
    font-family: 'Orbitron', monospace;
    font-size: 3.8rem; font-weight: 900;
    line-height: 1; text-align: center;
}
.metric-micro {
    font-size: 0.65rem; color: #2a6070;
    text-transform: uppercase; letter-spacing: 2px;
    text-align: center; margin-top: 4px;
}

/* ── agent trace log ── */
.trace-line {
    font-family: 'Courier New', monospace;
    font-size: 0.7rem; color: #00ffa3;
    background: rgba(0,0,0,0.35);
    border-left: 2px solid rgba(0,255,163,0.3);
    padding: 5px 10px; margin-bottom: 3px;
    border-radius: 0 4px 4px 0;
    animation: fadein 0.3s ease;
}
@keyframes fadein { from { opacity: 0; transform: translateX(-6px); } to { opacity: 1; } }

/* ── urgency bar ── */
.urg-high { background: linear-gradient(90deg,#ff3333,#ff7700); border-radius:3px; height:5px; animation: pulse 1.2s infinite; }
.urg-medium { background: linear-gradient(90deg,#ff9900,#ffcc00); border-radius:3px; height:5px; }
.urg-low { background: linear-gradient(90deg,#00e5ff,#00ffa3); border-radius:3px; height:5px; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.55} }

/* ── graph node diagram ── */
.graph-node {
    display: inline-block;
    background: rgba(0,229,255,0.07);
    border: 1px solid rgba(0,229,255,0.2);
    border-radius: 8px; padding: 6px 14px;
    font-size: 0.68rem; color: #00e5ff;
    font-family: 'Orbitron', monospace;
    white-space: nowrap;
}
.graph-node-active {
    background: rgba(0,229,255,0.18);
    border: 1px solid #00e5ff;
    color: #fff; box-shadow: 0 0 12px rgba(0,229,255,0.3);
}
.graph-arrow { color: #1a5a6a; font-size: 0.9rem; margin: 0 4px; }

/* ── divider ── */
.glow-hr {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,229,255,0.25), transparent);
    margin: 22px 0;
}

/* button override */
div.stButton > button {
    background: linear-gradient(90deg, #004455, #005566);
    color: #00e5ff; border: 1px solid rgba(0,229,255,0.35);
    border-radius: 8px; font-family: 'Orbitron', monospace;
    letter-spacing: 2px; font-size: 0.7rem; font-weight: 600;
    padding: 10px 24px; width: 100%;
    transition: all 0.2s;
}
div.stButton > button:hover {
    background: linear-gradient(90deg, #005566, #007788);
    border-color: #00e5ff;
    box-shadow: 0 0 16px rgba(0,229,255,0.25);
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MCP TOOL LAYER
#  In production: each @tool maps to an HTTP/SSE MCP server endpoint
#  Here: deterministic simulation of MCP server responses
# ══════════════════════════════════════════════════════════════════════════════

PARTNER_DB = {
    "Hinjewadi Ph 1": {
        "name": "TATA Power EV Hub · Hinjewadi",
        "partner_id": "TATA_PNQ_HIN_01",
        "charger_type": "DC Fast (150 kW)",
        "eta_minutes": 8,
        "amenities": ["McDonald's", "Starbucks", "Kids Zone"],
        "child_amenity": "AirConsole Gaming Lounge",
        "adult_amenity": "Starbucks",
        "child_offer_template": "45-min AirConsole Pro Session — Unlocked",
        "adult_offer_template": "Starbucks 20% Off + 2x Loyalty Points",
        "referral_fee_inr": 55,
        "available_slots": 3,
    },
    "Wakad Bypass": {
        "name": "Ather Grid · Wakad",
        "partner_id": "ATHER_PNQ_WKD_02",
        "charger_type": "DC Fast (100 kW)",
        "eta_minutes": 6,
        "amenities": ["Kids Zone", "Café Coffee Day", "Domino's"],
        "child_amenity": "Kids Zone",
        "adult_amenity": "Café Coffee Day",
        "child_offer_template": "Kids Zone Pass — 30 min Complimentary",
        "adult_offer_template": "Café Coffee Day Buy-1-Get-1 + Domino's ₹100 Off",
        "referral_fee_inr": 42,
        "available_slots": 5,
    },
    "Baner High St": {
        "name": "ChargeZone Premium · Baner",
        "partner_id": "CZ_PNQ_BAN_03",
        "charger_type": "AC + DC (22–100 kW)",
        "eta_minutes": 11,
        "amenities": ["GameOn Arcade", "Chaayos", "Subway", "Kids Corner"],
        "child_amenity": "GameOn Arcade",
        "adult_amenity": "Chaayos",
        "child_offer_template": "GameOn Arcade — 3 Free Game Credits",
        "adult_offer_template": "Chaayos ₹60 Voucher + Free Snack",
        "referral_fee_inr": 38,
        "available_slots": 2,
    },
}


@tool
def get_partner_stations(location: str) -> str:
    """[MCP:PartnerAPI v2.1] Retrieve available EV charging partner stations and amenity inventory for a location sector."""
    return json.dumps(PARTNER_DB.get(location, {"error": "No partner found"}))


@tool
def calculate_charging_urgency(soc: int, range_km: int, trip_km: int) -> str:
    """[MCP:BatteryMgmtSystem v3.0] Compute charging urgency level and score contribution from live battery telemetry."""
    deficit = trip_km - range_km
    if soc < 25 or deficit > 0:
        urgency, score = "HIGH", 40
    elif soc < 50:
        urgency, score = "MEDIUM", 30
    else:
        urgency, score = "LOW", 10
    return json.dumps({
        "urgency_level": urgency,
        "score_contribution": score,
        "range_deficit_km": max(0, deficit),
        "kwh_needed": round(max(0, deficit * 0.185), 1),
        "recommendation": "Immediate stop required" if urgency == "HIGH"
            else "Stop recommended within 20 km" if urgency == "MEDIUM"
            else "Optional lifestyle stop",
    })


@tool
def get_passenger_profile(passenger_type: str) -> str:
    """[MCP:OccupantProfiler v1.5] Retrieve service preference profile and dwell-time tolerance for a passenger type."""
    profiles = {
        "Child": {
            "entertainment_priority": "HIGH",
            "preferred_services": ["gaming", "kids_food", "play_zone"],
            "dwell_tolerance_min": 45,
            "attention_span_min": 8,
        },
        "Adult": {
            "entertainment_priority": "LOW",
            "preferred_services": ["coffee", "food", "wifi", "shopping"],
            "dwell_tolerance_min": 20,
            "upsell_affinity": "HIGH",
        },
        "Driver": {
            "entertainment_priority": "NONE",
            "preferred_services": ["navigation", "charging_status"],
            "dwell_tolerance_min": 15,
        },
    }
    return json.dumps(profiles.get(passenger_type, profiles["Adult"]))


@tool
def generate_voucher(partner_id: str, passenger_type: str, offer_template: str) -> str:
    """[MCP:PartnerNegotiation v2.0] Negotiate and generate a personalized discount voucher via partner API."""
    code = f"{'KID' if passenger_type == 'Child' else 'VIP'}-{partner_id[:4].upper()}-{uuid.uuid4().hex[:6].upper()}"
    return json.dumps({
        "voucher_code": code,
        "offer": offer_template,
        "valid_minutes": 120,
        "qr_enabled": True,
        "partner_id": partner_id,
    })


@tool
def push_hmi_alert(message: str, urgency: str) -> str:
    """[MCP:HMIBridge v4.1] Push contextual notification to vehicle center console and HUD."""
    return json.dumps({
        "status": "PUSHED",
        "displays": ["CENTER_CONSOLE", "HUD"],
        "message": message,
        "urgency": urgency,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  LANGGRAPH STATE
# ══════════════════════════════════════════════════════════════════════════════
class SDVState(TypedDict):
    # ── inputs ──
    soc: int
    range_km: int
    location: str
    trip_km: int
    destination: str
    passengers: List[dict]
    partner_active: bool
    # ── agent outputs ──
    occupancy_analysis: str
    charging_urgency: str
    partner_data: str
    passenger_offers: List[dict]
    hmi_message: str
    engagement_score: int
    # ── trace ──
    agent_logs: List[str]


def _llm():
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview", 
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.2,
    )


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 1 · OccupantAnalyzer
# ──────────────────────────────────────────────────────────────────────────────
def occupant_analyzer(state: SDVState) -> dict:
    logs = list(state.get("agent_logs", []))
    logs.append("🔍 [OccupantAnalyzer] Waking sensor fusion layer...")

    profiles = []
    for pax in state["passengers"]:
        raw = get_passenger_profile.invoke({"passenger_type": pax["type"]})
        profiles.append({"passenger": pax, "profile": json.loads(raw)})
        logs.append(f"   ↳ MCP:OccupantProfiler → {pax['type']} profile retrieved for seat '{pax['seat']}'")

    response = _llm().invoke([HumanMessage(content=f"""
You are the OccupantAnalysis specialist agent in a SDV system.
Passenger data with service profiles: {json.dumps(profiles)}

Return ONLY valid JSON (no markdown, no preamble):
{{
  "passenger_count": <int>,
  "has_children": <bool>,
  "has_adults": <bool>,
  "primary_need": "entertainment" | "comfort" | "mixed" | "navigation_only",
  "analysis": "<one concise sentence about the cabin composition and priority>"
}}
""")])

    try:
        txt = response.content.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        analysis = json.loads(txt)
    except Exception:
        analysis = {"passenger_count": len(state["passengers"]), "has_children": False,
                    "has_adults": False, "primary_need": "navigation_only",
                    "analysis": "Could not parse occupancy."}

    logs.append(
        f"✅ [OccupantAnalyzer] Done — {analysis.get('passenger_count')} occupants · "
        f"Children: {analysis.get('has_children')} · Adults: {analysis.get('has_adults')} · "
        f"Need: {analysis.get('primary_need')}"
    )
    return {"occupancy_analysis": json.dumps(analysis), "agent_logs": logs}


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 2 · ChargingOptimizer
# ──────────────────────────────────────────────────────────────────────────────
def charging_optimizer(state: SDVState) -> dict:
    logs = list(state.get("agent_logs", []))
    logs.append("⚡ [ChargingOptimizer] Querying BMS telemetry stream...")

    raw = calculate_charging_urgency.invoke({
        "soc": state["soc"],
        "range_km": state["range_km"],
        "trip_km": state["trip_km"],
    })
    data = json.loads(raw)
    logs.append(
        f"   ↳ MCP:BatteryMgmtSystem → Urgency: {data['urgency_level']} · "
        f"Score contribution: +{data['score_contribution']} · "
        f"Range deficit: {data['range_deficit_km']} km"
    )
    logs.append(f"✅ [ChargingOptimizer] Done — {data['recommendation']}")
    return {"charging_urgency": raw, "agent_logs": logs}


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 3 · PartnerNegotiator
# ──────────────────────────────────────────────────────────────────────────────
def partner_negotiator(state: SDVState) -> dict:
    logs = list(state.get("agent_logs", []))
    logs.append("🤝 [PartnerNegotiator] Opening partner API handshake...")

    station_raw = get_partner_stations.invoke({"location": state["location"]})
    station = json.loads(station_raw)
    logs.append(
        f"   ↳ MCP:PartnerAPI → Connected: {station.get('name')} "
        f"({station.get('available_slots')} slots · ETA {station.get('eta_minutes')} min)"
    )

    offers = []
    for pax in state["passengers"]:
        if pax["type"] == "Driver":
            offers.append({
                "passenger_id": pax["id"], "name": pax.get("name", "Driver"),
                "type": "Driver", "seat": pax["seat"],
                "offer": "Navigation routed to charging stop",
                "voucher_code": "—", "qr_enabled": False,
            })
            continue

        tmpl = station["child_offer_template"] if pax["type"] == "Child" else station["adult_offer_template"]
        voucher_raw = generate_voucher.invoke({
            "partner_id": station["partner_id"],
            "passenger_type": pax["type"],
            "offer_template": tmpl,
        })
        v = json.loads(voucher_raw)
        offers.append({
            "passenger_id": pax["id"],
            "name": pax.get("name") or pax["type"],
            "type": pax["type"],
            "seat": pax["seat"],
            "offer": v["offer"],
            "voucher_code": v["voucher_code"],
            "qr_enabled": v["qr_enabled"],
        })
        logs.append(f"   ↳ MCP:PartnerNegotiation → Voucher {v['voucher_code']} → {pax.get('name') or pax['type']}")

    logs.append(f"✅ [PartnerNegotiator] Done — {len(offers)} service packages ready")
    return {"partner_data": station_raw, "passenger_offers": offers, "agent_logs": logs}


# ──────────────────────────────────────────────────────────────────────────────
#  NODE 4 · HMIComposer
# ──────────────────────────────────────────────────────────────────────────────
def hmi_composer(state: SDVState) -> dict:
    logs = list(state.get("agent_logs", []))
    logs.append("📺 [HMIComposer] Computing final engagement score...")

    urgency_data = json.loads(state.get("charging_urgency", "{}"))
    occupancy_data = json.loads(state.get("occupancy_analysis", "{}"))
    station = json.loads(state.get("partner_data", "{}"))

    score = (
        urgency_data.get("score_contribution", 10)
        + (25 if occupancy_data.get("has_children") else 0)
        + (15 if occupancy_data.get("has_adults") else 0)
        + (20 if state.get("partner_active") else 0)
    )
    score = min(100, score)

    response = _llm().invoke([HumanMessage(content=f"""
You are the HMI Composer agent for a SDV vehicle.
Facts:
- Charging urgency: {urgency_data.get('urgency_level')}
- Occupants: {occupancy_data.get('analysis')}
- Partner stop: {station.get('name')} in {station.get('eta_minutes')} min
- Engagement score: {score}/100
- Services offered: {[o['offer'] for o in state.get('passenger_offers', [])]}

Write a single concise vehicle center-console message (max 16 words).
It must feel like a premium car HMI — calm, helpful, specific.
Return ONLY the message text, nothing else.
""")])

    print(f"DEBUG: Response type is {type(response.content)} and value is {response.content}")
    
    msg = response.content.strip().strip('"')
    push_hmi_alert.invoke({"message": msg, "urgency": urgency_data.get("urgency_level", "MEDIUM")})

    logs.append(f"   ↳ Engagement score: {score}/100")
    logs.append(f"   ↳ MCP:HMIBridge → PUSHED to CENTER_CONSOLE + HUD")
    logs.append(f'✅ [HMIComposer] Done — "{msg}"')
    return {"hmi_message": msg, "engagement_score": score, "agent_logs": logs}


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def build_graph():
    g = StateGraph(SDVState)
    g.add_node("occupant_analyzer", occupant_analyzer)
    g.add_node("charging_optimizer", charging_optimizer)
    g.add_node("partner_negotiator", partner_negotiator)
    g.add_node("hmi_composer", hmi_composer)
    g.add_edge(START, "occupant_analyzer")
    g.add_edge("occupant_analyzer", "charging_optimizer")
    g.add_edge("charging_optimizer", "partner_negotiator")
    g.add_edge("partner_negotiator", "hmi_composer")
    g.add_edge("hmi_composer", END)
    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
if "passengers" not in st.session_state:
    st.session_state.passengers = [
        {"id": "drv_001", "name": "Driver", "type": "Driver", "seat": "Front Left"}
    ]
if "result" not in st.session_state:
    st.session_state.result = None


# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown('<div class="hdr-title">⚡ SDV ORCHESTRATOR DEMO</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hdr-sub">LangGraph Multi-Agent · MCP Tool Layer · In-Cabin Intelligence</div>',
        unsafe_allow_html=True,
    )
with c2:
    nodes = ["OCCUPANT", "CHARGING", "PARTNER", "HMI"]
    st.markdown(
        "<div style='display:flex;align-items:center;gap:2px;margin-top:16px;flex-wrap:wrap'>"
        + "".join(
            f'<span class="graph-node">{n}</span><span class="graph-arrow">→</span>'
            if i < len(nodes) - 1
            else f'<span class="graph-node">{N}</span>'
            for i, (n, N) in enumerate(zip(nodes, nodes))
        )
        + "</div>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="glow-hr"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LAYOUT  ·  Cabin Setup  |  Telemetry  |  Run
# ══════════════════════════════════════════════════════════════════════════════
col_cabin, col_tel = st.columns([1, 1], gap="large")

# ─── CABIN SETUP ─────────────────────────────────────────────────────────────
with col_cabin:
    st.markdown('<div class="section-label">🛋️ In-Cabin Occupancy Setup</div>', unsafe_allow_html=True)

    # ── Seat map visual ──
    SEAT_SLOTS = ["Front Right", "Rear Left", "Rear Center", "Rear Right"]
    occupied = {p["seat"]: p for p in st.session_state.passengers}

    # Build seat map HTML
    seat_html = '<div class="seat-grid">'
    seat_html += '<div class="seat-driver">🧑‍✈️ DRIVER — Front Left</div>'
    for slot in SEAT_SLOTS:
        if slot in occupied:
            pax = occupied[slot]
            css = "seat-occupied-child" if pax["type"] == "Child" else "seat-occupied-adult"
            icon = "👦" if pax["type"] == "Child" else "🧑"
            label = pax.get("name") or pax["type"]
            seat_html += f'<div class="{css}">{icon} {slot}<br><b>{label}</b></div>'
        else:
            seat_html += f'<div class="seat-btn">○ {slot}<br><span style="color:#1a4a5a;font-size:0.65rem">empty</span></div>'
    seat_html += "</div>"

    st.markdown(
        f'<div class="card"><div class="metric-micro" style="text-align:left;margin-bottom:8px;">CABIN VIEW</div>{seat_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Add passenger form ──
    with st.expander("➕  Add Passenger", expanded=len(st.session_state.passengers) < 2):
        with st.form("add_pax_form", clear_on_submit=True):
            fa, fb, fc = st.columns([2, 1, 2])
            with fa:
                pax_name = st.text_input("Name (optional)", placeholder="e.g. Aanya")
            with fb:
                pax_type = st.selectbox("Type", ["Child", "Adult"])
            with fc:
                available_seats = [s for s in SEAT_SLOTS if s not in occupied]
                if available_seats:
                    pax_seat = st.selectbox("Seat", available_seats)
                else:
                    st.caption("All seats occupied")
                    pax_seat = None
            submitted = st.form_submit_button("Add to Cabin", use_container_width=True)
            if submitted and pax_seat:
                st.session_state.passengers.append({
                    "id": f"pax_{uuid.uuid4().hex[:6]}",
                    "name": pax_name.strip() or None,
                    "type": pax_type,
                    "seat": pax_seat,
                })
                st.rerun()

    # ── Current passengers list ──
    st.markdown('<div class="section-label" style="margin-top:16px;">CURRENT OCCUPANTS</div>', unsafe_allow_html=True)
    for pax in st.session_state.passengers:
        pc1, pc2 = st.columns([4, 1])
        icon = "🧑‍✈️" if pax["type"] == "Driver" else ("👦" if pax["type"] == "Child" else "🧑")
        label = pax.get("name") or pax["type"]
        accent = "#00e5ff" if pax["type"] == "Driver" else ("#c084fc" if pax["type"] == "Child" else "#4ade80")
        with pc1:
            st.markdown(
                f'<div class="card-accent"><span style="color:{accent};font-weight:600">{icon} {label}</span>'
                f'<span style="color:#2a5060;font-size:0.72rem;margin-left:10px">{pax["type"]} · {pax["seat"]}</span></div>',
                unsafe_allow_html=True,
            )
        with pc2:
            if pax["type"] != "Driver":
                if st.button("✕", key=f"del_{pax['id']}", help="Remove"):
                    st.session_state.passengers = [p for p in st.session_state.passengers if p["id"] != pax["id"]]
                    st.rerun()

# ─── TELEMETRY ────────────────────────────────────────────────────────────────
with col_tel:
    st.markdown('<div class="section-label">📡 Vehicle Telemetry</div>', unsafe_allow_html=True)

    soc = st.slider("🔋 Battery SoC %", 5, 100, 32)
    range_km = st.slider("📏 Estimated Range (km)", 10, 300, int(soc * 2.8))
    trip_km = st.slider("🗺️ Trip Distance (km)", 10, 150, 55)

    st.markdown('<div class="section-label" style="margin-top:18px;">📍 Geo-Context</div>', unsafe_allow_html=True)
    location = st.selectbox("Current Sector", ["Hinjewadi Ph 1", "Wakad Bypass", "Baner High St"])
    destination = st.text_input("Destination", "Pune Airport, T1")

    st.markdown('<div class="section-label" style="margin-top:18px;">🔗 Protocol Status</div>', unsafe_allow_html=True)
    c1p, c2p = st.columns(2)
    with c1p:
        partner_active = st.toggle("Partner Handshake", value=True)
    with c2p:
        v2x = st.toggle("V2X Grid Signal", value=True)

    # SoC urgency preview
    urg = "HIGH" if soc < 25 else ("MEDIUM" if soc < 50 else "LOW")
    urg_css = f"urg-{urg.lower()}"
    st.markdown(
        f'<div style="margin-top:14px"><div class="metric-micro" style="text-align:left;margin-bottom:6px;">'
        f'CHARGE URGENCY PREVIEW — {urg}</div>'
        f'<div class="{urg_css}"></div></div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  RUN BUTTON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="glow-hr"></div>', unsafe_allow_html=True)
run = st.button("⚙️  RUN LANGGRAPH ORCHESTRATION CYCLE", use_container_width=True)

if run:
    graph = build_graph()
    initial_state: SDVState = {
        "soc": soc, "range_km": range_km, "location": location,
        "trip_km": trip_km, "destination": destination,
        "passengers": st.session_state.passengers,
        "partner_active": partner_active,
        "occupancy_analysis": "", "charging_urgency": "",
        "partner_data": "", "passenger_offers": [],
        "hmi_message": "", "engagement_score": 0,
        "agent_logs": [],
    }

    # ── Stream execution node by node ──
    st.markdown('<div class="glow-hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">⚙️ LangGraph Execution Trace</div>', unsafe_allow_html=True)

    node_labels = {
        "occupant_analyzer": "NODE 1 · OccupantAnalyzer",
        "charging_optimizer": "NODE 2 · ChargingOptimizer",
        "partner_negotiator": "NODE 3 · PartnerNegotiator",
        "hmi_composer":       "NODE 4 · HMIComposer",
    }

    trace_placeholder = st.empty()
    all_logs = []
    final_state = None

    for step in graph.stream(initial_state, stream_mode="updates"):
        node_name = list(step.keys())[0]
        node_out = step[node_name]
        new_logs = node_out.get("agent_logs", [])
        fresh = [l for l in new_logs if l not in all_logs]
        all_logs = new_logs

        with trace_placeholder.container():
            label = node_labels.get(node_name, node_name)
            st.markdown(
                f'<div style="font-family:Orbitron,monospace;font-size:0.65rem;color:#1a7a8a;'
                f'letter-spacing:3px;margin:10px 0 6px;">▶ {label}</div>',
                unsafe_allow_html=True,
            )
            for line in fresh:
                st.markdown(f'<div class="trace-line">{line}</div>', unsafe_allow_html=True)

        if node_name == "hmi_composer":
            final_state = {**initial_state, **node_out}
            for prev_step in [initial_state]:
                pass
            # Accumulate all outputs
            final_state = node_out

    # Rebuild full final state from stream
    # Re-run synchronously to get full state
    full_result = graph.invoke(initial_state)
    st.session_state.result = full_result


# ══════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.result:
    r = st.session_state.result
    station = json.loads(r.get("partner_data", "{}"))
    urgency_data = json.loads(r.get("charging_urgency", "{}"))
    occupancy_data = json.loads(r.get("occupancy_analysis", "{}"))
    score = r.get("engagement_score", 0)
    offers = r.get("passenger_offers", [])

    st.markdown('<div class="glow-hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📊 Orchestration Output</div>', unsafe_allow_html=True)

    # ── Row 1: Score + Urgency + HMI ──
    ra, rb, rc = st.columns([1, 1, 2])

    with ra:
        color = "#00ffa3" if score > 65 else ("#ffb400" if score > 40 else "#ff4444")
        status = "SERVICES ON" if score > 65 else "STANDBY"
        st.markdown(f"""
        <div class="card" style="text-align:center">
            <div class="metric-micro">Engagement Score</div>
            <div class="score-big" style="color:{color}">{score}</div>
            <div class="metric-micro">/100 · <span style="color:{color}">{status}</span></div>
        </div>
        """, unsafe_allow_html=True)

    with rb:
        urg = urgency_data.get("urgency_level", "LOW")
        urg_color = {"HIGH": "#ff4444", "MEDIUM": "#ffb400", "LOW": "#00e5ff"}[urg]
        st.markdown(f"""
        <div class="card" style="text-align:center">
            <div class="metric-micro">Charging Urgency</div>
            <div class="score-big" style="color:#c8dae8;font-size:2.4rem;margin:8px 0">{soc}%</div>
            <div class="urg-{urg.lower()}" style="margin:8px 0"></div>
            <div class="metric-micro" style="color:{urg_color}">{urg}</div>
        </div>
        """, unsafe_allow_html=True)

    with rc:
        st.markdown(f"""
        <div class="card">
            <div class="metric-micro">HMI · Center Console</div>
            <div style="font-family:'Orbitron',monospace;font-size:1.1rem;color:#00e5ff;margin:12px 0 8px;line-height:1.4">
                📺 {r.get('hmi_message', '—')}
            </div>
            <div class="metric-micro">AI Reasoning</div>
            <div style="font-size:0.8rem;color:#6a9aaa;margin-top:4px">{occupancy_data.get('analysis','—')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="glow-hr"></div>', unsafe_allow_html=True)

    # ── Row 2: Passenger Offer Cards ──
    st.markdown('<div class="section-label">🎁 Personalised Passenger Experience</div>', unsafe_allow_html=True)

    if offers:
        cols = st.columns(len(offers))
        for idx, offer in enumerate(offers):
            ptype = offer["type"]
            css = "offer-child" if ptype == "Child" else ("offer-driver" if ptype == "Driver" else "offer-adult")
            icon = "🧑‍✈️" if ptype == "Driver" else ("👦" if ptype == "Child" else "🧑")
            accent = "#00e5ff" if ptype == "Driver" else ("#c084fc" if ptype == "Child" else "#4ade80")
            name = offer.get("name") or ptype
            with cols[idx]:
                voucher_html = (
                    f'<div class="voucher-code">{offer["voucher_code"]}</div>'
                    if offer.get("qr_enabled") else ""
                )
                st.markdown(f"""
                <div class="{css}">
                    <div style="font-size:1.6rem">{icon}</div>
                    <div style="font-family:'Orbitron',monospace;font-size:0.8rem;
                                color:{accent};font-weight:700;margin:6px 0 2px">{name}</div>
                    <div style="font-size:0.65rem;color:#2a5060;text-transform:uppercase;
                                letter-spacing:1px;margin-bottom:10px">{ptype} · {offer.get('seat','—')}</div>
                    <div style="font-size:0.88rem;color:#d0e8e0;font-weight:500;line-height:1.5">
                        {offer.get('offer','—')}
                    </div>
                    {voucher_html}
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<div class="glow-hr"></div>', unsafe_allow_html=True)

    # ── Row 3: Partner Station + Agent Trace ──
    rd, re = st.columns([1, 1], gap="large")

    with rd:
        st.markdown('<div class="section-label">📍 Recommended Partner Stop</div>', unsafe_allow_html=True)
        amenities_html = " ".join(
            [f'<span style="background:rgba(0,229,255,0.07);border:1px solid rgba(0,229,255,0.2);'
             f'border-radius:20px;padding:2px 10px;font-size:0.68rem;color:#00a0b0;margin-right:4px;">{a}</span>'
             for a in station.get("amenities", [])]
        )
        st.markdown(f"""
        <div class="card">
            <div style="font-family:'Orbitron',monospace;font-size:0.95rem;color:#00e5ff;font-weight:700">
                {station.get('name','—')}
            </div>
            <div style="font-size:0.7rem;color:#2a5060;margin:4px 0 14px">
                {station.get('partner_id','—')} · {station.get('charger_type','—')}
            </div>
            <div style="display:flex;gap:12px;margin-bottom:14px">
                <div style="text-align:center;flex:1;padding:10px;background:rgba(0,229,255,0.04);
                            border:1px solid rgba(0,229,255,0.1);border-radius:8px">
                    <div style="font-family:'Orbitron',monospace;font-size:1.4rem;color:#00e5ff">
                        {station.get('eta_minutes','?')}m
                    </div>
                    <div class="metric-micro">ETA</div>
                </div>
                <div style="text-align:center;flex:1;padding:10px;background:rgba(0,229,255,0.04);
                            border:1px solid rgba(0,229,255,0.1);border-radius:8px">
                    <div style="font-family:'Orbitron',monospace;font-size:1.4rem;color:#00e5ff">
                        ₹{station.get('referral_fee_inr','?')}
                    </div>
                    <div class="metric-micro">Referral</div>
                </div>
                <div style="text-align:center;flex:1;padding:10px;background:rgba(0,229,255,0.04);
                            border:1px solid rgba(0,229,255,0.1);border-radius:8px">
                    <div style="font-family:'Orbitron',monospace;font-size:1.4rem;color:#00e5ff">
                        {station.get('available_slots','?')}
                    </div>
                    <div class="metric-micro">Slots</div>
                </div>
            </div>
            <div style="font-size:0.68rem;color:#2a5060;margin-bottom:6px;letter-spacing:2px;text-transform:uppercase">
                Amenities
            </div>
            {amenities_html}
        </div>
        """, unsafe_allow_html=True)

    with re:
        st.markdown('<div class="section-label">🔧 Full Agent Execution Log</div>', unsafe_allow_html=True)
        all_logs = r.get("agent_logs", [])
        log_html = "".join(f'<div class="trace-line">{l}</div>' for l in all_logs)
        st.markdown(
            f'<div style="max-height:300px;overflow-y:auto;padding-right:4px">{log_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Business Value ──
    st.markdown('<div class="glow-hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">📈 Platform Business Layer</div>', unsafe_allow_html=True)
    bm1, bm2, bm3, bm4 = st.columns(4)
    for col, (val, label, sub) in zip(
        [bm1, bm2, bm3, bm4],
        [
            ("₹" + str(station.get("referral_fee_inr", "—")), "Referral / Stop", "F&B + Gaming + Charging"),
            ("₹2.3L", "Monthly Fleet Rev.", "Per 500 active vehicles"),
            ("18%", "Partner API Margin", "Net after gateway cost"),
            ("A2A Ready", "Protocol Layer", "Agent-to-Agent extensible"),
        ],
    ):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div style="font-family:'Orbitron',monospace;font-size:1.6rem;color:#00e5ff;font-weight:700">{val}</div>
                <div class="metric-micro" style="margin-top:4px">{label}</div>
                <div style="font-size:0.62rem;color:#1a3a4a;margin-top:3px">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ──
st.markdown('<div class="glow-hr"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;font-size:0.62rem;color:#1a3a4a;letter-spacing:2px;text-transform:uppercase">
    LangGraph StateGraph · 4 Specialist Agents · 5 MCP Tools · Gemini 2.5 Flash · Streamlit
    <br>OccupantAnalyzer → ChargingOptimizer → PartnerNegotiator → HMIComposer
</div>
""", unsafe_allow_html=True)
