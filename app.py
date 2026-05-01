# ─────────────────────────────────────────────────────────────────────────────
#  SDV Orchestrator  ·  LangGraph + MCP  ·  Route + Passenger-Aware Station Selection
#  pip install langgraph langchain-google-genai langchain-core streamlit
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import json
import uuid
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
MODEL_NAME = "gemini-3.1-flash-lite-preview"
SAFE_RANGE_FACTOR = 0.85          # keep 15% buffer
MIN_SAFE_BUFFER_KM = 8            # minimum absolute reserve after reaching charger
CRITICAL_SOC = 15
CRITICAL_SAFE_RANGE_KM = 22

st.set_page_config(page_title="SDV", page_icon="⚡", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
#  STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
*, html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0b0f18; color: #b0c4d4; }
section[data-testid="stSidebar"] { display: none; }

.app-title { font-family:'IBM Plex Mono',monospace; font-size:1.1rem; font-weight:600; color:#00d4ff; letter-spacing:3px; text-transform:uppercase; }
.app-sub   { font-size:.7rem; color:#2a5060; letter-spacing:2px; text-transform:uppercase; margin-top:2px; }

.graph-row { display:flex; align-items:center; gap:4px; flex-wrap:wrap; margin-top:8px; }
.gnode      { font-family:'IBM Plex Mono',monospace; font-size:.6rem; padding:4px 10px; border-radius:4px; background:#0f1520; border:1px solid #0e2535; color:#2a6070; white-space:nowrap; }
.gnode-done { background:#0a1a28; border:1px solid #00d4ff40; color:#00d4ff; }
.garrow     { color:#0e2535; font-size:.8rem; }

.sec { font-family:'IBM Plex Mono',monospace; font-size:.6rem; color:#1a6070; letter-spacing:3px; text-transform:uppercase; border-bottom:1px solid #0e2030; padding-bottom:8px; margin-bottom:16px; margin-top:28px; }

.card    { background:#0f1520; border:1px solid #0e2535; border-radius:8px; padding:16px 18px; margin-bottom:10px; }
.card-hi { background:#0a1a28; border:1px solid #0d3550; border-left:3px solid #00d4ff; border-radius:4px 8px 8px 4px; padding:14px 18px; margin-bottom:8px; }

/* station cards */
.station-chosen  { background:#0a1a20; border:1px solid #00d4ff50; border-left:3px solid #00d4ff; border-radius:4px 8px 8px 4px; padding:14px 16px; margin-bottom:8px; }
.station-skipped { background:#0d1520; border:1px solid #0e2030; border-radius:8px; padding:12px 16px; margin-bottom:6px; opacity:.75; }
.station-unreachable { background:#1a0d0d; border:1px solid #3a1010; border-radius:8px; padding:12px 16px; margin-bottom:6px; opacity:.45; }

.tag-child  { font-size:.62rem; background:#13102a; border:1px solid #5030a050; color:#9060e0; padding:2px 8px; border-radius:3px; margin-right:4px; }
.tag-adult  { font-size:.62rem; background:#0d1f18; border:1px solid #20604050; color:#40a060; padding:2px 8px; border-radius:3px; margin-right:4px; }
.tag-charge { font-size:.62rem; background:#0a1a28; border:1px solid #00d4ff30; color:#00a0cc; padding:2px 8px; border-radius:3px; margin-right:4px; }
.tag-route  { font-size:.62rem; background:#18180d; border:1px solid #60602050; color:#c0b060; padding:2px 8px; border-radius:3px; margin-right:4px; }
.tag-score  { font-size:.62rem; background:#102012; border:1px solid #40a06060; color:#60c080; padding:2px 8px; border-radius:3px; margin-right:4px; }

.seats { display:grid; grid-template-columns:1fr 1fr; gap:6px; }
.seat        { background:#0d1825; border:1px solid #0e2535; border-radius:6px; padding:10px 12px; font-size:.72rem; color:#2a5060; }
.seat-child  { background:#13102a; border:1px solid #5030a0; border-radius:6px; padding:10px 12px; font-size:.72rem; color:#9060e0; }
.seat-adult  { background:#0d1f18; border:1px solid #206040; border-radius:6px; padding:10px 12px; font-size:.72rem; color:#40a060; }
.seat-driver { background:#0a1a28; border:1px solid #0d4060; border-radius:6px; padding:10px 12px; font-size:.72rem; color:#00d4ff; grid-column:span 2; text-align:center; }

.offer        { border-radius:8px; padding:16px; margin-bottom:8px; }
.offer-child  { background:#13102a; border:1px solid #5030a0; }
.offer-adult  { background:#0d1f18; border:1px solid #206040; }
.offer-driver { background:#0a1a28; border:1px solid #0d4060; }

.voucher { font-family:'IBM Plex Mono',monospace; font-size:.68rem; color:#00d4ff; background:#0a1a28; border:1px solid #0d3550; padding:5px 10px; border-radius:4px; display:inline-block; margin-top:8px; letter-spacing:1px; }

.score-num { font-family:'IBM Plex Mono',monospace; font-size:3rem; font-weight:600; line-height:1; }
.stat { text-align:center; padding:14px; background:#0f1520; border:1px solid #0e2535; border-radius:6px; }
.stat-val { font-family:'IBM Plex Mono',monospace; font-size:1.4rem; color:#00d4ff; font-weight:600; }
.stat-lbl { font-size:.6rem; color:#1a5060; letter-spacing:2px; text-transform:uppercase; margin-top:3px; }

.biz-row { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-top:4px; }
.biz-card { background:#0a1520; border:1px solid #0d2535; border-radius:6px; padding:12px 14px; }
.biz-val  { font-family:'IBM Plex Mono',monospace; font-size:1.1rem; color:#00d4ff; font-weight:600; }
.biz-lbl  { font-size:.6rem; color:#1a5060; letter-spacing:1px; text-transform:uppercase; margin-top:2px; }
.biz-sub  { font-size:.65rem; color:#1a4050; margin-top:4px; }

.badge-on  { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:.6rem; padding:3px 10px; border-radius:3px; background:#00d4ff15; border:1px solid #00d4ff40; color:#00d4ff; letter-spacing:2px; }
.badge-off { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:.6rem; padding:3px 10px; border-radius:3px; background:#cc303015; border:1px solid #cc303040; color:#cc6060; letter-spacing:2px; }

.trace { font-family:'IBM Plex Mono',monospace; font-size:.65rem; color:#20a060; background:#080d12; border-left:2px solid #0a3020; padding:5px 10px; margin-bottom:2px; border-radius:0 4px 4px 0; }
.reasoning { font-size:.82rem; color:#7a9aaa; line-height:1.7; }

.u-critical { height:4px; background:linear-gradient(90deg,#8b0000,#ff3030); border-radius:2px; }
.u-high     { height:4px; background:linear-gradient(90deg,#cc2020,#ff5500); border-radius:2px; }
.u-medium   { height:4px; background:linear-gradient(90deg,#cc7700,#ffaa00); border-radius:2px; }
.u-low      { height:4px; background:linear-gradient(90deg,#007799,#00d4ff); border-radius:2px; }
.hr { height:1px; background:#0e2030; margin:24px 0; }

div.stButton > button { background:#00d4ff10; color:#00d4ff; border:1px solid #00d4ff40; border-radius:6px; width:100%; font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:2px; padding:10px; transition:all .2s; }
div.stButton > button:hover { background:#00d4ff20; border-color:#00d4ff; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def extract_text(response) -> str:
    c = response.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        for b in c:
            if isinstance(b, dict) and b.get("type") == "text":
                return b["text"]
    return str(c)


def llm():
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.2,
    )


def safe_json_from_llm(text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    """Robustly parse JSON returned by the model."""
    raw = text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return fallback


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTE-AWARE STATION NETWORK
#  Each station has:
#  - sector: physical sector where the station exists
#  - routes: km/detour from current origin sector
#  - amenity score: child/adult passenger experience
# ══════════════════════════════════════════════════════════════════════════════
ROUTE_CORRIDORS = {
    "Pune Airport": {
        "Hinjewadi Ph 1": ["Hinjewadi Ph 1", "Wakad Bypass", "Baner High St", "Aundh", "Kharadi", "Pune Airport"],
        "Wakad Bypass":   ["Wakad Bypass", "Baner High St", "Aundh", "Kharadi", "Pune Airport"],
        "Baner High St":  ["Baner High St", "Aundh", "Kharadi", "Pune Airport"],
        "Aundh":          ["Aundh", "Kharadi", "Pune Airport"],
    },
    "Pune City": {
        "Hinjewadi Ph 1": ["Hinjewadi Ph 1", "Wakad Bypass", "Baner High St", "Aundh", "Pune City"],
        "Wakad Bypass":   ["Wakad Bypass", "Baner High St", "Aundh", "Pune City"],
        "Baner High St":  ["Baner High St", "Aundh", "Pune City"],
        "Aundh":          ["Aundh", "Pune City"],
    },
}

ALL_STATIONS = [
    {
        "id": "TATA_HIN_01",
        "name": "TATA Power EV Hub · Hinjewadi",
        "sector": "Hinjewadi Ph 1",
        "partner_id": "TATA_PNQ_HIN_01",
        "charger": "DC Fast 150 kW",
        "max_kw": 150,
        "slots": 3,
        "fee": 55,
        "margin_pct": 19,
        "amenities": ["AirConsole Gaming Lounge", "McDonald's", "Starbucks"],
        "child_score": 9,
        "adult_score": 7,
        "child_offer": "AirConsole Pro — 45 min session unlocked",
        "adult_offer": "Starbucks 20% off · 2× loyalty points",
        "routes": {
            "Hinjewadi Ph 1": {"km": 2,  "detour_km": 0},
            "Wakad Bypass":   {"km": 5,  "detour_km": 1},
            "Baner High St":  {"km": 12, "detour_km": 3},
            "Aundh":          {"km": 8,  "detour_km": 2},
        },
    },
    {
        "id": "ATHER_WKD_02",
        "name": "Ather Grid · Wakad",
        "sector": "Wakad Bypass",
        "partner_id": "ATHER_PNQ_WKD_02",
        "charger": "DC Fast 100 kW",
        "max_kw": 100,
        "slots": 5,
        "fee": 42,
        "margin_pct": 16,
        "amenities": ["Kids Zone", "Café Coffee Day", "Domino's"],
        "child_score": 7,
        "adult_score": 6,
        "child_offer": "Kids Zone pass — 30 min complimentary",
        "adult_offer": "CCD buy-1-get-1 · Domino's ₹100 off",
        "routes": {
            "Hinjewadi Ph 1": {"km": 8,  "detour_km": 1},
            "Wakad Bypass":   {"km": 2,  "detour_km": 0},
            "Baner High St":  {"km": 7,  "detour_km": 1},
            "Aundh":          {"km": 5,  "detour_km": 1},
        },
    },
    {
        "id": "CZ_BAN_03",
        "name": "ChargeZone Premium · Baner",
        "sector": "Baner High St",
        "partner_id": "CZ_PNQ_BAN_03",
        "charger": "AC+DC 22–100 kW",
        "max_kw": 100,
        "slots": 2,
        "fee": 38,
        "margin_pct": 15,
        "amenities": ["GameOn Arcade", "Chaayos", "Subway"],
        "child_score": 8,
        "adult_score": 7,
        "child_offer": "GameOn Arcade — 3 free credits",
        "adult_offer": "Chaayos ₹60 voucher + free snack",
        "routes": {
            "Hinjewadi Ph 1": {"km": 14, "detour_km": 2},
            "Wakad Bypass":   {"km": 9,  "detour_km": 1},
            "Baner High St":  {"km": 2,  "detour_km": 0},
            "Aundh":          {"km": 4,  "detour_km": 0},
        },
    },
    {
        "id": "ZEON_AUN_04",
        "name": "Zeon Charge · Aundh",
        "sector": "Aundh",
        "partner_id": "ZEON_PNQ_AUN_04",
        "charger": "DC Fast 120 kW",
        "max_kw": 120,
        "slots": 4,
        "fee": 48,
        "margin_pct": 17,
        "amenities": ["SkyZone Trampoline", "Burger King", "Baskin-Robbins"],
        "child_score": 10,
        "adult_score": 5,
        "child_offer": "SkyZone Trampoline — 1 hr session FREE",
        "adult_offer": "Burger King Whopper combo ₹80 off",
        "routes": {
            "Hinjewadi Ph 1": {"km": 11, "detour_km": 2},
            "Wakad Bypass":   {"km": 7,  "detour_km": 2},
            "Baner High St":  {"km": 6,  "detour_km": 1},
            "Aundh":          {"km": 2,  "detour_km": 0},
        },
    },
    {
        "id": "TPDDL_KHR_05",
        "name": "TPDDL EV Point · Kharadi",
        "sector": "Kharadi",
        "partner_id": "TPDDL_PNQ_KHR_05",
        "charger": "AC 22 kW",
        "max_kw": 22,
        "slots": 6,
        "fee": 28,
        "margin_pct": 12,
        "amenities": ["Phoenix Mall Food Court", "PVR Cinema", "Timezone Arcade"],
        "child_score": 8,
        "adult_score": 9,
        "child_offer": "Timezone Arcade — ₹200 game credits",
        "adult_offer": "Phoenix Mall ₹500 voucher · PVR 30% off",
        "routes": {
            "Hinjewadi Ph 1": {"km": 28, "detour_km": 4},
            "Wakad Bypass":   {"km": 22, "detour_km": 3},
            "Baner High St":  {"km": 19, "detour_km": 2},
            "Aundh":          {"km": 24, "detour_km": 5},
        },
    },
]


def normalize_destination(destination: str) -> str:
    d = destination.lower()
    if "airport" in d or "t1" in d or "t2" in d:
        return "Pune Airport"
    return "Pune City"


def get_route_corridor(origin: str, destination: str) -> List[str]:
    dest_key = normalize_destination(destination)
    return ROUTE_CORRIDORS.get(dest_key, ROUTE_CORRIDORS["Pune City"]).get(origin, [origin])


def classify_route_alignment(station_sector: str, corridor: List[str]) -> Dict[str, Any]:
    """
    Route-aware classification:
    ON_ROUTE  : station sector is present in the origin-to-destination corridor.
    NEAR_ROUTE: not on modeled corridor but still considered nearby.
    OFF_ROUTE : fallback for future expansion.
    """
    if station_sector in corridor:
        idx = corridor.index(station_sector)
        progress_pct = int((idx / max(1, len(corridor) - 1)) * 100)
        return {
            "route_alignment": "ON_ROUTE",
            "route_progress_pct": progress_pct,
            "route_index": idx,
            "on_route": True,
        }
    return {
        "route_alignment": "NEAR_ROUTE",
        "route_progress_pct": 0,
        "route_index": 99,
        "on_route": False,
    }


def compute_station_score(
    station: Dict[str, Any],
    urgency_level: str,
    is_critical: bool,
    has_child: bool,
    has_adult: bool,
    safe_range_km: int,
) -> Dict[str, Any]:
    """
    Deterministic scoring layer.
    This is what fixes the earlier issue where the same nearest station was usually selected.
    Gemini is used later for explanation/HMI language, not for unstable core selection.
    """
    km = station["km_from_here"]
    detour = station["detour_km"]
    effective_km = km + detour
    buffer_after_stop = safe_range_km - effective_km

    if buffer_after_stop < 0:
        return {
            "decision_score": -999,
            "decision_mode": "OUT_OF_SAFE_RANGE",
            "score_breakdown": "Rejected: outside safe range",
        }

    child_score = station["child_score"]
    adult_score = station["adult_score"]
    slots = station["slots"]
    max_kw = station["max_kw"]
    alignment = station.get("route_alignment", "NEAR_ROUTE")
    progress = station.get("route_progress_pct", 0)

    # Safety-first mode: pick nearest safe station with charger reliability as secondary.
    if is_critical:
        score = 100
        score -= km * 4.0
        score -= detour * 9.0
        score += min(max_kw / 20, 8)
        score += min(slots * 2, 8)
        score += 12 if alignment == "ON_ROUTE" else -10
        breakdown = (
            f"critical safety mode: distance penalty {km} km, detour {detour} km, "
            f"route alignment {alignment}, charger {max_kw} kW, slots {slots}"
        )
        return {
            "decision_score": round(score, 1),
            "decision_mode": "SAFETY_FIRST",
            "score_breakdown": breakdown,
        }

    score = 0.0

    # Route fit: prefer same-route stations. Further on-route station can be better.
    if alignment == "ON_ROUTE":
        score += 22
        score += min(progress / 10, 8)     # encourages useful forward progress on same route
    else:
        score -= 12

    # Battery urgency: HIGH still cares about distance, but does not blindly force nearest.
    if urgency_level == "HIGH":
        score += max(0, 40 - km * 1.8)
        score -= detour * 6
        decision_mode = "RANGE_PROTECTED_OPTIMIZATION"
    elif urgency_level == "MEDIUM":
        score += max(0, 28 - km * 0.8)
        score -= detour * 4
        decision_mode = "BALANCED_ROUTE_PASSENGER_OPTIMIZATION"
    else:
        score += max(0, 18 - km * 0.35)
        score -= detour * 2.5
        decision_mode = "EXPERIENCE_OPTIMIZATION"

    # Passenger experience scoring.
    if has_child and has_adult:
        score += child_score * 5.0
        score += adult_score * 4.0
        pax_reason = "mixed cabin: child and adult scores balanced"
    elif has_child:
        score += child_score * 7.0
        score += adult_score * 1.0
        pax_reason = "child cabin: child experience heavily weighted"
    elif has_adult:
        score += adult_score * 6.0
        score += child_score * 0.5
        pax_reason = "adult cabin: adult comfort heavily weighted"
    else:
        score += adult_score * 2.0
        score += child_score * 1.0
        pax_reason = "driver-only: basic comfort weighted"

    # Charger quality and availability.
    score += min(max_kw / 10, 16)
    score += min(slots * 3, 12)

    # Safety reserve after reaching station.
    if buffer_after_stop >= MIN_SAFE_BUFFER_KM:
        score += min(buffer_after_stop / 5, 10)
    else:
        score -= 15

    breakdown = (
        f"{decision_mode}; {pax_reason}; route={alignment}; progress={progress}%; "
        f"distance={km} km; detour={detour} km; buffer_after_stop={buffer_after_stop} km; "
        f"charger={max_kw} kW; slots={slots}"
    )
    return {
        "decision_score": round(score, 1),
        "decision_mode": decision_mode,
        "score_breakdown": breakdown,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MCP TOOLS
# ══════════════════════════════════════════════════════════════════════════════
@tool
def get_stations_on_route(origin: str, destination: str, range_km: int) -> str:
    """[MCP:RouteAPI v3.1] Return partner stations reachable on/near the route with route position and detour details."""
    corridor = get_route_corridor(origin, destination)
    safe_range_km = int(range_km * SAFE_RANGE_FACTOR)
    reachable, unreachable = [], []

    for s in ALL_STATIONS:
        pos = s["routes"].get(origin, {"km": 99, "detour_km": 10})
        km = pos["km"]
        det = pos["detour_km"]
        effective_km = km + det
        eta = round(km / 30 * 60)   # ~30 km/h city average → minutes
        align = classify_route_alignment(s["sector"], corridor)

        entry = {
            **{k: v for k, v in s.items() if k != "routes"},
            "km_from_here": km,
            "detour_km": det,
            "effective_km": effective_km,
            "eta_min": eta,
            "safe_range_km": safe_range_km,
            **align,
        }

        if effective_km <= safe_range_km:
            reachable.append(entry)
        else:
            entry["reason"] = (
                f"Requires {effective_km} km including detour, "
                f"only {safe_range_km} km safe range available"
            )
            unreachable.append(entry)

    # First sort by route order and distance. Final selection is done in PartnerNegotiator.
    reachable.sort(key=lambda x: (x["route_index"], x["effective_km"]))
    unreachable.sort(key=lambda x: (x["route_index"], x["effective_km"]))

    return json.dumps({
        "reachable": reachable,
        "unreachable": unreachable,
        "origin": origin,
        "destination": destination,
        "destination_key": normalize_destination(destination),
        "corridor": corridor,
        "safe_range_km": safe_range_km,
    })


@tool
def calculate_urgency(soc: int, range_km: int, trip_km: int) -> str:
    """[MCP:BatteryMgmt v3.1] Compute charging urgency and safety mode from telemetry."""
    deficit = trip_km - range_km
    is_critical = soc <= CRITICAL_SOC or range_km <= CRITICAL_SAFE_RANGE_KM or deficit > 20

    if is_critical:
        level, contrib = "CRITICAL", 45
    elif soc < 25 or deficit > 0:
        level, contrib = "HIGH", 40
    elif soc < 50:
        level, contrib = "MEDIUM", 30
    else:
        level, contrib = "LOW", 10

    return json.dumps({
        "level": level,
        "score": contrib,
        "deficit_km": max(0, deficit),
        "range_km": range_km,
        "safe_range_km": int(range_km * SAFE_RANGE_FACTOR),
        "critical": is_critical,
    })


@tool
def generate_voucher(partner_id: str, pax_type: str, offer: str) -> str:
    """[MCP:PartnerNegotiation v2.0] Generate a personalised discount voucher."""
    tag = "KID" if pax_type == "Child" else "VIP"
    code = f"{tag}-{partner_id[:4].upper()}-{uuid.uuid4().hex[:5].upper()}"
    return json.dumps({"code": code, "offer": offer})


@tool
def push_hmi(message: str, urgency: str) -> str:
    """[MCP:HMIBridge v4.1] Push alert to vehicle center console and HUD."""
    return json.dumps({"status": "PUSHED", "message": message, "urgency": urgency})


@tool
def compute_business_projection(
    engagement_score: int,
    referral_fee: int,
    margin_pct: int,
    urgency_level: str,
    fleet_size: int = 500,
) -> str:
    """[MCP:BusinessIntelligence v1.0] Dynamic revenue projection from live trip context."""
    conv = 0.3 + (engagement_score / 100) * 0.5
    stops = {"CRITICAL": 2.4, "HIGH": 2.1, "MEDIUM": 1.4, "LOW": 0.7}.get(urgency_level, 1.0)
    monthly_stops = fleet_size * stops * 22 * conv
    monthly_rev = monthly_stops * referral_fee
    return json.dumps({
        "conversion_rate_pct": round(conv * 100, 1),
        "stops_per_vehicle_day": stops,
        "monthly_activated_stops": int(monthly_stops),
        "monthly_gross_inr": int(monthly_rev),
        "monthly_net_inr": int(monthly_rev * margin_pct / 100),
        "margin_pct": margin_pct,
    })


# ══════════════════════════════════════════════════════════════════════════════
#  LANGGRAPH STATE
# ══════════════════════════════════════════════════════════════════════════════
class SDVState(TypedDict):
    soc: int
    range_km: int
    trip_km: int
    location: str
    destination: str
    passengers: List[dict]
    partner_active: bool
    v2x_active: bool
    urgency: dict
    all_stations: dict
    chosen_station: dict
    station_reasoning: str
    offers: List[dict]
    score: int
    service_enabled: bool
    hmi_msg: str
    reasoning: str
    biz: dict
    logs: List[str]


# ── Node 1: OccupantAnalyzer ──────────────────────────────────────────────────
def node_occupant(state: SDVState) -> dict:
    logs = list(state.get("logs", []))
    logs.append("→ [OccupantAnalyzer] profiling cabin via MCP:OccupantProfiler")
    has_child = any(p["type"] == "Child" for p in state["passengers"])
    has_adult = any(p["type"] == "Adult" for p in state["passengers"])
    passenger_mix = "mixed" if has_child and has_adult else ("child" if has_child else ("adult" if has_adult else "driver-only"))
    logs.append(f"  cabin_profile:{passenger_mix}  children:{has_child}  adults:{has_adult}  v2x:{state.get('v2x_active')}")
    return {"logs": logs}


# ── Node 2: ChargingOptimizer ─────────────────────────────────────────────────
def node_charging(state: SDVState) -> dict:
    logs = list(state.get("logs", []))
    logs.append("→ [ChargingOptimizer] querying MCP:BatteryMgmt + MCP:RouteAPI")

    urg = json.loads(calculate_urgency.invoke({
        "soc": state["soc"],
        "range_km": state["range_km"],
        "trip_km": state["trip_km"],
    }))

    stations_raw = get_stations_on_route.invoke({
        "origin": state["location"],
        "destination": state["destination"],
        "range_km": state["range_km"],
    })
    all_stations = json.loads(stations_raw)
    reachable = all_stations.get("reachable", [])
    unreachable = all_stations.get("unreachable", [])

    logs.append(
        f"  urgency:{urg['level']}  range:{state['range_km']} km  safe_range:{urg['safe_range_km']} km  critical:{urg['critical']}"
    )
    logs.append(f"  route:{state['location']} → {state['destination']}")
    logs.append(f"  corridor:{' > '.join(all_stations.get('corridor', []))}")
    logs.append(f"  MCP:RouteAPI → {len(reachable)} reachable, {len(unreachable)} out of safe range")

    for s in reachable:
        logs.append(
            f"    ✓ {s['name']}  [{s['route_alignment']} · {s['km_from_here']} km · +{s['detour_km']} km detour · "
            f"child:{s['child_score']} adult:{s['adult_score']} · {s['max_kw']} kW · slots:{s['slots']}]")
    for s in unreachable:
        logs.append(f"    ✗ {s['name']}  [{s['reason']}]")

    return {"urgency": urg, "all_stations": all_stations, "logs": logs}


# ── Node 3: PartnerNegotiator — deterministic selection + Gemini explanation ──
def node_partner(state: SDVState) -> dict:
    logs = list(state.get("logs", []))
    logs.append("→ [PartnerNegotiator] scoring stations using route + passenger + battery context")

    reachable = state["all_stations"].get("reachable", [])
    urgency = state["urgency"]
    passengers = state["passengers"]
    has_child = any(p["type"] == "Child" for p in passengers)
    has_adult = any(p["type"] == "Adult" for p in passengers)
    safe_range_km = urgency.get("safe_range_km", int(state["range_km"] * SAFE_RANGE_FACTOR))

    if not reachable:
        logs.append("  ✗ No reachable stations — cannot recommend stop")
        return {
            "chosen_station": {},
            "station_reasoning": "No stations within safe range.",
            "offers": [],
            "logs": logs,
        }

    scored = []
    for s in reachable:
        scoring = compute_station_score(
            station=s,
            urgency_level=urgency["level"],
            is_critical=urgency.get("critical", False),
            has_child=has_child,
            has_adult=has_adult,
            safe_range_km=safe_range_km,
        )
        s2 = {**s, **scoring}
        scored.append(s2)

    scored.sort(key=lambda x: x["decision_score"], reverse=True)
    chosen = scored[0]

    for s in scored:
        logs.append(
            f"  score:{s['decision_score']}  {s['name']}  mode:{s['decision_mode']}  "
            f"route:{s['route_alignment']}  km:{s['km_from_here']}  detour:{s['detour_km']}  "
            f"child:{s['child_score']} adult:{s['adult_score']}"
        )

    second_best = scored[1] if len(scored) > 1 else None

    # Gemini explains the deterministic decision, but does not override the decision.
    candidates_summary = "\n".join([
        f"- {s['name']} | score:{s['decision_score']} | mode:{s['decision_mode']} | "
        f"route:{s['route_alignment']} progress:{s['route_progress_pct']}% | "
        f"km:{s['km_from_here']} detour:{s['detour_km']} | "
        f"child_score:{s['child_score']}/10 adult_score:{s['adult_score']}/10 | "
        f"charger:{s['charger']} slots:{s['slots']}"
        for s in scored
    ])

    prompt = f"""You are the SDV PartnerNegotiator agent.
Explain the station selection decision for a client demo.

VEHICLE STATE:
- Battery urgency: {urgency['level']}  critical={urgency.get('critical')}
- SoC: {state['soc']}%  range: {state['range_km']} km  safe_range: {safe_range_km} km
- Passengers: child_onboard={has_child}, adult_onboard={has_adult}
- Trip: {state['location']} → {state['destination']} ({state['trip_km']} km)

SCORING RESULT:
Chosen station: {chosen['name']}
Chosen station score breakdown: {chosen['score_breakdown']}

ALL SCORED CANDIDATES:
{candidates_summary}

Return ONLY valid JSON, no markdown:
{{
  "selection_reason": "2 sentences explaining why the chosen station is best. Mention route, passenger fit, battery safety, and why it can be farther than nearest if applicable.",
  "trade_off": "1 sentence explaining what was sacrificed, e.g. extra distance/detour for better child/adult experience or safety."
}}"""

    fallback_reason = {
        "selection_reason": (
            f"{chosen['name']} was selected with score {chosen['decision_score']} because it fits the route, "
            f"battery safety, passenger profile, charger speed, and slot availability better than alternatives."
        ),
        "trade_off": (
            f"Compared with {second_best['name'] if second_best else 'other stations'}, the system accepted the best overall trade-off."
        ),
    }

    try:
        resp = llm().invoke([HumanMessage(content=prompt)])
        decision = safe_json_from_llm(extract_text(resp), fallback_reason)
    except Exception:
        decision = fallback_reason

    logs.append(f"  selected:{chosen['name']}  final_score:{chosen['decision_score']}")
    logs.append(f"  reason:{decision.get('selection_reason', '')}")

    # Generate vouchers for the chosen station.
    offers = []
    for p in passengers:
        if p["type"] == "Driver":
            offers.append({**p, "offer": "Navigation routed to charging stop", "code": None})
            continue

        if not state.get("partner_active"):
            offers.append({**p, "offer": "Partner offers disabled", "code": None})
            continue

        tmpl = chosen["child_offer"] if p["type"] == "Child" else chosen["adult_offer"]
        v = json.loads(generate_voucher.invoke({
            "partner_id": chosen["partner_id"],
            "pax_type": p["type"],
            "offer": tmpl,
        }))
        offers.append({**p, "offer": v["offer"], "code": v["code"]})
        logs.append(f"  MCP:PartnerNegotiation → {v['code']} issued to {p.get('name') or p['type']}")

    return {
        "all_stations": {**state["all_stations"], "reachable": scored},
        "chosen_station": chosen,
        "station_reasoning": decision.get("selection_reason", "") + " " + decision.get("trade_off", ""),
        "offers": offers,
        "logs": logs,
    }


# ── Node 4: HMIComposer ────────────────────────────────────────────────────────
def node_hmi(state: SDVState) -> dict:
    logs = list(state.get("logs", []))
    logs.append("→ [HMIComposer] computing engagement score + business projection")

    urg = state.get("urgency", {})
    has_c = any(p["type"] == "Child" for p in state["passengers"])
    has_a = any(p["type"] == "Adult" for p in state["passengers"])
    station = state.get("chosen_station", {})

    score = min(100,
        urg.get("score", 10)
        + (25 if has_c else 0)
        + (15 if has_a else 0)
        + (20 if state.get("partner_active") else 0)
        + (5 if state.get("v2x_active") else 0)
        + (8 if station.get("route_alignment") == "ON_ROUTE" else 0)
        + min(int(station.get("decision_score", 0) / 15), 8)
    )
    enabled = score >= 65 and bool(station)

    if station:
        hmi_prompt = (
            f"Automotive HMI. Write a center-console notification, max 14 words. "
            f"urgency={urg.get('level')} station={station.get('name')} "
            f"ETA={station.get('eta_min')} min km_away={station.get('km_from_here')} "
            f"children_onboard={has_c}. Return ONLY the message text."
        )
        reason_prompt = (
            f"SDV product analyst. 2-sentence explanation of WHY the system chose "
            f"{station.get('name')} over other options. "
            f"Score:{score}/100 urgency:{urg.get('level')} SoC:{state['soc']}% "
            f"children:{has_c} adults:{has_a} route_alignment:{station.get('route_alignment')} "
            f"km_away:{station.get('km_from_here')} child_score:{station.get('child_score')} "
            f"adult_score:{station.get('adult_score')} detour:{station.get('detour_km')} km. "
            f"Return only 2 sentences."
        )
        try:
            resp = llm().invoke([HumanMessage(content=hmi_prompt)])
            msg = extract_text(resp).strip().strip('"')
        except Exception:
            msg = f"Recommended stop: {station.get('name')} in {station.get('eta_min')} minutes."

        try:
            resp2 = llm().invoke([HumanMessage(content=reason_prompt)])
            reasoning = extract_text(resp2).strip()
        except Exception:
            reasoning = state.get("station_reasoning", "")

        push_hmi.invoke({"message": msg, "urgency": urg.get("level", "MEDIUM")})

        biz = json.loads(compute_business_projection.invoke({
            "engagement_score": score,
            "referral_fee": station.get("fee", 42),
            "margin_pct": station.get("margin_pct", 16),
            "urgency_level": urg.get("level", "MEDIUM"),
            "fleet_size": 500,
        }))
    else:
        msg = "No safe charging stop available on current route."
        reasoning = "No station is within the safe range buffer. The system should alert the driver and request a route change."
        biz = {}

    logs.append(f"  score:{score}/100  service_enabled:{enabled}")
    if biz:
        logs.append(f"  MCP:BusinessIntelligence → monthly gross: ₹{biz['monthly_gross_inr']:,}")
    logs.append(f"  MCP:HMIBridge PUSHED → \"{msg}\"")
    logs.append("✓ Orchestration complete")

    return {
        "score": score,
        "service_enabled": enabled,
        "hmi_msg": msg,
        "reasoning": reasoning,
        "biz": biz,
        "logs": logs,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD GRAPH
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def build_graph():
    g = StateGraph(SDVState)
    g.add_node("occupant", node_occupant)
    g.add_node("charging", node_charging)
    g.add_node("partner", node_partner)
    g.add_node("hmi", node_hmi)
    g.add_edge(START, "occupant")
    g.add_edge("occupant", "charging")
    g.add_edge("charging", "partner")
    g.add_edge("partner", "hmi")
    g.add_edge("hmi", END)
    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION
# ══════════════════════════════════════════════════════════════════════════════
if "passengers" not in st.session_state:
    st.session_state.passengers = [{"id": "drv", "name": "Driver", "type": "Driver", "seat": "Front-L"}]
if "result" not in st.session_state:
    st.session_state.result = None

SEAT_SLOTS = ["Front-R", "Rear-L", "Rear-C", "Rear-R"]

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
h1, h2 = st.columns([2, 3])
with h1:
    st.markdown(
        '<div class="app-title">⚡SDV Orchestrator</div>'
        '<div class="app-sub">Route-Aware · Passenger-Personalised · LangGraph + MCP</div>',
        unsafe_allow_html=True,
    )
with h2:
    nodes = ["OccupantAnalyzer", "ChargingOptimizer", "PartnerNegotiator", "HMIComposer"]
    done = 4 if st.session_state.result else 0
    html = '<div class="graph-row"><span class="gnode">START</span><span class="garrow">→</span>'
    for i, n in enumerate(nodes):
        css = "gnode-done" if i < done else "gnode"
        html += f'<span class="{css}">{n}</span>'
        html += '<span class="garrow">→</span>'
    html += '<span class="gnode-done" style="opacity:.4">END</span></div>'
    st.markdown(html, unsafe_allow_html=True)

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1], gap="large")

# ─── LEFT ─────────────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="sec">In-Cabin Occupancy</div>', unsafe_allow_html=True)
    occupied = {p["seat"]: p for p in st.session_state.passengers}
    seat_html = '<div class="seats"><div class="seat-driver">Driver · Front-L</div>'
    for slot in SEAT_SLOTS:
        if slot in occupied:
            p = occupied[slot]
            css = "seat-child" if p["type"] == "Child" else "seat-adult"
            ico = "Child" if p["type"] == "Child" else "Adult"
            seat_html += f'<div class="{css}">{ico}: {p.get("name") or p["type"]}<br><span style="opacity:.5">{slot}</span></div>'
        else:
            seat_html += f'<div class="seat">Open · {slot}</div>'
    seat_html += '</div>'
    st.markdown(seat_html, unsafe_allow_html=True)

    with st.expander("＋ Add passenger", expanded=len(st.session_state.passengers) == 1):
        with st.form("pax", clear_on_submit=True):
            c1, c2, c3 = st.columns([2, 1, 2])
            name = c1.text_input("Name", placeholder="optional")
            ptype = c2.selectbox("Type", ["Child", "Adult"])
            free = [s for s in SEAT_SLOTS if s not in occupied]
            seat = c3.selectbox("Seat", free) if free else None
            if st.form_submit_button("Add", use_container_width=True) and seat:
                st.session_state.passengers.append({
                    "id": uuid.uuid4().hex[:6],
                    "name": name.strip() or None,
                    "type": ptype,
                    "seat": seat,
                })
                st.rerun()

    for p in st.session_state.passengers:
        if p["type"] == "Driver":
            continue
        label = p.get("name") or p["type"]
        color = "#9060e0" if p["type"] == "Child" else "#40a060"
        pc1, pc2 = st.columns([5, 1])
        pc1.markdown(
            f'<div class="card-hi" style="border-left-color:{color}">'
            f'<b style="color:{color}">{label}</b>'
            f'<span style="color:#1a4050;font-size:.7rem;margin-left:8px">{p["type"]} · {p["seat"]}</span></div>',
            unsafe_allow_html=True,
        )
        if pc2.button("✕", key=p["id"]):
            st.session_state.passengers = [x for x in st.session_state.passengers if x["id"] != p["id"]]
            st.rerun()

    st.markdown('<div class="sec">Vehicle Telemetry</div>', unsafe_allow_html=True)
    soc = st.slider("Battery SoC %", 5, 100, 32)
    range_km = st.slider("Estimated range (km)", 10, 300, int(soc * 2.8))
    trip_km = st.slider("Trip distance (km)", 10, 150, 55)

    st.markdown('<div class="sec">Geo / Partner</div>', unsafe_allow_html=True)
    location = st.selectbox("Current sector", ["Hinjewadi Ph 1", "Wakad Bypass", "Baner High St", "Aundh"])
    destination = st.text_input("Destination", "Pune Airport, T1")
    tc1, tc2 = st.columns(2)
    partner_on = tc1.toggle("Partner API", value=True)
    v2x_on = tc2.toggle("V2X Signal", value=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    run = st.button("RUN ORCHESTRATION CYCLE", use_container_width=True)

# ─── RIGHT ────────────────────────────────────────────────────────────────────
with right:
    if run:
        with st.spinner("Running LangGraph agents…"):
            result = build_graph().invoke({
                "soc": soc,
                "range_km": range_km,
                "trip_km": trip_km,
                "location": location,
                "destination": destination,
                "passengers": st.session_state.passengers,
                "partner_active": partner_on,
                "v2x_active": v2x_on,
                "urgency": {},
                "all_stations": {},
                "chosen_station": {},
                "station_reasoning": "",
                "offers": [],
                "score": 0,
                "service_enabled": False,
                "hmi_msg": "",
                "reasoning": "",
                "biz": {},
                "logs": [],
            })
        st.session_state.result = result

    r = st.session_state.result

    if not r:
        st.markdown(
            '<div style="height:260px;display:flex;align-items:center;justify-content:center;'
            'color:#0e2535;font-family:\'IBM Plex Mono\',monospace;font-size:.75rem;letter-spacing:2px">'
            'AWAITING ORCHESTRATION CYCLE</div>',
            unsafe_allow_html=True,
        )
    else:
        urg = r.get("urgency", {})
        chosen = r.get("chosen_station", {})
        all_st = r.get("all_stations", {})
        offers = r.get("offers", [])
        score = r.get("score", 0)
        enabled = r.get("service_enabled", False)
        hmi_msg = r.get("hmi_msg", "")
        reasoning = r.get("reasoning", "")
        stn_reason = r.get("station_reasoning", "")
        biz = r.get("biz", {})
        logs = r.get("logs", [])
        urg_level = urg.get("level", "LOW")
        score_color = "#00d4ff" if score > 65 else ("#ffaa00" if score > 40 else "#cc3030")
        badge_html = f'<span class="{"badge-on" if enabled else "badge-off"}">{"SERVICES ON" if enabled else "STANDBY"}</span>'

        # ── Scores ──
        st.markdown('<div class="sec">Result</div>', unsafe_allow_html=True)
        sa, sb, sc = st.columns(3)
        sa.markdown(
            f'<div class="stat"><div class="score-num" style="color:{score_color}">{score}</div>'
            f'<div style="margin-top:6px">{badge_html}</div></div>',
            unsafe_allow_html=True,
        )
        sb.markdown(
            f'<div class="stat"><div class="stat-val">{soc}%</div>'
            f'<div class="u-{urg_level.lower()}" style="margin:6px 0"></div>'
            f'<div class="stat-lbl">{urg_level}</div></div>',
            unsafe_allow_html=True,
        )
        sc.markdown(
            f'<div class="stat"><div class="stat-val">{chosen.get("eta_min", "?")}m</div>'
            f'<div class="stat-lbl">{chosen.get("km_from_here", "?")} km away</div></div>',
            unsafe_allow_html=True,
        )

        # ── HMI ──
        st.markdown(
            f'<div class="card" style="margin-top:12px">'
            f'<div style="font-size:.6rem;color:#1a5060;letter-spacing:2px;text-transform:uppercase;font-family:\'IBM Plex Mono\',monospace">HMI · Center Console</div>'
            f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:.95rem;color:#00d4ff;margin-top:8px">{hmi_msg}</div></div>',
            unsafe_allow_html=True,
        )

        # ── Station selection ──
        st.markdown('<div class="sec">Station Selection · Route + Passenger-Aware Decision</div>', unsafe_allow_html=True)

        reachable = all_st.get("reachable", [])
        unreachable = all_st.get("unreachable", [])

        for s in reachable:
            is_chosen = s["id"] == chosen.get("id")
            css = "station-chosen" if is_chosen else "station-skipped"
            prefix = "✓ CHOSEN" if is_chosen else "○ considered"
            prefix_c = "#00d4ff" if is_chosen else "#2a5060"
            child_tag = f'<span class="tag-child">child:{s["child_score"]}/10</span>'
            adult_tag = f'<span class="tag-adult">adult:{s["adult_score"]}/10</span>'
            km_tag = f'<span class="tag-charge">{s["km_from_here"]} km · {s["eta_min"]} min</span>'
            det_tag = f'<span class="tag-charge">+{s["detour_km"]} km detour</span>'
            route_tag = f'<span class="tag-route">{s.get("route_alignment", "-")} · {s.get("route_progress_pct", 0)}%</span>'
            score_tag = f'<span class="tag-score">score:{s.get("decision_score", "-")}</span>'
            mode_tag = f'<span class="tag-score">{s.get("decision_mode", "-")}</span>'

            st.markdown(
                f'<div class="{css}">'
                f'<div style="font-size:.62rem;color:{prefix_c};font-family:\'IBM Plex Mono\',monospace;letter-spacing:1px;margin-bottom:4px">{prefix}</div>'
                f'<div style="font-weight:500;color:#c8dce8;font-size:.85rem">{s["name"]}</div>'
                f'<div style="margin-top:6px">{score_tag}{mode_tag}{route_tag}{km_tag}{det_tag}{child_tag}{adult_tag}</div>'
                + (f'<div style="font-size:.72rem;color:#4a8090;margin-top:8px;line-height:1.5">{stn_reason}</div>' if is_chosen else '')
                + f'<div style="font-size:.62rem;color:#294d5a;margin-top:6px;line-height:1.4">{s.get("score_breakdown", "")}</div>'
                + '</div>',
                unsafe_allow_html=True,
            )

        for s in unreachable:
            st.markdown(
                f'<div class="station-unreachable">'
                f'<div style="font-size:.62rem;color:#3a1a1a;font-family:\'IBM Plex Mono\',monospace;letter-spacing:1px;margin-bottom:4px">✗ out of safe range</div>'
                f'<div style="font-weight:500;color:#6a4040;font-size:.85rem">{s["name"]}</div>'
                f'<div style="font-size:.65rem;color:#3a2020;margin-top:4px">{s.get("reason", "")}</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Reasoning ──
        if reasoning:
            st.markdown(
                f'<div class="card"><div style="font-size:.6rem;color:#1a5060;letter-spacing:2px;text-transform:uppercase;font-family:\'IBM Plex Mono\',monospace;margin-bottom:8px">AI Reasoning</div>'
                f'<div class="reasoning">{reasoning}</div></div>',
                unsafe_allow_html=True,
            )

        # ── Passenger offers ──
        st.markdown('<div class="sec">Passenger Experience</div>', unsafe_allow_html=True)
        for o in offers:
            ptype = o["type"]
            css = "offer-child" if ptype == "Child" else ("offer-driver" if ptype == "Driver" else "offer-adult")
            color = "#00d4ff" if ptype == "Driver" else ("#9060e0" if ptype == "Child" else "#40a060")
            name = o.get("name") or ptype
            code_html = f'<div class="voucher">{o["code"]}</div>' if o.get("code") else ""
            st.markdown(
                f'<div class="offer {css}">'
                f'<div><b style="color:{color}">{name}</b> <span style="font-size:.65rem;color:#1a4050">{ptype} · {o.get("seat", "")}</span></div>'
                f'<div style="font-size:.82rem;color:#8ab0b8;margin-top:6px">{o.get("offer", "")}</div>'
                f'{code_html}</div>',
                unsafe_allow_html=True,
            )

        # ── Business ──
        st.markdown('<div class="sec">Business Projection · 500-vehicle fleet</div>', unsafe_allow_html=True)

        def fmt_inr(v):
            if v >= 100000:
                return f"₹{v / 100000:.1f}L"
            if v >= 1000:
                return f"₹{v / 1000:.0f}K"
            return f"₹{v}"

        st.markdown(f"""
        <div class="biz-row">
            <div class="biz-card"><div class="biz-val">{fmt_inr(biz.get('monthly_gross_inr', 0))}</div><div class="biz-lbl">Monthly Gross</div><div class="biz-sub">@{biz.get('conversion_rate_pct', 0)}% conversion</div></div>
            <div class="biz-card"><div class="biz-val">{fmt_inr(biz.get('monthly_net_inr', 0))}</div><div class="biz-lbl">Monthly Net</div><div class="biz-sub">{biz.get('margin_pct', 0)}% margin</div></div>
            <div class="biz-card"><div class="biz-val">{biz.get('monthly_activated_stops', 0):,}</div><div class="biz-lbl">Activated Stops</div><div class="biz-sub">per month</div></div>
            <div class="biz-card"><div class="biz-val">MCP</div><div class="biz-lbl">Tool Layer</div><div class="biz-sub">Route + Battery + HMI</div></div>
        </div>
        <div style="font-size:.6rem;color:#0e2535;margin-top:6px;font-family:'IBM Plex Mono',monospace">
            Conversion scales with score ({score}/100) · urgency:{urg_level} · {biz.get('stops_per_vehicle_day', 0)} stops/vehicle/day
        </div>""", unsafe_allow_html=True)

        # ── Agent log ──
        st.markdown('<div class="sec">Agent Execution Log</div>', unsafe_allow_html=True)
        log_html = "".join(f'<div class="trace">{l}</div>' for l in logs)
        st.markdown(f'<div style="max-height:180px;overflow-y:auto">{log_html}</div>', unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:.6rem;color:#0e2535;font-family:\'IBM Plex Mono\',monospace">'
            'OccupantAnalyzer → ChargingOptimizer → PartnerNegotiator → HMIComposer · Route-aware scoring · MCP Tools · LangGraph</div>',
            unsafe_allow_html=True,
        )
