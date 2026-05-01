# ─────────────────────────────────────────────────────────────────────────────
#  SDV Orchestrator  ·  LangGraph + MCP  ·  Clean Edition
#  pip install langgraph langchain-google-genai langchain-core streamlit
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import json, uuid
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="SDV Orchestrator", page_icon="⚡", layout="wide")

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
.sec { font-family:'IBM Plex Mono',monospace; font-size:.6rem; color:#1a6070; letter-spacing:3px; text-transform:uppercase; border-bottom:1px solid #0e2030; padding-bottom:8px; margin-bottom:16px; margin-top:28px; }

.card    { background:#0f1520; border:1px solid #0e2535; border-radius:8px; padding:16px 18px; margin-bottom:10px; }
.card-hi { background:#0a1a28; border:1px solid #0d3550; border-left:3px solid #00d4ff; border-radius:4px 8px 8px 4px; padding:14px 18px; margin-bottom:8px; }

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
.label { font-family:'IBM Plex Mono',monospace; font-size:.58rem; color:#1a5060; letter-spacing:2px; text-transform:uppercase; margin-top:4px; }
.stat { text-align:center; padding:14px; background:#0f1520; border:1px solid #0e2535; border-radius:6px; }
.stat-val { font-family:'IBM Plex Mono',monospace; font-size:1.4rem; color:#00d4ff; font-weight:600; }
.stat-lbl { font-size:.6rem; color:#1a5060; letter-spacing:2px; text-transform:uppercase; margin-top:3px; }

.trace { font-family:'IBM Plex Mono',monospace; font-size:.65rem; color:#20a060; background:#080d12; border-left:2px solid #0a3020; padding:5px 10px; margin-bottom:2px; border-radius:0 4px 4px 0; }

.u-high   { height:4px; background:linear-gradient(90deg,#cc2020,#ff5500); border-radius:2px; }
.u-medium { height:4px; background:linear-gradient(90deg,#cc7700,#ffaa00); border-radius:2px; }
.u-low    { height:4px; background:linear-gradient(90deg,#007799,#00d4ff); border-radius:2px; }

.hr { height:1px; background:#0e2030; margin:24px 0; }

div.stButton > button { background:#00d4ff10; color:#00d4ff; border:1px solid #00d4ff40; border-radius:6px; width:100%; font-family:'IBM Plex Mono',monospace; font-size:.72rem; letter-spacing:2px; padding:10px; transition:all .2s; }
div.stButton > button:hover { background:#00d4ff20; border-color:#00d4ff; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def extract_text(response) -> str:
    """Handle both str and list[dict] content from Gemini."""
    c = response.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                return block["text"]
    return str(c)

def parse_json(raw: str) -> dict:
    txt = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    return json.loads(txt)

def llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=st.secrets["GEMINI_API_KEY"],
        temperature=0.2,
    )

# ══════════════════════════════════════════════════════════════════════════════
#  PARTNER DATA
# ══════════════════════════════════════════════════════════════════════════════
PARTNERS = {
    "Hinjewadi Ph 1": {"name":"TATA Power EV Hub · Hinjewadi","partner_id":"TATA_PNQ_HIN_01","charger":"DC Fast 150 kW","eta":8,"slots":3,"amenities":["AirConsole Lounge","McDonald's","Starbucks"],"child_offer":"AirConsole Pro — 45 min session unlocked","adult_offer":"Starbucks 20% off · 2× loyalty points","fee":55},
    "Wakad Bypass":   {"name":"Ather Grid · Wakad","partner_id":"ATHER_PNQ_WKD_02","charger":"DC Fast 100 kW","eta":6,"slots":5,"amenities":["Kids Zone","Café Coffee Day","Domino's"],"child_offer":"Kids Zone pass — 30 min complimentary","adult_offer":"CCD buy-1-get-1 · Domino's ₹100 off","fee":42},
    "Baner High St":  {"name":"ChargeZone Premium · Baner","partner_id":"CZ_PNQ_BAN_03","charger":"AC+DC 22–100 kW","eta":11,"slots":2,"amenities":["GameOn Arcade","Chaayos","Subway"],"child_offer":"GameOn Arcade — 3 free credits","adult_offer":"Chaayos ₹60 voucher + free snack","fee":38},
}

# ══════════════════════════════════════════════════════════════════════════════
#  MCP TOOLS
# ══════════════════════════════════════════════════════════════════════════════
@tool
def get_partner_station(location: str) -> str:
    """[MCP:PartnerAPI] Get EV partner station details for a location."""
    return json.dumps(PARTNERS.get(location, {}))

@tool
def calculate_urgency(soc: int, range_km: int, trip_km: int) -> str:
    """[MCP:BatteryMgmt] Compute charging urgency from battery telemetry."""
    deficit = trip_km - range_km
    if soc < 25 or deficit > 0: level, contrib = "HIGH", 40
    elif soc < 50:               level, contrib = "MEDIUM", 30
    else:                        level, contrib = "LOW", 10
    return json.dumps({"level":level,"score":contrib,"deficit_km":max(0,deficit)})

@tool
def generate_voucher(partner_id: str, pax_type: str, offer: str) -> str:
    """[MCP:PartnerNegotiation] Generate a personalised discount voucher."""
    tag  = "KID" if pax_type == "Child" else "VIP"
    code = f"{tag}-{partner_id[:4].upper()}-{uuid.uuid4().hex[:5].upper()}"
    return json.dumps({"code":code,"offer":offer})

@tool
def push_hmi(message: str, urgency: str) -> str:
    """[MCP:HMIBridge] Push alert to vehicle center console."""
    return json.dumps({"status":"PUSHED","message":message,"urgency":urgency})

# ══════════════════════════════════════════════════════════════════════════════
#  LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════
class SDVState(TypedDict):
    soc: int; range_km: int; trip_km: int
    location: str; destination: str
    passengers: List[dict]; partner_active: bool
    urgency: dict; station: dict; offers: List[dict]
    score: int; hmi_msg: str; logs: List[str]

def node_occupant(state: SDVState) -> dict:
    logs = list(state.get("logs", []))
    logs.append("→ [OccupantAnalyzer] profiling cabin occupants via MCP:OccupantProfiler")
    has_child = any(p["type"]=="Child" for p in state["passengers"])
    has_adult = any(p["type"]=="Adult" for p in state["passengers"])
    logs.append(f"  detected — children:{has_child}  adults:{has_adult}")
    return {"logs": logs}

def node_charging(state: SDVState) -> dict:
    logs = list(state.get("logs", []))
    logs.append("→ [ChargingOptimizer] querying MCP:BatteryMgmt telemetry stream")
    raw = calculate_urgency.invoke({"soc":state["soc"],"range_km":state["range_km"],"trip_km":state["trip_km"]})
    urg = json.loads(raw)
    logs.append(f"  urgency:{urg['level']}  score_contribution:+{urg['score']}  deficit:{urg['deficit_km']} km")
    return {"urgency": urg, "logs": logs}

def node_partner(state: SDVState) -> dict:
    logs = list(state.get("logs", []))
    logs.append("→ [PartnerNegotiator] opening MCP:PartnerAPI handshake")
    station = json.loads(get_partner_station.invoke({"location":state["location"]}))
    logs.append(f"  connected: {station.get('name')}  slots:{station.get('slots')}  ETA:{station.get('eta')} min")
    offers = []
    for p in state["passengers"]:
        if p["type"] == "Driver":
            offers.append({**p,"offer":"Navigation routed to charging stop","code":None})
            continue
        tmpl = station["child_offer"] if p["type"]=="Child" else station["adult_offer"]
        v    = json.loads(generate_voucher.invoke({"partner_id":station["partner_id"],"pax_type":p["type"],"offer":tmpl}))
        offers.append({**p,"offer":v["offer"],"code":v["code"]})
        logs.append(f"  MCP:PartnerNegotiation → {v['code']} issued to {p.get('name') or p['type']}")
    return {"station":station,"offers":offers,"logs":logs}

def node_hmi(state: SDVState) -> dict:
    logs  = list(state.get("logs", []))
    logs.append("→ [HMIComposer] computing engagement score")
    urg   = state.get("urgency", {})
    has_c = any(p["type"]=="Child" for p in state["passengers"])
    has_a = any(p["type"]=="Adult" for p in state["passengers"])
    score = min(100, urg.get("score",10) + (25 if has_c else 0) + (15 if has_a else 0) + (20 if state.get("partner_active") else 0))
    st8   = state.get("station",{})
    resp  = llm().invoke([HumanMessage(content=
        f"Automotive HMI. Write a center-console notification max 14 words. "
        f"urgency={urg.get('level')} partner={st8.get('name')} ETA={st8.get('eta')} min. "
        f"Return ONLY the message text.")])
    msg = extract_text(resp).strip().strip('"')
    push_hmi.invoke({"message":msg,"urgency":urg.get("level","MEDIUM")})
    logs.append(f"  engagement score: {score}/100")
    logs.append(f"  MCP:HMIBridge PUSHED → \"{msg}\"")
    logs.append("✓ Orchestration complete")
    return {"score":score,"hmi_msg":msg,"logs":logs}

@st.cache_resource
def build_graph():
    g = StateGraph(SDVState)
    g.add_node("occupant", node_occupant)
    g.add_node("charging", node_charging)
    g.add_node("partner",  node_partner)
    g.add_node("hmi",      node_hmi)
    g.add_edge(START,      "occupant")
    g.add_edge("occupant", "charging")
    g.add_edge("charging", "partner")
    g.add_edge("partner",  "hmi")
    g.add_edge("hmi",      END)
    return g.compile()

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION
# ══════════════════════════════════════════════════════════════════════════════
if "passengers" not in st.session_state:
    st.session_state.passengers = [{"id":"drv","name":"Driver","type":"Driver","seat":"Front-L"}]
if "result" not in st.session_state:
    st.session_state.result = None

SEAT_SLOTS = ["Front-R","Rear-L","Rear-C","Rear-R"]

# ══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="app-title">⚡SDV Orchestrator</div>'
    '<div class="app-sub">LangGraph · 4 Agents · MCP Tool Layer · Gemini 1.5 Flash</div>',
    unsafe_allow_html=True)
st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="sec">In-Cabin Occupancy</div>', unsafe_allow_html=True)
    occupied = {p["seat"]: p for p in st.session_state.passengers}

    seat_html = '<div class="seats"><div class="seat-driver">🧑‍✈️ Driver · Front-L</div>'
    for slot in SEAT_SLOTS:
        if slot in occupied:
            p   = occupied[slot]
            css = "seat-child" if p["type"]=="Child" else "seat-adult"
            ico = "👦" if p["type"]=="Child" else "🧑"
            seat_html += f'<div class="{css}">{ico} {p.get("name") or p["type"]}<br><span style="opacity:.5">{slot}</span></div>'
        else:
            seat_html += f'<div class="seat">○ {slot}</div>'
    seat_html += '</div>'
    st.markdown(seat_html, unsafe_allow_html=True)

    with st.expander("＋ Add passenger", expanded=len(st.session_state.passengers)==1):
        with st.form("pax", clear_on_submit=True):
            c1, c2, c3 = st.columns([2,1,2])
            name  = c1.text_input("Name", placeholder="optional")
            ptype = c2.selectbox("Type", ["Child","Adult"])
            free  = [s for s in SEAT_SLOTS if s not in occupied]
            seat  = c3.selectbox("Seat", free) if free else None
            if st.form_submit_button("Add", use_container_width=True) and seat:
                st.session_state.passengers.append({"id":uuid.uuid4().hex[:6],"name":name.strip() or None,"type":ptype,"seat":seat})
                st.rerun()

    for p in st.session_state.passengers:
        if p["type"] == "Driver": continue
        ico   = "👦" if p["type"]=="Child" else "🧑"
        color = "#9060e0" if p["type"]=="Child" else "#40a060"
        label = p.get("name") or p["type"]
        pc1, pc2 = st.columns([5,1])
        pc1.markdown(f'<div class="card-hi" style="border-left-color:{color}">{ico} <b style="color:{color}">{label}</b><span style="color:#1a4050;font-size:.7rem;margin-left:8px">{p["type"]} · {p["seat"]}</span></div>', unsafe_allow_html=True)
        if pc2.button("✕", key=p["id"]):
            st.session_state.passengers = [x for x in st.session_state.passengers if x["id"]!=p["id"]]
            st.rerun()

    st.markdown('<div class="sec">Vehicle Telemetry</div>', unsafe_allow_html=True)
    soc      = st.slider("Battery SoC %", 5, 100, 32)
    range_km = st.slider("Estimated range (km)", 10, 300, int(soc*2.8))
    trip_km  = st.slider("Trip distance (km)", 10, 150, 55)

    st.markdown('<div class="sec">Geo / Partner</div>', unsafe_allow_html=True)
    location    = st.selectbox("Current sector", list(PARTNERS.keys()))
    destination = st.text_input("Destination", "Pune Airport, T1")
    partner_on  = st.toggle("Partner API active", value=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    run = st.button("RUN ORCHESTRATION CYCLE", use_container_width=True)

with right:
    if run:
        with st.spinner("Running LangGraph agents…"):
            result = build_graph().invoke({
                "soc":soc,"range_km":range_km,"trip_km":trip_km,
                "location":location,"destination":destination,
                "passengers":st.session_state.passengers,"partner_active":partner_on,
                "urgency":{},"station":{},"offers":[],"score":0,"hmi_msg":"","logs":[],
            })
        st.session_state.result = result

    r = st.session_state.result

    if not r:
        st.markdown(
            '<div style="height:260px;display:flex;align-items:center;justify-content:center;'
            'color:#0e2535;font-family:\'IBM Plex Mono\',monospace;font-size:.75rem;letter-spacing:2px">'
            'AWAITING ORCHESTRATION CYCLE</div>', unsafe_allow_html=True)
    else:
        urg     = r.get("urgency",{})
        station = r.get("station",{})
        offers  = r.get("offers",[])
        score   = r.get("score",0)
        hmi_msg = r.get("hmi_msg","")
        logs    = r.get("logs",[])

        score_color = "#00d4ff" if score>65 else ("#ffaa00" if score>40 else "#cc3030")
        urg_level   = urg.get("level","LOW")

        st.markdown('<div class="sec">Result</div>', unsafe_allow_html=True)
        sa, sb, sc = st.columns(3)
        sa.markdown(f'<div class="stat"><div class="score-num" style="color:{score_color}">{score}</div><div class="stat-lbl">Engagement</div></div>', unsafe_allow_html=True)
        sb.markdown(f'<div class="stat"><div class="stat-val">{soc}%</div><div class="u-{urg_level.lower()}" style="margin:6px 0"></div><div class="stat-lbl">{urg_level}</div></div>', unsafe_allow_html=True)
        sc.markdown(f'<div class="stat"><div class="stat-val">{station.get("eta","?")}m</div><div class="stat-lbl">ETA to stop</div></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="card" style="margin-top:12px"><div class="label">HMI · Center Console</div><div style="font-family:\'IBM Plex Mono\',monospace;font-size:.95rem;color:#00d4ff;margin-top:8px;line-height:1.5">{hmi_msg}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="sec">Passenger Experience</div>', unsafe_allow_html=True)
        for o in offers:
            ptype = o["type"]
            css   = "offer-child" if ptype=="Child" else ("offer-driver" if ptype=="Driver" else "offer-adult")
            ico   = "🧑‍✈️" if ptype=="Driver" else ("👦" if ptype=="Child" else "🧑")
            color = "#00d4ff" if ptype=="Driver" else ("#9060e0" if ptype=="Child" else "#40a060")
            name  = o.get("name") or ptype
            code_html = f'<div class="voucher">{o["code"]}</div>' if o.get("code") else ""
            st.markdown(f'<div class="offer {css}"><div>{ico} <b style="color:{color}">{name}</b> <span style="font-size:.65rem;color:#1a4050">{ptype} · {o.get("seat","")}</span></div><div style="font-size:.82rem;color:#8ab0b8;margin-top:6px">{o.get("offer","")}</div>{code_html}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec">Partner Stop</div>', unsafe_allow_html=True)
        amenities = " · ".join(station.get("amenities",[]))
        st.markdown(f'<div class="card"><div style="font-weight:500;color:#c8dce8">{station.get("name","")}</div><div style="font-size:.7rem;color:#1a5060;margin:2px 0 10px">{station.get("partner_id","")} · {station.get("charger","")}</div><div style="display:flex;gap:10px"><div class="stat" style="flex:1"><div class="stat-val">₹{station.get("fee","")}</div><div class="stat-lbl">Referral</div></div><div class="stat" style="flex:1"><div class="stat-val">{station.get("slots","")}</div><div class="stat-lbl">Slots</div></div></div><div style="font-size:.7rem;color:#1a4050;margin-top:10px">{amenities}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="sec">Agent Execution Log</div>', unsafe_allow_html=True)
        log_html = "".join(f'<div class="trace">{l}</div>' for l in logs)
        st.markdown(f'<div style="max-height:200px;overflow-y:auto">{log_html}</div>', unsafe_allow_html=True)
