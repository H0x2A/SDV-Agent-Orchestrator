# SDV Orchestrator Demo

A proof-of-concept demo for **Software Defined Vehicle (SDV) service orchestration** using:

- **LangGraph** for agent orchestration
- **4-agent decision flow**
- **MCP-style tool layer**
- **Gemini 3.1 Flash-Lite Preview** for lightweight reasoning
- **Streamlit** for interactive HMI-style UI

The demo shows how an SDV can make a context-aware in-vehicle service recommendation by combining vehicle telemetry, passenger profile, route context, charging station availability, partner offers, and business projection.

---

## Demo Objective

This PoC demonstrates how SDV services can move beyond static rule-based recommendations and use an agentic orchestration flow to generate intelligent, context-aware decisions.

The system does **not simply recommend the nearest charger**.

Instead, it evaluates:

- Vehicle battery SoC
- Estimated remaining range
- Trip distance
- Current route sector
- Destination
- Passenger type, such as child/adult onboard
- Charging station reachability
- Route detour
- Charger speed
- Available charging slots
- Passenger experience score
- Partner offer value
- Business projection

If the battery condition is critical, the system prioritizes the nearest safe charging option.  
If the battery range is safe, the system may recommend a slightly farther on-route station when it provides a better passenger experience or business value.

---

## Key Demo Scenario

Example behavior:

| Scenario | Expected Decision |
|---|---|
| Very low SoC / critical range | Nearest safe charger is selected |
| Child passenger onboard + sufficient range | Child-friendly station may be selected, even if slightly farther |
| Adult passenger onboard + sufficient range | Adult-friendly location such as mall/cafe/cinema may be preferred |
| Mixed passengers | Balanced station selection based on child and adult scores |
| Station outside safe range | Station is shown as unavailable and not selected |

---

## Architecture Overview

```text
Vehicle Telemetry + Passenger Profile + Route Context
                    |
                    v
          LangGraph Orchestration
                    |
    ------------------------------------------------
    |              |               |               |
Occupant      Charging       Partner          HMI
Analyzer      Optimizer      Negotiator       Composer
    |              |               |               |
    ------------------------------------------------
                    |
                    v
              MCP-style Tool Layer
                    |
    ------------------------------------------------
    | RouteAPI | BatteryMgmt | PartnerNegotiation |
    | HMIBridge | BusinessIntelligence           |
    ------------------------------------------------
                    |
                    v
        HMI Recommendation + Offers + Business Output
```

---

## Agent Flow

The demo uses a 4-agent orchestration flow.

### 1. OccupantAnalyzer

Analyzes the in-cabin passenger profile.

Responsibilities:

- Detects whether child passengers are present
- Detects whether adult passengers are present
- Adds cabin context for downstream decision-making

---

### 2. ChargingOptimizer

Evaluates battery and route context.

Responsibilities:

- Calculates charging urgency
- Checks safe remaining range
- Identifies reachable charging stations
- Identifies out-of-range stations
- Calculates route distance, ETA, and detour

---

### 3. PartnerNegotiator

Selects the most suitable charging station.

Responsibilities:

- Applies deterministic station scoring
- Considers battery urgency, route safety, passenger profile, charger speed, detour, and slots
- Uses Gemini to generate reasoning for the selected station
- Generates passenger-specific offers and vouchers

---

### 4. HMIComposer

Generates final SDV output.

Responsibilities:

- Creates HMI notification text
- Calculates engagement score
- Pushes HMI message through the simulated HMI bridge
- Computes business projection for a fleet scenario
- Displays final decision and agent execution log

---

## MCP-style Tool Layer

This PoC uses local Python tools that simulate an MCP-style service layer.

| Tool | Purpose |
|---|---|
| `get_stations_on_route` | Simulates Route API and returns reachable/on-route stations |
| `calculate_urgency` | Simulates Battery Management and computes charging urgency |
| `generate_voucher` | Simulates Partner Negotiation and offer generation |
| `push_hmi` | Simulates HMI Bridge message push |
| `compute_business_projection` | Simulates Business Intelligence revenue projection |

> Note: In this demo, the MCP layer is implemented as local tool abstractions.  
> For production use, these tools can be replaced with real MCP server tools or backend APIs.

---

## Decision Logic

The station selection is based on a hybrid approach:

1. **Deterministic safety and scoring logic**
2. **LLM-based explanation and HMI message generation**

This keeps the demo decision reliable while still showing AI reasoning.

### Priority Rules

| Battery State | Decision Priority |
|---|---|
| Critical | Select nearest safe charger |
| High | Prefer safe and nearby station, but allow better scored option if range permits |
| Medium | Balance route, passenger experience, charger capability, and detour |
| Low | Prioritize passenger experience and partner value if route safety is satisfied |

---

## Passenger-aware Behavior

The orchestrator can change its recommendation depending on who is inside the vehicle.

### Child Onboard

The system increases weightage for:

- Child-friendly amenities
- Gaming zone
- Kids zone
- Arcade
- Trampoline park
- Short waiting-time experience

### Adult Onboard

The system increases weightage for:

- Cafe availability
- Mall access
- Food options
- Comfort
- Lower detour
- Better charger reliability

### Mixed Cabin

The system balances child score and adult score.

---

## Route-aware Behavior

The station recommendation is not based only on current geo-location.

The system considers:

- Origin sector
- Destination
- Station position on route
- Detour distance
- Safe battery margin
- Whether the station is reachable without range risk

A farther station may be selected when:

- It is still within safe battery range
- It is on the route corridor
- Detour is acceptable
- Passenger experience score is significantly better
- Charger speed and slot availability are suitable

---

## Technology Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Agent Orchestration | LangGraph |
| LLM | Gemini 3.1 Flash-Lite Preview |
| Tool Abstraction | LangChain tools |
| State Management | LangGraph StateGraph |
| Language | Python |

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

For Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit langgraph langchain-google-genai langchain-core
```

---

## Gemini API Key Setup

Create a Streamlit secrets file:

```bash
mkdir -p .streamlit
nano .streamlit/secrets.toml
```

Add your Gemini API key:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

---

## Run the App

```bash
streamlit run app.py
```

If your file has a different name, use:

```bash
streamlit run sdv_orchestrator_route_passenger_aware.py
```

---

## How to Use the Demo

1. Start the Streamlit app.
2. Add a passenger:
   - Child
   - Adult
   - Or both
3. Adjust battery SoC.
4. Adjust estimated range and trip distance.
5. Select current sector.
6. Enter destination.
7. Enable/disable Partner API and V2X signal.
8. Click **Run Orchestration Cycle**.
9. Observe:
   - Selected charging station
   - Reason for selection
   - Reachable and out-of-range stations
   - Passenger offer/voucher
   - HMI notification
   - Business projection
   - Agent execution log

---

## Recommended Demo Cases

### Case 1: Critical Battery

Set:

- SoC: 10–20%
- Range: Low
- Passenger: Any

Expected result:

- The system should select the nearest safe charging station.

---

### Case 2: Child Passenger with Enough Range

Set:

- SoC: 35–60%
- Add child passenger
- Range: Sufficient

Expected result:

- The system may select a child-friendly station, even if it is slightly farther.

---

### Case 3: Adult Passenger with Enough Range

Set:

- SoC: 40–70%
- Add adult passenger
- Range: Sufficient

Expected result:

- The system may prefer a station with adult-friendly amenities such as cafe, mall, food court, or cinema.

---

### Case 4: Mixed Cabin

Set:

- Add child passenger
- Add adult passenger
- Keep battery range safe

Expected result:

- The system balances passenger experience and route suitability.

---

### Case 5: Out-of-range Station

Set:

- Low estimated range
- Longer trip distance

Expected result:

- Some stations are marked out of range and excluded from selection.

---

## Example Client Explanation

> This PoC demonstrates how an SDV can use agentic orchestration to make real-time service recommendations.  
> The system combines vehicle telemetry, passenger context, route intelligence, partner services, and HMI output.  
> It does not always choose the nearest charging station. If battery safety allows, it selects the best on-route option based on passenger experience, detour, charger capability, and business value.

---

## Current Scope

Included in this PoC:

- Streamlit-based SDV demo UI
- LangGraph agent flow
- Passenger-aware recommendation
- Route-aware charging station selection
- Battery urgency calculation
- MCP-style local tool layer
- Gemini-based reasoning and HMI message generation
- Passenger-specific offers
- Business projection
- Agent execution trace

---

## Future Enhancements

Possible next steps:

- Replace simulated MCP tools with real MCP server integration
- Integrate live EV charger availability APIs
- Add real map and route API integration
- Add V2X signal simulation
- Add vehicle profile and driver preference memory
- Add multi-stop route planning
- Add telemetry replay from recorded vehicle data
- Add OEM-specific HMI templates
- Add safety guardrails for driver-distraction control
- Add cloud-to-vehicle service orchestration flow

---

## Disclaimer

This is a proof-of-concept demo intended for showcasing SDV orchestration concepts.

The current implementation uses static sample station data and simulated MCP-style tools.  
It is not intended for direct production deployment without integration with real vehicle systems, route APIs, charger networks, safety validation, and OEM HMI compliance checks.

---

## Summary

This demo shows how SDV service orchestration can combine:

- Vehicle context
- Route intelligence
- Passenger personalization
- Partner ecosystem integration
- LLM-assisted reasoning
- HMI communication
- Business intelligence

The result is a more intelligent and personalized in-vehicle service recommendation experience.

