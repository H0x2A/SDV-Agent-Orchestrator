import streamlit as st
import google.generativeai as genai
import json

# 1. Connect to the AI
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')

# 2. Mock Data (Simulating MCP tool results)
def get_charging_stations():
    return [
        {"name": "Shell Recharge", "amenities": "Coffee, Food Court", "offer": "20% Coffee Discount"},
        {"name": "Tata Power Hub", "amenities": "AirConsole Gaming, Kids Zone", "offer": "Game Skin Unlock"}
    ]

# 3. UI Setup
st.set_page_config(page_title="Agent Demo", layout="centered")
st.title("🚗 Agent Demo")

# Sidebar Sensors
st.sidebar.header("📡 Vehicle Sensors")
battery = st.sidebar.slider("Battery %", 0, 100, 20)
passengers = st.sidebar.multiselect("Occupants", ["Driver", "Adult Passenger", "Child"], default=["Driver"])

# 4. Logic Execution
if st.button("Generate Recommendation", use_container_width=True):
    stations = get_charging_stations()
    
    # The Prompt - How the AI "thinks"
    prompt = f"""
    Context: Battery {battery}%, Passengers: {passengers}.
    Available Stops: {stations}.
    Task: If battery < 30%, suggest the best stop. 
    If a child is present, prioritize the Gaming stop. 
    Otherwise, suggest the Coffee stop. 
    Speak like a premium car assistant.
    """

    with st.spinner("AI is reasoning..."):
        response = model.generate_content(prompt)
        st.chat_message("assistant").write(response.text)

    # Developer Log for the interview
    with st.expander("🛠️ Developer Log (JSON Payload)"):
        st.json({"sensors": {"bat": battery, "occ": passengers}, "mcp_data": stations})
