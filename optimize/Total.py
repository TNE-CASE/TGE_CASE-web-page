# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 15:50:25 2025

@author: LENOVO
"""

# ================================================================
#  merged_app.py (FINAL)
# ================================================================

import os
import random
import re
import streamlit as st
import pandas as pd
import plotly.express as px
import streamlit.components.v1 as components
import gurobipy as gp
import inspect

from sc1_app import run_sc1
from sc2_app import run_sc2
from Scenario_Setting_For_SC1F import run_scenario as run_SC1F
from Scenario_Setting_For_SC2F import run_scenario as run_SC2F
# MASTER model import (supports mode-share enforcement & parametric versions)

from MASTER import run_scenario_master
from collections import defaultdict



# ================================================================
# PAGE CONFIG (only once!)
# ================================================================
st.set_page_config(
    page_title="Supply Chain Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ================================================================
# SIDEBAR LOGO (add above navigation)
# ================================================================
BASE_DIR = os.path.dirname(__file__)
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.png")

if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)
    st.sidebar.markdown("---")  # small separator so menu stays clean
else:
    st.sidebar.warning("Logo not found: assets/logo.png")

# ================================================================
# SIDEBAR NAVIGATION WITH COLLAPSIBLE GROUPS
# ================================================================
st.sidebar.title("📌 Navigation")

# Make the two navigation groups mutually exclusive.
# Otherwise, once a Factory Model page is selected, routing always stops there
# and the user cannot navigate to the Optimization dashboard.
def _on_factory_change():
    st.session_state["optimization_radio"] = None

def _on_optimization_change():
    st.session_state["factory_radio"] = None

# Collapsible "Factory Model" group
with st.sidebar.expander("🏭 Factory Model", expanded=True):
    factory_choice = st.radio(
        "Select model:",
        [
            "SC1 – Existing Facilities",
            "SC2 – New Facilities"
        ],
        index=None,
        key="factory_radio",
        on_change=_on_factory_change,
    )

# Collapsible "Optimization" group
with st.sidebar.expander("📊 Optimization", expanded=True):
    opt_choice = st.radio(
        "Select:",
        ["Optimization Dashboard"],
        index=None,
        key="optimization_radio",
        on_change=_on_optimization_change,
    )

# ================================================================
# ROUTING LOGIC
# ================================================================
if factory_choice == "SC1 – Existing Facilities":
    run_sc1()
    st.stop()

elif factory_choice == "SC2 – New Facilities":
    run_sc2()
    st.stop()

elif opt_choice == "Optimization Dashboard":
    pass  # Continue into optimization block below

else:
    st.write("👈 Select a page from the Navigation menu.")
    st.stop()

# ================================================================
# OPTIMIZATION DASHBOARD
# ================================================================

st.title("🌍 Global Supply Chain Optimization (Gurobi)")

# ------------------------------------------------------------
# Google Analytics Injection (safe)
# ------------------------------------------------------------
GA_MEASUREMENT_ID = "G-78BY82MRZ3"

components.html(f"""
<script>
(function() {{
    const targetDoc = window.parent.document;

    const old1 = targetDoc.getElementById("ga-tag");
    const old2 = targetDoc.getElementById("ga-src");
    if (old1) old1.remove();
    if (old2) old2.remove();

    const s1 = targetDoc.createElement('script');
    s1.id = "ga-src";
    s1.async = true;
    s1.src = "https://www.googletagmanager.com/gtag/js?id={GA_MEASUREMENT_ID}";
    targetDoc.head.appendChild(s1);

    const s2 = targetDoc.createElement('script');
    s2.id = "ga-tag";
    s2.innerHTML = `
        window.dataLayer = window.dataLayer || [];
        function gtag() {{ dataLayer.push(arguments); }}
        gtag('js', new Date());
        gtag('config', '{GA_MEASUREMENT_ID}', {{
            send_page_view: true
        }});
    `;
    targetDoc.head.appendChild(s2);

    console.log("GA injected successfully");
}})();
</script>
""", height=0)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def positive_input(label, default):
    """Clean numeric input helper."""
    val_str = st.text_input(label, value=str(default))
    try:
        val = float(val_str)
        return max(val, 0)
    except:
        st.warning(f"{label} must be numeric. Using {default}.")
        return default


def run_filtered(func, kwargs: dict):
    """Call a scenario function with only the kwargs it supports (signature-safe)."""
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in allowed}
    return func(**filtered)


def run_master_filtered(master_kwargs: dict):
    """Backward-compatible alias for MASTER."""
    return run_filtered(run_scenario_master, master_kwargs)

    
# ------------------------------------------------------------
# Helpers (NEW): compute node activity from flows
# ------------------------------------------------------------
EPS = 1e-6

# IMPORTANT: Map displayed City labels -> model facility keys used in variable names
# Adjust these to match YOUR model naming.
CITY_TO_KEYS = {
    # Plants (model keys)
    "Shanghai": ["Shanghai"],
    "Taiwan": ["Taiwan"],  

    # Cross-docks 
    "Paris": ["Paris"],
    "Gdansk": ["Gdansk"],
    "Vienna": ["Vienna"],

    # DCs 
    "Pardubice": ["Pardubice"],
    "Lille": ["Lille"],
    "Riga": ["Riga"],
    "LaGomera": ["LaGomera"],

    # Retailers 
    "Cologne": ["Cologne"],
    "Antwerp": ["Antwerp"],
    "Krakow": ["Krakow"],
    "Kaunas": ["Kaunas"],
    "Oslo": ["Oslo"],
    "Dublin": ["Dublin"],
    "Stockholm": ["Stockholm"],
}

def _parse_inside_brackets(varname: str):
    # "f2[ATVIE,GMZ,air]" -> ["ATVIE","GMZ","air"]
    i = varname.find("[")
    j = varname.rfind("]")
    if i == -1 or j == -1 or j <= i:
        return None
    inside = varname[i+1:j]
    return [x.strip() for x in inside.split(",")]

def compute_key_throughput(model) -> dict:
    """
    Returns dict: facility_key -> total flow touching the node (in+out aggregated)
    Based on f1, f2, f2_2, f3 variable values.
    """
    thr = defaultdict(float)
    for v in model.getVars():
        n = v.VarName

        if n.startswith("f1[") or n.startswith("f2[") or n.startswith("f2_2[") or n.startswith("f3["):
            parts = _parse_inside_brackets(n)
            if not parts or len(parts) < 2:
                continue

            o, d = parts[0], parts[1]
            try:
                x = float(v.X)
            except Exception:
                x = 0.0

            if x > EPS:
                thr[o] += x
                thr[d] += x

    return thr

def city_is_active(city: str, key_thr: dict) -> bool:
    keys = CITY_TO_KEYS.get(city, [])
    return sum(key_thr.get(k, 0.0) for k in keys) > EPS


# ------------------------------------------------------------
# Helpers (NEW): transport flow totals by mode + cost/emission charts
# ------------------------------------------------------------

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def sum_flows_by_mode_model(model, prefix: str):
    """Sum air/sea/road units for a given flow prefix like 'f1', 'f2', 'f2_2', or 'f3' from a Gurobi model."""
    totals = {"air": 0.0, "sea": 0.0, "road": 0.0}
    if model is None:
        return totals

    for v in model.getVars():
        n = getattr(v, "VarName", "")
        if not n.startswith(prefix + "["):
            continue
        parts = _parse_inside_brackets(n)
        if not parts or len(parts) < 3:
            continue
        mode = str(parts[-1]).lower()
        if mode in totals:
            totals[mode] += _safe_float(getattr(v, "X", 0.0))

    return totals


def display_layer_summary_model(model, title: str, prefix: str, include_road: bool = True):
    totals = sum_flows_by_mode_model(model, prefix)
    st.markdown(f"### {title}")
    cols = st.columns(3 if include_road else 2)
    cols[0].metric("🚢 Sea", f"{totals['sea']:,.0f} units")
    cols[1].metric("✈️ Air", f"{totals['air']:,.0f} units")
    if include_road:
        cols[2].metric("🚛 Road", f"{totals['road']:,.0f} units")

    if sum(totals.values()) <= EPS:
        st.info("No transport activity recorded for this layer.")
    st.markdown("---")


def render_transport_flows_by_mode(model):
    st.markdown("## 🚚 Transport Flows by Mode")
    display_layer_summary_model(model, "Layer 1: Plants → Cross-docks", "f1", include_road=False)
    display_layer_summary_model(model, "Layer 2a: Cross-docks → DCs", "f2", include_road=True)
    display_layer_summary_model(model, "Layer 2b: New Facilities → DCs", "f2_2", include_road=True)
    display_layer_summary_model(model, "Layer 3: DCs → Retailer Hubs", "f3", include_road=True)


def render_cost_emission_distribution(results: dict):
    """Replicates the SC1/SC2-style Cost & Emission Distribution charts for optimization outputs."""
    st.markdown("## 💰 Cost and 🌿 Emission Distribution")

    col1, col2 = st.columns(2)

    # --- Cost Distribution ---
    with col1:
        st.subheader("Cost Distribution")

        transport_cost = (
            _safe_float(results.get("Transport_L1", 0))
            + _safe_float(results.get("Transport_L2", 0))
            + _safe_float(results.get("Transport_L2_new", results.get("Transport_L2_new", 0)))
            + _safe_float(results.get("Transport_L3", 0))
        )
        if transport_cost <= 0 and "Transportation Cost" in results:
            transport_cost = _safe_float(results.get("Transportation Cost", 0))

        sourcing_handling_cost = (
            _safe_float(results.get("Sourcing_L1", 0))
            + _safe_float(results.get("Handling_L2_total", 0))
            + _safe_float(results.get("Handling_L3", 0))
        )
        if sourcing_handling_cost <= 0 and "Sourcing/Handling Cost" in results:
            sourcing_handling_cost = _safe_float(results.get("Sourcing/Handling Cost", 0))

        co2_cost_production = (
            _safe_float(results.get("CO2_Manufacturing_State1", 0))
            + _safe_float(results.get("CO2_Cost_L2_2", 0))
        )
        if co2_cost_production <= 0:
            co2_cost_production = _safe_float(results.get("CO2 Cost in Production", results.get("CO2_Cost_in_Production", 0)))

        inventory_cost = (
            _safe_float(results.get("Inventory_L1", 0))
            + _safe_float(results.get("Inventory_L2", 0))
            + _safe_float(results.get("Inventory_L2_new", 0))
            + _safe_float(results.get("Inventory_L3", 0))
        )
        if inventory_cost <= 0 and "Transit Inventory Cost" in results:
            inventory_cost = _safe_float(results.get("Transit Inventory Cost", 0))

        cost_parts = {
            "Transportation Cost": transport_cost,
            "Sourcing/Handling Cost": sourcing_handling_cost,
            "CO₂ Cost in Production": co2_cost_production,
            "Inventory Cost": inventory_cost,
        }

        df_cost_dist = pd.DataFrame({
            "Category": list(cost_parts.keys()),
            "Value": list(cost_parts.values()),
        })

        fig_cost = px.bar(
            df_cost_dist,
            x="Category",
            y="Value",
            text="Value",
            color="Category",
            color_discrete_sequence=["#A7C7E7", "#B0B0B0", "#F8C471", "#5D6D7E"],
        )

        fig_cost.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
        )
        fig_cost.update_layout(
            template="plotly_white",
            showlegend=False,
            xaxis_tickangle=-35,
            yaxis_title="€",
            height=400,
            yaxis_tickformat=",",
        )

        st.plotly_chart(fig_cost, use_container_width=True)

    # --- Emission Distribution ---
    with col2:
        st.subheader("Emission Distribution")

        e_air = _safe_float(results.get("E_air", results.get("E_Air", 0)))
        e_sea = _safe_float(results.get("E_sea", results.get("E_Sea", 0)))
        e_road = _safe_float(results.get("E_road", results.get("E_Road", 0)))
        e_last = _safe_float(results.get("E_lastmile", results.get("E_Last-mile", results.get("E_last_mile", 0))))
        e_total = _safe_float(results.get("CO2_Total", results.get("Total Emissions", 0)))

        e_prod = _safe_float(results.get("E_production", results.get("E_Production", 0)))
        if e_prod <= 0 and e_total > 0:
            e_prod = max(e_total - e_air - e_sea - e_road - e_last, 0.0)

        total_transport = e_air + e_sea + e_road

        emission_data = {
            "Production": e_prod,
            "Last-mile": e_last,
            "Air": e_air,
            "Sea": e_sea,
            "Road": e_road,
            "Total Transport": total_transport,
        }

        df_emission = pd.DataFrame({
            "Source": list(emission_data.keys()),
            "Emission (tons)": list(emission_data.values()),
        })

        fig_emission = px.bar(
            df_emission,
            x="Source",
            y="Emission (tons)",
            text="Emission (tons)",
            color="Source",
            color_discrete_sequence=["#4B8A08", "#2E8B57", "#808080", "#FFD700", "#90EE90", "#000000"],
        )

        fig_emission.update_traces(
            texttemplate="%{text:,.2f}",
            textposition="outside",
            marker_line_color="black",
            marker_line_width=0.5,
        )

        fig_emission.update_layout(
            template="plotly_white",
            showlegend=False,
            xaxis_tickangle=-35,
            yaxis_title="Tons of CO₂",
            height=400,
            yaxis_tickformat=",",
        )

        st.plotly_chart(fig_emission, use_container_width=True)

# ------------------------------------------------------------
# Mode selection (Normal vs Gamification)
# ------------------------------------------------------------
mode = st.radio("Select mode:", ["Normal Mode", "Gamification Mode"])

# default scenario flags
suez_flag = oil_flag = volcano_flag = trade_flag = False
tariff_rate_used = 1.0

# ------------------------------------------------------------
# GAMIFICATION MODE LOGIC 
# ------------------------------------------------------------
if mode == "Gamification Mode":
    st.subheader("🧩 Gamification Mode: Design Your Network")

    st.markdown(
        "Turn facilities and transport modes on/off and see how the optimal network "
        "and emissions change. This uses the parametric `MASTER` model."
    )

    # --- Scenario events as toggles ---
    st.markdown("#### Scenario events")
    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        suez_flag = st.checkbox(
            "Suez Canal Blockade (no sea from plants to Europe)",
            value=False,
            key="gm_suez"
        )
        oil_flag = st.checkbox(
            "Oil Crisis (increase all transport costs)",
            value=False,
            key="gm_oil"
        )
    with col_ev2:
        volcano_flag = st.checkbox(
            "Volcanic Eruption (no air shipments)",
            value=False,
            key="gm_volcano"
        )
        trade_flag = st.checkbox(
            "Trade War (more expensive sourcing)",
            value=False,
            key="gm_trade"
        )

    tariff_rate_used = 1.0
    if trade_flag:
        tariff_rate_used = st.slider(
            "Tariff multiplier on sourcing cost",
            min_value=1.0,
            max_value=2.0,
            value=1.3,
            step=0.05,
            help="1.0 = no tariff, 2.0 = sourcing cost doubles",
        )

    # --- Facility activation ---
    st.markdown("#### Facility activation")
    



    plants_all = ["Taiwan", "Shanghai"]
    crossdocks_all = ["Vienna", "Gdansk", "Paris"]
    dcs_all = ["Pardubice", "Lille", "Riga", "LaGomera"]
    new_locs_all = ["Budapest", "Prague", "Dublin", "Helsinki", "Warsaw"]
    st.info("✅ In Gamification Mode, all Distribution Centers (DCs) are assumed active.")

    col_p, col_c, col_n = st.columns(3)
    with col_p:
        st.caption("Plants")
        gm_active_plants = [
            p for p in plants_all
            if st.checkbox(p, value=True, key=f"gm_pl_{p}")
        ]
    with col_c:
        st.caption("Cross-docks")
        gm_active_crossdocks = [
            c for c in crossdocks_all
            if st.checkbox(c, value=True, key=f"gm_cd_{c}")
        ]
    with col_n:
        st.caption("New production sites")
        gm_active_new_locs = [
            n for n in new_locs_all
            if st.checkbox(n, value=True, key=f"gm_new_{n}")
        ]

    # All DCs active (no selection in UI)
    gm_active_dcs = list(dcs_all)
    st.session_state["gm_active_new_locs"]   = gm_active_new_locs
    
    # Map selections -> MASTER boolean flags (isHUDTG, isCZMCT, ...)
    gm_newloc_flag_kwargs = {f"is{code}": (code in gm_active_new_locs) for code in new_locs_all}

    # --- Mode activation ---
    st.markdown("#### Allowed transport modes per layer")

    all_modes = ["air", "sea", "road"]
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        gm_modes_L1 = st.multiselect(
            "Plant → Cross-dock (Road not allowed)",
            options=["air", "sea"],
            default=["air", "sea"],
            key="gm_modes_L1",
            help="Layer 1 (Plant → Cross-dock) does not allow road transport.",
        )
    with col_m2:
        gm_modes_L2 = st.multiselect(
            "Cross-dock / New → DC",
            options=all_modes,
            default=all_modes,
            key="gm_modes_L2",
        )
    with col_m3:
        gm_modes_L3 = st.multiselect(
            "DC → Retailer",
            options=all_modes,
            default=all_modes,
            key="gm_modes_L3",
        )

    # --- Mode share enforcement (UPDATED): per-node shares for MASTER ---
    st.markdown("#### Transport mode shares (enforced on Layer 1 & 2)")

    enforce_mode_shares = st.checkbox(
        "Enforce transport-mode shares on Layer 1 & 2 (per facility)",
        value=False,
        key="gm_enforce_mode_shares",
        help=(
            "If enabled, the optimizer is forced to match these percentages separately for each active node. "
            "Layer 1 is per-plant (air/sea only). Layer 2 is per-origin (crossdock or new facility) (air/sea/road)."
        ),
    )

    gm_mode_share_L1_by_plant = None
    gm_mode_share_L2_by_origin = None

    def _pct(x: float) -> str:
        return f"{100.0 * float(x):.1f}%"

    def _l1_share_ui_for_plant(plant: str, key_prefix: str):
        # Only air/sea; we ask for sea and auto-fill air
        sea = st.slider(
            f"{plant} – Sea share (L1)",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
            key=f"{key_prefix}_sea",
        )
        air = 1.0 - float(sea)
        st.write(f"{plant} – Air share (L1, auto): **{_pct(air)}**")
        # Use None remainder semantics for robustness on the MASTER side
        return {"sea": float(sea), "air": None}

    def _l2_share_ui_for_origin(origin: str, key_prefix: str):
        # Ask for sea, then air up to remaining; road is remainder
        sea = st.slider(
            f"{origin} – Sea share (L2)",
            min_value=0.0,
            max_value=1.0,
            value=0.20,
            step=0.01,
            key=f"{key_prefix}_sea",
        )
        rem_after_sea = 1.0 - float(sea)

        if rem_after_sea <= 1e-12:
            air = 0.0
            st.write(f"{origin} – Air share (L2): **{_pct(air)}** (fixed because Sea is 100%)")
        else:
            air = st.slider(
                f"{origin} – Air share (L2)",
                min_value=0.0,
                max_value=float(rem_after_sea),
                value=0.00,
                step=0.01,
                key=f"{key_prefix}_air",
            )

        road = max(0.0, 1.0 - float(sea) - float(air))
        st.write(f"{origin} – Road share (L2, auto): **{_pct(road)}**")

        # Road is auto remainder
        return {"sea": float(sea), "air": float(air), "road": None}

    if enforce_mode_shares:
        st.caption(
            "For each node: you set shares for some modes; the remaining mode is auto-filled to reach 100%. "
            "Setting a mode to 100% should NOT error."
        )

        # -------------------------
        # L1: per-plant (air/sea)
        # -------------------------
        st.markdown("**Layer 1 (Plant → Cross-dock): per-plant shares (Road is forbidden)**")
        if len(gm_active_plants) == 0:
            st.info("No active plants selected.")
        else:
            gm_mode_share_L1_by_plant = {}
            for p in gm_active_plants:
                with st.expander(f"🌱 {p}", expanded=False):
                    gm_mode_share_L1_by_plant[p] = _l1_share_ui_for_plant(p, key_prefix=f"gm_l1_{p}")

        st.markdown("---")

        # -----------------------------------------
        # L2: per-origin (crossdock + new facility)
        # -----------------------------------------
        st.markdown("**Layer 2 (Cross-dock / New → DC): per-origin shares**")
        active_origins = list(gm_active_crossdocks) + list(gm_active_new_locs)
        if len(active_origins) == 0:
            st.info("No active cross-docks or new facilities selected.")
        else:
            gm_mode_share_L2_by_origin = {}
            for o in active_origins:
                with st.expander(f"🏷️ {o}", expanded=False):
                    gm_mode_share_L2_by_origin[o] = _l2_share_ui_for_origin(o, key_prefix=f"gm_l2_{o}")

        # Ensure required modes are enabled in the mode lists (except road on L1)
        gm_modes_L1 = sorted(set(gm_modes_L1) | {"air", "sea"})
        gm_modes_L2 = sorted(set(gm_modes_L2) | {"air", "sea", "road"})

    st.session_state["gm_mode_share_L1_by_plant"] = gm_mode_share_L1_by_plant
    st.session_state["gm_mode_share_L2_by_origin"] = gm_mode_share_L2_by_origin


    # Make sure lists exist even if user deselects everything
    gm_active_plants = gm_active_plants or []
    gm_active_crossdocks = gm_active_crossdocks or []
    gm_active_dcs = gm_active_dcs or []
    gm_active_new_locs = gm_active_new_locs or []
    gm_modes_L1 = gm_modes_L1 or []
    gm_modes_L2 = gm_modes_L2 or []
    gm_modes_L3 = gm_modes_L3 or []

# For Normal Mode we keep the default flags (all False, tariff 1.0)

# ------------------------------------------------------------
# Parameter Inputs
# ------------------------------------------------------------
st.subheader("📊 Scenario Parameters")

co2_pct = positive_input("CO₂ Reduction Target (%)", 50.0) / 100

model_choice = st.selectbox(
    "Optimization model:",
    ["SC1F – Existing Facilities Only", "SC2F – Allow New Facilities"]
)

# Base sourcing costs (same as MASTER defaults)
BASE_SOURCING_COST = {"Taiwan": 3.343692308, "Shanghai": 3.423384615}

# Expose sourcing-cost multiplier only for SC2F (and for Gamification Mode / MASTER).
# For SC1F in Normal Mode, keep the old behavior (no multiplier UI).
if (mode == "Gamification Mode") or ("SC2F" in model_choice):
    sourcing_cost_multiplier = st.slider(
        "Sourcing Cost Multiplier (Layer 1)",
        min_value=0.5,
        max_value=4.0,
        value=1.0,
        step=0.01,
        help="Scales plant sourcing costs on Layer 1: effective_cost = base_cost × multiplier.",
    )
else:
    sourcing_cost_multiplier = 1.0

scaled_sourcing_cost = {k: v * float(sourcing_cost_multiplier) for k, v in BASE_SOURCING_COST.items()}

if "service_level" not in st.session_state:
    st.session_state["service_level"] = 0.90


# Only let user edit it in Normal Mode + SC1F (your requirement)
if (mode == "Normal Mode") and ("SC1F" in model_choice):
    st.session_state["service_level"] = st.slider(
        "Service Level",
        min_value=0.50,
        max_value=0.99,
        value=float(st.session_state["service_level"]),
        step=0.01,
        help="Used by SC1F and also passed to MASTER (Gamification) for inventory/safety stock logic.",
    )

# Always use the persisted value everywhere (including MASTER run)
service_level = float(st.session_state["service_level"])


# Keep both defined (MASTER uses both; UI edits the relevant one)
co2_cost_per_ton = 37.5
co2_cost_per_ton_New = 60.0

if "SC1F" in model_choice:
    co2_cost_per_ton = positive_input("CO₂ Cost per ton (€)", 37.5)
else:
    co2_cost_per_ton_New = positive_input("CO₂ Cost per ton (New Facility)", 60.0)
# ------------------------------------------------------------
# RUN OPTIMIZATION
# ------------------------------------------------------------
if st.button("Run Optimization"):
    with st.spinner("⚙ Optimizing with Gurobi..."):
        try:
            # 1) Choose which model to run
            
            if mode == "Gamification Mode":
                # Use the MASTER model (compatible with multiple MASTER variants via kw filtering)
                master_kwargs = dict(
                    active_plants=gm_active_plants,
                    active_crossdocks=gm_active_crossdocks,
                    # All DCs active (UI no longer allows selecting DCs)
                    active_dcs=dcs_all,
                    # Candidate set of new locations (availability controlled via isXXX flags)
                    active_new_locs=new_locs_all,

                    active_modes_L1=gm_modes_L1,
                    active_modes_L2=gm_modes_L2,
                    active_modes_L3=gm_modes_L3,

                    # NEW: enforce transport-mode shares (per-node). None => ignored.
                    mode_share_L1_by_plant=st.session_state.get("gm_mode_share_L1_by_plant", None),
                    mode_share_L2_by_origin=st.session_state.get("gm_mode_share_L2_by_origin", None),

                    # Scenario params
                    CO_2_percentage=co2_pct,
                    co2_cost_per_ton=co2_cost_per_ton,
                    co2_cost_per_ton_New=co2_cost_per_ton_New,
                    suez_canal=suez_flag,
                    oil_crises=oil_flag,
                    volcano=volcano_flag,
                    trade_war=trade_flag,
                    tariff_rate=tariff_rate_used,
                    sourcing_cost=scaled_sourcing_cost,
                    service_level=service_level,
                    print_results="NO",
                )

                # Add per-new-location switches (isHUDTG/isCZMCT/...)
                master_kwargs.update(gm_newloc_flag_kwargs)

                results, model = run_master_filtered(master_kwargs)

                # ------------------------------------------------------------
                # Benchmarking
                # ------------------------------------------------------------
                try:
                    # Always benchmark against SC2F optimal (Allow New Facilities)
                    benchmark_label = "SC2F Optimal (Allow New Facilities)"
                
                    # Use the same CO₂ price the user entered
                    # - SC1F seçiliyse: co2_cost_per_ton var
                    # - SC2F seçiliyse: co2_cost_per_ton_New var
                    bench_co2_new      = co2_cost_per_ton_New if "SC2F" in model_choice else co2_cost_per_ton
                
                    bench_kwargs = dict(
                        CO_2_percentage=co2_pct,
                        co2_cost_per_ton_New=bench_co2_new,
                        suez_canal=suez_flag,
                        oil_crises=oil_flag,
                        volcano=volcano_flag,
                        trade_war=trade_flag,
                        tariff_rate=tariff_rate_used,
                        sourcing_cost=scaled_sourcing_cost,
                        print_results="NO",
                        service_level=service_level,
                    )

                    benchmark_results, benchmark_model = run_filtered(run_SC2F, bench_kwargs)
                
                except Exception as _bench_e:
                    benchmark_results = None
                    benchmark_model = None
                    benchmark_label = None
                    st.warning(f"Benchmark run failed (showing only gamification results). Reason: {_bench_e}")
                
                
                

            elif "SC1F" in model_choice:
                # Existing facilities only
                sc1_kwargs = dict(
                    CO_2_percentage=co2_pct,
                    co2_cost_per_ton=co2_cost_per_ton,
                    suez_canal=suez_flag,
                    oil_crises=oil_flag,
                    volcano=volcano_flag,
                    trade_war=trade_flag,
                    tariff_rate=tariff_rate_used,
                    sourcing_cost=scaled_sourcing_cost,
                    print_results="NO",
                    service_level=service_level,
                )
                results, model = run_filtered(run_SC1F, sc1_kwargs)
            else:
                # Allow new EU facilities (SC2F)
                sc2_kwargs = dict(
                    CO_2_percentage=co2_pct,
                    co2_cost_per_ton_New=co2_cost_per_ton_New,
                    suez_canal=suez_flag,
                    oil_crises=oil_flag,
                    volcano=volcano_flag,
                    trade_war=trade_flag,
                    tariff_rate=tariff_rate_used,
                    sourcing_cost=scaled_sourcing_cost,
                    print_results="NO",
                    service_level=service_level,
                )
                results, model = run_filtered(run_SC2F, sc2_kwargs)


            st.success("Optimization complete! ✅")

            # ===========================================
            # Objective + Emissions
            # ===========================================
            st.metric("💰 Objective Value (€)", f"{results['Objective_value']:,.2f}")

            # ------------------------------------------------------------
            # Show gap vs optimal (only in Gamification Mode)
            # ------------------------------------------------------------
            if mode == "Gamification Mode" and benchmark_results is not None:
                try:
                    stud_obj = float(results.get("Objective_value", 0.0))
                    opt_obj  = float(benchmark_results.get("Objective_value", 0.0))
                    gap = stud_obj - opt_obj
                    gap_pct = (gap / opt_obj * 100.0) if opt_obj != 0 else 0.0
            
                    st.subheader("🏁 Gap vs Optimal")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Your (Gamification) Objective (€)", f"{stud_obj:,.2f}")
                    c2.metric(benchmark_label or "Optimal Objective (€)", f"{opt_obj:,.2f}")
                    c3.metric("Gap (You − Optimal)", f"{gap:,.2f}", delta=f"{gap_pct:+.2f}%")
            
                    with st.expander("See benchmark breakdown"):
                        st.json({
                            "Benchmark": benchmark_label,
                            "Benchmark Objective": opt_obj,
                            "Your Objective": stud_obj,
                            "Absolute Gap": gap,
                            "Gap (%)": gap_pct,
                        })
                except Exception:
                    pass





            st.subheader("🌿 CO₂ Emissions")
            st.json({
                "Air": results.get("E_air", 0),
                "Sea": results.get("E_sea", 0),
                "Road": results.get("E_road", 0),
                "Last-mile": results.get("E_lastmile", 0),
                "Production": results.get("E_production", 0),
                "Total": results.get("CO2_Total", 0),
            })

            # ===========================================
            # 🌍 MAP (no more pd errors!)
            # ===========================================
            st.markdown("## 🌍 Global Supply Chain Map")

            nodes = [
                ("Plant", 31.230416, 121.473701, "Shanghai"),
                ("Plant", 23.553100, 121.021100, "Taiwan"),
                ("Cross-dock", 48.856610, 2.352220, "Paris"),
                ("Cross-dock", 54.352100, 18.646400, "Gdansk"),
                ("Cross-dock", 48.208500, 16.372100, "Vienna"),
                ("DC", 50.040750, 15.776590, "Pardubice"),
                ("DC", 50.629250, 3.057256, "Lille"),
                ("DC", 56.946285, 24.105078, "Riga"),
                ("DC", 28.116667, -17.216667, "LaGomera"),
                ("Retail", 50.935173, 6.953101, "Cologne"),
                ("Retail", 51.219890, 4.403460, "Antwerp"),
                ("Retail", 50.061430, 19.936580, "Krakow"),
                ("Retail", 54.902720, 23.909610, "Kaunas"),
                ("Retail", 59.911491, 10.757933, "Oslo"),
                ("Retail", 53.350140, -6.266155, "Dublin"),
                ("Retail", 59.329440, 18.068610, "Stockholm"),
            ]


            locations = pd.DataFrame(nodes, columns=["Type", "Lat", "Lon", "City"])

            # ================================================================
            # 🌍 FULL GLOBAL MAP (with new facilities + events)
            # ================================================================
            
            # New facilities (only if active)
            facility_coords = {
                "Budapest": (47.497913, 19.040236, "Budapest"),
                "Prague": (50.088040, 14.420760, "Prague"),
                "Dublin": (53.350140, -6.266155, "Dublin"),
                "Helsinki": (60.169520, 24.935450, "Helsinki"),
                "Warsaw": (52.229770, 21.011780, "Warsaw"),
            }

            
            for name, (lat, lon, city) in facility_coords.items():
                var = model.getVarByName(f"f2_2_bin[{name}]")
                if var is not None and var.X > 0.5:
                    nodes.append(("New Production Facility", lat, lon, city))
            
            # Build DataFrame
            locations = pd.DataFrame(nodes, columns=["Type", "Lat", "Lon", "City"])
            
            # ================================================================
            # Add EVENT MARKERS to the map
            # ================================================================
            event_nodes = []
            
            if suez_flag:
                event_nodes.append(("Event: Suez Canal Blockade", 30.59, 32.27, "Suez Canal Crisis"))
            
            if volcano_flag:
                event_nodes.append(("Event: Volcano Eruption", 63.63, -19.62, "Volcanic Ash Zone"))
            
            if oil_flag:
                event_nodes.append(("Event: Oil Crisis", 28.60, 47.80, "Oil Supply Shock"))
            
            if trade_flag:
                event_nodes.append(("Event: Trade War", 55.00, 60.00, "Trade War Impact Zone"))
            
            if event_nodes:
                df_events = pd.DataFrame(event_nodes, columns=["Type", "Lat", "Lon", "City"])
                locations = pd.concat([locations, df_events], ignore_index=True)
            
            # ================================================================
            # Marker colors & sizes
            # ================================================================
            color_map = {
                "Plant": "purple",
                "Cross-dock": "dodgerblue",
                "DC": "black",
                "Retail": "red",
                "New Production Facility": "deepskyblue",
                "Event: Suez Canal Blockade": "gold",
                "Event: Volcano Eruption": "orange",
                "Event: Oil Crisis": "brown",
                "Event: Trade War": "green",
            }
            
            size_map = {
                "Plant": 15,
                "Cross-dock": 14,
                "DC": 16,
                "Retail": 20,
                "New Production Facility": 14,
                "Event: Suez Canal Blockade": 18,
                "Event: Volcano Eruption": 18,
                "Event: Oil Crisis": 18,
                "Event: Trade War": 18,
            }

            
            # ================================================================
            # Build MAP
            # ================================================================
            fig_map = px.scatter_geo(
                locations,
                lat="Lat",
                lon="Lon",
                color="Type",
                text="City",
                hover_name="City",
                color_discrete_map=color_map,
                projection="natural earth",
                scope="world",
                title="Global Supply Chain Structure",
            )
            
            # compute activity once
            key_thr = compute_key_throughput(model)
            
            for trace in fig_map.data:
                trace.marker.update(
                    size=size_map.get(trace.name, 12),
                    line=dict(width=0.5, color="white"),
                )
            
                if trace.name.startswith("Event:") or trace.name == "New Production Facility":
                    trace.marker.update(opacity=0.9)
                    continue
            
                if hasattr(trace, "text") and trace.text is not None:
                    per_point_opacity = [
                        0.9 if city_is_active(city, key_thr) else 0.25
                        for city in trace.text
                    ]
                    trace.marker.update(opacity=per_point_opacity)
                else:
                    trace.marker.update(opacity=0.9)

            
                # Events and New Production Facility -> always bright (unchanged behaviour)
                if trace.name.startswith("Event:") or trace.name == "New Production Facility":
                    trace.marker.update(opacity=0.9)
                    continue
            
                # For other facility types: per-point opacity based on City activity
                # px.scatter_geo puts city labels into trace.text
                if hasattr(trace, "text") and trace.text is not None:
                    per_point_opacity = [
                        0.9 if city_is_active(city, key_thr) else 0.25
                        for city in trace.text
                    ]
                    trace.marker.update(opacity=per_point_opacity)
                else:
                    trace.marker.update(opacity=0.9)

            
            fig_map.update_geos(
                showcountries=True,
                countrycolor="lightgray",
                showland=True,
                landcolor="rgb(245,245,245)",
                fitbounds="locations",
            )
            
            fig_map.update_layout(
                height=600,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            
            st.plotly_chart(fig_map, use_container_width=True)
            
            
            
            # ================================================================
            # 🏭 PRODUCTION OUTBOUND PIE CHART
            # ================================================================
            st.markdown("## 🏭 Production Outbound Breakdown")
            
            TOTAL_MARKET_DEMAND = 111000
            
            f1_vars = [v for v in model.getVars() if v.VarName.startswith("f1[")]
            f2_2_vars = [v for v in model.getVars() if v.VarName.startswith("f2_2[")]
            
            prod_sources = {}
            
            # Existing plants
            for plant in ["Taiwan", "Shanghai"]:
                total = sum(v.X for v in f1_vars if v.VarName.startswith(f"f1[{plant},"))
                prod_sources[plant] = total
            
            # New EU facilities
            for fac in ["Budapest", "Prague", "Dublin", "Helsinki", "Warsaw"]:
                total = sum(v.X for v in f2_2_vars if v.VarName.startswith(f"f2_2[{fac},"))
                prod_sources[fac] = total
            
            total_produced = sum(prod_sources.values())
            unmet = max(TOTAL_MARKET_DEMAND - total_produced, 0)
            
            labels = list(prod_sources.keys()) + ["Unmet Demand"]
            values = list(prod_sources.values()) + [unmet]
            
            df_prod = pd.DataFrame({"Source": labels, "Units Produced": values})
            
            fig_prod = px.pie(
                df_prod,
                names="Source",
                values="Units Produced",
                hole=0.3,
                title="Production Share by Source",
            )
            
            color_map = {name: col for name, col in zip(df_prod["Source"], px.colors.qualitative.Set2)}
            color_map["Unmet Demand"] = "lightgrey"
            
            fig_prod.update_traces(
                textinfo="label+percent",
                textfont_size=13,
                marker=dict(colors=[color_map[s] for s in df_prod["Source"]])
            )
            
            fig_prod.update_layout(
                showlegend=True,
                height=400,
                template="plotly_white",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_prod, use_container_width=True)
            st.markdown("#### 📦 Production Summary Table")
            st.dataframe(df_prod.round(2), use_container_width=True)
            
            
            
            # ================================================================
            # 🚚 CROSS-DOCK OUTBOUND PIE CHART
            # ================================================================
            st.markdown("## 🚚 Cross-dock Outbound Breakdown")
            
            f2_vars = [v for v in model.getVars() if v.VarName.startswith("f2[")]
            
            crossdocks = ["Vienna", "Gdansk", "Paris"]
            crossdock_flows = {}
            
            for cd in crossdocks:
                total = sum(v.X for v in f2_vars if v.VarName.startswith(f"f2[{cd},"))
                crossdock_flows[cd] = total
            
            if sum(crossdock_flows.values()) == 0:
                st.info("No cross-dock activity.")
            else:
                df_crossdock = pd.DataFrame({
                    "Crossdock": list(crossdock_flows.keys()),
                    "Shipped (units)": list(crossdock_flows.values()),
                })
                df_crossdock["Share (%)"] = df_crossdock["Shipped (units)"] / df_crossdock["Shipped (units)"].sum() * 100
            
                fig_crossdock = px.pie(
                    df_crossdock,
                    names="Crossdock",
                    values="Shipped (units)",
                    hole=0.3,
                    title="Cross-dock Outbound Share"
                )
            
                fig_crossdock.update_layout(
                    showlegend=True,
                    height=400,
                    template="plotly_white",
                    margin=dict(l=20, r=20, t=40, b=20),
                )
            
                st.plotly_chart(fig_crossdock, use_container_width=True)
            
                st.markdown("#### 🚚 Cross-dock Outbound Table")
                st.dataframe(df_crossdock.round(2), use_container_width=True)


            # ================================================================
            # 🚚 Transport Flows by Mode (match SC1/SC2 apps)
            # ================================================================
            render_transport_flows_by_mode(model)

            # ================================================================
            # 💰🌿 Cost & Emission Distribution (match SC1/SC2 apps)
            # ================================================================
            render_cost_emission_distribution(results)


        except Exception as e:
            # --------------------------------------------------
            # PRIMARY MODEL FAILED
            # --------------------------------------------------
            st.error(f"❌ Primary optimization failed: {e}")

            # In Gamification Mode we DO NOT run SC1F/SC2F_uns,
            # because they ignore the student's facility/mode choices.
            if mode == "Gamification Mode":
                st.warning(
                    "Fallback models are only defined for SC1F/SC2F. "
                    "In Gamification Mode, please adjust your facility / mode "
                    "selection or relax the CO₂ target and try again."
                )

            else:
                st.warning("⚠ Running fallback model to compute maximum satisfiable demand...")

                try:
                    # --------------------------------------------------
                    # CHOOSE CORRECT FALLBACK MODEL
                    # --------------------------------------------------
                    if "SC2F" in model_choice:
                        from Scenario_Setting_For_SC2F_uns import run_scenario as run_Uns
                        results_uns, model_uns = run_Uns(
                            CO_2_percentage=co2_pct,
                            co2_cost_per_ton_New=co2_cost_per_ton_New,
                            suez_canal=suez_flag,
                            oil_crises=oil_flag,
                            volcano=volcano_flag,
                            trade_war=trade_flag,
                            tariff_rate=tariff_rate_used,
                            print_results="NO",
                        )
                    else:
                        from Scenario_Setting_For_SC1F_uns import run_scenario as run_Uns
                        results_uns, model_uns = run_Uns(
                            CO_2_percentage=co2_pct,
                            co2_cost_per_ton=co2_cost_per_ton,
                            suez_canal=suez_flag,
                            oil_crises=oil_flag,
                            volcano=volcano_flag,
                            trade_war=trade_flag,
                            tariff_rate=tariff_rate_used,
                            print_results="NO",
                        )

                    # --------------------------------------------------
                    # SUCCESS DISPLAY (FALLBACK MODEL)
                    # --------------------------------------------------
                    st.success("Fallback optimization successful! ✅")

                    # ===================================================
                    # 📦 MAXIMUM SATISFIABLE DEMAND
                    # ===================================================
                    st.markdown("## 📦 Maximum Satisfiable Demand (Fallback Model)")

                    st.metric(
                        "Satisfied Demand (%)",
                        f"{results_uns['Satisfied_Demand_pct'] * 100:.2f}%"
                    )

                    st.metric(
                        "Satisfied Demand (Units)",
                        f"{results_uns['Satisfied_Demand_units']:,.0f}"
                    )

                    # ===================================================
                    # 💰 OBJECTIVE
                    # ===================================================
                    st.markdown("## 💰 Objective Value (Excluding Slack Penalty)")
                    st.metric(
                        "Objective (€)",
                        f"{results_uns['Objective_value']:,.2f}"
                    )

                    # ===================================================
                    # 🌍 MAP
                    # ===================================================
                    st.markdown("## 🌍 Global Supply Chain Map (Fallback Model)")

                    nodes = [
                    ("Plant", 31.23, 121.47, "Shanghai"),
                    ("Plant", 22.32, 114.17, "Taiwan"),
                    ("Cross-dock", 48.85, 2.35, "Paris"),
                    ("Cross-dock", 50.11, 8.68, "Gdansk"),
                    ("Cross-dock", 37.98, 23.73, "Vienna"),
                    ("DC", 47.50, 19.04, "Pardubice"),
                    ("DC", 48.14, 11.58, "Lille"),
                    ("DC", 46.95, 7.44, "Riga"),
                    ("DC", 45.46, 9.19, "LaGomera"),
                    ("Retail", 55.67, 12.57, "Cologne"),
                    ("Retail", 53.35, -6.26, "Antwerp"),
                    ("Retail", 51.50, -0.12, "Krakow"),
                    ("Retail", 49.82, 19.08, "Kaunas"),
                    ("Retail", 45.76, 4.83, "Oslo"),
                    ("Retail", 43.30, 5.37, "Dublin"),
                    ("Retail", 40.42, -3.70, "Stockholm"),
                    ]

                    # Add new facilities from fallback model
                    facility_coords = {
                        "Budapest": (49.61, 6.13, "Budapest"),
                        "Prague": (44.83, 20.42, "Prague"),
                        "Dublin": (47.09, 16.37, "Dublin"),
                        "Helsinki": (50.45, 14.50, "Helsinki"),
                        "Warsaw": (42.70, 12.65, "Warsaw"),
                    }

                    for name, (lat, lon, city) in facility_coords.items():
                        var = model_uns.getVarByName(f"f2_2_bin[{name}]")
                        if var is not None and var.X > 0.5:
                            nodes.append(("New Production Facility", lat, lon, city))

                    locations = pd.DataFrame(nodes, columns=["Type", "Lat", "Lon", "City"])

                    # Event overlays
                    event_nodes = []
                    if suez_flag:
                        event_nodes.append(
                            ("Event: Suez Canal Blockade", 30.59, 32.27, "Suez Canal Crisis")
                        )
                    if volcano_flag:
                        event_nodes.append(
                            ("Event: Volcano Eruption", 63.63, -19.62, "Volcanic Ash Zone")
                        )
                    if oil_flag:
                        event_nodes.append(
                            ("Event: Oil Crisis", 28.60, 47.80, "Oil Supply Shock")
                        )
                    if trade_flag:
                        event_nodes.append(
                            ("Event: Trade War", 55.00, 60.00, "Trade War Impact Zone")
                        )

                    if event_nodes:
                        df_events = pd.DataFrame(event_nodes, columns=["Type", "Lat", "Lon", "City"])
                        locations = pd.concat([locations, df_events], ignore_index=True)

                    color_map = {
                        "Plant": "purple",
                        "Cross-dock": "dodgerblue",
                        "DC": "black",
                        "Retail": "red",
                        "New Production Facility": "deepskyblue",
                        "Event: Suez Canal Blockade": "gold",
                        "Event: Volcano Eruption": "orange",
                        "Event: Oil Crisis": "brown",
                        "Event: Trade War": "green",
                    }

                    size_map = {
                        "Plant": 15,
                        "Cross-dock": 14,
                        "DC": 16,
                        "Retail": 20,
                        "New Production Facility": 14,
                        "Event: Suez Canal Blockade": 18,
                        "Event: Volcano Eruption": 18,
                        "Event: Oil Crisis": 18,
                        "Event: Trade War": 18,
                    }

                    fig_map = px.scatter_geo(
                        locations,
                        lat="Lat",
                        lon="Lon",
                        color="Type",
                        text="City",
                        hover_name="City",
                        color_discrete_map=color_map,
                        projection="natural earth",
                        scope="world",
                        title="Global Supply Chain Structure (Fallback Model)",
                    )

                    for trace in fig_map.data:
                        trace.marker.update(
                            size=size_map.get(trace.name, 12),
                            opacity=0.9,
                            line=dict(width=0.5, color="white"),
                        )

                    fig_map.update_geos(
                        showcountries=True,
                        countrycolor="lightgray",
                        showland=True,
                        landcolor="rgb(245,245,245)",
                        fitbounds="locations",
                    )

                    fig_map.update_layout(
                        height=600,
                        margin=dict(l=0, r=0, t=40, b=0),
                    )

                    st.plotly_chart(fig_map, use_container_width=True)

                    # ===================================================
                    # 🏭 PRODUCTION OUTBOUND PIE CHART
                    # ===================================================
                    st.markdown("## 🏭 Production Outbound Breakdown (Fallback Model)")

                    f1_vars = [v for v in model_uns.getVars() if v.VarName.startswith("f1[")]
                    f2_2_vars = [v for v in model_uns.getVars() if v.VarName.startswith("f2_2[")]

                    prod_sources = {}

                    # Existing plants
                    for plant in ["Taiwan", "Shanghai"]:
                        total = sum(v.X for v in f1_vars if v.VarName.startswith(f"f1[{plant},"))
                        prod_sources[plant] = total

                    # New EU facilities
                    for fac in ["Budapest", "Prague", "Dublin", "Helsinki", "Warsaw"]:
                        total = sum(v.X for v in f2_2_vars if v.VarName.startswith(f"f2_2[{fac},"))
                        prod_sources[fac] = total

                    TOTAL_MARKET_DEMAND = 111000
                    total_produced = sum(prod_sources.values())
                    unmet = max(TOTAL_MARKET_DEMAND - total_produced, 0)

                    labels = list(prod_sources.keys()) + ["Unmet Demand"]
                    values = list(prod_sources.values()) + [unmet]

                    df_prod = pd.DataFrame({"Source": labels, "Units Produced": values})

                    fig_prod = px.pie(
                        df_prod,
                        names="Source",
                        values="Units Produced",
                        hole=0.3,
                        title="Production Share by Source (Fallback Model)",
                    )

                    fig_prod.update_traces(
                        textinfo="label+percent",
                        textfont_size=13,
                    )

                    st.plotly_chart(fig_prod, use_container_width=True)
                    st.dataframe(df_prod.round(2), use_container_width=True)

                    # ===================================================
                    # 🚚 CROSS-DOCK OUTBOUND PIE CHART
                    # ===================================================
                    st.markdown("## 🚚 Cross-dock Outbound Breakdown (Fallback Model)")

                    f2_vars = [v for v in model_uns.getVars() if v.VarName.startswith("f2[")]

                    crossdocks = ["Vienna", "Gdansk", "Paris"]
                    crossdock_flows = {}

                    for cd in crossdocks:
                        total = sum(v.X for v in f2_vars if v.VarName.startswith(f"f2[{cd},"))
                        crossdock_flows[cd] = total

                    if sum(crossdock_flows.values()) == 0:
                        st.info("No cross-dock activity.")
                    else:
                        df_crossdock = pd.DataFrame({
                            "Crossdock": list(crossdock_flows.keys()),
                            "Shipped (units)": list(crossdock_flows.values()),
                        })
                        df_crossdock["Share (%)"] = (
                            df_crossdock["Shipped (units)"] /
                            df_crossdock["Shipped (units)"].sum()
                        ) * 100

                        fig_crossdock = px.pie(
                            df_crossdock,
                            names="Crossdock",
                            values="Shipped (units)",
                            hole=0.3,
                            title="Cross-dock Outbound Share (Fallback Model)",
                        )

                        st.plotly_chart(fig_crossdock, use_container_width=True)
                        st.dataframe(df_crossdock.round(2), use_container_width=True)

                    # ================================================================
                    # 🚚 Transport Flows by Mode (match SC1/SC2 apps)
                    # ================================================================
                    render_transport_flows_by_mode(model_uns)

                    # ================================================================
                    # 💰🌿 Cost & Emission Distribution (match SC1/SC2 apps)
                    # ================================================================
                    render_cost_emission_distribution(results_uns)

                except Exception as e2:
                    st.error(f"❌ Fallback model also failed: {e2}")



