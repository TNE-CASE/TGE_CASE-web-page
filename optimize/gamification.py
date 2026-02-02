# -*- coding: utf-8 -*-
"""gamification.py

Gamification Mode was originally implemented inline inside Total.py.
This file extracts that block so Total.py can simply import it and
toggle it on/off with a single flag.

⚠️ Design goals
- Do **not** change widget keys or session_state names.
- Do **not** change defaults or business logic.
- Return a single dict that Total.py can use to build MASTER kwargs.
"""

from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def render_gamification_mode() -> Dict[str, Any]:
    """Render the Gamification Mode UI and return the collected configuration.

    Returns a dict containing:
      - scenario flags + tariff_rate_used
      - active facility lists
      - active transport modes per layer
      - per-node mode-share dicts (stored in st.session_state too)
      - per-new-location boolean flags for MASTER (isBudapest, isPrague, ...)
      - convenience lists: dcs_all, new_locs_all
    """

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
            key="gm_suez",
        )
        oil_flag = st.checkbox(
            "Oil Crisis (increase all transport costs)",
            value=False,
            key="gm_oil",
        )
    with col_ev2:
        volcano_flag = st.checkbox(
            "Volcanic Eruption (no air shipments)",
            value=False,
            key="gm_volcano",
        )
        trade_flag = st.checkbox(
            "Trade War (more expensive sourcing)",
            value=False,
            key="gm_trade",
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
    dcs_all = ["Pardubice", "Calais", "Riga", "LaGomera"]
    new_locs_all = ["Budapest", "Prague", "Cork", "Helsinki", "Warsaw"]

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
            if st.checkbox(n, value=False, key=f"gm_new_{n}")
        ]

    # All DCs active (no selection in UI)
    gm_active_dcs = list(dcs_all)
    st.session_state["gm_active_new_locs"] = gm_active_new_locs

    # Map selections -> MASTER boolean flags (isBudapest, isPrague, ...)
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

    # --- Mode share enforcement: per-node shares for MASTER ---
    # Always enforced in Gamification Mode (no checkbox).
    st.markdown("#### Transport mode shares (enforced on Layer 1 & 2)")

    gm_mode_share_L1_by_plant = None
    gm_mode_share_L2_by_origin = None

    def _pct(x: float) -> str:
        return f"{100.0 * float(x):.1f}%"

    def _l1_share_ui_for_plant(plant: str, key_prefix: str) -> Dict[str, Any]:
        # Only air/sea; we ask for sea and auto-fill air
        sea_pct = st.slider(
            f"{plant} – Sea share (L1) (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key=f"{key_prefix}_sea",
        )
        sea = float(sea_pct) / 100.0
        air = 1.0 - float(sea)
        st.write(f"{plant} – Air share (L1, auto): **{_pct(air)}**")
        # Use None remainder semantics for robustness on the MASTER side
        return {"sea": float(sea), "air": None}

    def _l2_share_ui_for_origin(origin: str, key_prefix: str) -> Dict[str, Any]:
        # Ask for sea, then air up to remaining; road is remainder
        sea_pct = st.slider(
            f"{origin} – Sea share (L2) (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=1,
            key=f"{key_prefix}_sea",
        )
        sea = float(sea_pct) / 100.0
        rem_after_sea = 1.0 - float(sea)

        if rem_after_sea <= 1e-12:
            air = 0.0
            st.write(f"{origin} – Air share (L2): **{_pct(air)}** (fixed because Sea is 100%)")
        else:
            # Default to 50-50-0 (sea-air-road)
            air_default = min(0.50, float(rem_after_sea))
            air_pct = st.slider(
                f"{origin} – Air share (L2) (%)",
                min_value=0,
                max_value=int(round(100.0 * float(rem_after_sea))),
                value=int(round(100.0 * float(air_default))),
                step=1,
                key=f"{key_prefix}_air",
            )
            air = float(air_pct) / 100.0

        road = max(0.0, 1.0 - float(sea) - float(air))
        st.write(f"{origin} – Road share (L2, auto): **{_pct(road)}**")

        # Road is auto remainder
        return {"sea": float(sea), "air": float(air), "road": None}

    st.caption(
        "For each node: you set shares for some modes; the remaining mode is auto-filled to reach 100%. "
        "(Default: L1=50/50, L2=50/50/0)"
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
    active_origins: List[str] = list(gm_active_crossdocks) + list(gm_active_new_locs)
    if len(active_origins) == 0:
        st.info("No active cross-docks or new facilities selected.")
    else:
        gm_mode_share_L2_by_origin = {}
        for o in active_origins:
            with st.expander(f"🏷️ {o}", expanded=False):
                gm_mode_share_L2_by_origin[o] = _l2_share_ui_for_origin(o, key_prefix=f"gm_l2_{o}")

    # Ensure required modes are enabled in the mode lists (except road on L1)
    # (Mode-share constraints require these variables to exist.)
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

    return {
        # scenario
        "suez_flag": suez_flag,
        "oil_flag": oil_flag,
        "volcano_flag": volcano_flag,
        "trade_flag": trade_flag,
        "tariff_rate_used": tariff_rate_used,
        # facilities
        "plants_all": plants_all,
        "crossdocks_all": crossdocks_all,
        "dcs_all": dcs_all,
        "new_locs_all": new_locs_all,
        "gm_active_plants": gm_active_plants,
        "gm_active_crossdocks": gm_active_crossdocks,
        "gm_active_dcs": gm_active_dcs,
        "gm_active_new_locs": gm_active_new_locs,
        "gm_newloc_flag_kwargs": gm_newloc_flag_kwargs,
        # modes
        "gm_modes_L1": gm_modes_L1,
        "gm_modes_L2": gm_modes_L2,
        "gm_modes_L3": gm_modes_L3,
        # mode shares
        "gm_mode_share_L1_by_plant": gm_mode_share_L1_by_plant,
        "gm_mode_share_L2_by_origin": gm_mode_share_L2_by_origin,
    }
