# -*- coding: utf-8 -*-
"""gamification.py

Gamification Mode was originally implemented inline inside Total.py.
This file extracts that block so Total.py can simply import it and
toggle it on/off with a single flag.

⚠️ Design goals
- Do **not** change existing widget keys or session_state names.
- Do **not** change defaults or business logic.
- Return a single dict that Total.py can use to build MASTER kwargs.

UI update (Feb 2026)
- Make the layout more compact (avoid many repeated expanders/tabs)
- Keep the same state keys and returned structure
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

    # -----------------------
    # Scenario events (compact)
    # -----------------------
    st.markdown("#### Scenario events")
    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        suez_flag = st.checkbox(
            "Suez Canal Blockade (no Water from plants to Europe)",
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
            "Sourcing Cost Surcharge (Trade War)",
            min_value=1.0,
            max_value=2.0,
            value=1.3,
            step=0.05,
            help="1.0 = no surcharge, 2.0 = sourcing cost doubles",
        )

    st.markdown("---")

    # ----------------
    # Facility selection
    # ----------------
    st.markdown("#### Facility activation")

    plants_all = ["Taiwan", "Shanghai"]
    crossdocks_all = ["Vienna", "Gdansk", "Paris"]
    dcs_all = ["Pardubice", "Calais", "Riga", "LaGomera"]
    new_locs_all = ["Budapest", "Prague", "Cork", "Helsinki", "Warsaw"]

    st.info("✅ In Gamification Mode, all Distribution Centers (DCs) are assumed active.")

    # Keep original checkbox keys, but hide them inside compact expanders
    col_p, col_c, col_n = st.columns(3)

    with col_p:
        st.caption("Plants")
        with st.expander("Select plants", expanded=False):
            gm_active_plants = [
                p for p in plants_all
                if st.checkbox(p, value=True, key=f"gm_pl_{p}")
            ]

    with col_c:
        st.caption("Cross-docks")
        with st.expander("Select cross-docks", expanded=False):
            gm_active_crossdocks = [
                c for c in crossdocks_all
                if st.checkbox(c, value=True, key=f"gm_cd_{c}")
            ]

    with col_n:
        st.caption("New production sites")
        with st.expander("Select new sites", expanded=False):
            gm_active_new_locs = [
                n for n in new_locs_all
                if st.checkbox(n, value=False, key=f"gm_new_{n}")
            ]

    # All DCs active (no selection in UI)
    gm_active_dcs = list(dcs_all)
    st.session_state["gm_active_new_locs"] = gm_active_new_locs

    # Map selections -> MASTER boolean flags (isBudapest, isPrague, ...)
    gm_newloc_flag_kwargs = {f"is{code}": (code in gm_active_new_locs) for code in new_locs_all}

    st.markdown("---")

    # ---------------------------
    # Allowed transport modes
    # ---------------------------
    st.markdown("#### Allowed transport modes per layer")

    all_modes = ["air", "Water", "road"]
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        gm_modes_L1 = st.multiselect(
            "Plant → Cross-dock (Road not allowed)",
            options=["air", "Water"],
            default=["air", "Water"],
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

    st.markdown("---")

    # -----------------------------------------
    # Mode share enforcement: per-node shares
    # -----------------------------------------
    st.markdown("#### Transport mode shares (enforced on Layer 1 & 2)")
    st.caption(
        "You set shares for some modes; the remaining mode is auto-filled to reach 100%. "
        "(Defaults: L1=50/50, L2=50/50/0)"
    )

    def _pct(x: float) -> str:
        return f"{100.0 * float(x):.1f}%"

    def _ensure_int_state(key: str, default: int) -> int:
        """Ensure st.session_state[key] exists as an int."""
        if key not in st.session_state or st.session_state[key] is None:
            st.session_state[key] = int(default)
        try:
            st.session_state[key] = int(st.session_state[key])
        except Exception:
            st.session_state[key] = int(default)
        return int(st.session_state[key])

    gm_mode_share_L1_by_plant: Dict[str, Any] | None = None
    gm_mode_share_L2_by_origin: Dict[str, Any] | None = None

    # -------------------------
    # Layer 1: per-plant shares
    # -------------------------
    st.markdown("**Layer 1 (Plant → Cross-dock): per-plant shares (Road is forbidden)**")
    if len(gm_active_plants) == 0:
        st.info("No active plants selected.")
        gm_mode_share_L1_by_plant = None
    else:
        # Build shares for all active plants from session_state (defaults if unseen)
        gm_mode_share_L1_by_plant = {}
        for p in gm_active_plants:
            keyW = f"gm_l1_{p}_Water"
            Water_pct = _ensure_int_state(keyW, default=50)
            Water = float(Water_pct) / 100.0
            gm_mode_share_L1_by_plant[p] = {"Water": float(Water), "air": None}

        # Compact editor: choose one plant to edit at a time
        edit_col, table_col = st.columns([1, 2])
        with edit_col:
            plant_to_edit = st.selectbox(
                "Edit a plant",
                options=list(gm_active_plants),
                index=0,
                key="gm_l1_edit_plant",
            )
            keyW = f"gm_l1_{plant_to_edit}_Water"
            # Render the existing slider with the same key (important!)
            Water_pct = st.slider(
                f"{plant_to_edit} – Water share (L1) (%)",
                min_value=0,
                max_value=100,
                value=_ensure_int_state(keyW, default=50),
                step=1,
                key=keyW,
            )
            Water = float(Water_pct) / 100.0
            air = 1.0 - float(Water)
            st.write(f"{plant_to_edit} – Air share (L1, auto): **{_pct(air)}**")

            # Update dict for the edited plant
            gm_mode_share_L1_by_plant[plant_to_edit] = {"Water": float(Water), "air": None}

        with table_col:
            rows = []
            for p in gm_active_plants:
                keyW = f"gm_l1_{p}_Water"
                Water_pct = _ensure_int_state(keyW, default=50)
                Water = float(Water_pct) / 100.0
                air = 1.0 - float(Water)
                rows.append({"Plant": p, "Water": _pct(Water), "Air (auto)": _pct(air)})
            st.dataframe(rows, use_container_width=True, hide_index=True)

    st.markdown("---")

    # -----------------------------------------
    # Layer 2: per-origin shares
    # -----------------------------------------
    st.markdown("**Layer 2 (Cross-dock / New → DC): per-origin shares**")
    active_origins: List[str] = list(gm_active_crossdocks) + list(gm_active_new_locs)

    if len(active_origins) == 0:
        st.info("No active cross-docks or new facilities selected.")
        gm_mode_share_L2_by_origin = None
    else:
        gm_mode_share_L2_by_origin = {}

        # Ensure defaults exist for all origins (even if user doesn't edit each)
        for o in active_origins:
            keyW = f"gm_l2_{o}_Water"
            Water_pct = _ensure_int_state(keyW, default=50)
            Water = float(Water_pct) / 100.0
            rem = max(0.0, 1.0 - float(Water))

            keyA = f"gm_l2_{o}_air"
            if rem <= 1e-12:
                # Air fixed to 0 when Water=100%
                if keyA not in st.session_state:
                    st.session_state[keyA] = 0
                st.session_state[keyA] = 0
            else:
                air_default = min(0.50, float(rem))
                air_pct_default = int(round(100.0 * float(air_default)))
                _ensure_int_state(keyA, default=air_pct_default)
                # Clamp air to remaining share if needed
                max_air_pct = int(round(100.0 * float(rem)))
                if int(st.session_state[keyA]) > max_air_pct:
                    st.session_state[keyA] = max_air_pct

            # Build dict
            air = float(int(st.session_state[keyA])) / 100.0 if rem > 1e-12 else 0.0
            gm_mode_share_L2_by_origin[o] = {"Water": float(Water), "air": float(air), "road": None}

        # Compact editor: choose one origin to edit at a time
        edit_col, table_col = st.columns([1, 2])
        with edit_col:
            origin_to_edit = st.selectbox(
                "Edit an origin",
                options=list(active_origins),
                index=0,
                key="gm_l2_edit_origin",
            )

            keyW = f"gm_l2_{origin_to_edit}_Water"
            Water_pct = st.slider(
                f"{origin_to_edit} – Water share (L2) (%)",
                min_value=0,
                max_value=100,
                value=_ensure_int_state(keyW, default=50),
                step=1,
                key=keyW,
            )
            Water = float(Water_pct) / 100.0
            rem = max(0.0, 1.0 - float(Water))

            keyA = f"gm_l2_{origin_to_edit}_air"
            if rem <= 1e-12:
                st.session_state[keyA] = 0
                air = 0.0
                st.write(f"{origin_to_edit} – Air share (L2): **{_pct(air)}** (fixed because Water is 100%)")
            else:
                max_air_pct = int(round(100.0 * float(rem)))
                # Clamp state before rendering slider to avoid Streamlit bounds errors
                _ensure_int_state(keyA, default=int(round(100.0 * min(0.50, float(rem)))))
                if int(st.session_state[keyA]) > max_air_pct:
                    st.session_state[keyA] = max_air_pct

                air_pct = st.slider(
                    f"{origin_to_edit} – Air share (L2) (%)",
                    min_value=0,
                    max_value=max_air_pct,
                    value=int(st.session_state[keyA]),
                    step=1,
                    key=keyA,
                )
                air = float(air_pct) / 100.0

            road = max(0.0, 1.0 - float(Water) - float(air))
            st.write(f"{origin_to_edit} – Road share (L2, auto): **{_pct(road)}**")

            # Update dict for edited origin
            gm_mode_share_L2_by_origin[origin_to_edit] = {"Water": float(Water), "air": float(air), "road": None}

        with table_col:
            rows = []
            for o in active_origins:
                keyW = f"gm_l2_{o}_Water"
                keyA = f"gm_l2_{o}_air"
                Water_pct = _ensure_int_state(keyW, default=50)
                Water = float(Water_pct) / 100.0
                rem = max(0.0, 1.0 - float(Water))
                air = (float(_ensure_int_state(keyA, default=0)) / 100.0) if rem > 1e-12 else 0.0
                # Clamp air to remaining (for display)
                if air > rem:
                    air = rem
                road = max(0.0, 1.0 - float(Water) - float(air))
                rows.append({"Origin": o, "Water": _pct(Water), "Air": _pct(air), "Road (auto)": _pct(road)})
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # Ensure required modes are enabled in the mode lists (except road on L1)
    # (Mode-share constraints require these variables to exist.)
    gm_modes_L1 = sorted(set(gm_modes_L1) | {"air", "Water"})
    gm_modes_L2 = sorted(set(gm_modes_L2) | {"air", "Water", "road"})

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
