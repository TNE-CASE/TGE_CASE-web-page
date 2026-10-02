"""Shared model reference: the parameter tables the three pages show, and the cost distribution
chart they all draw.

Two jobs live here because both are needed by `Total.py` *and* by `sc1_app.py`/`sc2_app.py`, and
`Total.py` already imports those two — a helper placed there could not be imported back without a
cycle.

The numbers below mirror `Total.py::_puzzle_defaults()`. They are duplicated rather than imported
for the same reason: importing `Total` runs Streamlit at module load. Puzzle mode hands us its
live `cfg` so that page can never drift; the dashboards are parquet-backed (their numbers come
from `Scenario_Setting_For_SC*.py`), so the mirror is the honest source there. If a parameter
changes in `_puzzle_defaults()`, change it here too.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
# Parameters — mirror of Total.py::_puzzle_defaults()
# ---------------------------------------------------------------------------

PLANTS_ALL = ["Taiwan", "Shanghai"]
NEW_LOCS_ALL = ["Budapest", "Prague", "Cork", "Helsinki", "Warsaw"]

SOURCING_COST = {"Taiwan": 3.343692308, "Shanghai": 3.423384615}
CO2_PROD_KG_PER_UNIT = {"Taiwan": 6.3, "Shanghai": 9.8}

NEW_LOC_CAPACITY = {"Budapest": 37000, "Prague": 35500, "Cork": 46000,
                    "Helsinki": 35000, "Warsaw": 26500}
NEW_LOC_OPENING_COST = {"Budapest": 2.775e6, "Prague": 2.6625e6, "Cork": 3.45e6,
                        "Helsinki": 2.625e6, "Warsaw": 1.9875e6}
NEW_LOC_OPERATION_COST = {"Budapest": 250000, "Prague": 305000, "Cork": 450000,
                          "Helsinki": 420000, "Warsaw": 412500}
NEW_LOC_CO2 = {"Budapest": 3.2, "Prague": 2.8, "Cork": 4.6, "Helsinki": 5.8, "Warsaw": 6.2}

# The engine spreads a fixed 90,000 € budget over a site's capacity to get its per-unit rate
# (Total.py: `new_loc_unitCost = (1 / capacity) * 90000`), so the rate is derived, never written —
# and derived in that same order, which for Helsinki is one ULP away from `90000 / capacity`.
NEW_LOC_COST_BUDGET = 90000.0

TAU = {"air": 0.0105, "Water": 0.0013, "road": 0.0054}          # € per kg-km
CO2_EMISSION_FACTOR = {"air": 0.000971, "Water": 0.000027, "road": 0.000076}  # ton CO2 / ton-km

# Every delivered unit also travels the last leg to the customer, charged per unit rather than by
# distance (Total.py: `lastmile_CO2_kg`).
LAST_MILE_CO2_KG = 2.68

MODE_LABELS = {"air": "✈️ Air", "Water": "🚢 Water", "road": "🚛 Road"}

_DEMAND = {"Cologne": 17000, "Antwerp": 9000, "Krakow": 13000, "Kaunas": 19000,
           "Oslo": 15000, "Dublin": 20000, "Stockholm": 18000}

_DIST1 = pd.DataFrame(
    [[8997.94617146616, 8558.96520835034, 9812.38584027454],
     [8468.71339377354, 7993.62774285959, 9240.26233801075]],
    index=["Taiwan", "Shanghai"],
    columns=["Vienna", "Gdansk", "Paris"],
)
_DIST2 = pd.DataFrame(
    [[220.423995674989, 1019.43140587827, 1098.71652257982, 1262.62587924823],
     [519.161031102087, 1154.87176862626, 440.338211856603, 1855.94939751482],
     [962.668288266132, 149.819604703365, 1675.455462176, 2091.1437090641]],
    index=["Vienna", "Gdansk", "Paris"],
    columns=["Pardubice", "Calais", "Riga", "Algeciras"],
)
_DIST2_NEW = pd.DataFrame(
    [[367.762425639798, 1216.10262027458, 1098.57245368619, 1120.13248546123],
     [98.034644813461, 818.765381327031, 987.72775809091, 1529.9990581232],
     [1558.60889112091, 714.077816812742, 1949.83469918776, 2854.35402610261],
     [1265.72892702748, 1758.18103997611, 367.698822815676, 2461.59771450036],
     [437.686419974076, 1271.77800922148, 554.373376462774, 1592.14058614186]],
    index=["Budapest", "Prague", "Cork", "Helsinki", "Warsaw"],
    columns=["Pardubice", "Calais", "Riga", "Algeciras"],
)
_DIST3 = pd.DataFrame(
    [[1184.65051865833, 933.730015948432, 557.144058480586, 769.757089072695, 2147.98445345001, 2315.79621115423, 1590.07662902924],
     [311.994969562194, 172.326685809878, 622.433010022067, 1497.40239816531, 1387.73696467636, 1585.6370207201, 1984.31926933368],
     [1702.34810062205, 1664.62283033352, 942.985120680279, 222.318687415142, 2939.50970842422, 3128.54724287652, 713.715034612432],
     [2452.23922908608, 2048.41487682505, 2022.91355628344, 1874.11994156457, 2774.73634842816, 2848.65086298747, 2806.05576441898]],
    index=["Pardubice", "Calais", "Riga", "Algeciras"],
    columns=list(_DEMAND.keys()),
)

_FALLBACK = {
    "demand": _DEMAND,
    "plants_all": PLANTS_ALL,
    "new_locs_all": NEW_LOCS_ALL,
    "sourcing_cost": SOURCING_COST,
    "co2_prod_kg_per_unit": CO2_PROD_KG_PER_UNIT,
    "new_loc_capacity": NEW_LOC_CAPACITY,
    "new_loc_openingCost": NEW_LOC_OPENING_COST,
    "new_loc_operationCost": NEW_LOC_OPERATION_COST,
    "new_loc_CO2": NEW_LOC_CO2,
    "co2_emission_factor": CO2_EMISSION_FACTOR,
    "tau": TAU,
    "dist1": _DIST1,
    "dist2": _DIST2,
    "dist2_new": _DIST2_NEW,
    "dist3": _DIST3,
}

_MATRIX_TITLES = {
    "dist1": "Manufacturers → Cross-docks",
    "dist2": "Cross-docks → DCs",
    "dist2_new": "Alternative facilities → DCs",
    "dist3": "DCs → Retail hubs",
}

# Scenario 1 runs on the existing network alone — no alternative site can be opened there, so
# pricing them on that page would offer the class options the scenario does not have.
_SCOPE_MATRICES = {
    "puzzle": ["dist1", "dist2", "dist2_new", "dist3"],
    "sc1": ["dist1", "dist2", "dist3"],
    "sc2": ["dist1", "dist2", "dist2_new", "dist3"],
}

EM_DASH = "—"


# ---------------------------------------------------------------------------
# Reference panels
# ---------------------------------------------------------------------------

def _facility_rows(src: dict, scope: str) -> list[dict]:
    """Prices, one row per facility the page's scenario can actually use."""
    demand_total = sum(src["demand"].values())
    rows = []
    for p in src["plants_all"]:
        rows.append({
            "Facility": f"🏭 {p}",
            # The two manufacturers are already running: nothing to open, nothing to keep open.
            "Opening (€)": EM_DASH,
            "Operating (€/yr)": EM_DASH,
            "Prod. (€/unit)": f"{src['sourcing_cost'][p]:.3f}",
            # They have no site limit of their own, so the ceiling is total demand.
            "Capacity": f"{demand_total:,}",
        })
    if scope != "sc1":
        for n in src["new_locs_all"]:
            cap = src["new_loc_capacity"][n]
            rows.append({
                "Facility": f"🏗️ {n}",
                "Opening (€)": f"{src['new_loc_openingCost'][n]:,.0f}",
                "Operating (€/yr)": f"{src['new_loc_operationCost'][n]:,.0f}",
                # Written exactly as the engine writes it, down to the last bit.
                "Prod. (€/unit)": f"{(1.0 / cap) * NEW_LOC_COST_BUDGET:.3f}",
                "Capacity": f"{cap:,}",
            })
    return rows


def _matrix_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Whole kilometres with the row name as a first column, so `hide_index` still applies."""
    out = df.round(0).astype(int).reset_index()
    out = out.rename(columns={out.columns[0]: "km"})
    return out


def render_reference_panels(scope: str, cfg: dict | None = None) -> None:
    """Three collapsed panels — Prices, Emissions, Distances — for the parameters the model
    charges against.

    `scope` is "puzzle", "sc1" or "sc2": Scenario 1 cannot open alternative facilities, so it
    sees neither their rows nor their distance matrix. `cfg` is a live `_puzzle_defaults()` dict
    when the caller has one; otherwise the mirror above is used.
    """
    src = cfg or _FALLBACK

    st.markdown("### 📐 Cost & emission factors")

    with st.expander("💶 Prices", expanded=False):
        st.markdown("**Facilities**")
        st.dataframe(pd.DataFrame(_facility_rows(src, scope)),
                     hide_index=True, use_container_width=True)
        st.markdown("**Transport tariff**")
        st.dataframe(
            pd.DataFrame([{"Mode": MODE_LABELS[m], "€/kg-km": f"{src['tau'][m]:.4f}"}
                          for m in ["air", "Water", "road"]]),
            hide_index=True, use_container_width=True,
        )

    with st.expander("🌿 Emissions", expanded=False):
        st.markdown("**Production**")
        prod_rows = [{"Facility": f"🏭 {p}", "kg CO₂e/unit": f"{src['co2_prod_kg_per_unit'][p]:.1f}"}
                     for p in src["plants_all"]]
        if scope != "sc1":
            prod_rows += [{"Facility": f"🏗️ {n}", "kg CO₂e/unit": f"{src['new_loc_CO2'][n]:.1f}"}
                          for n in src["new_locs_all"]]
        st.dataframe(pd.DataFrame(prod_rows), hide_index=True, use_container_width=True)

        st.markdown("**Transport**")
        # Shown in grams: the engine's own unit is tons per ton-km (0.000027 and friends), which
        # is unreadable side by side. 971 / 27 / 76 makes "air is ~36× water" visible at a glance.
        st.dataframe(
            pd.DataFrame([{"Mode": MODE_LABELS[m],
                           "g CO₂/ton-km": f"{src['co2_emission_factor'][m] * 1_000_000:,.0f}"}
                          for m in ["air", "Water", "road"]]),
            hide_index=True, use_container_width=True,
        )
        st.caption("The model stores these as tons CO₂ per ton-km (air 0.000971, water 0.000027, "
                   "road 0.000076).")

        st.markdown("**Last-mile**")
        st.dataframe(
            pd.DataFrame([{"Leg": "🚐 DC → customer", "kg CO₂e/unit": f"{LAST_MILE_CO2_KG:.2f}"}]),
            hide_index=True, use_container_width=True,
        )

    with st.expander("📏 Distances (km)", expanded=False):
        for key in _SCOPE_MATRICES.get(scope, _SCOPE_MATRICES["puzzle"]):
            st.markdown(f"**{_MATRIX_TITLES[key]}**")
            st.dataframe(_matrix_frame(src[key]), hide_index=True, use_container_width=True)
        st.caption("Rounded to the nearest kilometre; the model uses the full values.")


# ---------------------------------------------------------------------------
# Facility cost, and the chart that has to add up to the headline total
# ---------------------------------------------------------------------------

def _safe(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if out != out else out  # NaN


def opened_new_locs_from_model(model) -> list[str]:
    """Which alternative facilities a solved model opened, read off its `f2_2_bin` flags."""
    opened = []
    for name in NEW_LOCS_ALL:
        try:
            var = model.getVarByName(f"f2_2_bin[{name}]")
            if var is not None and var.X > 0.5:
                opened.append(name)
        except Exception:
            # SC1F models have no such variables at all.
            return []
    return opened


def facility_cost_split(results, opened_new_locs=None) -> tuple[float, float, float]:
    """Alternative-facility cost as (opening, operating, production).

    Two key families reach this: puzzle mode reports the three parts separately, while the
    optimizer and the parquet sheets report `FixedCost_NewLocs`, which already holds opening plus
    operating combined. In the second case the opening half is rebuilt from the sites that were
    opened and operating is taken as *the remainder*, so the two segments always add back to the
    figure the source itself reports — even if a site's cost table ever changes.
    """
    if "Cost_NewLocs_opening" in results or "Cost_NewLocs_operating" in results:
        return (_safe(results.get("Cost_NewLocs_opening")),
                _safe(results.get("Cost_NewLocs_operating")),
                _safe(results.get("Cost_NewLocs_prod")))

    fixed = _safe(results.get("FixedCost_NewLocs"))
    prod = _safe(results.get("ProdCost_NewLocs"))
    if opened_new_locs:
        opening = sum(NEW_LOC_OPENING_COST.get(s, 0.0) for s in opened_new_locs)
        return (opening, fixed - opening, prod)
    # Nothing to split by: keep the whole thing in one segment rather than guess. The column
    # still carries the right total.
    return (0.0, fixed, prod)


FACILITY_CATEGORY = "Facility Cost"
# The four existing bar colours, pinned per category so trace order cannot reshuffle them.
BASE_COLORS = ["#A7C7E7", "#B0B0B0", "#F8C471", "#5D6D7E"]
OPENING_COLOR = "#5E35B1"
OPERATING_COLOR = "#B39DDB"


def build_cost_distribution_figure(cost_parts: dict, facility_parts: dict | None = None):
    """The Cost Distribution bar chart, with facility cost as one vertically split column.

    `cost_parts` is the page's own category → € mapping (its wording is kept as given).
    `facility_parts` is {"Opening": €, "Operating": €} or None. Segments worth nothing are left
    out, and with no facility cost at all the chart is exactly the four-bar one it has always
    been, legend included.
    """
    rows = [{"Category": cat, "Segment": cat, "Value_MEUR": _safe(val) / 1_000_000.0}
            for cat, val in cost_parts.items()]

    parts = {k: _safe(v) for k, v in (facility_parts or {}).items()}
    segments = [(name, parts.get(name, 0.0)) for name in ("Opening", "Operating")]
    segments = [(name, val) for name, val in segments if abs(val) > 1e-9]
    facility_total = sum(val for _, val in segments)
    for name, val in segments:
        rows.append({"Category": FACILITY_CATEGORY, "Segment": name,
                     "Value_MEUR": val / 1_000_000.0})

    df = pd.DataFrame(rows)
    df["Label"] = df["Value_MEUR"].map(lambda v: f"{v:.2f} M€")

    categories = list(cost_parts.keys())
    color_map = {cat: BASE_COLORS[i % len(BASE_COLORS)] for i, cat in enumerate(categories)}
    color_map["Opening"] = OPENING_COLOR
    color_map["Operating"] = OPERATING_COLOR
    if segments:
        categories = categories + [FACILITY_CATEGORY]

    fig = px.bar(
        df,
        x="Category",
        y="Value_MEUR",
        color="Segment",
        text="Label",
        color_discrete_map=color_map,
        # Opening before Operating: Plotly stacks in trace order, so opening sits at the bottom.
        category_orders={"Category": categories,
                         "Segment": list(cost_parts.keys()) + ["Opening", "Operating"]},
    )

    for trace in fig.data:
        if trace.name in ("Opening", "Operating"):
            # Inside the segment it belongs to — a number above the column would read as the
            # whole bar. The white edge keeps the two fills visibly apart.
            trace.textposition = "inside"
            trace.insidetextanchor = "middle"
            trace.marker.line.color = "white"
            trace.marker.line.width = 2
            trace.showlegend = True
        else:
            trace.textposition = "outside"
            trace.cliponaxis = False
            trace.showlegend = False

    tallest = max([_safe(v) / 1_000_000.0 for v in cost_parts.values()]
                  + [facility_total / 1_000_000.0, 0.0])

    fig.update_layout(
        barmode="stack",
        template="plotly_white",
        showlegend=bool(segments),
        legend=dict(title_text="", orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, traceorder="normal"),
        xaxis_title=None,
        xaxis_tickangle=-35,
        yaxis_title="Million €",
        height=400,
        yaxis_tickformat=".2f",
        # Hide a label that will not fit rather than shrink it into a smudge.
        uniformtext=dict(mode="hide", minsize=9),
    )
    if tallest > 0:
        # Headroom for the outside labels on the tallest bar.
        fig.update_yaxes(range=[0, tallest * 1.18])
    return fig
