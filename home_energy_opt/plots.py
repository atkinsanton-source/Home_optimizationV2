from __future__ import annotations

from pathlib import Path

import pandas as pd


_CP_SHADE_COLORS = {
    "home": "#cbeecb",
    "driving": "#d7e3fb",
    "workplace": "#ffe8b8",
    "public": "#e8d8ff",
    "fast75": "#ffd7c7",
    "fast150": "#ffc9c9",
}


def _charging_point_series(data: pd.DataFrame, index: pd.Index) -> pd.Series:
    if "charging_point_effective" not in data.columns:
        return pd.Series("none", index=index, dtype="string")
    return data["charging_point_effective"].astype("string").fillna("none").str.strip().str.lower()


def _collect_cp_intervals(index: pd.Index, cp: pd.Series) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp, str]], list[str]]:
    """Collect continuous charging-point intervals (excluding 'none')."""
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    states_present: list[str] = []
    seen: set[str] = set()
    if len(index) == 0:
        return intervals, states_present

    if len(index) > 1:
        step = index[1] - index[0]
    else:
        step = pd.Timedelta(minutes=15)

    start = 0
    current = cp.iloc[0]
    for i in range(1, len(cp) + 1):
        changed = i == len(cp) or cp.iloc[i] != current
        if not changed:
            continue

        state = str(current)
        if state != "none" and state in _CP_SHADE_COLORS:
            x0 = index[start]
            x1 = index[i] if i < len(index) else index[-1] + step
            intervals.append((x0, x1, state))
            if state not in seen:
                states_present.append(state)
                seen.add(state)

        if i < len(cp):
            start = i
            current = cp.iloc[i]

    return intervals, states_present


def _build_cp_overlay_shapes(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str]],
    row_specs: list[tuple[str, float, float]],
) -> list[dict]:
    """Build CP overlay rectangles in one bulk list for fast layout assignment."""
    shapes: list[dict] = []
    if not intervals:
        return shapes

    for xref, y0, y1 in row_specs:
        for x0, x1, state in intervals:
            color = _CP_SHADE_COLORS.get(state)
            if color is None:
                continue
            shapes.append(
                {
                    "type": "rect",
                    "xref": xref,
                    "yref": "paper",
                    "x0": x0,
                    "x1": x1,
                    "y0": float(y0),
                    "y1": float(y1),
                    "fillcolor": color,
                    "opacity": 0.28,
                    "layer": "below",
                    "line": {"width": 0},
                }
            )
    return shapes


def _add_cp_legend_traces(fig, states_present: list[str]) -> None:
    """Add legend-only entries for CP overlay colors."""
    import plotly.graph_objects as go

    for state in states_present:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=f"CP area: {state}",
                marker={"symbol": "square", "size": 10, "color": _CP_SHADE_COLORS[state]},
                legendgroup=f"cp_area_{state}",
                showlegend=True,
                hoverinfo="skip",
            ),
        )


def _add_system_panel_plotly(
    fig,
    row: int,
    index: pd.Index,
    data: pd.DataFrame,
    result: pd.DataFrame,
    show_battery: bool,
    show_pv: bool,
) -> None:
    import plotly.graph_objects as go

    if show_battery:
        fig.add_trace(
            go.Scattergl(
                x=index,
                y=result["bat_energy_kwh"],
                name="Battery SOC (kWh)",
                mode="lines",
                line={"color": "#1f77b4", "width": 1.5},
            ),
            row=row,
            col=1,
            secondary_y=False,
        )
    fig.add_trace(
        go.Scattergl(
            x=index,
            y=result["ev_energy_kwh"],
            name="EV SOC (kWh)",
            mode="lines",
            line={"color": "#2ca02c", "width": 1.5},
        ),
        row=row,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scattergl(
            x=index,
            y=data["ev_reserve_kwh"],
            name="EV required next 24h (clamped) (kWh)",
            mode="lines",
            line={"color": "#17becf", "width": 1.3, "dash": "dash"},
        ),
        row=row,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scattergl(
            x=index,
            y=data["EV_required_kWh_24h"],
            name="EV required next 24h (raw) (kWh)",
            mode="lines",
            line={"color": "#0b7285", "width": 1.2, "dash": "dot"},
        ),
        row=row,
        col=1,
        secondary_y=False,
    )
    if show_pv:
        fig.add_trace(
            go.Scattergl(
                x=index,
                y=data["pv_ac_kw"],
                name="PV generation (kW)",
                mode="lines",
                line={"color": "#8c564b", "width": 1.1},
                opacity=0.65,
            ),
            row=row,
            col=1,
            secondary_y=True,
        )
    fig.add_trace(
        go.Scattergl(
            x=index,
            y=result["grid_import_kw"],
            name="Grid import (kW)",
            mode="lines",
            line={"color": "#4f7a94", "width": 2.1},
            opacity=0.9,
        ),
        row=row,
        col=1,
        secondary_y=True,
    )
    fig.add_trace(
        go.Scattergl(
            x=index,
            y=data["load_kw"],
            name="House load (kW)",
            mode="lines",
            line={"color": "#ff7f0e", "width": 1.4},
            opacity=0.9,
        ),
        row=row,
        col=1,
        secondary_y=True,
    )


def _is_active_series(values: pd.Series, eps: float = 1e-9) -> bool:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return bool((numeric.abs() > eps).any())


def _get_series(df: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in df.columns:
        return None
    return pd.to_numeric(df[column], errors="coerce")


def _add_flow_balance_panel_plotly(
    fig,
    row: int,
    index: pd.Index,
    data: pd.DataFrame,
    result: pd.DataFrame,
    show_battery: bool,
    show_pv: bool,
    legend_seen: set[str],
) -> None:
    import plotly.graph_objects as go

    sink_specs = [
        ("House load (sink)", _get_series(data, "load_kw"), 1.0, "#ff7f0e"),
        ("EV home charging (sink)", _get_series(result, "ev_home_ch_kw"), 1.0, "#bcbd22"),
        ("EV external charging (sink)", _get_series(result, "ev_ext_ch_kw"), 1.0, "#9467bd"),
        ("Battery charging (sink)", _get_series(result, "bat_ch_kw"), 1.0, "#17becf"),
        ("Grid export (sink)", _get_series(result, "grid_export_kw"), 1.0, "#2ca02c"),
    ]
    source_specs = [
        ("Grid import (source)", _get_series(result, "grid_import_kw"), -1.0, "#4f7a94"),
        ("PV generation (source)", _get_series(data, "pv_ac_kw"), -1.0, "#8c564b"),
        ("Battery discharge (source)", _get_series(result, "bat_dis_kw"), -1.0, "#1f77b4"),
        ("EV discharge to house (source)", _get_series(result, "ev_dis_to_home_kw"), -1.0, "#d62728"),
        ("EV discharge to grid (source)", _get_series(result, "ev_dis_to_grid_kw"), -1.0, "#7f7f7f"),
        ("External charging supply (source)", _get_series(result, "ev_ext_ch_kw"), -1.0, "#6baed6"),
    ]

    for name, series, sign, color in sink_specs + source_specs:
        if series is None:
            continue
        if ("Battery" in name and not show_battery) or ("PV generation" in name and not show_pv):
            continue
        y = sign * series
        if not _is_active_series(y):
            continue
        showlegend = name not in legend_seen
        if showlegend:
            legend_seen.add(name)
        fig.add_trace(
            go.Scattergl(
                x=index,
                y=y,
                name=name,
                legendgroup=name,
                showlegend=showlegend,
                mode="lines",
                line={"color": color, "width": 1.8},
            ),
            row=row,
            col=1,
        )


def _build_system_connection_overview_plotly_figure(
    index: pd.Index,
    data: pd.DataFrame,
    baseline: pd.DataFrame,
    mpc: pd.DataFrame,
    show_battery: bool,
    show_pv: bool,
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Baseline", "MPC", "Home Grid Price"),
        specs=[[{"secondary_y": True}], [{"secondary_y": True}], [{}]],
    )
    _add_system_panel_plotly(fig, 1, index, data, baseline, show_battery=show_battery, show_pv=show_pv)
    _add_system_panel_plotly(fig, 2, index, data, mpc, show_battery=show_battery, show_pv=show_pv)
    price_series = _get_series(data, "import_price_eur_per_kwh")
    if price_series is None:
        price_series = _get_series(mpc, "home_grid_price_total_eur_per_kwh")
    if price_series is not None:
        fig.add_trace(
            go.Scattergl(
                x=index,
                y=price_series,
                name="Home grid price (EUR/kWh)",
                mode="lines",
                line={"color": "#e4572e", "width": 1.7},
            ),
            row=3,
            col=1,
        )

    cp = _charging_point_series(data, index)
    intervals, states_present = _collect_cp_intervals(index, cp)
    row_specs = [("x", fig.layout.yaxis.domain[0], fig.layout.yaxis.domain[1])]
    if hasattr(fig.layout, "yaxis3") and fig.layout.yaxis3 is not None:
        row_specs.append(("x2", fig.layout.yaxis3.domain[0], fig.layout.yaxis3.domain[1]))
    if hasattr(fig.layout, "yaxis5") and fig.layout.yaxis5 is not None:
        row_specs.append(("x3", fig.layout.yaxis5.domain[0], fig.layout.yaxis5.domain[1]))
    cp_shapes = _build_cp_overlay_shapes(intervals, row_specs)
    if cp_shapes:
        existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        fig.update_layout(shapes=existing_shapes + cp_shapes)
    _add_cp_legend_traces(fig, states_present)

    fig.update_yaxes(title_text="Energy (kWh)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Power (kW)", row=1, col=1, secondary_y=True, rangemode="tozero")
    fig.update_yaxes(title_text="Energy (kWh)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1, secondary_y=True, rangemode="tozero")
    fig.update_yaxes(title_text="Price (EUR/kWh)", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_layout(
        height=1100,
        width=1400,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 60, "t": 80, "b": 50},
    )
    return fig


def _build_flow_balance_figure(
    index: pd.Index,
    data: pd.DataFrame,
    baseline: pd.DataFrame,
    mpc: pd.DataFrame,
    show_battery: bool,
    show_pv: bool,
):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Baseline", "MPC", "Home Grid Price"),
    )

    legend_seen: set[str] = set()
    _add_flow_balance_panel_plotly(
        fig,
        1,
        index,
        data,
        baseline,
        show_battery=show_battery,
        show_pv=show_pv,
        legend_seen=legend_seen,
    )
    _add_flow_balance_panel_plotly(
        fig,
        2,
        index,
        data,
        mpc,
        show_battery=show_battery,
        show_pv=show_pv,
        legend_seen=legend_seen,
    )
    price_series = _get_series(data, "import_price_eur_per_kwh")
    if price_series is None:
        price_series = _get_series(mpc, "home_grid_price_total_eur_per_kwh")
    if price_series is not None:
        fig.add_trace(
            go.Scattergl(
                x=index,
                y=price_series,
                name="Home grid price (EUR/kWh)",
                legendgroup="Home grid price (EUR/kWh)",
                showlegend="Home grid price (EUR/kWh)" not in legend_seen,
                mode="lines",
                line={"color": "#e4572e", "width": 1.7},
            ),
            row=3,
            col=1,
        )
        legend_seen.add("Home grid price (EUR/kWh)")

    cp = _charging_point_series(data, index)
    intervals, states_present = _collect_cp_intervals(index, cp)
    row_specs = [("x", fig.layout.yaxis.domain[0], fig.layout.yaxis.domain[1])]
    if hasattr(fig.layout, "yaxis2") and fig.layout.yaxis2 is not None:
        row_specs.append(("x2", fig.layout.yaxis2.domain[0], fig.layout.yaxis2.domain[1]))
    if hasattr(fig.layout, "yaxis3") and fig.layout.yaxis3 is not None:
        row_specs.append(("x3", fig.layout.yaxis3.domain[0], fig.layout.yaxis3.domain[1]))
    cp_shapes = _build_cp_overlay_shapes(intervals, row_specs)
    if cp_shapes:
        existing_shapes = list(fig.layout.shapes) if fig.layout.shapes else []
        fig.update_layout(shapes=existing_shapes + cp_shapes)
    _add_cp_legend_traces(fig, states_present)

    fig.update_yaxes(title_text="Power (kW)", row=1, col=1)
    fig.update_yaxes(title_text="Power (kW)", row=2, col=1)
    fig.update_yaxes(title_text="Price (EUR/kWh)", row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.add_hline(y=0.0, line_width=1.0, line_dash="dot", line_color="#555555", row=1, col=1)
    fig.add_hline(y=0.0, line_width=1.0, line_dash="dot", line_color="#555555", row=2, col=1)
    fig.update_layout(
        height=1100,
        width=1400,
        template="plotly_white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
        margin={"l": 60, "r": 60, "t": 80, "b": 50},
    )
    return fig


def _ensure_local_plotly_js(out_path: str) -> str:
    """Ensure a local plotly.min.js exists next to the dashboard (offline usage)."""
    from plotly.offline.offline import get_plotlyjs

    out_file = Path(out_path).resolve()
    js_path = out_file.parent / "plotly.min.js"
    js_path.parent.mkdir(parents=True, exist_ok=True)
    if not js_path.exists():
        js_path.write_text(get_plotlyjs(), encoding="utf-8")
    return js_path.name


def save_system_connection_interactive_html(
    index: pd.Index,
    data: pd.DataFrame,
    baseline: pd.DataFrame,
    mpc: pd.DataFrame,
    out_path: str,
    show_battery: bool = True,
    show_pv: bool = True,
) -> None:
    """Save an interactive HTML plot with persistent zoom/pan and dynamic axis scaling."""
    overview_fig = _build_system_connection_overview_plotly_figure(
        index, data, baseline, mpc, show_battery=show_battery, show_pv=show_pv
    )
    flow_fig = _build_flow_balance_figure(
        index, data, baseline, mpc, show_battery=show_battery, show_pv=show_pv
    )

    _local_plotly_js = _ensure_local_plotly_js(out_path)
    plot_config = {"responsive": True, "scrollZoom": True, "displaylogo": False}
    overview_html = overview_fig.to_html(full_html=False, include_plotlyjs=False, config=plot_config)
    flow_html = flow_fig.to_html(full_html=False, include_plotlyjs=False, config=plot_config)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>System Connection Dashboard</title>
  <script src="{_local_plotly_js}"></script>
  <style>
    body {{
      margin: 0;
      padding: 16px 18px 24px 18px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #f7f8fa;
      color: #1f2328;
    }}
    .section {{
      background: #ffffff;
      border: 1px solid #d0d7de;
      border-radius: 8px;
      padding: 14px 14px 6px 14px;
      margin-bottom: 14px;
    }}
    h1 {{
      font-size: 21px;
      margin: 2px 0 14px 2px;
      font-weight: 650;
    }}
    h2 {{
      font-size: 17px;
      margin: 2px 0 8px 2px;
      font-weight: 620;
    }}
    p {{
      margin: 2px 0 10px 2px;
      color: #57606a;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <h1>System Connection Dashboard</h1>
  <div class="section">
    <h2>Overview</h2>
    <p>System connection overview including SOC, EV reserve targets, house load, grid import, PV (if active), and home grid price.</p>
    {overview_html}
  </div>
  <div class="section">
    <h2>Flow Balance</h2>
    <p>Power flow balance: sources (negative) and sinks (positive), with home grid price at the bottom.</p>
    {flow_html}
  </div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
