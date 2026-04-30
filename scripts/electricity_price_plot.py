from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from home_energy_opt.config import EnergySystemConfig



class ElectricityPriceData:
    def __init__(
        self,
        time,
        electricity_prices,
        home_buy_prices,
        day_ahead_price,
        ev_at_home_mask,
        discharging_mask,
        grid_import_kwh,
    ):
        self.time = time
        self.electricity_prices = electricity_prices
        self.home_buy_prices = home_buy_prices
        self.day_ahead_price = day_ahead_price
        self.ev_at_home_mask = ev_at_home_mask
        self.discharging_mask = discharging_mask
        self.grid_import_kwh = grid_import_kwh


    @classmethod
    def from_csv(cls, results_data_path, initial_data_path):
        data_results = pd.read_csv(results_data_path)
        data_initial = pd.read_csv(initial_data_path)


        time = pd.to_datetime(data_results["local_time"])
        electricity_prices = data_results.get("home_grid_price_total_eur_per_kwh", data_results["import_price_eur_per_kwh"])
        home_buy_prices = data_results.get("ev_home_import_price_eur_per_kwh", electricity_prices)
        day_ahead_price = data_initial["day_ahead_price"] / 1000.0
        ev_at_home_mask = data_results["ev_state"] == "home"
        discharging_mask = data_results["grid_export_kwh"] > 0
        grid_import_kwh = data_results["grid_import_kwh"]

        return cls(
            time,
            electricity_prices,
            home_buy_prices,
            day_ahead_price,
            ev_at_home_mask,
            discharging_mask,
            grid_import_kwh,
        )

    @classmethod
    def find_metrics_comparison_files(cls, base_path):
        base_path = Path(base_path)
        metrics_files = list(base_path.rglob("metrics_comparison.csv"))
        return metrics_files

    @classmethod
    def from_output_metrics(cls, base_path, wanted_columns):
        base_path = Path(base_path)
        metrics_files = list(base_path.rglob("metrics_comparison.csv"))

        frames = []

        for file_path in metrics_files:
            df = pd.read_csv(file_path)
            df = df.rename(columns={df.columns[0]: "model"})
            df = df[["model"] + wanted_columns].copy()
            df["scenario"] = file_path.parent.name
            frames.append(df)

        combined_df = pd.concat(frames, ignore_index=True)
        return combined_df


    @property
    def electricity_prices_at_home(self):
        return self.electricity_prices[self.ev_at_home_mask]

    @property
    def time_at_home(self):
        return self.time[self.ev_at_home_mask]

    @property
    def home_buy_prices_at_home(self):
        return self.home_buy_prices[self.ev_at_home_mask]




def plot_electricity_price(ax, time, electricity_price):
    avg_price=electricity_price.mean()
    rolling_avg_price_24h=electricity_price.rolling(window=96).mean()

    ax.plot(time, electricity_price, label="EPEX 15 Minute Electricity Price", color="blue")
    ax.axhline(avg_price, color="red", linestyle="--", label=f"Average = {avg_price:.2f}")
    ax.plot(time, rolling_avg_price_24h, label="Rolling average", color="green", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price Euro/KWh")
    ax.set_title("Dynamic Electricity Price Germany 2025 with a Octopus Energy Contract")
    ax.legend()
    ax.grid(True)

    return

def plot_priced_hours(ax, electricity_price, title):
    cfg=EnergySystemConfig()
    static_electricity_price_eur_per_kwh=cfg.baseline_static_import_price_eur_per_kwh
    sorted_electricity_prices=sorted(electricity_price)
    hours = [x * 0.25 for x in range(1, len(sorted_electricity_prices) + 1)]

    intersection = None
    for index, value in enumerate(sorted_electricity_prices):
        if value >= static_electricity_price_eur_per_kwh:
            intersection=hours[index]
            break

    ax.plot(hours, sorted_electricity_prices, label="Sorted Dynamic Electricity Price", color="blue")
    ax.axhline(static_electricity_price_eur_per_kwh, label=f"Static Electricity Price at {static_electricity_price_eur_per_kwh:.2f}Euro/KWh", color="green")
    if intersection is not None:
        ax.axvline(intersection, linestyle="--", color="black", label=f"Intersection at {intersection:.2f} h")
    ax.legend()
    ax.set_xlabel("Hours")
    ax.set_ylabel("Price Euro/KWh")
    ax.set_title(title)
    ax.grid(True)

    return

def plot_weighted_grid_import_price(ax, grid_import_kwh, electricity_prices):
    cfg = EnergySystemConfig()
    static_electricity_price_eur_per_kwh = cfg.baseline_static_import_price_eur_per_kwh

    data = pd.DataFrame({
        "grid_import_kwh": grid_import_kwh,
        "electricity_price": electricity_prices,
    })

    data = data[data["grid_import_kwh"] > 0]
    data = data.sort_values("electricity_price")
    data["cumulative_grid_import_kwh"] = data["grid_import_kwh"].cumsum()

    weighted_average_price = (
        (data["grid_import_kwh"] * data["electricity_price"]).sum()
        / data["grid_import_kwh"].sum()
    )

    intersection = None
    for row in data.itertuples():
        if row.electricity_price >= static_electricity_price_eur_per_kwh:
            intersection = row.cumulative_grid_import_kwh
            break

    ax.plot(data["cumulative_grid_import_kwh"], data["electricity_price"], color="blue", label="Sorted bought electricity")
    ax.axhline(weighted_average_price, color="red", linestyle="--", label=f"Weighted average = {weighted_average_price:.2f}")
    ax.axhline(static_electricity_price_eur_per_kwh, color="green", label=f"Static Electricity Price at {static_electricity_price_eur_per_kwh:.2f}Euro/KWh")
    if intersection is not None:
        ax.axvline(intersection, linestyle="--", color="black", label=f"Intersection at {intersection:.2f} KWh")
    ax.set_xlabel("Cumulative grid import [KWh]")
    ax.set_ylabel("Price Euro/KWh")
    ax.set_title("Weighted Average Price of Imported Grid Electricity in 2025")
    ax.legend()
    ax.grid(True)

    return

def plot_sorted_buy_prices_with_best_future_sell_price(ax, home_buy_price_total_eur_per_kwh):
    cfg = EnergySystemConfig()
    ev_battery_deg=cfg.ev_degradation_eur_per_kwh_charged
    extra_costs_electricity=cfg.import_price_adder_eur_per_kwh
    netzentgelt=cfg.export_price_adder_eur_per_kwh
    stromsteuer = 0.0244
    Mehrwehrtsteuer=cfg.import_price_adder_pct + 1.0
    roundtrip_efficiency = cfg.ev_eta_ch*cfg.ev_eta_dis
    threshold = extra_costs_electricity-netzentgelt-stromsteuer+ev_battery_deg

    home_buy_prices = list(home_buy_price_total_eur_per_kwh)
    day_ahead_prices = [
        (value - extra_costs_electricity) / Mehrwehrtsteuer
        for value in home_buy_prices
    ]
    best_future_sell_prices = []

    for current_index, current_day_ahead_price in enumerate(day_ahead_prices):
        best_future_sell_price = None
        end_index = min(current_index + 96, len(day_ahead_prices) - 1)
        future_prices = day_ahead_prices[current_index + 1:end_index + 1]

        if future_prices:
            highest_future_price = max(future_prices)

            if highest_future_price*roundtrip_efficiency - current_day_ahead_price * Mehrwehrtsteuer > threshold:
                best_future_sell_price = highest_future_price

        best_future_sell_prices.append(best_future_sell_price)

    data = pd.DataFrame({
        "buy_price": home_buy_prices,
        "best_future_sell_price": best_future_sell_prices,
    })

    data = data.sort_values("buy_price")
    data["hours"] = [x * 0.25 for x in range(1, len(data) + 1)]
    profitable_hours = data["best_future_sell_price"].notna().sum() * 0.25

    ax.plot(data["hours"], data["buy_price"], color="blue", label="Sorted buy price")
    profitable_data = data[data["best_future_sell_price"].notna()]
    ax.scatter(profitable_data["hours"], profitable_data["best_future_sell_price"], color="green", s=8, label="Best profitable sell price within 24h")
    ax.set_xlabel("Hours")
    ax.set_ylabel("Price Euro/KWh")
    ax.set_title(
        f"Sorted Buy Prices and Best Future Sell Prices\n"
        f"Profitable Buy Windows: {profitable_hours:.1f} h"
    )
    ax.legend()
    ax.grid(True)

    return

def plot_price_deltas_with_profit_in24h(ax, time, day_ahead_price, buy_price_total_eur_per_kwh, discharging_mask=None, ev_at_home_mask=None, at_home_only=False):
    profitable_sell_mask = []
    cfg = EnergySystemConfig()
    ev_battery_deg=cfg.ev_degradation_eur_per_kwh_charged
    extra_costs_electricity=cfg.import_price_adder_eur_per_kwh
    netzentgelt=cfg.export_price_adder_eur_per_kwh
    stromsteuer = 0.0244
    Mehrwehrtsteuer=cfg.import_price_adder_pct + 1.0
    roundtrip_efficiency = cfg.ev_eta_ch*cfg.ev_eta_dis
    
    threshold = extra_costs_electricity-netzentgelt-stromsteuer+ev_battery_deg
    
    current_buy_prices = pd.Series(buy_price_total_eur_per_kwh, dtype="float64").tolist()

    for current_index, current_buy_price in enumerate(current_buy_prices):
        profitable = False
        if at_home_only and not ev_at_home_mask.iloc[current_index]:
            profitable_sell_mask.append(False)
            continue

        current_day_ahead_price = (float(current_buy_price) - extra_costs_electricity) / Mehrwehrtsteuer
        start_index = max(0, current_index - 96)
        for past_index in range(current_index - 1, start_index - 1, -1):
            past_buy_price = float(current_buy_prices[past_index])
            past_day_ahead_price = (past_buy_price - extra_costs_electricity) / Mehrwehrtsteuer

            if at_home_only and not ev_at_home_mask.iloc[past_index]:
                continue
            elif current_day_ahead_price*roundtrip_efficiency - past_day_ahead_price * Mehrwehrtsteuer > threshold:
                profitable = True
                break

        profitable_sell_mask.append(profitable)

    ax.plot(time, current_buy_prices, label="EV Home Charge Price", color="blue")

    window_start = None
    for index, profitable in enumerate(profitable_sell_mask):
        if profitable and window_start is None:
            window_start = index
        elif not profitable and window_start is not None:
            ax.axvspan(time.iloc[window_start], time.iloc[index - 1], color="lightgreen", alpha=0.35)
            window_start = None

    if window_start is not None:
        ax.axvspan(time.iloc[window_start], time.iloc[len(time) - 1], color="lightgreen", alpha=0.35)
    
    if discharging_mask is not None:
        window_start = None
        for index, discharging in enumerate(discharging_mask):
            if discharging and window_start is None:
                window_start = index
            elif not discharging and window_start is not None:
                ax.axvspan(time.iloc[window_start], time.iloc[index - 1], color="#ff9999", alpha=0.35)
                window_start = None

        if window_start is not None:
            ax.axvspan(time.iloc[window_start], time.iloc[len(time) - 1], color="#ff9999", alpha=0.35)

    ax.set_xlabel("Time")
    ax.set_ylabel("Price Euro/KWh")
    if at_home_only:
        ax.set_title(
            f"Sell Windows, in which V2G would be profitable, when EV is at home,\n"
            f"Profitable Sell Windows: {sum(profitable_sell_mask)*0.25} h, "
            f"Actual discharge: {sum(discharging_mask)*0.25} h"
        )
    else:
        ax.set_title(
            f"Sell windows, in which V2G would be profitable \n"
            f"Profitable Sell Windows: {sum(profitable_sell_mask)*0.25} h,"
        )
    ax.legend()
    ax.grid(True)

    return

def plot_indifference_curve_v2g(ax,x):
    cfg = EnergySystemConfig()
    Netzentgelt=0.0997
    Mehrwehrtsteuer=cfg.import_price_adder_pct
    stromsteuer = 0.0244
    ev_battery_deg=cfg.ev_degradation_eur_per_kwh_charged
    extra_costs_electricity=cfg.import_price_adder_eur_per_kwh
    roundtrip_efficiency = cfg.ev_eta_ch*cfg.ev_eta_dis


    y = [(value * (1.0 + Mehrwehrtsteuer) + extra_costs_electricity + ev_battery_deg - Netzentgelt - stromsteuer)/roundtrip_efficiency for value in x]
    ax.plot(x, y, label="Indifference Curve for V2G")
    ax.set_xlabel("Buy Day Ahead Price [Eur/KWh]")
    ax.set_ylabel("Sale Day Ahead Price [Eur/KWh]")

    ax.set_title("Indifference Curve for V2G")
    ax.legend()
    ax.minorticks_on()
    ax.grid(True, which="major", linewidth=0.8)
    ax.grid(True, which="minor", linewidth=0.4, alpha=0.3)

    return 

def plot_carpet(ax, time, values, title, colorbar_label, fig, cmap="viridis", clip_values=True, vmin=None, vmax=None, show_colorbar=True):
    data = pd.DataFrame({
        "time": time,
        "value": pd.to_numeric(values, errors="coerce"),
    })

    data["date"] = data["time"].dt.date
    data["time_of_day"] = data["time"].dt.hour + data["time"].dt.minute / 60
    carpet_data = data.pivot_table(index="time_of_day", columns="date", values="value", aggfunc="mean")
    carpet_data = carpet_data.sort_index()

    if clip_values:
        values_for_clipping = carpet_data.stack()
        if vmin is None:
            vmin = values_for_clipping.quantile(0.01)
        if vmax is None:
            vmax = values_for_clipping.quantile(0.99)

    image = ax.imshow(carpet_data, aspect="auto", origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    if show_colorbar:
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label(colorbar_label)

    dates = pd.to_datetime(carpet_data.columns)
    month_ticks = []
    month_labels = []
    for index, date in enumerate(dates):
        if date.day == 1:
            month_ticks.append(index)
            month_labels.append(date.strftime("%b"))

    hour_ticks = []
    hour_labels = []
    for hour in range(0, 25, 4):
        closest_time_index = min(range(len(carpet_data.index)), key=lambda index: abs(carpet_data.index[index] - hour))
        hour_ticks.append(closest_time_index)
        hour_labels.append(f"{hour:02d}")

    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels, fontsize=7)
    ax.set_yticks(hour_ticks)
    ax.set_yticklabels(hour_labels, fontsize=7)
    ax.set_xlabel("Month", fontsize=8)
    ax.set_ylabel("Time of day", fontsize=8)
    ax.set_title(title)

    return image

def calculate_best_future_v2g_profit_margin(home_buy_price_total_eur_per_kwh, scenario_name):
    cfg = EnergySystemConfig()
    extra_costs_electricity = cfg.import_price_adder_eur_per_kwh
    mehrwertsteuer = cfg.import_price_adder_pct + 1.0
    netzentgelt = cfg.export_price_adder_eur_per_kwh
    stromsteuer = 0.0244
    if "0.5degcost" in scenario_name:
        ev_battery_deg = cfg.ev_degradation_eur_per_kwh_charged / 2
    elif "0deg" in scenario_name:
        ev_battery_deg = 0
    else:
        ev_battery_deg = cfg.ev_degradation_eur_per_kwh_charged
    roundtrip_efficiency = cfg.ev_eta_ch * cfg.ev_eta_dis
    threshold = extra_costs_electricity - netzentgelt - stromsteuer + ev_battery_deg

    day_ahead_prices = [
        (value - extra_costs_electricity) / mehrwertsteuer
        for value in home_buy_price_total_eur_per_kwh
    ]

    best_future_profit_margins = []

    for current_index, current_day_ahead_price in enumerate(day_ahead_prices):
        end_index = min(current_index + 96, len(day_ahead_prices) - 1)
        future_prices = day_ahead_prices[current_index + 1:end_index + 1]

        if future_prices:
            highest_future_price = max(future_prices)
            best_future_profit_margin = (
                highest_future_price * roundtrip_efficiency
                - current_day_ahead_price * mehrwertsteuer
                - threshold
            )
        else:
            best_future_profit_margin = -threshold

        best_future_profit_margins.append(best_future_profit_margin)

    return best_future_profit_margins

def plot_carpet_plots(output_folder_path):
    cfg = EnergySystemConfig()
    carpet_specs = [
        ("EV Home Charge Price", "Euro/KWh", "ev_home_import_price_eur_per_kwh", "viridis", True, None, None),
        ("EV State of Charge", "SOC [%]", "ev_soc_pct", "YlGn", False, 0, 100),
        ("Grid Import Usage", "KW", "grid_import_kw", "Blues", False, None, None),
        ("Grid Export Usage", "KW", "grid_export_kw", "Oranges", False, 0, 12),
        ("EV Discharge to Home", "KW", "ev_dis_to_home_kw", "Greens", False, 0, None),
        ("EV Charge and Discharge", "KW", "ev_charge_discharge_kw", "coolwarm", False, -12, 44),
    ]

    scenario_paths = sorted(
        [
            path
            for path in Path(output_folder_path).glob("outputs_*")
            if (path / "mpc_results.csv").exists()
        ]
    )

    if not scenario_paths:
        print(f"No scenario folders with mpc_results.csv found in {output_folder_path}")
        return

    def get_scenario_label(scenario_name):
        if "0.5degcost" in scenario_name:
            return "Noncommuter with 0.5 Battery Deg. Costs"
        if "0deg" in scenario_name:
            return "Noncommuter with 0 Battery Deg. Costs"
        if "Noncommuter" in scenario_name:
            return "Noncommuter Profile"
        return "Commuter Profile"

    carpet_spec_groups = [carpet_specs[:3], carpet_specs[3:]]

    for carpet_group in carpet_spec_groups:
        fig, axes = plt.subplots(len(carpet_group), len(scenario_paths), figsize=(16.8, 5.8), squeeze=False)
        fig.subplots_adjust(left=0.07, right=0.95, top=0.87, bottom=0.10, wspace=0.14, hspace=0.40)
        row_images = [None] * len(carpet_group)

        for column_index, scenario_path in enumerate(scenario_paths):
            result_path = scenario_path / "mpc_results.csv"
            data = pd.read_csv(result_path, low_memory=False)
            time = pd.to_datetime(data["local_time"])
            data["ev_soc_pct"] = data["ev_energy_kwh"] / cfg.ev_cap_kwh * 100
            data["ev_charge_discharge_kw"] = data["ev_home_ch_kw"] + data["ev_ext_ch_kw"] - data["ev_dis_to_home_kw"] - data["ev_dis_to_grid_kw"]

            for row_index, (title, colorbar_label, column_name, cmap, clip_values, vmin, vmax) in enumerate(carpet_group):
                image = plot_carpet(
                    axes[row_index, column_index],
                    time,
                    data[column_name],
                    "",
                    colorbar_label,
                    fig,
                    cmap=cmap,
                    clip_values=clip_values,
                    vmin=vmin,
                    vmax=vmax,
                    show_colorbar=False,
                )
                row_images[row_index] = image

                if column_index == 0:
                    axes[row_index, column_index].set_ylabel("Time of day")
                else:
                    axes[row_index, column_index].set_ylabel("")

                if row_index == len(carpet_group) - 1:
                    axes[row_index, column_index].set_xlabel("Month")
                else:
                    axes[row_index, column_index].set_xlabel("")

                if row_index == 0:
                    axes[row_index, column_index].text(0.5, 1.10, scenario_path.name, ha="center", va="bottom", fontsize=6, transform=axes[row_index, column_index].transAxes)
                    axes[row_index, column_index].text(0.5, 1.03, get_scenario_label(scenario_path.name), ha="center", va="bottom", fontsize=8, fontweight="bold", transform=axes[row_index, column_index].transAxes)

        for row_index, (title, colorbar_label, _, _, _, _, _) in enumerate(carpet_group):
            colorbar = fig.colorbar(row_images[row_index], ax=axes[row_index, :], fraction=0.015, pad=0.01)
            colorbar.set_label(f"{title}\n{colorbar_label}")

    profitability_specs = [
        ("V2G Profitability within 24h", "Euro/KWh above break-even", "best_future_v2g_profit_margin", "coolwarm", False, None, None),
    ]

    fig, axes = plt.subplots(1, len(scenario_paths), figsize=(16.8, 2.4), squeeze=False)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.82, bottom=0.18, wspace=0.14, hspace=0.30)
    row_images = [None]

    for column_index, scenario_path in enumerate(scenario_paths):
        result_path = scenario_path / "mpc_results.csv"
        data = pd.read_csv(result_path, low_memory=False)
        time = pd.to_datetime(data["local_time"])
        data["best_future_v2g_profit_margin"] = calculate_best_future_v2g_profit_margin(
            data["ev_home_import_price_eur_per_kwh"].tolist(),
            scenario_path.name,
        )

        for row_index, (title, colorbar_label, column_name, cmap, clip_values, vmin, vmax) in enumerate(profitability_specs):
            values = data[column_name]
            max_abs_value = max(abs(values.min()), abs(values.max()))
            profitability_cmap = plt.cm.get_cmap("viridis").copy()
            profitability_cmap.set_under("darkgray")
            image = plot_carpet(
                axes[row_index, column_index],
                time,
                values,
                "",
                colorbar_label,
                fig,
                cmap=profitability_cmap,
                clip_values=clip_values,
                vmin=0,
                vmax=max_abs_value,
                show_colorbar=False,
            )
            row_images[row_index] = image

            if column_index == 0:
                axes[row_index, column_index].set_ylabel("Time of day")
            else:
                axes[row_index, column_index].set_ylabel("")

            axes[row_index, column_index].set_xlabel("Month")

            axes[row_index, column_index].text(0.5, 1.10, scenario_path.name, ha="center", va="bottom", fontsize=6, transform=axes[row_index, column_index].transAxes)
            axes[row_index, column_index].text(0.5, 1.03, get_scenario_label(scenario_path.name), ha="center", va="bottom", fontsize=8, fontweight="bold", transform=axes[row_index, column_index].transAxes)

    for row_index, (title, colorbar_label, _, _, _, _, _) in enumerate(profitability_specs):
        colorbar = fig.colorbar(row_images[row_index], ax=axes[row_index, :], fraction=0.015, pad=0.01)
        colorbar.set_label(f"{title}\n{colorbar_label}")

    plt.show()

    return

def plot_costs_and_revenues(cost_and_revenue_data):
    models = ["baseline_static", "baseline_dynamic", "mpc_static", "mpc_dynamic_v1", "mpc"]
    model_display_labels = {
        "baseline_static": "Static Baseline",
        "baseline_dynamic": "Dyn. Baseline",
        "mpc_static": "Static MPC",
        "mpc_dynamic_v1": "Dyn. MPC v1",
        "mpc": "Dyn. MPC",
    }

    cost_and_revenue_data = cost_and_revenue_data.copy()
    cost_and_revenue_data["model"] = pd.Categorical(cost_and_revenue_data["model"], categories=models, ordered=True)
    cost_and_revenue_data = cost_and_revenue_data.sort_values(["scenario", "model"]).reset_index(drop=True)
    scenarios = cost_and_revenue_data["scenario"].unique()
    scenario_labels = ["Commuter Profile", "Noncommuter Profile", "Noncommuter with 0.5 Battery Deg. Costs", "Noncommuter with 0 Battery Deg. Costs"]

    x_labels = [
        model_display_labels.get(row.model, row.model)
        for row in cost_and_revenue_data.itertuples()
    ]

    public_charge_costs = cost_and_revenue_data["external_charge_public_cost_eur"].tolist()
    workplace_charge_costs = cost_and_revenue_data["external_charge_workplace_cost_eur"].tolist()
    fast75_charge_costs = cost_and_revenue_data["external_charge_fast75_cost_eur"].tolist()
    fast150_charge_costs = cost_and_revenue_data["external_charge_fast150_cost_eur"].tolist()
    fast_charge_costs = [a + b for a, b in zip(fast75_charge_costs, fast150_charge_costs)]
    ev_battery_degradation_costs = cost_and_revenue_data["ev_battery_degradation_cost_eur"].tolist()
    ev_home_charge_costs = cost_and_revenue_data["ev_home_charge_cost_eur"].tolist()
    home_load_costs = [a + b for a, b in zip(cost_and_revenue_data["home_load_cost_eur"].tolist(), ev_home_charge_costs)]
    ev_discharge_grid_revenues = cost_and_revenue_data["ev_discharge_grid_revenue_eur"].tolist()
    external_charge_costs = [a + b + c for a, b, c in zip(public_charge_costs, workplace_charge_costs, fast_charge_costs)]

    model_gap = 0.12
    x = [value * (1 + model_gap) for value in range(len(x_labels))]
    width = 0.4
    bar_gap = 0.04
    bar_style = {"edgecolor": "black", "linewidth": 0.8}

    fig, ax = plt.subplots(figsize=(14, 7))

    bars_home_load = ax.bar(x, home_load_costs, width=width, label="Home load cost", color="#7f7f7f", **bar_style)

    bars_public_charge = ax.bar(x, public_charge_costs, width=width, bottom=home_load_costs, label="Public charge cost", color="#f28e2b", **bar_style)

    bottom_workplace_charge = [a + b for a, b in zip(home_load_costs, public_charge_costs)]
    bars_workplace_charge = ax.bar(x, workplace_charge_costs, width=width, bottom=bottom_workplace_charge, label="Workplace charge cost", color="#ffbe7d", **bar_style)

    bottom_fast_charge = [a + b for a, b in zip(bottom_workplace_charge, workplace_charge_costs)]
    bars_fast_charge = ax.bar(x, fast_charge_costs, width=width, bottom=bottom_fast_charge, label="EV Fast Charge Cost", color="#d65f00", **bar_style)

    bottom_ev_battery_deg = [a + b for a, b in zip(home_load_costs, external_charge_costs)]
    bars_ev_battery_deg = ax.bar(x, ev_battery_degradation_costs, width=width, bottom=bottom_ev_battery_deg, label="EV battery degradation cost", color="#b8860b", **bar_style)

    revenue_x = [value + width + bar_gap for value in x]
    bars_revenue = ax.bar(revenue_x, ev_discharge_grid_revenues, width=width, label="EV discharge grid revenue", color="#2ca02c", **bar_style)

    total_costs = [a + b + c for a, b, c in zip(home_load_costs, external_charge_costs, ev_battery_degradation_costs)]
    net_values = [cost - revenue for cost, revenue in zip(total_costs, ev_discharge_grid_revenues)]

    net_label_added = False
    for x_value, net_value, revenue in zip(x, net_values, ev_discharge_grid_revenues):
        if revenue <= 0:
            continue
        if not net_label_added:
            ax.hlines(net_value, x_value - width / 2, x_value + width / 2, colors="#d62728", linewidth=2, label ="Net")
            net_label_added = True
        else:
            ax.hlines(net_value, x_value - width / 2, x_value + width / 2, colors="#d62728", linewidth=2)
        if net_value > 100:
            ax.text(x_value - width / 2 - 0.08, net_value, f"{net_value:.0f}", color="#d62728", fontsize=6.5, fontweight="bold", ha="right", va="center", bbox={"facecolor": "white", "edgecolor": "#d62728", "boxstyle": "square,pad=0.10"})

    for x_value, total_cost in zip(x, total_costs):
        if total_cost > 100:
            ax.text(x_value, total_cost + 7, f"{total_cost:.0f}", color="black", fontsize=9, fontweight="bold", ha="center", va="bottom")

    for x_value, revenue in zip(revenue_x, ev_discharge_grid_revenues):
        if revenue > 100:
            ax.text(x_value, revenue + 5, f"{revenue:.0f}", color="black", fontsize=9, fontweight="bold", ha="center", va="bottom")

    ax.bar_label(bars_home_load, labels=[f"{value:.0f}" if value > 100 else "" for value in home_load_costs], label_type="center", fontsize=7)
    ax.bar_label(bars_public_charge, labels=[f"{value:.0f}" if value > 100 else "" for value in public_charge_costs], label_type="center", fontsize=7)
    ax.bar_label(bars_workplace_charge, labels=[f"{value:.0f}" if value > 100 else "" for value in workplace_charge_costs], label_type="center", fontsize=7)
    ax.bar_label(bars_fast_charge, labels=[f"{value:.0f}" if value > 100 else "" for value in fast_charge_costs], label_type="center", fontsize=7)
    ax.bar_label(bars_ev_battery_deg, labels=[f"{value:.0f}" if value > 100 else "" for value in ev_battery_degradation_costs], label_type="center", fontsize=7)
    ax.bar_label(bars_revenue, labels=[f"{value:.0f}" if value > 100 else "" for value in ev_discharge_grid_revenues], label_type="center", fontsize=7)

    ax.set_xticks([value + (width + bar_gap) / 2 for value in x])
    ax.set_xticklabels(x_labels, fontsize=8, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("Operational costs / revenues (€)", fontweight="bold")
    ax.set_title("Costs and revenues (2025)")
    for scenario_index, scenario in enumerate(scenarios):
        scenario_rows = cost_and_revenue_data.index[cost_and_revenue_data["scenario"] == scenario].tolist()
        scenario_start = scenario_rows[0]
        scenario_end = scenario_rows[-1]
        scenario_center = (x[scenario_start] + x[scenario_end]) / 2 + (width + bar_gap) / 2
        if scenario_index > 0:
            previous_pair_right = revenue_x[scenario_start - 1] + width / 2
            next_pair_left = x[scenario_start] - width / 2
            separator_x = (previous_pair_right + next_pair_left) / 2
            ax.axvline(separator_x, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(scenario_center, -0.11, scenario_labels[scenario_index], ha="center", va="top", fontsize=8, fontweight="bold", transform=ax.get_xaxis_transform())
        ax.text(scenario_center, -0.20, scenario, ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())
    
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="lightgray", linestyle="--", linewidth=1, alpha=0.8)

    plt.tight_layout()
    plt.show()

    return


def plot_energy_sinks_sources(cost_and_revenue_data):
    models = ["baseline_static", "baseline_dynamic", "mpc_static", "mpc_dynamic_v1", "mpc"]
    model_display_labels = {
        "baseline_static": "Static Baseline",
        "baseline_dynamic": "Dyn. Baseline",
        "mpc_static": "Static MPC",
        "mpc_dynamic_v1": "Dyn. MPC V1",
        "mpc": "Dyn. MPC",
    }

    cost_and_revenue_data = cost_and_revenue_data.copy()
    cost_and_revenue_data["model"] = pd.Categorical(cost_and_revenue_data["model"], categories=models, ordered=True)
    cost_and_revenue_data = cost_and_revenue_data.sort_values(["scenario", "model"]).reset_index(drop=True)
    scenarios = cost_and_revenue_data["scenario"].unique()
    scenario_labels = ["Commuter Profile", "Noncommuter Profile", "Noncommuter with 0.5 Battery Deg. Costs", "Noncommuter with 0 Battery Deg. Costs"]
    
    x_labels = [
        model_display_labels.get(row.model, row.model)
        for row in cost_and_revenue_data.itertuples()
    ]

    ev_public_charge_kwh = cost_and_revenue_data["external_charge_public_kwh"].tolist()
    ev_workplace_charge_kwh = cost_and_revenue_data["external_charge_workplace_kwh"].tolist()
    ev_fast75_charge_kwh = cost_and_revenue_data["external_charge_fast75_kwh"].tolist()
    ev_fast150_charge_kwh = cost_and_revenue_data["external_charge_fast150_kwh"].tolist()
    ev_fast_charge_kwh = [a + b for a, b in zip(ev_fast75_charge_kwh, ev_fast150_charge_kwh)]
    grid_import_kwh = cost_and_revenue_data["grid_import_kwh"].tolist()
    ev_consumption_kwh = cost_and_revenue_data["ev_consumption_kwh"].tolist()
    ev_discharge_home_kwh = cost_and_revenue_data["ev_discharge_to_home_kwh"].tolist()
    ev_discharge_grid_kwh = cost_and_revenue_data["ev_discharge_to_grid_kwh"].tolist()
    home_load_grid_import_kwh = cost_and_revenue_data["home_load_grid_import_kwh"].tolist()
    ev_charge_home_kwh = cost_and_revenue_data["home_ev_charge_kwh"].tolist()
    ev_initial_energy_kwh = cost_and_revenue_data["ev_initial_energy_kwh"].tolist()
    ev_final_energy_kwh = cost_and_revenue_data["ev_final_energy_kwh"].tolist()
    ev_external_charge_kwh = [a + b + c for a, b, c in zip(ev_public_charge_kwh, ev_workplace_charge_kwh, ev_fast_charge_kwh)]

    total_ev_charged = [a + b for a, b in zip(ev_external_charge_kwh, ev_charge_home_kwh)]
    ev_charging_losses = [value * 0.08 for value in total_ev_charged]

    total_ev_discharged = [a + b for a, b in zip(ev_discharge_home_kwh, ev_discharge_grid_kwh)]
    ev_discharging_losses = [(value / 0.92) - value for value in total_ev_discharged]



    model_gap = 0.12
    x = [value * (1 + model_gap) for value in range(len(x_labels))]
    width = 0.4
    bar_gap = 0.04
    bar_style = {"edgecolor": "black", "linewidth": 0.8}

    fig, ax = plt.subplots(figsize=(14, 7))

    #Energy in bar
    bars_grid_import_kwh = ax.bar(x, grid_import_kwh, width=width, label="Grid Import", color="grey", **bar_style)

    bottom_ev_external_charge = grid_import_kwh
    bars_ev_public_charge = ax.bar(x, ev_public_charge_kwh, bottom=bottom_ev_external_charge, width=width, label="EV Public Charge", color="#f28e2b", **bar_style)

    bottom_ev_workplace_charge = [a+b for a,b in zip(bottom_ev_external_charge, ev_public_charge_kwh)]
    bars_ev_workplace_charge = ax.bar(x, ev_workplace_charge_kwh, bottom=bottom_ev_workplace_charge, width=width, label="EV Workplace Charge", color="#ffbe7d", **bar_style)

    bottom_ev_fast_charge = [a+b for a,b in zip(bottom_ev_workplace_charge, ev_workplace_charge_kwh)]
    bars_ev_fast_charge = ax.bar(x, ev_fast_charge_kwh, bottom=bottom_ev_fast_charge, width=width, label="EV Fast Charge", color="#d65f00", **bar_style)

    bottom_ev_initial_energy = [a+b for a,b in zip(grid_import_kwh, ev_external_charge_kwh)]
    bars_ev_initial_energy = ax.bar(x, ev_initial_energy_kwh, bottom=bottom_ev_initial_energy, width=width, label="EV Initial Energy", color="lightgreen", **bar_style)

    #Energy Out bar
    energy_out_x = [value + width + bar_gap for value in x]
    bars_home_load_kwh = ax.bar(energy_out_x, home_load_grid_import_kwh, width=width, label="Home Load From Grid", color="orange", **bar_style)
    bars_ev_consumption_kwh = ax.bar(energy_out_x, ev_consumption_kwh, bottom=home_load_grid_import_kwh, width=width, label="EV Consumption", color="blue", **bar_style)
    
    bottom_ev_discharge_home = [a+b for a,b in zip(home_load_grid_import_kwh, ev_consumption_kwh)]
    bars_ev_discharge_home_kwh = ax.bar(energy_out_x, ev_discharge_home_kwh, bottom=bottom_ev_discharge_home, width=width, label="EV Discharge To Home", **bar_style)

    bottom_ev_discharge_grid = [a+b+c for a,b,c in zip(home_load_grid_import_kwh, ev_consumption_kwh, ev_discharge_home_kwh)]
    bars_ev_discharge_grid_kwh = ax.bar(energy_out_x, ev_discharge_grid_kwh, bottom=bottom_ev_discharge_grid, width=width, label="EV Discharge To Grid", **bar_style)

    bottom_charging_losses = [a+b+c+d for a,b,c,d in zip(home_load_grid_import_kwh, ev_consumption_kwh, ev_discharge_home_kwh, ev_discharge_grid_kwh)]
    bars_ev_charging_losses = ax.bar(energy_out_x, ev_charging_losses, bottom=bottom_charging_losses, width=width, label="EV Charging Losses", **bar_style)

    bottom_discharging_losses = [a+b for a,b in zip(bottom_charging_losses, ev_charging_losses)]
    bars_ev_discharging_losses = ax.bar(energy_out_x, ev_discharging_losses, bottom=bottom_discharging_losses, width=width, label="EV Discharging Losses", **bar_style)

    bottom_ev_final_energy = [a+b for a,b in zip(bottom_discharging_losses, ev_discharging_losses)]
    bars_ev_final_energy = ax.bar(energy_out_x, ev_final_energy_kwh, bottom=bottom_ev_final_energy, width=width, label="EV Final Energy", color="darkgreen", **bar_style)
    
    #Total Energy Consumed
    total_energy_out = [a+b+c+d+e+f+g for a,b,c,d,e,f,g in zip(home_load_grid_import_kwh, ev_consumption_kwh, ev_discharge_home_kwh, ev_discharge_grid_kwh, ev_charging_losses, ev_discharging_losses, ev_final_energy_kwh)]
    
    #Description for total energy out above the bar pair
    for x_value, total_energy_out in zip(energy_out_x, total_energy_out):
        label_x = x_value - ((width + bar_gap) / 2)
        ax.text(label_x, total_energy_out + 7, f"{total_energy_out:.0f}", color="black", fontsize=11, fontweight="bold", ha="center", va="bottom")
    
    #Plot bars
    ax.bar_label(bars_ev_initial_energy, labels=[f"{value:.0f}" if value >= 200 else "" for value in ev_initial_energy_kwh], label_type="center", fontsize=7)
    ax.bar_label(bars_grid_import_kwh, labels=[f"{value:.0f}" if value >= 200 else "" for value in grid_import_kwh], label_type="center", fontsize=7)
    ax.bar_label(bars_ev_public_charge, labels=[f"{value:.0f}" if value >= 200 else "" for value in ev_public_charge_kwh], label_type="center", fontsize=7)
    ax.bar_label(bars_ev_workplace_charge, labels=[f"{value:.0f}" if value >= 200 else "" for value in ev_workplace_charge_kwh], label_type="center", fontsize=7)
    ax.bar_label(bars_ev_fast_charge, labels=[f"{value:.0f}" if value >= 200 else "" for value in ev_fast_charge_kwh], label_type="center", fontsize=7)
    ax.bar_label(bars_home_load_kwh, labels=[f"{value:.0f}" if value >= 200 else "" for value in home_load_grid_import_kwh], label_type="center", fontsize=7)
    ax.bar_label(bars_ev_consumption_kwh, labels=[f"{value:.0f}" if value >= 200 else "" for value in ev_consumption_kwh], label_type="center", fontsize=7)
    ax.bar_label(bars_ev_discharge_home_kwh, labels=[f"{value:.0f}" if value >= 200 else "" for value in ev_discharge_home_kwh], label_type="center", fontsize=7)
    ax.bar_label(bars_ev_discharge_grid_kwh, labels=[f"{value:.0f}" if value >= 200 else "" for value in ev_discharge_grid_kwh], label_type="center", fontsize=7)
    ax.bar_label(bars_ev_charging_losses, labels=[f"{value:.0f}" if value >= 200 else "" for value in ev_charging_losses], label_type="center", fontsize=7)
    ax.bar_label(bars_ev_discharging_losses, labels=[f"{value:.0f}" if value >= 200 else "" for value in ev_discharging_losses], label_type="center", fontsize=7)
    ax.bar_label(bars_ev_final_energy, labels=[f"{value:.0f}" if value >= 200 else "" for value in ev_final_energy_kwh], label_type="center", fontsize=7)

    #Naming plot and adding labels
    ax.set_xticks([value + (width + bar_gap) / 2 for value in x])
    ax.set_xticklabels(x_labels, fontsize=8, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("KWh", fontweight="bold")
    ax.set_title("Energy Sources and Sinks for Driver Profiles with a Tesla Model 3")

    for scenario_index, scenario in enumerate(scenarios):
        scenario_rows = cost_and_revenue_data.index[cost_and_revenue_data["scenario"] == scenario].tolist()
        scenario_start = scenario_rows[0]
        scenario_end = scenario_rows[-1]
        scenario_center = (x[scenario_start] + x[scenario_end]) / 2 + (width + bar_gap) / 2
        if scenario_index > 0:
            previous_pair_right = energy_out_x[scenario_start - 1] + width / 2
            next_pair_left = x[scenario_start] - width / 2
            separator_x = (previous_pair_right + next_pair_left) / 2
            ax.axvline(separator_x, color="gray", linestyle="--", linewidth=1, alpha=0.5)
        ax.text(scenario_center, -0.11, scenario_labels[scenario_index], ha="center", va="top", fontsize=8, fontweight="bold", transform=ax.get_xaxis_transform())
        ax.text(scenario_center, -0.20, scenario, ha="center", va="top", fontsize=7, transform=ax.get_xaxis_transform())

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()

    return

def main():
    print("Which plots do you want to see?")
    print("1: Electricity price plots")
    print("2: Energy Sources and Sinks")
    print("3: Costs and Revenue Analysis")
    print("4: Carpet plots")
    choice = input("Enter 1, 2, 3 or 4: ")

    #For the Electricity Price Data
    results_data_path=("/Users/anton.atkins/Documents/TU Berlin/Bachelor Arbeit/code/New_ModelV2/Home_optimizationV2/outputs_incl_mpcdynamic/outputs_3_Tesla3_V3_79.5KWh_Noncommuter_incl_mpcdynamic_0.5degcost/mpc_results.csv")
    initial_data_path=("/Users/anton.atkins/Documents/TU Berlin/Bachelor Arbeit/code/New_ModelV2/Home_optimizationV2/data/LPG_FlexEhome_2025_Tesla3_79.5_Commuter.csv")
    output_folder_path = "/Users/anton.atkins/Documents/TU Berlin/Bachelor Arbeit/code/New_ModelV2/Home_optimizationV2/outputs_incl_mpcdynamic"

    if choice == "1":
        data = ElectricityPriceData.from_csv(results_data_path, initial_data_path)

        #Plotting Electricity Prices and Profitability Windows
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        x = [0.0, 0.1, 0.2]
        plot_electricity_price(axes[0, 0], data.time, data.electricity_prices)
        plot_priced_hours(axes[0, 1], data.electricity_prices, "Sorted Dynamic Electricity Price")
        plot_price_deltas_with_profit_in24h(axes[1, 0], data.time, data.day_ahead_price, data.home_buy_prices)
        plot_price_deltas_with_profit_in24h(axes[1, 1], data.time, data.day_ahead_price, data.home_buy_prices, data.discharging_mask, data.ev_at_home_mask, True)
        plot_priced_hours(axes[0, 2], data.home_buy_prices_at_home, "Sorted Dynamic EV Home Charge Price When EV Is At Home")
        plot_weighted_grid_import_price(axes[0, 3], data.grid_import_kwh, data.electricity_prices)
        plot_indifference_curve_v2g(axes[1, 2], x)
        plot_sorted_buy_prices_with_best_future_sell_price(axes[1, 3], data.home_buy_prices)
        plt.tight_layout()
        plt.show()

    elif choice == "2":
        wanted_columns = ["external_charge_public_kwh", "external_charge_workplace_kwh", "external_charge_fast75_kwh", "external_charge_fast150_kwh", "grid_import_kwh", "ev_consumption_kwh", "ev_discharge_to_home_kwh", "ev_discharge_to_grid_kwh", "home_load_grid_import_kwh", "home_ev_charge_kwh", "ev_initial_energy_kwh", "ev_final_energy_kwh"]
        costs_revenue_metrics_combined_data= ElectricityPriceData.from_output_metrics(output_folder_path, wanted_columns)
        plot_energy_sinks_sources(costs_revenue_metrics_combined_data)

        
    elif choice == "3":
        wanted_columns_stacked = ["external_charge_public_cost_eur", "external_charge_workplace_cost_eur", "external_charge_fast75_cost_eur", "external_charge_fast150_cost_eur", "ev_battery_degradation_cost_eur", "ev_home_charge_cost_eur", "home_load_cost_eur", "ev_discharge_grid_revenue_eur"]
        costs_revenue_metrics_combined_data= ElectricityPriceData.from_output_metrics(output_folder_path, wanted_columns_stacked)
        plot_costs_and_revenues(costs_revenue_metrics_combined_data)

    elif choice == "4":
        plot_carpet_plots(output_folder_path)

    else:
        print("Invalid choice. Please enter 1, 2, 3 or 4.")


    return

if __name__ =="__main__":
    main()


    



        
