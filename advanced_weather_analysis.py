"""Advanced weather analysis - anomalies, hourly patterns, extremes, trends, and records."""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression


def load_data(csv_path="data/chicago_temps.csv"):
    """Load and prepare temperature data."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["year"] = df["timestamp"].dt.year
    df["week_number"] = df["timestamp"].dt.isocalendar().week
    return df


def detect_anomalies(df, std_threshold=2.5, output_dir="reports"):
    """Detect temperature anomalies using z-score method."""
    print("\n🔍 Detecting temperature anomalies...")

    mean_temp = df["temperature_celsius"].mean()
    std_temp = df["temperature_celsius"].std()

    df["z_score"] = (df["temperature_celsius"] - mean_temp) / std_temp
    df["is_anomaly"] = abs(df["z_score"]) > std_threshold

    anomalies = df[df["is_anomaly"]]
    print(f"   Found {len(anomalies)} anomalies (>{std_threshold} std devs)")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    # Normal data
    normal = df[~df["is_anomaly"]]
    ax.scatter(
        normal["timestamp"],
        normal["temperature_celsius"],
        c="#3498DB",
        alpha=0.4,
        s=10,
        label="Normal",
    )

    # Anomalies
    hot_anomalies = anomalies[anomalies["z_score"] > 0]
    cold_anomalies = anomalies[anomalies["z_score"] < 0]

    ax.scatter(
        hot_anomalies["timestamp"],
        hot_anomalies["temperature_celsius"],
        c="#E74C3C",
        s=50,
        marker="^",
        label=f"Hot Anomalies ({len(hot_anomalies)})",
        edgecolors="black",
        linewidths=1,
    )
    ax.scatter(
        cold_anomalies["timestamp"],
        cold_anomalies["temperature_celsius"],
        c="#3498DB",
        s=50,
        marker="v",
        label=f"Cold Anomalies ({len(cold_anomalies)})",
        edgecolors="black",
        linewidths=1,
    )

    ax.axhline(
        mean_temp,
        color="gray",
        linestyle="--",
        alpha=0.5,
        label=f"Mean: {mean_temp:.1f}°C",
    )
    ax.axhline(
        mean_temp + std_threshold * std_temp, color="red", linestyle=":", alpha=0.3
    )
    ax.axhline(
        mean_temp - std_threshold * std_temp, color="blue", linestyle=":", alpha=0.3
    )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(
        f"Temperature Anomaly Detection (±{std_threshold}σ threshold)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/anomaly_detection.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ Created: {output_dir}/anomaly_detection.png")
    plt.close()

    return anomalies


def analyze_hourly_patterns(df, output_dir="reports"):
    """Analyze average temperature by hour of day across seasons."""
    print("\n🕐 Analyzing hourly patterns...")

    # Assign seasons
    def assign_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df["season"] = df["month"].apply(assign_season)

    # Calculate hourly averages by season
    hourly_by_season = (
        df.groupby(["season", "hour"])["temperature_celsius"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    season_colors = {
        "Winter": "#3498DB",
        "Spring": "#2ECC71",
        "Summer": "#F39C12",
        "Fall": "#E74C3C",
    }

    season_order = ["Winter", "Spring", "Summer", "Fall"]
    available_seasons = [
        s for s in season_order if s in hourly_by_season["season"].unique()
    ]

    for season in available_seasons:
        season_data = hourly_by_season[hourly_by_season["season"] == season]
        ax.plot(
            season_data["hour"],
            season_data["mean"],
            marker="o",
            linewidth=2.5,
            label=season,
            color=season_colors[season],
            markersize=6,
        )

        # Add confidence bands
        ax.fill_between(
            season_data["hour"],
            season_data["mean"] - season_data["std"],
            season_data["mean"] + season_data["std"],
            alpha=0.2,
            color=season_colors[season],
        )

    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Average Temperature (°C)", fontsize=12)
    ax.set_title(
        "Daily Temperature Cycle by Season - Chicago", fontsize=14, fontweight="bold"
    )
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hourly_patterns.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ Created: {output_dir}/hourly_patterns.png")
    plt.close()


def detect_extreme_weather(df, output_dir="reports"):
    """Detect heatwaves and cold snaps."""
    print("\n🔥❄️  Detecting heatwaves and cold snaps...")

    # Define thresholds
    HEATWAVE_TEMP = 30.0  # °C
    COLD_SNAP_TEMP = 0.0  # °C
    MIN_DURATION = 3  # consecutive days

    # Daily max/min
    daily_stats = (
        df.groupby("date")["temperature_celsius"].agg(["min", "max"]).reset_index()
    )

    # Detect heatwaves (3+ consecutive days with max >= 30°C)
    daily_stats["is_hot"] = daily_stats["max"] >= HEATWAVE_TEMP
    daily_stats["is_cold"] = daily_stats["min"] <= COLD_SNAP_TEMP

    # Find consecutive periods
    daily_stats["hot_group"] = (
        daily_stats["is_hot"] != daily_stats["is_hot"].shift()
    ).cumsum()
    daily_stats["cold_group"] = (
        daily_stats["is_cold"] != daily_stats["is_cold"].shift()
    ).cumsum()

    heatwaves = (
        daily_stats[daily_stats["is_hot"]]
        .groupby("hot_group")
        .filter(lambda x: len(x) >= MIN_DURATION)
    )
    cold_snaps = (
        daily_stats[daily_stats["is_cold"]]
        .groupby("cold_group")
        .filter(lambda x: len(x) >= MIN_DURATION)
    )

    print(
        f"   Found {len(heatwaves)} days in heatwaves ({len(heatwaves)//MIN_DURATION}+ events)"
    )
    print(
        f"   Found {len(cold_snaps)} days in cold snaps ({len(cold_snaps)//MIN_DURATION}+ events)"
    )

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Heatwave plot
    ax1.plot(
        daily_stats["date"], daily_stats["max"], color="gray", alpha=0.5, linewidth=1
    )
    ax1.axhline(
        HEATWAVE_TEMP,
        color="red",
        linestyle="--",
        label=f"Heatwave Threshold ({HEATWAVE_TEMP}°C)",
    )
    if not heatwaves.empty:
        ax1.scatter(
            heatwaves["date"],
            heatwaves["max"],
            color="red",
            s=50,
            label=f"Heatwave Days ({len(heatwaves)})",
            zorder=5,
        )
    ax1.set_ylabel("Daily Max Temperature (°C)", fontsize=12)
    ax1.set_title(
        "Heatwave Detection (3+ consecutive days ≥30°C)", fontsize=13, fontweight="bold"
    )
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Cold snap plot
    ax2.plot(
        daily_stats["date"], daily_stats["min"], color="gray", alpha=0.5, linewidth=1
    )
    ax2.axhline(
        COLD_SNAP_TEMP,
        color="blue",
        linestyle="--",
        label=f"Cold Snap Threshold ({COLD_SNAP_TEMP}°C)",
    )
    if not cold_snaps.empty:
        ax2.scatter(
            cold_snaps["date"],
            cold_snaps["min"],
            color="blue",
            s=50,
            label=f"Cold Snap Days ({len(cold_snaps)})",
            zorder=5,
        )
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Daily Min Temperature (°C)", fontsize=12)
    ax2.set_title(
        "Cold Snap Detection (3+ consecutive days ≤0°C)", fontsize=13, fontweight="bold"
    )
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/extreme_weather.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ Created: {output_dir}/extreme_weather.png")
    plt.close()


def analyze_temperature_trends(df, output_dir="reports"):
    """Analyze temperature trends over time."""
    print("\n📈 Analyzing temperature trends...")

    # Create ordinal time for regression
    df["days_since_start"] = (
        df["timestamp"] - df["timestamp"].min()
    ).dt.total_seconds() / 86400

    # Linear regression
    X = df["days_since_start"].values.reshape(-1, 1)
    y = df["temperature_celsius"].values

    model = LinearRegression()
    model.fit(X, y)
    trend_line = model.predict(X)

    slope_per_day = model.coef_[0]
    slope_per_year = slope_per_day * 365.25

    print(f"   Trend: {slope_per_year:+.4f}°C per year")

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Scatter with trend line
    ax1.scatter(
        df["timestamp"],
        df["temperature_celsius"],
        alpha=0.3,
        s=5,
        color="#3498DB",
        label="Data Points",
    )
    ax1.plot(
        df["timestamp"],
        trend_line,
        color="red",
        linewidth=3,
        label=f"Trend: {slope_per_year:+.4f}°C/year",
    )
    ax1.set_ylabel("Temperature (°C)", fontsize=12)
    ax1.set_title(
        "Temperature Trend Analysis - Chicago", fontsize=14, fontweight="bold"
    )
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Monthly averages
    monthly_avg = df.groupby([df["timestamp"].dt.to_period("M")])[
        "temperature_celsius"
    ].mean()
    monthly_dates = [period.to_timestamp() for period in monthly_avg.index]

    ax2.plot(
        monthly_dates,
        monthly_avg.values,
        marker="o",
        linewidth=2,
        markersize=6,
        color="#2ECC71",
        label="Monthly Average",
    )
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Temperature (°C)", fontsize=12)
    ax2.set_title("Monthly Average Temperature", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/temperature_trends.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ Created: {output_dir}/temperature_trends.png")
    plt.close()


def analyze_hourly_volatility(df, output_dir="reports"):
    """Analyze hour-to-hour temperature volatility."""
    print("\n📏 Analyzing hourly volatility...")

    # Calculate hour-to-hour change
    df_sorted = df.sort_values("timestamp")
    df_sorted["temp_change"] = df_sorted["temperature_celsius"].diff()
    df_sorted["abs_temp_change"] = abs(df_sorted["temp_change"])

    # By hour and season
    def assign_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"

    df_sorted["season"] = df_sorted["month"].apply(assign_season)

    # Volatility heatmap data
    volatility = (
        df_sorted.groupby(["season", "hour"])["abs_temp_change"].mean().reset_index()
    )
    pivot = volatility.pivot(index="season", columns="hour", values="abs_temp_change")

    # Reorder seasons
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    pivot = pivot.reindex([s for s in season_order if s in pivot.index])

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_xlabel("Hour of Day", fontsize=12)
    ax.set_ylabel("Season", fontsize=12)
    ax.set_title(
        "Hourly Temperature Volatility Heatmap (°C change)",
        fontsize=14,
        fontweight="bold",
    )

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Avg Absolute Change (°C)", fontsize=11)

    # Add values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(
                j,
                i,
                f"{pivot.values[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(f"{output_dir}/hourly_volatility.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ Created: {output_dir}/hourly_volatility.png")
    plt.close()


def track_records(df, output_dir="reports"):
    """Track temperature records over time."""
    print("\n🏆 Tracking temperature records...")

    # Sort by time
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    # Initialize records
    df_sorted["cumulative_max"] = df_sorted["temperature_celsius"].cummax()
    df_sorted["cumulative_min"] = df_sorted["temperature_celsius"].cummin()

    # Find when records were broken
    df_sorted["new_high"] = (
        df_sorted["temperature_celsius"] == df_sorted["cumulative_max"]
    )
    df_sorted["new_low"] = (
        df_sorted["temperature_celsius"] == df_sorted["cumulative_min"]
    )

    # Filter to actual record-breaking moments (not just ties)
    df_sorted["is_new_high"] = (df_sorted["new_high"]) & (
        df_sorted["cumulative_max"] != df_sorted["cumulative_max"].shift()
    )
    df_sorted["is_new_low"] = (df_sorted["new_low"]) & (
        df_sorted["cumulative_min"] != df_sorted["cumulative_min"].shift()
    )

    high_records = df_sorted[df_sorted["is_new_high"]]
    low_records = df_sorted[df_sorted["is_new_low"]]

    print(f"   High temperature records broken: {len(high_records)}")
    print(f"   Low temperature records broken: {len(low_records)}")
    print(f"   Current record high: {df_sorted['cumulative_max'].iloc[-1]:.2f}°C")
    print(f"   Current record low: {df_sorted['cumulative_min'].iloc[-1]:.2f}°C")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(
        df_sorted["timestamp"],
        df_sorted["cumulative_max"],
        color="#E74C3C",
        linewidth=2.5,
        label="Record High",
        alpha=0.8,
    )
    ax.plot(
        df_sorted["timestamp"],
        df_sorted["cumulative_min"],
        color="#3498DB",
        linewidth=2.5,
        label="Record Low",
        alpha=0.8,
    )

    # Mark record-breaking events
    ax.scatter(
        high_records["timestamp"],
        high_records["temperature_celsius"],
        color="red",
        s=80,
        marker="^",
        edgecolors="black",
        linewidths=1.5,
        label=f"New High Records ({len(high_records)})",
        zorder=5,
    )
    ax.scatter(
        low_records["timestamp"],
        low_records["temperature_celsius"],
        color="blue",
        s=80,
        marker="v",
        edgecolors="black",
        linewidths=1.5,
        label=f"New Low Records ({len(low_records)})",
        zorder=5,
    )

    ax.fill_between(
        df_sorted["timestamp"],
        df_sorted["cumulative_min"],
        df_sorted["cumulative_max"],
        alpha=0.1,
        color="gray",
    )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(
        "Temperature Records Progression - Chicago", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/temperature_records.png", dpi=300, bbox_inches="tight")
    print(f"   ✅ Created: {output_dir}/temperature_records.png")
    plt.close()


def main():
    """Run all advanced weather analyses."""
    print("=" * 60)
    print("🌡️  ADVANCED WEATHER ANALYSIS - CHICAGO")
    print("=" * 60)

    # Ensure output directory exists
    Path("reports").mkdir(exist_ok=True)

    # Load data
    print("\n📊 Loading data...")
    df = load_data()
    print(f"   Loaded {len(df)} temperature readings")
    print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Run all analyses
    detect_anomalies(df)
    analyze_hourly_patterns(df)
    detect_extreme_weather(df)
    analyze_temperature_trends(df)
    analyze_hourly_volatility(df)
    track_records(df)

    print("\n" + "=" * 60)
    print("✅ All advanced analyses complete!")
    print("📁 Check the 'reports/' directory for visualizations")
    print("=" * 60)


if __name__ == "__main__":
    main()
