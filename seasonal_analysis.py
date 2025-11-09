"""Seasonal temperature analysis for Chicago weather data."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def assign_season(month):
    """Assign season based on month (Northern Hemisphere)."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Fall"


def load_and_prepare_data(csv_path):
    """Load data and add season information."""
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["timestamp"].dt.month
    df["season"] = df["month"].apply(assign_season)
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = df["timestamp"].dt.date
    return df


def calculate_seasonal_stats(df):
    """Calculate statistics for each season."""
    stats = df.groupby("season")["temperature_celsius"].agg(
        ["mean", "std", "min", "max", "count"]
    )
    stats.columns = ["Mean (°C)", "Std Dev", "Min (°C)", "Max (°C)", "Count"]
    return stats


def create_seasonal_plots(df, output_dir="reports"):
    """Create multiple seasonal comparison plots."""
    Path(output_dir).mkdir(exist_ok=True)

    # Define season order and colors
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    season_colors = {
        "Winter": "#3498DB",
        "Spring": "#2ECC71",
        "Summer": "#F39C12",
        "Fall": "#E74C3C",
    }

    # Filter to only seasons present in data
    available_seasons = [s for s in season_order if s in df["season"].unique()]

    # Plot 1: Box plot comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    boxes = []
    positions = []
    colors = []

    for i, season in enumerate(available_seasons):
        season_data = df[df["season"] == season]["temperature_celsius"]
        boxes.append(season_data)
        positions.append(i)
        colors.append(season_colors[season])

    bp = ax.boxplot(boxes, positions=positions, widths=0.6, patch_artist=True)

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(available_seasons)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title("Temperature Distribution by Season - Chicago", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/seasonal_boxplot.png", dpi=300, bbox_inches="tight")
    print(f"✅ Created: {output_dir}/seasonal_boxplot.png")

    # Plot 2: Mean temperature by season
    fig, ax = plt.subplots(figsize=(12, 7))
    seasonal_means = df.groupby("season")["temperature_celsius"].mean()
    seasonal_means = seasonal_means.reindex(available_seasons)

    bars = ax.bar(
        available_seasons,
        seasonal_means.values,
        color=[season_colors[s] for s in available_seasons],
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    # Add value labels on bars
    for bar, value in zip(bars, seasonal_means.values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.1f}°C",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Average Temperature (°C)", fontsize=12)
    ax.set_title("Average Temperature by Season - Chicago", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/seasonal_averages.png", dpi=300, bbox_inches="tight")
    print(f"✅ Created: {output_dir}/seasonal_averages.png")

    # Plot 3: Temperature over time colored by season
    fig, ax = plt.subplots(figsize=(14, 7))

    for season in available_seasons:
        season_data = df[df["season"] == season]
        ax.scatter(
            season_data["timestamp"],
            season_data["temperature_celsius"],
            c=season_colors[season],
            label=season,
            alpha=0.6,
            s=10,
        )

    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Temperature (°C)", fontsize=12)
    ax.set_title(
        "Temperature Over Time by Season - Chicago", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/seasonal_timeline.png", dpi=300, bbox_inches="tight")
    print(f"✅ Created: {output_dir}/seasonal_timeline.png")

    # Plot 4: Daily temperature range by season
    fig, ax = plt.subplots(figsize=(12, 7))
    daily_range = df.groupby(["date", "season"])["temperature_celsius"].agg(
        ["min", "max"]
    )
    daily_range["range"] = daily_range["max"] - daily_range["min"]
    daily_range = daily_range.reset_index()

    seasonal_ranges = daily_range.groupby("season")["range"].mean()
    seasonal_ranges = seasonal_ranges.reindex(available_seasons)

    bars = ax.bar(
        available_seasons,
        seasonal_ranges.values,
        color=[season_colors[s] for s in available_seasons],
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
    )

    for bar, value in zip(bars, seasonal_ranges.values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.1f}°C",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_ylabel("Average Daily Temperature Range (°C)", fontsize=12)
    ax.set_title(
        "Average Daily Temperature Variation by Season - Chicago",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/seasonal_variation.png", dpi=300, bbox_inches="tight")
    print(f"✅ Created: {output_dir}/seasonal_variation.png")

    plt.close("all")


def main():
    """Run seasonal analysis."""
    print("🌡️  Starting seasonal analysis...")

    # Load data
    df = load_and_prepare_data("data/chicago_temps.csv")
    print(f"📊 Loaded {len(df)} temperature readings")
    print(f"📅 Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Calculate statistics
    stats = calculate_seasonal_stats(df)
    print("\n📈 Seasonal Statistics:")
    print(stats)

    # Create plots
    print("\n🎨 Creating plots...")
    create_seasonal_plots(df)

    print("\n✅ Analysis complete! Check the 'reports/' directory for plots.")


if __name__ == "__main__":
    main()
