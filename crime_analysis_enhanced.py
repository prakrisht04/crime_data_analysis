"""Enhanced Crime Data Analytics for Urban Safety.

Downloads the LAPD Crime Data from 2020 to Present, performs data-quality
checks and feature engineering, and writes publication-ready figures plus
summary tables to outputs/.

Source: Los Angeles Open Data Portal dataset 2nrs-mtv8.
"""

from pathlib import Path
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

DATA_URL = "https://data.lacity.org/api/views/2nrs-mtv8/rows.csv?accessType=DOWNLOAD"
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_FILE = DATA_DIR / "Crime_Data_from_2020_to_Present.csv"


def load_data():
    DATA_DIR.mkdir(exist_ok=True)
    if not DATA_FILE.exists():
        print("Downloading official LAPD crime dataset...")
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    df = pd.read_csv(DATA_FILE, low_memory=False)
    return df


def clean_data(df):
    df = df.copy()
    df["DATE OCC"] = pd.to_datetime(df["DATE OCC"], errors="coerce")
    df["Date Rptd"] = pd.to_datetime(df["Date Rptd"], errors="coerce")

    # TIME OCC is usually HHMM; coerce safely rather than treating it as a date.
    time_num = pd.to_numeric(df["TIME OCC"], errors="coerce").fillna(0).astype(int)
    df["Hour"] = (time_num // 100).clip(0, 23)
    df["Month"] = df["DATE OCC"].dt.month
    df["Month Name"] = df["DATE OCC"].dt.month_name().str[:3]
    df["Year"] = df["DATE OCC"].dt.year
    df["Day"] = df["DATE OCC"].dt.day_name().str[:3]
    df["Time Slot"] = pd.cut(
        df["Hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["Night (00-05)", "Morning (06-11)", "Afternoon (12-17)", "Evening (18-23)"],
    )

    age = pd.to_numeric(df.get("Vict Age"), errors="coerce")
    age = age.where(age.between(0, 100))
    df["Age Group"] = pd.cut(
        age,
        bins=[-1, 17, 30, 45, 60, 100],
        labels=["0-17", "18-30", "31-45", "46-60", "61+"],
    ).astype("object").fillna("Unknown")

    df["Vict Sex Clean"] = df.get("Vict Sex", pd.Series(index=df.index, dtype="object"))
    df["Vict Sex Clean"] = df["Vict Sex Clean"].replace({"X": "Unknown", "": "Unknown"}).fillna("Unknown")
    df["Area"] = df["AREA NAME"].fillna("Unknown")
    df["Crime Type"] = df["Crm Cd Desc"].fillna("Unknown")
    df["Status"] = df["Status Desc"].fillna("Unknown")

    # Coordinates (0,0) are treated as missing for spatial analysis.
    lat = pd.to_numeric(df.get("LAT"), errors="coerce")
    lon = pd.to_numeric(df.get("LON"), errors="coerce")
    df["Geo Valid"] = lat.between(33, 35) & lon.between(-119, -117)
    return df


def save_tables(df):
    OUT_DIR.mkdir(exist_ok=True)
    yearly = df.groupby("Year").size().rename("Crime Incidents").reset_index()
    area = df["Area"].value_counts().head(15).rename_axis("Area").reset_index(name="Crime Incidents")
    crimes = df["Crime Type"].value_counts().head(15).rename_axis("Crime Type").reset_index(name="Crime Incidents")
    hourly = df["Hour"].value_counts().sort_index().rename_axis("Hour").reset_index(name="Crime Incidents")
    yearly.to_csv(OUT_DIR / "crime_by_year.csv", index=False)
    area.to_csv(OUT_DIR / "top_areas.csv", index=False)
    crimes.to_csv(OUT_DIR / "top_crime_types.csv", index=False)
    hourly.to_csv(OUT_DIR / "crime_by_hour.csv", index=False)

    kpis = pd.DataFrame({
        "Metric": ["Total incidents", "Years covered", "Unique areas", "Unique crime types", "Geo-valid records"],
        "Value": [
            len(df),
            df["Year"].nunique(),
            df["Area"].nunique(),
            df["Crime Type"].nunique(),
            int(df["Geo Valid"].sum()),
        ],
    })
    kpis.to_csv(OUT_DIR / "kpis.csv", index=False)


def make_figures(df):
    OUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Annual trend
    yearly = df.groupby("Year").size()
    fig, ax = plt.subplots(figsize=(10, 5))
    yearly.plot(ax=ax, marker="o")
    ax.set_title("Crime Incidents by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Incidents")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "01_yearly_trend.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Monthly seasonality
    monthly = df.groupby("Month").size().reindex(range(1, 13), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly.index, monthly.values, marker="o")
    ax.set_xticks(range(1, 13))
    ax.set_title("Monthly Crime Pattern")
    ax.set_xlabel("Month")
    ax.set_ylabel("Incidents")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "02_monthly_pattern.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 3. Top areas
    top_area = df["Area"].value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    top_area.plot.barh(ax=ax)
    ax.set_title("Top 12 Areas by Reported Crime Incidents")
    ax.set_xlabel("Incidents")
    ax.set_ylabel("Area")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "03_top_areas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Top crime types
    top_crime = df["Crime Type"].value_counts().head(12).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    top_crime.plot.barh(ax=ax)
    ax.set_title("Top 12 Crime Types")
    ax.set_xlabel("Incidents")
    ax.set_ylabel("Crime Type")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "04_top_crime_types.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 5. Hourly pattern
    hourly = df.groupby("Hour").size().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hourly.index, hourly.values, marker="o")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Crime Incidents by Hour of Day")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Incidents")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "05_hourly_pattern.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 6. Area x time-slot heatmap
    heat = pd.crosstab(df["Area"], df["Time Slot"])
    top_areas = df["Area"].value_counts().head(15).index
    heat = heat.loc[heat.index.intersection(top_areas)]
    heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(heat, cmap="YlOrRd", annot=False, ax=ax)
    ax.set_title("Crime Concentration by Area and Time Slot")
    ax.set_xlabel("Time Slot")
    ax.set_ylabel("Area")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "06_area_time_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 7. Victim age groups
    age_counts = df["Age Group"].value_counts().reindex(["0-17", "18-30", "31-45", "46-60", "61+", "Unknown"], fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 5))
    age_counts.plot.bar(ax=ax)
    ax.set_title("Victim Age-Group Distribution")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Incidents")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "07_victim_age_groups.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 8. Spatial scatter for valid coordinates; capped for rendering speed.
    geo = df.loc[df["Geo Valid"], ["LAT", "LON"]].dropna()
    if len(geo) > 20000:
        geo = geo.sample(20000, random_state=42)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(geo["LON"], geo["LAT"], s=2, alpha=0.15)
    ax.set_title("Geographic Distribution of Reported Crime (sampled points)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "08_geographic_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    df = clean_data(load_data())
    save_tables(df)
    make_figures(df)
    print(f"Analysis complete: {len(df):,} records processed.")
    print(f"Figures and summary tables: {OUT_DIR}")


if __name__ == "__main__":
    main()
