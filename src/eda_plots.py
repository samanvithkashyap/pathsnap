"""
Generate EDA Visualisations and Interactive Folium Maps for PathSnap.

Outputs saved to outputs/eda/:
- plots/01_partition_overview.png      : Observation counts, node/arc counts, route lengths (train vs student_test)
- plots/02_sampling_and_quality.png    : Sampling interval (dt) distribution, GPS-to-route distance CDF & outliers
- plots/03_global_distribution.png     : Geographic distribution of all 90 trips worldwide
- plots/04_road_network_topology.png   : Road network connectivity (WCC/SCC, arc lengths, bidirectionality)
- plots/05_sample_trajectories.png     : Local metric plots of clean vs loop vs anomalous GPS tracks + road networks
- maps/trip_00000000_clean.html        : Interactive Folium map (student_test/00000000)
- maps/trip_00000056_loop.html         : Interactive Folium map (student_test/00000056 - bidirectional loop)
- maps/trip_00000054_anomaly.html      : Interactive Folium map (train/00000054 - off-road stationary drift)
"""

from __future__ import annotations

from pathlib import Path

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_loader import LocalMetricProjector, list_trips, load_trip

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
EDA_DIR = PROJECT_ROOT / "outputs" / "eda"
PLOTS_DIR = EDA_DIR / "plots"
MAPS_DIR = EDA_DIR / "maps"


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
    })


def plot_partition_overview(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    palette = {"train": "#2563eb", "student_test": "#ea580c"}

    metrics = [
        ("n_obs", "GPS Observations per Trip (log scale)", True),
        ("n_arcs", "Road-Network Arcs per Trip (log scale)", True),
        ("n_route_arcs", "Reference Route Length (Arcs, log scale)", True),
        ("route_total_len_km", "Reference Route Physical Length (km)", False),
    ]

    for ax, (col, title, use_log) in zip(axes.flat, metrics):
        for part in ["train", "student_test"]:
            vals = df[df["partition"] == part][col].dropna()
            ax.hist(
                np.log10(np.maximum(vals, 1e-3)) if use_log else vals,
                bins=20,
                alpha=0.65,
                label=f"{part} (n={len(vals)}, med={vals.median():.1f})",
                color=palette[part],
                edgecolor="white",
            )
        ax.set_title(title)
        ax.set_xlabel(f"log10({col})" if use_log else col)
        ax.set_ylabel("Number of Trips")
        ax.legend(fontsize=8)

    fig.suptitle("PathSnap Dataset Overview: Train (80 trips) vs Student-Test (10 trips)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_partition_overview.png", bbox_inches="tight")
    plt.close(fig)


def plot_sampling_and_quality(df: pd.DataFrame) -> None:
    # Gather raw dt intervals across train and student_test
    dts_train, dts_test = [], []
    for part, bucket in [("train", dts_train), ("student_test", dts_test)]:
        for td in list_trips(DATASET_ROOT, part):
            trip = load_trip(td, part, load_route=False)
            ts = trip.track["timestamp"].dropna().to_numpy()
            if len(ts) >= 2:
                bucket.extend(np.diff(ts).tolist())

    dts_train = np.array(dts_train)
    dts_test = np.array(dts_test)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    # 1. Sampling interval buckets
    bins = [0.5, 1.5, 2.5, 5.5, 15.5, 60.5, 1e9]
    labels = ["1s", "2s", "3-5s", "6-15s", "16-60s", ">60s"]
    train_pct = [100.0 * np.mean((dts_train >= bins[i]) & (dts_train < bins[i + 1])) for i in range(len(labels))]
    test_pct = [100.0 * np.mean((dts_test >= bins[i]) & (dts_test < bins[i + 1])) for i in range(len(labels))]

    x = np.arange(len(labels))
    w = 0.38
    axes[0].bar(x - w / 2, train_pct, width=w, label="Train (80 trips)", color="#2563eb")
    axes[0].bar(x + w / 2, test_pct, width=w, label="Student-Test (10 trips)", color="#ea580c")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("% of Consecutive Steps (log scale)")
    axes[0].set_title("GPS Sampling Interval (dt) Distribution")
    axes[0].legend(fontsize=8)

    # 2. Per-trip median vs p95 GPS distance to reference route
    for part, color, marker in [("train", "#2563eb", "o"), ("student_test", "#ea580c", "^")]:
        sub = df[df["partition"] == part]
        axes[1].scatter(
            sub["gps_to_route_dist_median_m"],
            sub["gps_to_route_dist_p95_m"],
            alpha=0.8,
            s=45,
            c=color,
            marker=marker,
            label=part,
        )
    # Annotate extreme outliers
    for _, row in df[df["gps_to_route_dist_p95_m"] > 100].iterrows():
        axes[1].annotate(
            str(row["trip_id"]).zfill(8),
            (row["gps_to_route_dist_median_m"], row["gps_to_route_dist_p95_m"]),
            fontsize=7.5,
            xytext=(4, 3),
            textcoords="offset points",
        )
    axes[1].set_xlabel("Median GPS-to-Route Distance (m)")
    axes[1].set_ylabel("95th-Percentile GPS-to-Route Distance (m)")
    axes[1].set_title("GPS Noise & Off-Road Drift Outliers by Trip")
    axes[1].legend(fontsize=8)

    # 3. Coverage vs candidate radius
    radii = [15, 30, 50, 100]
    train_cov = [df[df["partition"] == "train"][f"pct_gps_within_{r}m_route"].mean() for r in radii]
    test_cov = [df[df["partition"] == "student_test"][f"pct_gps_within_{r}m_route"].mean() for r in radii]
    axes[2].plot(radii, train_cov, marker="o", linewidth=2, color="#2563eb", label="Train (mean across 80 trips)")
    axes[2].plot(radii, test_cov, marker="^", linewidth=2, color="#ea580c", label="Student-Test (mean across 10 trips)")
    for r, tr, te in zip(radii, train_cov, test_cov):
        axes[2].annotate(f"{tr:.1f}%", (r, tr), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color="#2563eb")
        axes[2].annotate(f"{te:.1f}%", (r, te), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color="#ea580c")
    axes[2].set_ylim(88, 101.5)
    axes[2].set_xlabel("Distance Threshold to Reference Route (m)")
    axes[2].set_ylabel("Mean % GPS Points Within Threshold")
    axes[2].set_title("GPS Proximity to Reference Route")
    axes[2].legend(fontsize=8)

    fig.suptitle("Sampling Intervals, GPS Noise & Data Quality Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_sampling_and_quality.png", bbox_inches="tight")
    plt.close(fig)


def plot_global_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for part, color, marker, size in [("train", "#2563eb", "o", 55), ("student_test", "#ea580c", "D", 75)]:
        sub = df[df["partition"] == part]
        ax.scatter(
            sub["lon_center"],
            sub["lat_center"],
            c=color,
            marker=marker,
            s=size,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.6,
            label=f"{part} ({len(sub)} trips)",
        )
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_xlim(-120, 155)
    ax.set_ylim(-45, 70)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.set_title("Geographical Distribution of PathSnap Trips (Global Coverage → Local Metric Projection Required)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_global_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_road_network_topology(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # 1. Nodes vs Arcs scaling
    for part, color in [("train", "#2563eb"), ("student_test", "#ea580c")]:
        sub = df[df["partition"] == part]
        axes[0].scatter(sub["n_nodes"], sub["n_arcs"], alpha=0.75, c=color, label=part)
    max_n = df["n_nodes"].max()
    xs = np.linspace(0, max_n, 100)
    axes[0].plot(xs, 2.0 * xs, "--", color="gray", alpha=0.7, label="arcs = 2 × nodes")
    axes[0].set_xlabel("Number of Road Nodes")
    axes[0].set_ylabel("Number of Directed Road Arcs")
    axes[0].set_title("Road Graph Density (Nodes vs Arcs)")
    axes[0].legend(fontsize=8)

    # 2. Bidirectional arcs percentage
    sns.boxplot(data=df, x="partition", y="pct_bidirectional_arcs", hue="partition", legend=False, ax=axes[1], palette={"train": "#93c5fd", "student_test": "#fdba74"})
    sns.stripplot(data=df, x="partition", y="pct_bidirectional_arcs", ax=axes[1], color="#1e293b", size=4, alpha=0.6)
    axes[1].set_title("Bidirectional Road Segments (%)")
    axes[1].set_ylabel("% of Directed Arcs Having Reverse Arc")
    axes[1].set_xlabel("Partition")

    # 3. Weakly connected components vs Largest SCC %
    for part, color in [("train", "#2563eb"), ("student_test", "#ea580c")]:
        sub = df[df["partition"] == part]
        axes[2].scatter(sub["n_wcc"], sub["largest_scc_pct"], alpha=0.75, c=color, label=part)
    axes[2].set_xlabel("Number of Weakly Connected Components")
    axes[2].set_ylabel("Largest Strongly Connected Component (%)")
    axes[2].set_title("Road-Network Connectivity")
    axes[2].legend(fontsize=8)

    fig.suptitle("Road-Network Size, Directionality & Connectivity Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_road_network_topology.png", bbox_inches="tight")
    plt.close(fig)


def plot_sample_trajectories() -> None:
    samples = [
        ("student_test", "00000000", "Clean Urban/Suburban Trip (student_test/00000000)"),
        ("student_test", "00000056", "Bidirectional Loop Trip (student_test/00000056)"),
        ("train", "00000054", "Off-Road Stationary Drift Anomaly (train/00000054)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    for ax, (part, tid, title) in zip(axes, samples):
        trip = load_trip(DATASET_ROOT / part / tid, part, load_route=True)
        proj = LocalMetricProjector.from_trip(trip)
        tx, ty = proj.to_xy(trip.track["lon"], trip.track["lat"])
        nxy = proj.to_xy_array(trip.nodes)

        # Plot subset of nearby road network arcs for clarity
        pad = 300.0
        xmin, xmax = tx.min() - pad, tx.max() + pad
        ymin, ymax = ty.min() - pad, ty.max() + pad
        u, v = trip.arcs[:, 0], trip.arcs[:, 1]
        ax_x, ax_y = nxy[u, 0], nxy[u, 1]
        bx_x, bx_y = nxy[v, 0], nxy[v, 1]
        mask = (
            (ax_x >= xmin) & (ax_x <= xmax) & (ax_y >= ymin) & (ax_y <= ymax)
        )
        idx = np.where(mask)[0]
        if len(idx) > 6000:
            idx = idx[:: max(1, len(idx) // 6000)]

        # Road segments in light gray
        xs_bg = np.column_stack([ax_x[idx], bx_x[idx], np.full(len(idx), np.nan)]).ravel() / 1000.0
        ys_bg = np.column_stack([ax_y[idx], bx_y[idx], np.full(len(idx), np.nan)]).ravel() / 1000.0
        ax.plot(xs_bg, ys_bg, color="#cbd5e1", linewidth=0.5, zorder=1, label="Road network")

        # Reference route in green
        r = trip.route
        xs_r = np.column_stack([ax_x[r], bx_x[r], np.full(len(r), np.nan)]).ravel() / 1000.0
        ys_r = np.column_stack([ax_y[r], bx_y[r], np.full(len(r), np.nan)]).ravel() / 1000.0
        ax.plot(xs_r, ys_r, color="#16a34a", linewidth=2.4, zorder=2, label="Reference route")

        # Raw GPS observations in blue
        step = max(1, len(tx) // 1500)
        ax.scatter(tx[::step] / 1000.0, ty[::step] / 1000.0, s=4, color="#2563eb", alpha=0.6, zorder=3, label="GPS points")

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("East (km)")
        ax.set_ylabel("North (km)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(fontsize=7.5, loc="best")

    fig.suptitle("Representative GPS Trajectories vs Road Network & Reference Routes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_sample_trajectories.png", bbox_inches="tight")
    plt.close(fig)


def create_folium_eda_map(partition: str, trip_id: str, out_path: Path, title: str) -> None:
    trip = load_trip(DATASET_ROOT / partition / trip_id, partition, load_route=True)
    lons = trip.track["lon"].to_numpy()
    lats = trip.track["lat"].to_numpy()
    center_lat = float(np.median(lats))
    center_lon = float(np.median(lons))

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

    # Layer 1: Nearby road network (capped for crisp browser performance)
    road_layer = folium.FeatureGroup(name="Nearby Road Network", show=True)
    u, v = trip.arcs[:, 0], trip.arcs[:, 1]
    lat_min, lat_max = lats.min() - 0.004, lats.max() + 0.004
    lon_min, lon_max = lons.min() - 0.005, lons.max() + 0.005
    nlon, nlat = trip.nodes[:, 0], trip.nodes[:, 1]
    in_box = (
        (nlon[u] >= lon_min) & (nlon[u] <= lon_max) &
        (nlat[u] >= lat_min) & (nlat[u] <= lat_max)
    )
    arc_ids = np.where(in_box)[0]
    if len(arc_ids) > 2500:
        arc_ids = arc_ids[:: max(1, len(arc_ids) // 2500)]
    road_segments = [
        [(float(nlat[u[i]]), float(nlon[u[i]])), (float(nlat[v[i]]), float(nlon[v[i]]))]
        for i in arc_ids
    ]
    folium.PolyLine(road_segments, color="#94a3b8", weight=1.5, opacity=0.6).add_to(road_layer)
    road_layer.add_to(m)

    # Layer 2: Reference route
    route_layer = folium.FeatureGroup(name="Reference Route (.route)", show=True)
    r = trip.route
    route_segments = [
        [(float(nlat[u[i]]), float(nlon[u[i]])), (float(nlat[v[i]]), float(nlon[v[i]]))]
        for i in r
    ]
    folium.PolyLine(route_segments, color="#16a34a", weight=5, opacity=0.85, tooltip="Reference Route").add_to(route_layer)
    route_layer.add_to(m)

    # Layer 3: Raw GPS observations
    gps_layer = folium.FeatureGroup(name="Raw GPS Observations (.track)", show=True)
    step = max(1, len(lats) // 600)
    for idx in range(0, len(lats), step):
        folium.CircleMarker(
            location=[float(lats[idx]), float(lons[idx])],
            radius=2.5,
            color="#2563eb",
            fill=True,
            fill_color="#2563eb",
            fill_opacity=0.7,
            popup=f"idx={idx}, t={trip.track['timestamp'].iloc[idx]:.0f}s",
        ).add_to(gps_layer)
    gps_layer.add_to(m)

    # Start / End markers
    folium.Marker(
        [float(lats[0]), float(lons[0])],
        popup=f"Start (t={trip.track['timestamp'].iloc[0]:.0f}s)",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        [float(lats[-1]), float(lons[-1])],
        popup=f"End (t={trip.track['timestamp'].iloc[-1]:.0f}s)",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    # Fit bounds & legend
    m.fit_bounds([[float(lats.min()), float(lons.min())], [float(lats.max()), float(lons.max())]])
    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999; background: white;
                padding: 10px 14px; border: 2px solid #cbd5e1; border-radius: 8px; font-size: 12px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);">
      <b>{title}</b><br>
      <span style="color:#2563eb;">●</span> Raw GPS Observations (n={len(lats)})<br>
      <span style="color:#16a34a;">━━</span> Reference Route ({len(r)} arcs)<br>
      <span style="color:#94a3b8;">──</span> Road Network Subset
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(str(out_path))


def main() -> None:
    setup_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    MAPS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(EDA_DIR / "trip_inventory_eda.csv")
    print("Generating EDA plots...")
    plot_partition_overview(df)
    plot_sampling_and_quality(df)
    plot_global_distribution(df)
    plot_road_network_topology(df)
    plot_sample_trajectories()

    print("Generating interactive Folium EDA maps...")
    create_folium_eda_map(
        "student_test", "00000000", MAPS_DIR / "trip_00000000_clean.html",
        "EDA: Clean Trip (student_test / 00000000)",
    )
    create_folium_eda_map(
        "student_test", "00000056", MAPS_DIR / "trip_00000056_loop.html",
        "EDA: Bidirectional Loop Trip (student_test / 00000056)",
    )
    create_folium_eda_map(
        "train", "00000054", MAPS_DIR / "trip_00000054_anomaly.html",
        "EDA: Off-Road Stationary Drift Anomaly (train / 00000054)",
    )
    print(f"Done! Plots saved to {PLOTS_DIR}, Maps saved to {MAPS_DIR}")


if __name__ == "__main__":
    main()
