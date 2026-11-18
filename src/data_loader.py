"""
PathSnap Dataset Loader & Geodesic Utilities.

Handles loading raw trip files (.track, .nodes, .arcs, .route) and converting
WGS84 (longitude, latitude) coordinates into local metric Cartesian coordinates
(East, North in meters) using WGS84 ellipsoid curvature radii centered at each
trip's reference origin.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# WGS84 ellipsoid constants
WGS84_A = 6378137.0  # semi-major axis (meters)
WGS84_E2 = 6.69437999014e-3  # first eccentricity squared


@dataclass
class TripData:
    trip_id: str
    partition: str
    trip_dir: Path
    track: pd.DataFrame  # columns: lon, lat, timestamp
    nodes: np.ndarray  # shape (N_nodes, 2): [lon, lat]
    arcs: np.ndarray  # shape (N_arcs, 2): [from_node, to_node]
    route: Optional[np.ndarray]  # shape (N_route,): arc_id sequence, or None
    file_sizes: Dict[str, int]
    missing_files: List[str]


def list_trips(dataset_root: Path, partition: str) -> List[Path]:
    """Return sorted list of trip directories in a given partition."""
    part_dir = Path(dataset_root) / partition
    if not part_dir.exists():
        return []
    return sorted([p for p in part_dir.iterdir() if p.is_dir()])


def load_trip(trip_dir: Path, partition: str = "", load_route: bool = True) -> TripData:
    """
    Load all files for a single trip directory.

    Parameters
    ----------
    trip_dir : Path
        Directory containing <trip_id>.track, .nodes, .arcs, and optionally .route.
    partition : str
        Partition name (e.g. 'train', 'student_test').
    load_route : bool
        Whether to load the .route file if present. Set False in inference pipelines
        to strictly guarantee ground-truth separation.
    """
    trip_dir = Path(trip_dir)
    trip_id = trip_dir.name
    exts = ["track", "nodes", "arcs", "route"]
    file_sizes: Dict[str, int] = {}
    missing_files: List[str] = []

    paths: Dict[str, Path] = {}
    for ext in exts:
        p = trip_dir / f"{trip_id}.{ext}"
        if p.exists():
            file_sizes[ext] = p.stat().st_size
            paths[ext] = p
        else:
            file_sizes[ext] = 0
            missing_files.append(ext)

    # Load .track: lon, lat, timestamp
    if "track" in paths and file_sizes["track"] > 0:
        track = pd.read_csv(
            paths["track"],
            sep=r"\s+",
            header=None,
            names=["lon", "lat", "timestamp"],
            engine="c",
        )
        for col in ["lon", "lat", "timestamp"]:
            track[col] = pd.to_numeric(track[col], errors="coerce")
    else:
        track = pd.DataFrame(columns=["lon", "lat", "timestamp"])

    # Load .nodes: lon, lat (0-based row index = node_id)
    if "nodes" in paths and file_sizes["nodes"] > 0:
        nodes_df = pd.read_csv(
            paths["nodes"],
            sep=r"\s+",
            header=None,
            names=["lon", "lat"],
            engine="c",
        )
        nodes = nodes_df.to_numpy(dtype=np.float64)
    else:
        nodes = np.empty((0, 2), dtype=np.float64)

    # Load .arcs: from_node_id, to_node_id (0-based row index = arc_id)
    if "arcs" in paths and file_sizes["arcs"] > 0:
        arcs_df = pd.read_csv(
            paths["arcs"],
            sep=r"\s+",
            header=None,
            names=["from_node", "to_node"],
            engine="c",
        )
        arcs = arcs_df.to_numpy(dtype=np.int64)
    else:
        arcs = np.empty((0, 2), dtype=np.int64)

    # Load .route: chronological ground-truth arc_id sequence
    route: Optional[np.ndarray] = None
    if load_route and "route" in paths and file_sizes["route"] > 0:
        route_df = pd.read_csv(
            paths["route"],
            sep=r"\s+",
            header=None,
            names=["arc_id"],
            engine="c",
        )
        route = route_df["arc_id"].to_numpy(dtype=np.int64)

    return TripData(
        trip_id=trip_id,
        partition=partition,
        trip_dir=trip_dir,
        track=track,
        nodes=nodes,
        arcs=arcs,
        route=route,
        file_sizes=file_sizes,
        missing_files=missing_files,
    )


class LocalMetricProjector:
    """
    Projects WGS84 (lon, lat) degrees to local tangent-plane (x_east, y_north) in meters
    using exact WGS84 meridional (M) and prime-vertical (N) radii of curvature at lat0.

    Accuracy over a 50 km trip extent is < 0.05% distortion, works globally across any
    UTM zone without zone-boundary discontinuities, and requires only numpy.
    """

    def __init__(self, lon0: float, lat0: float):
        self.lon0 = float(lon0)
        self.lat0 = float(lat0)
        lat_rad = np.deg2rad(self.lat0)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        denom = np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        # Prime vertical radius of curvature (East-West)
        n_radius = WGS84_A / denom
        # Meridional radius of curvature (North-South)
        m_radius = WGS84_A * (1.0 - WGS84_E2) / (denom ** 3)
        self.meters_per_deg_lon = (np.pi / 180.0) * n_radius * cos_lat
        self.meters_per_deg_lat = (np.pi / 180.0) * m_radius

    @classmethod
    def from_trip(cls, trip: TripData) -> "LocalMetricProjector":
        valid_lon = trip.track["lon"].dropna().to_numpy()
        valid_lat = trip.track["lat"].dropna().to_numpy()
        if len(valid_lon) > 0:
            return cls(float(np.median(valid_lon)), float(np.median(valid_lat)))
        return cls(float(np.median(trip.nodes[:, 0])), float(np.median(trip.nodes[:, 1])))

    def to_xy(self, lon: np.ndarray, lat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = (np.asarray(lon, dtype=np.float64) - self.lon0) * self.meters_per_deg_lon
        y = (np.asarray(lat, dtype=np.float64) - self.lat0) * self.meters_per_deg_lat
        return x, y

    def to_xy_array(self, lonlat: np.ndarray) -> np.ndarray:
        x, y = self.to_xy(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y])


def haversine_m(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Vectorized Haversine distance in meters between WGS84 coordinates."""
    r = 6371008.8
    lon1_r, lat1_r = np.deg2rad(lon1), np.deg2rad(lat1)
    lon2_r, lat2_r = np.deg2rad(lon2), np.deg2rad(lat2)
    dlon = lon2_r - lon1_r
    dlat = lat2_r - lat1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def point_to_segments_distance_and_proj(
    px: float,
    py: float,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute perpendicular distance from point (px, py) to line segments A->B.

    Returns
    -------
    dist : np.ndarray
        Euclidean distance in meters from point to closest point on each segment.
    t : np.ndarray
        Clamped projection fraction along segment in [0, 1].
    proj_x, proj_y : np.ndarray
        Coordinates of the closest point on each segment.
    """
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    # Avoid division by zero for degenerate zero-length segments
    safe_len_sq = np.where(seg_len_sq > 1e-12, seg_len_sq, 1.0)
    t = ((px - ax) * dx + (py - ay) * dy) / safe_len_sq
    t = np.where(seg_len_sq > 1e-12, np.clip(t, 0.0, 1.0), 0.0)
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    dist = np.hypot(px - proj_x, py - proj_y)
    return dist, t, proj_x, proj_y
