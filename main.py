# step_service/main.py
#
# SAFE GEOMETRY STEP microservice for Alex-IO foam layouts.
# Guarantees at least one valid solid is exported.
#
# FIX:
# - Circle cavities supported (cylindrical cuts)
# - Coordinate system alignment: editor uses top-left origin (y down),
#   CAD uses bottom-left origin (y up). Flip Y when placing cavities.
# - NEW (Path A): Rounded-rect cavities supported via optional corner radius.
#   If a cavity includes cornerRadiusIn (or corner_radius_in), we generate a
#   filleted rectangular pocket instead of a sharp-corner box cut.
#
# LAYER CAVITY FIX (Path A):
# - Per-layer STEP exports must ONLY use layer.cavities.
# - Do NOT merge layout.cavities into the first layer.
#   layout.cavities is a mirror of the active layer in the editor and can
#   incorrectly stamp cavities onto blank layers when exporting them alone.
#
# CROPPED-CORNER BLOCK SUPPORT (Path A):
# - Outer perimeter chamfer (two corners: upper-left and lower-right).
# - Accept global intent:
#     - layout.block.croppedCorners true (or snake-case), OR
#     - layout.block.cornerStyle == "chamfer" (or snake-case)
# - NEW (Path A): Per-layer override
#     - If a layer sets cropCorners true/false (or snake-case), STEP honors it
#       for THAT layer only. If omitted, we fall back to the global intent.
# - chamferIn is in inches, default 1" if omitted.
#
# NEW (Path A) - ROUNDED OUTER BLOCK CORNERS:
# - Per-layer roundCorners + roundRadiusIn (also snake-case variants).
# - If enabled for a layer, the outer block is filleted on the vertical edges.
# - SAFE fallback: if fillet fails, fall back to square block (no rounding).
# - Rounded corners take precedence over chamfer/crop for that layer.
#
# NESTED CAVITY DEPTH FIX (Path A) - 12/27:
# - When cavities overlap (one inside another), boolean cut ORDER matters.
#   To preserve "stepped" pockets (large shallow + smaller deep), we cut
#   shallow cavities first, then deeper cavities last.
# - This does NOT change non-overlapping cavities.
# - Depth reference remains the layer's top surface (correct).
#
# NEW (Path A) - THROUGH CUT:
# - If a cavity depth is >= layer thickness, cut through the full layer.
# - Implemented by setting cut depth to T_mm + small epsilon (ensures exit).
#
# NEW (Path A) - POLYGON CAVITIES:
# - If shape == "poly" and points[] is present (0..1 normalized, top-left origin),
#   we cut the pocket using that polygon instead of a rectangle.
# - Points are flipped to CAD Y-up coordinates and scaled into mm.
#
# NEW - ISLAND SUPPORT:
# - A cavity may carry islands[]: raised, uncut regions inside its footprint
#   (the frontend calls these "nestedCavities" and renders them with
#   fillRule="evenodd" -- an island is the OPPOSITE of a deeper nested cut).
# - Implemented as cut-then-union-back, NOT a single hole-in-face extrusion:
#   after cutting the parent cavity as usual, each island's polygon is
#   extruded from the cavity floor (z_cut) back up to the layer's original
#   top surface (z_top) and unioned into the solid. Full-height boss only
#   (no partial-height islands in this version).
# - Ordering is critical: island union happens immediately after ITS OWN
#   cavity's cut, in the same per-cavity loop iteration -- not in a second
#   pass after all cavities are cut -- so a later sibling cavity can never
#   slice into an island before it's been restored.
#
# NEW - PRE-FLIGHT VALIDATION (validate_layout):
# - Runs BEFORE any CAD kernel calls (separate from the existing per-cavity
#   try/except "SAFE GEOMETRY" swallowing, which is for genuine unexpected
#   CAD kernel failures). Raises ValueError -> HTTP 400 with a clear message
#   for:
#     - an island not fully contained within its parent cavity's footprint
#     - a wall/gap thinner than MIN_WALL_IN, between sibling cavities in the
#       same layer, or between an island and its parent cavity's wall
#   Sibling cavities that are legitimately fully nested (the existing
#   "stepped pocket" feature) are exempt from the gap check -- there is no
#   side wall between them, only a floor step.
# - This is a heuristic guard, not a fix for the underlying OCCT boolean-cut
#   robustness issue with closely-spaced features (that remains separate,
#   future work). It exists only to fail loudly instead of silently
#   exporting degenerate geometry.

from typing import List, Optional, Tuple
import math
import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, validator
import cadquery as cq
from stl import mesh as stlmesh

INCH_TO_MM = 25.4

DEPTH_CLAMP_RATIO = 0.95
THROUGH_CUT_EPS_MM = 0.25
FILLET_EPS_MM = 1e-3
DEFAULT_OUTER_ROUND_IN = 0.25

# MIN-WALL VALIDATION GUARD: minimum allowed gap (inches) between two cavity
# footprints in the same layer (sibling cavities, or an island and its
# parent cavity's wall). Chosen conservatively above the ~0.1in wall
# thickness observed to trigger boolean-cut instability in this service.
# This is a heuristic pre-flight check, not a fix for that instability --
# see the ISLAND SUPPORT / PRE-FLIGHT VALIDATION notes above.
MIN_WALL_IN = 0.125
MIN_WALL_MM = MIN_WALL_IN * INCH_TO_MM

# STL corner heal tolerance (inches). Used only in stl_to_faces_json().
STL_HEAL_TOL_IN = 0.015625  # 1/64"


class Point2D(BaseModel):
    # Normalized coordinates in editor space: x,y in [0..1], origin top-left (y down)
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)


class Island(BaseModel):
    """
    A raised, uncut region ("boss") inside a parent cavity's footprint --
    the inverse of a cavity cutout. Rises from the parent cavity's cut floor
    back up to the layer's original top surface (full-height boss only; no
    partial-height islands in this version). Points use the same normalized
    [0..1], top-left-origin convention as Cavity.points.
    """

    points: List[Point2D]

    @validator("points")
    def min_points(cls, v: List[Point2D]) -> List[Point2D]:
        if len(v) < 3:
            raise ValueError("Island needs at least 3 points")
        return v


class Cavity(BaseModel):
    lengthIn: float
    widthIn: float
    depthIn: float
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)

    shape: Optional[str] = None
    diameterIn: Optional[float] = None

    cornerRadiusIn: Optional[float] = None
    corner_radius_in: Optional[float] = None

    # NEW: polygon cavities (normalized points, top-left origin)
    points: Optional[List[Point2D]] = None

    # NEW: islands (raised, uncut bosses within this cavity's footprint)
    islands: Optional[List[Island]] = None

    @validator("lengthIn", "widthIn", "depthIn")
    def positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Dimension must be > 0")
        return v


class FoamLayer(BaseModel):
    thicknessIn: float
    label: Optional[str] = None
    cavities: Optional[List[Cavity]] = None

    cropCorners: Optional[bool] = None
    crop_corners: Optional[bool] = None

    roundCorners: Optional[bool] = None
    round_corners: Optional[bool] = None
    roundRadiusIn: Optional[float] = None
    round_radius_in: Optional[float] = None

    @validator("thicknessIn")
    def positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Layer thickness must be > 0")
        return v


class Block(BaseModel):
    lengthIn: float
    widthIn: float
    thicknessIn: float

    croppedCorners: Optional[bool] = None
    cropped_corners: Optional[bool] = None

    chamferIn: Optional[float] = None
    chamfer_in: Optional[float] = None

    cornerStyle: Optional[str] = None
    corner_style: Optional[str] = None

    @validator("lengthIn", "widthIn", "thicknessIn")
    def positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Block dimensions must be > 0")
        return v

    @validator("chamferIn", "chamfer_in")
    def chamfer_positive(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if v <= 0:
            raise ValueError("Chamfer must be > 0 when provided")
        return v


class Layout(BaseModel):
    block: Block
    stack: List[FoamLayer]
    cavities: Optional[List[Cavity]] = None


class StepRequest(BaseModel):
    layout: Layout
    quoteNo: str
    materialLegend: Optional[str] = None


app = FastAPI(title="Alex-IO STEP microservice")


@app.get("/health")
async def health():
    return {"ok": True}


def _safe_pos(v: Optional[float]) -> Optional[float]:
    try:
        if v is None:
            return None
        n = float(v)
        return n if n > 0 else None
    except Exception:
        return None


def _truthy_bool(v: Optional[bool]) -> bool:
    return bool(v is True)


def _resolve_corner_style(block: Block) -> str:
    s = block.cornerStyle or block.corner_style
    return str(s).strip().lower() if s is not None else ""


def _resolve_cropped_global(block: Block) -> bool:
    if _truthy_bool(block.croppedCorners) or _truthy_bool(block.cropped_corners):
        return True
    corner_style = _resolve_corner_style(block)
    return corner_style == "chamfer"


def _resolve_cropped_for_layer(layer: FoamLayer, cropped_global: bool) -> bool:
    if layer.cropCorners is True or layer.crop_corners is True:
        return True
    if layer.cropCorners is False or layer.crop_corners is False:
        return False
    return bool(cropped_global)


def _resolve_chamfer_in(block: Block) -> float:
    v = _safe_pos(block.chamferIn) or _safe_pos(block.chamfer_in)
    return float(v) if v is not None else 1.0


def _resolve_round_for_layer(layer: FoamLayer) -> bool:
    return bool(layer.roundCorners is True or layer.round_corners is True)


def _resolve_round_radius_in(layer: FoamLayer) -> float:
    v = _safe_pos(layer.roundRadiusIn) or _safe_pos(layer.round_radius_in)
    return float(v) if v is not None else DEFAULT_OUTER_ROUND_IN


def _normalize_poly_points_mm(points, L_mm: float, W_mm: float) -> List[Tuple[float, float]]:
    """
    Convert a list of normalized [0..1] editor-space points (top-left
    origin) into CAD mm space (bottom-left origin, Y-up), matching the same
    convention used for cavity.points in build_cad_from_layout's poly-cavity
    branch. Clamps to block bounds and drops consecutive duplicate points.

    Used by island validation and island geometry construction. Kept as an
    independent implementation from the existing poly-cavity cut branch
    (deliberately not shared/refactored) so this addition cannot perturb the
    already-working cavity cut code path.
    """
    pts_mm: List[Tuple[float, float]] = []
    last = None
    for p in points:
        try:
            xn = float(p.x)
            yn = float(p.y)
        except Exception:
            continue

        xn = max(0.0, min(1.0, xn))
        yn = max(0.0, min(1.0, yn))

        x_mm = xn * L_mm
        y_mm = (1.0 - yn) * W_mm

        x_mm = max(0.0, min(L_mm, x_mm))
        y_mm = max(0.0, min(W_mm, y_mm))

        pt = (x_mm, y_mm)
        if last is None or pt != last:
            pts_mm.append(pt)
            last = pt

    return pts_mm


def _cavity_footprint_mm(cav: "Cavity", L_mm: float, W_mm: float) -> dict:
    """
    Approximate 2D footprint of a cavity's cut region in CAD mm space
    (bottom-left origin, Y-up), for the pre-flight validation pass ONLY.

    Deliberately mirrors -- but does not share code with -- the placement
    math in build_cad_from_layout, so validation can never perturb the
    existing, working cut/extrude code paths.

    Rounded-rect footprints are approximated by their sharp-corner bounding
    rectangle (slightly larger than the true filleted cut): a deliberately
    conservative approximation that can only make the guard MORE cautious
    near rounded corners, never less.
    """
    shape = (cav.shape or "rect").strip().lower()

    if shape == "circle":
        dia_in = cav.diameterIn or min(cav.lengthIn, cav.widthIn)
        r_mm = (float(dia_in) * INCH_TO_MM) / 2.0

        x_left = cav.x * L_mm
        y_top_cad = W_mm * (1.0 - cav.y) - (2.0 * r_mm)
        x_left = max(0.0, min(L_mm - 2.0 * r_mm, x_left))
        y_top_cad = max(0.0, min(W_mm - 2.0 * r_mm, y_top_cad))

        cx = x_left + r_mm
        cy = y_top_cad + r_mm
        return {"type": "circle", "cx": cx, "cy": cy, "r": r_mm}

    if shape == "poly" and cav.points is not None and len(cav.points) >= 3:
        pts_mm = _normalize_poly_points_mm(cav.points, L_mm, W_mm)
        if len(pts_mm) >= 3:
            return {"type": "poly", "pts": pts_mm}
        # degenerate poly points -> fall through to rect approximation below

    cav_L = float(cav.lengthIn) * INCH_TO_MM
    cav_W = float(cav.widthIn) * INCH_TO_MM

    x_left = cav.x * L_mm
    y_top_cad = W_mm * (1.0 - cav.y) - cav_W
    x_left = max(0.0, min(L_mm - cav_L, x_left))
    y_top_cad = max(0.0, min(W_mm - cav_W, y_top_cad))

    pts = [
        (x_left, y_top_cad),
        (x_left + cav_L, y_top_cad),
        (x_left + cav_L, y_top_cad + cav_W),
        (x_left, y_top_cad + cav_W),
    ]
    return {"type": "poly", "pts": pts}


def _point_in_poly(pt: Tuple[float, float], poly_pts: List[Tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test. poly_pts need not be closed."""
    pts = poly_pts[:-1] if (len(poly_pts) > 1 and poly_pts[0] == poly_pts[-1]) else poly_pts
    n = len(pts)
    if n < 3:
        return False

    x, y = pt
    inside = False
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_int = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_int:
                inside = not inside
    return inside


def _point_in_footprint(pt: Tuple[float, float], fp: dict) -> bool:
    if fp["type"] == "circle":
        dx = pt[0] - fp["cx"]
        dy = pt[1] - fp["cy"]
        return (dx * dx + dy * dy) <= (fp["r"] * fp["r"]) + 1e-9
    return _point_in_poly(pt, fp["pts"])


def _polygon_fully_inside(pts: List[Tuple[float, float]], fp: dict) -> bool:
    return all(_point_in_footprint(p, fp) for p in pts)


def _footprint_fully_inside(inner_fp: dict, outer_fp: dict) -> bool:
    """
    True if inner_fp is entirely inside outer_fp -- the legitimate
    "stepped pocket" case (a smaller/deeper cavity nested inside a larger,
    shallower one). Such pairs share no side wall and are exempt from the
    sibling min-wall gap check.
    """
    if inner_fp["type"] == "poly":
        return _polygon_fully_inside(inner_fp["pts"], outer_fp)

    # circle: sample the center plus 8 points around the circumference
    cx, cy, r = inner_fp["cx"], inner_fp["cy"], inner_fp["r"]
    sample_pts = [
        (cx + r * math.cos(a), cy + r * math.sin(a))
        for a in (i * math.pi / 4.0 for i in range(8))
    ]
    sample_pts.append((cx, cy))
    return all(_point_in_footprint(p, outer_fp) for p in sample_pts)


def _point_to_seg_dist(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 <= 1e-15:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    cx_, cy_ = ax + t * dx, ay + t * dy
    return math.hypot(px - cx_, py - cy_)


def _seg_seg_dist(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    q1: Tuple[float, float],
    q2: Tuple[float, float],
) -> float:
    """Minimum distance between segments p1-p2 and q1-q2 (2D). 0.0 if they cross/touch."""

    def ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])

    d1 = ccw(q1, q2, p1)
    d2 = ccw(q1, q2, p2)
    d3 = ccw(p1, p2, q1)
    d4 = ccw(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 != 0 and d2 != 0:
        return 0.0

    return min(
        _point_to_seg_dist(p1, q1, q2),
        _point_to_seg_dist(p2, q1, q2),
        _point_to_seg_dist(q1, p1, p2),
        _point_to_seg_dist(q2, p1, p2),
    )


def _poly_edges(pts: List[Tuple[float, float]]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    p = pts[:-1] if (len(pts) > 1 and pts[0] == pts[-1]) else list(pts)
    n = len(p)
    if n < 2:
        return []
    return [(p[i], p[(i + 1) % n]) for i in range(n)]


def _min_gap_mm(fp_a: dict, fp_b: dict) -> float:
    """
    Approximate minimum gap (mm) between two footprints. <= 0.0 if they
    overlap/touch. Heuristic for the min-wall guard, not exact CAD
    tolerancing.
    """
    if fp_a["type"] == "circle" and fp_b["type"] == "circle":
        d = math.hypot(fp_a["cx"] - fp_b["cx"], fp_a["cy"] - fp_b["cy"])
        return d - fp_a["r"] - fp_b["r"]

    if fp_a["type"] == "circle" or fp_b["type"] == "circle":
        circle_fp, poly_fp = (fp_a, fp_b) if fp_a["type"] == "circle" else (fp_b, fp_a)
        cx, cy, r = circle_fp["cx"], circle_fp["cy"], circle_fp["r"]
        edges = _poly_edges(poly_fp["pts"])
        if not edges:
            return float("inf")
        best = min(_point_to_seg_dist((cx, cy), a, b) for a, b in edges)
        if _point_in_poly((cx, cy), poly_fp["pts"]):
            return -best - r
        return best - r

    edges_a = _poly_edges(fp_a["pts"])
    edges_b = _poly_edges(fp_b["pts"])
    if not edges_a or not edges_b:
        return float("inf")
    return min(_seg_seg_dist(a1, a2, b1, b2) for a1, a2 in edges_a for b1, b2 in edges_b)


def validate_layout(layout: "Layout") -> List[str]:
    """
    Pre-flight validation, run BEFORE any CAD kernel calls.

    Hard blocks (raise ValueError, caller maps this to HTTP 400) -- these
    are genuinely invalid/malformed data, not a spacing judgment call:
      - an island with fewer than 3 distinct points after normalization
      - an island not fully contained within its parent cavity's footprint

    Non-fatal warnings (returned as a list of strings, export proceeds):
      - a wall/gap thinner than MIN_WALL_IN, between sibling cavities in the
        same layer, or between an island and its parent cavity's wall.
        This is a heuristic guard against known boolean-cut instability
        with closely-spaced features, not a correctness issue -- a
        real-world layout can legitimately need a wall this thin (see
        Q-AI-20260803-104756), so it's surfaced for a human to weigh
        rather than blocking export outright.

    Sibling cavities that are legitimately fully nested inside one another
    (the existing "stepped pocket" feature) are exempt from the sibling gap
    check -- there is no side wall between them, only a floor step.
    """
    L_mm = layout.block.lengthIn * INCH_TO_MM
    W_mm = layout.block.widthIn * INCH_TO_MM

    warnings: List[str] = []

    for layer_idx, layer in enumerate(layout.stack):
        cavities = layer.cavities or []
        layer_label = layer.label or f"layer[{layer_idx}]"

        footprints: List[dict] = []
        for cav_idx, cav in enumerate(cavities):
            fp = _cavity_footprint_mm(cav, L_mm, W_mm)
            footprints.append(fp)

            for isl_idx, isl in enumerate(cav.islands or []):
                isl_pts_mm = _normalize_poly_points_mm(isl.points, L_mm, W_mm)
                if len(isl_pts_mm) < 3:
                    raise ValueError(
                        f"{layer_label} cavity[{cav_idx}] island[{isl_idx}] has "
                        f"fewer than 3 distinct points after normalization"
                    )

                if not _polygon_fully_inside(isl_pts_mm, fp):
                    raise ValueError(
                        f"{layer_label} cavity[{cav_idx}] island[{isl_idx}] is not "
                        f"fully contained within its parent cavity's footprint"
                    )

                isl_fp = {"type": "poly", "pts": isl_pts_mm}
                gap_mm = _min_gap_mm(isl_fp, fp)
                if gap_mm < MIN_WALL_MM:
                    gap_in = gap_mm / INCH_TO_MM
                    warnings.append(
                        f"{layer_label} cavity[{cav_idx}] island[{isl_idx}] wall is "
                        f"{gap_in:.4f}in, thinner than the minimum {MIN_WALL_IN}in guard"
                    )

        for i in range(len(footprints)):
            for j in range(i + 1, len(footprints)):
                fp_a, fp_b = footprints[i], footprints[j]
                if _footprint_fully_inside(fp_a, fp_b) or _footprint_fully_inside(fp_b, fp_a):
                    continue  # legitimate nested/stepped pocket -- no side wall

                gap_mm = _min_gap_mm(fp_a, fp_b)
                if gap_mm < MIN_WALL_MM:
                    gap_in = gap_mm / INCH_TO_MM
                    warnings.append(
                        f"{layer_label} cavity[{i}] and cavity[{j}] are {gap_in:.4f}in "
                        f"apart, thinner than the minimum {MIN_WALL_IN}in guard"
                    )

    return warnings


def build_layer_block(
    L_mm: float,
    W_mm: float,
    T_mm: float,
    z: float,
    cropped: bool,
    chamfer_mm: float,
    rounded: bool,
    radius_mm: float,
):
    def _square():
        return (
            cq.Workplane("XY")
            .box(L_mm, W_mm, T_mm, centered=(False, False, False))
            .translate((0, 0, z))
        )

    if rounded:
        r = float(radius_mm) if radius_mm else 0.0
        if r > 0:
            max_r = (min(L_mm, W_mm) / 2.0) - FILLET_EPS_MM
            r = max(0.0, min(r, max_r)) if max_r > 0 else 0.0

        if r > 0:
            try:
                solid = (
                    cq.Workplane("XY")
                    .box(L_mm, W_mm, T_mm, centered=(False, False, False))
                    .edges("|Z")
                    .fillet(r)
                    .translate((0, 0, z))
                )
                if solid.val().Solids():
                    return solid
            except Exception:
                return _square()

        return _square()

    if not cropped:
        return _square()

    c = float(chamfer_mm)
    if not (c > 0):
        c = 1.0 * INCH_TO_MM

    if L_mm <= 2.0 * c or W_mm <= 2.0 * c:
        return _square()

    pts = [
        (0.0, 0.0),
        (L_mm - c, 0.0),
        (L_mm, c),
        (L_mm, W_mm),
        (c, W_mm),
        (0.0, W_mm - c),
    ]

    return (
        cq.Workplane("XY")
        .polyline(pts)
        .close()
        .extrude(T_mm)
        .translate((0, 0, z))
    )


def build_cad_from_layout(layout: Layout) -> cq.Workplane:
    L_mm = layout.block.lengthIn * INCH_TO_MM
    W_mm = layout.block.widthIn * INCH_TO_MM

    cropped_global = _resolve_cropped_global(layout.block)
    chamfer_in = _resolve_chamfer_in(layout.block)
    chamfer_mm = chamfer_in * INCH_TO_MM

    z_bottom = 0.0
    valid_solids: List[cq.Workplane] = []

    for idx, layer in enumerate(layout.stack):
        T_mm = layer.thicknessIn * INCH_TO_MM
        if T_mm <= 0:
            continue

        cropped_layer = _resolve_cropped_for_layer(layer, cropped_global)
        rounded_layer = _resolve_round_for_layer(layer)
        radius_in = _resolve_round_radius_in(layer)
        radius_mm = float(radius_in) * INCH_TO_MM if radius_in else 0.0

        base = build_layer_block(
            L_mm,
            W_mm,
            T_mm,
            z_bottom,
            cropped=cropped_layer,
            chamfer_mm=chamfer_mm,
            rounded=rounded_layer,
            radius_mm=radius_mm,
        )
        working = base

        cavities = list(layer.cavities or [])

        def _eff_depth_mm(c: Cavity) -> float:
            try:
                d_mm = float(c.depthIn) * INCH_TO_MM
            except Exception:
                return 0.0
            if d_mm >= T_mm:
                return T_mm + THROUGH_CUT_EPS_MM
            return min(d_mm, T_mm * DEPTH_CLAMP_RATIO)

        cavities.sort(key=_eff_depth_mm)

        for cav in cavities:
            try:
                req_mm = float(cav.depthIn) * INCH_TO_MM
            except Exception:
                continue

            cav_D = (T_mm + THROUGH_CUT_EPS_MM) if req_mm >= T_mm else min(req_mm, T_mm * DEPTH_CLAMP_RATIO)
            shape = (cav.shape or "rect").strip().lower()
            corner_r_in = _safe_pos(cav.cornerRadiusIn) or _safe_pos(cav.corner_radius_in) or None

            z_top = z_bottom + T_mm
            z_cut = z_top - cav_D

            try:
                if shape == "circle":
                    dia_in = cav.diameterIn or min(cav.lengthIn, cav.widthIn)
                    dia_mm = float(dia_in) * INCH_TO_MM
                    r_mm = dia_mm / 2.0

                    x_left = cav.x * L_mm
                    y_top_cad = W_mm * (1.0 - cav.y) - (2.0 * r_mm)

                    x_left = max(0.0, min(L_mm - 2.0 * r_mm, x_left))
                    y_top_cad = max(0.0, min(W_mm - 2.0 * r_mm, y_top_cad))

                    cx = x_left + r_mm
                    cy = y_top_cad + r_mm

                    cavity = (
                        cq.Workplane("XY")
                        .workplane(offset=z_cut)
                        .center(cx, cy)
                        .circle(r_mm)
                        .extrude(cav_D)
                    )

                elif shape == "poly" and cav.points is not None and len(cav.points) >= 3:
                    # Points are normalized [0..1] in editor space (top-left origin).
                    # Convert to CAD XY in mm with bottom-left origin (Y-up).
                    pts_mm: List[tuple] = []

                    last = None
                    for p in cav.points:
                        try:
                            xn = float(p.x)
                            yn = float(p.y)
                        except Exception:
                            continue

                        if xn < 0.0:
                            xn = 0.0
                        if xn > 1.0:
                            xn = 1.0
                        if yn < 0.0:
                            yn = 0.0
                        if yn > 1.0:
                            yn = 1.0

                        x_mm = xn * L_mm
                        y_mm = (1.0 - yn) * W_mm  # flip Y

                        # clamp to block bounds
                        x_mm = max(0.0, min(L_mm, x_mm))
                        y_mm = max(0.0, min(W_mm, y_mm))

                        pt = (x_mm, y_mm)
                        if last is None or pt != last:
                            pts_mm.append(pt)
                            last = pt

                    # Need at least 3 distinct points
                    if len(pts_mm) >= 3:
                        cavity = (
                            cq.Workplane("XY")
                            .workplane(offset=z_cut)
                            .polyline(pts_mm)
                            .close()
                            .extrude(cav_D)
                        )
                    else:
                        # Fallback to rect if points are degenerate
                        cav_L = float(cav.lengthIn) * INCH_TO_MM
                        cav_W = float(cav.widthIn) * INCH_TO_MM
                        if cav_L >= L_mm or cav_W >= W_mm:
                            continue

                        x_left = cav.x * L_mm
                        y_top_cad = W_mm * (1.0 - cav.y) - cav_W

                        x_left = max(0.0, min(L_mm - cav_L, x_left))
                        y_top_cad = max(0.0, min(W_mm - cav_W, y_top_cad))

                        cavity = (
                            cq.Workplane("XY")
                            .box(cav_L, cav_W, cav_D, centered=(False, False, False))
                            .translate((x_left, y_top_cad, z_cut))
                        )

                else:
                    cav_L = float(cav.lengthIn) * INCH_TO_MM
                    cav_W = float(cav.widthIn) * INCH_TO_MM
                    if cav_L >= L_mm or cav_W >= W_mm:
                        continue

                    x_left = cav.x * L_mm
                    y_top_cad = W_mm * (1.0 - cav.y) - cav_W

                    x_left = max(0.0, min(L_mm - cav_L, x_left))
                    y_top_cad = max(0.0, min(W_mm - cav_W, y_top_cad))

                    r_mm = 0.0
                    if corner_r_in is not None and corner_r_in > 0:
                        r_mm = float(corner_r_in) * INCH_TO_MM
                        max_r = (min(cav_L, cav_W) / 2.0) - FILLET_EPS_MM
                        r_mm = max(0.0, min(r_mm, max_r)) if max_r > 0 else 0.0

                    if shape in ("roundedrect", "rounded_rect", "rounded-rect") or r_mm > 0:
                        cx = x_left + (cav_L / 2.0)
                        cy = y_top_cad + (cav_W / 2.0)

                        cavity = (
                            cq.Workplane("XY")
                            .workplane(offset=z_cut)
                            .center(cx, cy)
                            .rect(cav_L, cav_W)
                            .extrude(cav_D)
                        )
                        if r_mm > 0:
                            cavity = cavity.edges("|Z").fillet(r_mm)
                    else:
                        cavity = (
                            cq.Workplane("XY")
                            .box(cav_L, cav_W, cav_D, centered=(False, False, False))
                            .translate((x_left, y_top_cad, z_cut))
                        )

                cut_result = working.cut(cavity)
                if cut_result.val().Solids():
                    working = cut_result

                    # ISLAND SUPPORT: cut-then-union-back. Each island is
                    # extruded from this cavity's own floor (z_cut) back up
                    # to the layer's original top (z_top) and unioned into
                    # `working` immediately -- before the loop moves on to
                    # the next cavity -- so a later sibling cavity can never
                    # slice into an island before it's restored.
                    #
                    # Clamp the island's base to z_bottom: for a THROUGH cut,
                    # z_cut is intentionally pushed below z_bottom (by
                    # THROUGH_CUT_EPS_MM) to guarantee a clean exit face on
                    # the cut. Without this clamp, the island union would
                    # inherit that same overshoot and leave a sliver of
                    # material protruding below the layer's actual bottom
                    # face.
                    isl_z_base = max(z_cut, z_bottom)
                    isl_height = z_top - isl_z_base

                    for isl in (cav.islands or []):
                        try:
                            if isl_height <= 0:
                                continue

                            isl_pts_mm = _normalize_poly_points_mm(isl.points, L_mm, W_mm)
                            if len(isl_pts_mm) < 3:
                                continue

                            island_solid = (
                                cq.Workplane("XY")
                                .workplane(offset=isl_z_base)
                                .polyline(isl_pts_mm)
                                .close()
                                .extrude(isl_height)
                            )

                            union_result = working.union(island_solid)
                            if union_result.val().Solids():
                                working = union_result
                        except Exception:
                            continue

            except Exception:
                continue

        if not working.val().Solids():
            working = base

        valid_solids.append(working)
        z_bottom += T_mm

    if not valid_solids:
        raise ValueError("No valid solids generated from layout")

    solid = valid_solids[0]
    for other in valid_solids[1:]:
        solid = solid.union(other)

    if not solid.val().Solids():
        raise ValueError("Final solid is empty after union")

    return solid


def stl_to_faces_json(stl_bytes: bytes):
    """
    STL → faces_json extraction (top-face boundary edges) with:
    - Corner healing via vertex snapping (tolerance grid)
    - Reduced tolerance (1/64") to avoid collapsing small cavities
    - Angle-based loop walking
    - Junction-aware bridge split to handle “throat” connections
    - Snap+simplify+canonical signature de-dupe
    """
    from collections import defaultdict
    import math

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        stl_path = tmp.name
        tmp.write(stl_bytes)

    try:
        m = stlmesh.Mesh.from_file(stl_path)
        tris = m.vectors

        top_tris = []
        for tri in tris:
            v1, v2, v3 = tri
            ux, uy, uz = v2 - v1
            vx, vy, vz = v3 - v1
            nz = ux * vy - uy * vx
            if float(nz) > 0.0:
                top_tris.append(tri)

        if not top_tris:
            raise ValueError("No upward-facing triangles found")

        planes = defaultdict(list)
        for tri in top_tris:
            z = float((tri[0][2] + tri[1][2] + tri[2][2]) / 3.0)
            planes[round(z, 4)].append(tri)

        def tri_area(t):
            a, b, c = t
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            cx, cy = float(c[0]), float(c[1])
            return abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) * 0.5

        plane_tris = max(planes.values(), key=lambda lst: sum(tri_area(t) for t in lst))

        raw_xs = []
        raw_ys = []
        for tri in plane_tris:
            raw_xs.extend([float(tri[0][0]), float(tri[1][0]), float(tri[2][0])])
            raw_ys.extend([float(tri[0][1]), float(tri[1][1]), float(tri[2][1])])

        if not raw_xs or not raw_ys:
            raise ValueError("Selected top plane has no XY vertices")

        raw_span = max(max(raw_xs) - min(raw_xs), max(raw_ys) - min(raw_ys))
        assume_mm = float(raw_span) > 120.0
        scale = (1.0 / INCH_TO_MM) if assume_mm else 1.0

        tol_native = float(STL_HEAL_TOL_IN) * (INCH_TO_MM if assume_mm else 1.0)
        if not (tol_native > 0.0):
            tol_native = 1e-6

        min_edge = 0.15 * tol_native

        edge_count = defaultdict(int)
        canon = {}

        def snap_xy(x, y):
            qx = int(round(float(x) / tol_native))
            qy = int(round(float(y) / tol_native))
            key = (qx, qy)
            if key in canon:
                return canon[key]
            sx = float(qx) * tol_native
            sy = float(qy) * tol_native
            canon[key] = (sx, sy)
            return canon[key]

        def edge_key(a, b):
            ax, ay = snap_xy(a[0], a[1])
            bx, by = snap_xy(b[0], b[1])
            if math.hypot(bx - ax, by - ay) < min_edge:
                return None
            p1 = (float(ax), float(ay))
            p2 = (float(bx), float(by))
            return tuple(sorted((p1, p2)))

        for tri in plane_tris:
            pts = [
                (tri[0][0], tri[0][1]),
                (tri[1][0], tri[1][1]),
                (tri[2][0], tri[2][1]),
            ]
            for i in range(3):
                e = edge_key(pts[i], pts[(i + 1) % 3])
                if e is None:
                    continue
                edge_count[e] += 1

        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if not boundary_edges:
            raise ValueError("No boundary edges found on selected top plane")

        from collections import defaultdict as _dd

        adj = _dd(list)
        for a, b in boundary_edges:
            adj[a].append(b)
            adj[b].append(a)

        def prune_spurs(edges_in):
            if not edges_in:
                return []
            g = _dd(set)
            for a2, b2 in edges_in:
                g[a2].add(b2)
                g[b2].add(a2)

            leaves = [v for v, nbs in g.items() if len(nbs) == 1]
            while leaves:
                v = leaves.pop()
                if v not in g or len(g[v]) != 1:
                    continue
                u = next(iter(g[v]))
                g[u].discard(v)
                g[v].discard(u)
                if v in g and len(g[v]) == 0:
                    del g[v]
                if u in g and len(g[u]) == 1:
                    leaves.append(u)
                if u in g and len(g[u]) == 0:
                    del g[u]

            out = []
            seen = set()
            for a2, nbs in g.items():
                for b2 in nbs:
                    e = (a2, b2) if a2 <= b2 else (b2, a2)
                    if e in seen:
                        continue
                    seen.add(e)
                    out.append(e)
            return out

        def turn_angle(prev_pt, cur_pt, nxt_pt):
            ax = cur_pt[0] - prev_pt[0]
            ay = cur_pt[1] - prev_pt[1]
            bx = nxt_pt[0] - cur_pt[0]
            by = nxt_pt[1] - cur_pt[1]
            cross = ax * by - ay * bx
            dot = ax * bx + ay * by
            ang = math.atan2(cross, dot)
            if ang < 0:
                ang += 2.0 * math.pi
            return ang

        def extract_loops_from_adj(adj_in):
            def walk(prefer: str):
                used_dir = set()
                loops_out = []
                edge_ct = sum(len(v) for v in adj_in.values()) // 2
                max_steps = max(1000, edge_ct * 2)

                def dir_id(p, q):
                    return (p, q)

                for start in list(adj_in.keys()):
                    for nxt in adj_in[start]:
                        if dir_id(start, nxt) in used_dir:
                            continue

                        loop = [start, nxt]
                        used_dir.add(dir_id(start, nxt))

                        prev = start
                        cur = nxt
                        steps = 0

                        while steps < max_steps:
                            steps += 1
                            nbrs = adj_in[cur]
                            if not nbrs:
                                break

                            cands = [p for p in nbrs if p != prev]
                            if not cands:
                                cands = [prev]

                            best = None
                            best_ang = None
                            for cand in cands:
                                if dir_id(cur, cand) in used_dir:
                                    continue
                                ang = turn_angle(prev, cur, cand)
                                if best is None:
                                    best = cand
                                    best_ang = ang
                                else:
                                    if prefer == "min":
                                        if ang < best_ang:
                                            best = cand
                                            best_ang = ang
                                    else:
                                        if ang > best_ang:
                                            best = cand
                                            best_ang = ang

                            if best is None:
                                break

                            used_dir.add(dir_id(cur, best))
                            loop.append(best)
                            prev, cur = cur, best

                            if cur == start:
                                break

                        if len(loop) >= 4 and loop[0] == loop[-1]:
                            loops_out.append(loop)

                return loops_out

            def canonicalize_loop_pts(loop_pts):
                pts = loop_pts[:-1] if (len(loop_pts) > 1 and loop_pts[0] == loop_pts[-1]) else list(loop_pts)
                if len(pts) < 3:
                    return loop_pts

                def rotate_to_min(seq):
                    mi = min(range(len(seq)), key=lambda i: (seq[i][0], seq[i][1]))
                    return seq[mi:] + seq[:mi]

                fwd = rotate_to_min(pts)
                rev = rotate_to_min(list(reversed(pts)))
                rep_fwd = tuple((round(p[0], 6), round(p[1], 6)) for p in fwd)
                rep_rev = tuple((round(p[0], 6), round(p[1], 6)) for p in rev)
                chosen = fwd if rep_fwd <= rep_rev else rev
                return chosen + [chosen[0]]

            all_loops = []
            all_loops.extend(walk("min"))
            all_loops.extend(walk("max"))

            uniq = {}
            for lp in all_loops:
                can = canonicalize_loop_pts(lp)
                sig = tuple((round(p[0], 6), round(p[1], 6)) for p in can)
                if sig not in uniq:
                    uniq[sig] = can

            return list(uniq.values())

        def _loop_area(loop_pts):
            pts = loop_pts[:-1] if (len(loop_pts) > 1 and loop_pts[0] == loop_pts[-1]) else list(loop_pts)
            if len(pts) < 3:
                return 0.0
            a2 = 0.0
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % len(pts)]
                a2 += x1 * y2 - x2 * y1
            return abs(a2) * 0.5

        # ---------------------------------------------------------------------
        # Heal small "leaf gaps" BEFORE spur pruning.
        # ---------------------------------------------------------------------
        def heal_leaf_gaps(adj_in):
            if not adj_in:
                return adj_in

            g = _dd(set)
            for a0, nbs0 in adj_in.items():
                for b0 in nbs0:
                    g[a0].add(b0)

            def to_list_adj(g_in):
                out2 = _dd(list)
                for a0, nbs0 in g_in.items():
                    out2[a0] = list(nbs0)
                return out2

            def leaves(g_in):
                return [v for v, nbs0 in g_in.items() if len(nbs0) == 1]

            base_loops = extract_loops_from_adj(to_list_adj(g))
            base_count = len(base_loops)
            base_outer_area = max((_loop_area(lp) for lp in base_loops), default=0.0)

            axis_eps = 2.0 * tol_native
            max_bridge = max(3.0 * tol_native, 0.05 * float(raw_span))

            changed = True
            guard = 0
            while changed and guard < 25:
                guard += 1
                changed = False

                lf = leaves(g)
                if len(lf) < 2:
                    break

                best = None  # (dist, a, b)
                for i in range(len(lf)):
                    a = lf[i]
                    for j in range(i + 1, len(lf)):
                        b = lf[j]
                        dx = float(b[0]) - float(a[0])
                        dy = float(b[1]) - float(a[1])

                        if not (abs(dx) <= axis_eps or abs(dy) <= axis_eps):
                            continue

                        d = math.hypot(dx, dy)
                        if d <= 0.0 or d > max_bridge:
                            continue

                        if best is None or d < best[0]:
                            best = (d, a, b)

                if best is None:
                    break

                _, a, b = best

                g[a].add(b)
                g[b].add(a)

                test_loops = extract_loops_from_adj(to_list_adj(g))
                test_count = len(test_loops)
                test_outer = max((_loop_area(lp) for lp in test_loops), default=0.0)

                ok_outer = (base_outer_area <= 0.0) or (test_outer >= (base_outer_area * 0.98))
                ok_gain = test_count > base_count

                if ok_outer and ok_gain:
                    base_count = test_count
                    base_outer_area = max(base_outer_area, test_outer)
                    changed = True
                else:
                    g[a].discard(b)
                    g[b].discard(a)

            return to_list_adj(g)

        adj = heal_leaf_gaps(adj)

        # Convert adj -> edges then prune spurs
        boundary_edges2 = []
        seen = set()
        for a0, nbs0 in adj.items():
            for b0 in nbs0:
                e = (a0, b0) if a0 <= b0 else (b0, a0)
                if e in seen:
                    continue
                seen.add(e)
                boundary_edges2.append(e)

        pruned_edges = prune_spurs(boundary_edges2)
        if pruned_edges:
            adj = _dd(list)
            for a, b in pruned_edges:
                adj[a].append(b)
                adj[b].append(a)

        def bridge_split_junctions(adj_in):
            g = _dd(set)
            for a0, nbs0 in adj_in.items():
                for b0 in nbs0:
                    g[a0].add(b0)

            def to_list_adj(g_in):
                out2 = _dd(list)
                for a0, nbs0 in g_in.items():
                    out2[a0] = list(nbs0)
                return out2

            base_loops = extract_loops_from_adj(to_list_adj(g))
            base_count = len(base_loops)
            base_outer_area = max((_loop_area(lp) for lp in base_loops), default=0.0)

            max_iters = 25
            it = 0
            changed = True

            while changed and it < max_iters:
                it += 1
                changed = False
                for v in list(g.keys()):
                    if v not in g or len(g[v]) < 3:
                        continue
                    for u in list(g[v]):
                        if u not in g or v not in g[u]:
                            continue

                        g[v].discard(u)
                        g[u].discard(v)
                        if v in g and len(g[v]) == 0:
                            del g[v]
                        if u in g and len(g[u]) == 0:
                            del g[u]

                        test_loops = extract_loops_from_adj(to_list_adj(g))
                        if len(test_loops) > base_count:
                            test_outer = max((_loop_area(lp) for lp in test_loops), default=0.0)

                            if base_outer_area > 0 and test_outer < (base_outer_area * 0.98):
                                if v not in g:
                                    g[v] = set()
                                if u not in g:
                                    g[u] = set()
                                g[v].add(u)
                                g[u].add(v)
                            else:
                                base_count = len(test_loops)
                                base_outer_area = max(base_outer_area, test_outer)
                                changed = True
                        else:
                            if v not in g:
                                g[v] = set()
                            if u not in g:
                                g[u] = set()
                            g[v].add(u)
                            g[u].add(v)

            return to_list_adj(g)

        adj = bridge_split_junctions(adj)
        loops_native = extract_loops_from_adj(adj)
        if not loops_native:
            raise ValueError("Failed to assemble loops from boundary edges")

        def poly_area_xy(pts_xy):
            if len(pts_xy) < 3:
                return 0.0
            pts = pts_xy
            if pts[0] != pts[-1]:
                pts = pts_xy + [pts_xy[0]]
            a2 = 0.0
            for i in range(len(pts) - 1):
                a2 += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]
            return abs(a2) * 0.5

        # Shift to (0,0) (native), then scale to inches
        xs = [p[0] for loop in loops_native for p in loop]
        ys = [p[1] for loop in loops_native for p in loop]
        min_x, min_y = min(xs), min(ys)

        loops_in = []
        for loop in loops_native:
            pts_in = [((p[0] - min_x) * scale, (p[1] - min_y) * scale) for p in loop]
            loops_in.append(pts_in)

        # ---------------------------------------------------------------------
        # 8) Snap + simplify + canonical signature de-dup
        # ---------------------------------------------------------------------
        snap_in = max(STL_HEAL_TOL_IN * 0.25, 1e-6)  # inches
        col_eps = snap_in * snap_in  # cross-product threshold scale

        def _snap_pt(p):
            return (
                round(float(p[0]) / snap_in) * snap_in,
                round(float(p[1]) / snap_in) * snap_in,
            )

        def _simplify_closed(loop_pts):
            if not loop_pts or len(loop_pts) < 4:
                return loop_pts

            pts = loop_pts[:-1] if loop_pts[0] == loop_pts[-1] else list(loop_pts)

            pts = [_snap_pt(p) for p in pts]

            ded = []
            for p in pts:
                if not ded or p != ded[-1]:
                    ded.append(p)
            pts = ded

            if len(pts) < 3:
                return loop_pts

            changed = True
            guard = 0
            while changed and guard < 10:
                guard += 1
                changed = False
                if len(pts) <= 3:
                    break

                out = []
                n = len(pts)
                for i in range(n):
                    a = pts[(i - 1) % n]
                    b = pts[i]
                    c = pts[(i + 1) % n]

                    abx = b[0] - a[0]
                    aby = b[1] - a[1]
                    bcx = c[0] - b[0]
                    bcy = c[1] - b[1]

                    cross = abx * bcy - aby * bcx
                    if abs(cross) <= col_eps:
                        changed = True
                        continue
                    out.append(b)

                if len(out) >= 3:
                    pts = out
                else:
                    break

            pts = pts + [pts[0]]
            return pts

        def _canonicalize(loop_pts):
            if len(loop_pts) < 4 or loop_pts[0] != loop_pts[-1]:
                return loop_pts

            pts = loop_pts[:-1]
            if len(pts) < 3:
                return loop_pts

            def rotate_to_min(seq):
                mi = min(range(len(seq)), key=lambda i: (seq[i][0], seq[i][1]))
                return seq[mi:] + seq[:mi]

            fwd = rotate_to_min(pts)
            rev = rotate_to_min(list(reversed(pts)))

            rep_fwd = tuple((round(x, 6), round(y, 6)) for (x, y) in fwd)
            rep_rev = tuple((round(x, 6), round(y, 6)) for (x, y) in rev)

            chosen = fwd if rep_fwd <= rep_rev else rev
            return chosen + [chosen[0]]

        dedup = {}
        for lp in loops_in:
            simp = _simplify_closed(lp)
            can = _canonicalize(simp)
            if len(can) < 4 or can[0] != can[-1]:
                continue
            sig = tuple((round(p[0], 6), round(p[1], 6)) for p in can)
            if sig not in dedup:
                dedup[sig] = can

        loops_can = list(dedup.values())
        if not loops_can:
            raise ValueError("No valid loops after simplify/dedupe")

        # ---------------------------------------------------------------------
        # FINAL RULE (approved):
        # - Interior loops are NEVER dropped by area filtering.
        # - Area threshold is used ONLY to select/validate the OUTER loop index.
        # ---------------------------------------------------------------------
        areas_can = [poly_area_xy(l) for l in loops_can]
        outer_idx = int(max(range(len(loops_can)), key=lambda i: areas_can[i]))

        # If the chosen outer is microscopic (noise), attempt to pick the next-largest
        # that clears the threshold. Do NOT drop interior loops.
        min_area_in2 = (STL_HEAL_TOL_IN * STL_HEAL_TOL_IN) * 0.10
        if areas_can[outer_idx] < min_area_in2:
            candidates = [i for i, a in enumerate(areas_can) if a >= min_area_in2]
            if candidates:
                outer_idx = int(max(candidates, key=lambda i: areas_can[i]))

        out_loops = []
        for idx, loop in enumerate(loops_can):
            pts = [{"x": float(p[0]), "y": float(p[1])} for p in loop]
            out_loops.append({"idx": idx, "closed": True, "points": pts})

        return {"units": "in", "outerLoopIndex": outer_idx, "loops": out_loops}

    finally:
        try:
            os.remove(stl_path)
        except OSError:
            pass


def export_step_text(solid: cq.Workplane) -> str:
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
        path = tmp.name

    try:
        cq.exporters.export(solid, path)
        with open(path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

    return data.decode("utf-8", errors="ignore")


@app.post("/faces-from-stl")
async def faces_from_stl(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty STL upload")

        faces = stl_to_faces_json(data)
        return {"ok": True, "faces_json": faces}
    except HTTPException:
        raise
    except Exception as exc:
        print("[STEP-SVC] STL faces error:", repr(exc))
        raise HTTPException(status_code=400, detail=f"Failed to extract faces from STL: {exc}")


@app.post("/step-from-layout")
async def step_from_layout(payload: StepRequest):
    try:
        warnings = validate_layout(payload.layout)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Layout validation failed: {exc}")

    try:
        solid = build_cad_from_layout(payload.layout)
        step_text = export_step_text(solid)
    except Exception as exc:
        print("[STEP-SVC] Geometry error:", repr(exc))
        raise HTTPException(status_code=400, detail=f"Failed to build STEP geometry: {exc}")

    if not step_text.strip():
        raise HTTPException(500, "STEP export produced empty text")

    return {
        "ok": True,
        "step": step_text,
        "quoteNo": payload.quoteNo,
        "materialLegend": payload.materialLegend,
        "warnings": warnings,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
