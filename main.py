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

from typing import List, Optional
import os
import tempfile

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field, validator
import cadquery as cq
from stl import mesh as stlmesh

INCH_TO_MM = 25.4

# Legacy behavior: never cut full thickness (blind pockets)
DEPTH_CLAMP_RATIO = 0.95

# Through-cut epsilon: ensures the cut cleanly exits the bottom face
THROUGH_CUT_EPS_MM = 0.25

# Small epsilon to avoid fillet edge-case when radius ~= half the side
FILLET_EPS_MM = 1e-3

# Default outer round radius (inches) when roundCorners is true but radius omitted/invalid
DEFAULT_OUTER_ROUND_IN = 0.25


class Cavity(BaseModel):
    lengthIn: float
    widthIn: float
    depthIn: float
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)

    shape: Optional[str] = None           # "rect" | "circle" | "roundedRect" (optional)
    diameterIn: Optional[float] = None    # for circle cavities

    # NEW (Path A): optional rounded-rect radius (inches)
    cornerRadiusIn: Optional[float] = None
    corner_radius_in: Optional[float] = None

    @validator("lengthIn", "widthIn", "depthIn")
    def positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Dimension must be > 0")
        return v


class FoamLayer(BaseModel):
    thicknessIn: float
    label: Optional[str] = None
    cavities: Optional[List[Cavity]] = None

    # Per-layer cropped corner override (optional)
    cropCorners: Optional[bool] = None
    crop_corners: Optional[bool] = None

    # NEW: per-layer rounded outer corners (optional)
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

    # Global crop-corners intent
    croppedCorners: Optional[bool] = None
    cropped_corners: Optional[bool] = None

    # Keep chamfer size support (inches)
    chamferIn: Optional[float] = None
    chamfer_in: Optional[float] = None

    # Accept "cornerStyle" to match SVG/DXF export wiring
    cornerStyle: Optional[str] = None        # "square" | "chamfer"
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
    """
    Per-layer override:
      - If layer explicitly sets cropCorners true/false, honor it
      - If omitted (None), fall back to global intent
    """
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
    """
    Build one layer block.

    Coordinates are CAD bottom-left origin.
    Base block is always from (0,0, z) to (L, W, z+T).

    Precedence:
      - If rounded: fillet vertical edges (SAFE fallback if fillet fails)
      - Else if cropped: chamfer two corners (UL and LR) using polygon profile
      - Else: square box
    """
    # Always start from a square base in the correct coordinate system
    def _square():
        return (
            cq.Workplane("XY")
            .box(L_mm, W_mm, T_mm, centered=(False, False, False))
            .translate((0, 0, z))
        )

    # Rounded outer corners (takes precedence over crop)
    if rounded:
        r = float(radius_mm) if radius_mm else 0.0
        if r > 0:
            max_r = (min(L_mm, W_mm) / 2.0) - FILLET_EPS_MM
            if max_r > 0:
                r = max(0.0, min(r, max_r))
            else:
                r = 0.0

        if r > 0:
            try:
                # Build square block at (0,0) then fillet vertical edges only
                solid = (
                    cq.Workplane("XY")
                    .box(L_mm, W_mm, T_mm, centered=(False, False, False))
                    .edges("|Z")
                    .fillet(r)
                    .translate((0, 0, z))
                )
                # Ensure we still have a solid; otherwise fallback
                if solid.val().Solids():
                    return solid
            except Exception:
                # SAFE fallback
                return _square()

        # If radius invalid/zero, just return square
        return _square()

    # Square block
    if not cropped:
        return _square()

    # Cropped corners (existing behavior)
    c = float(chamfer_mm)
    if not (c > 0):
        c = 1.0 * INCH_TO_MM

    if L_mm <= 2.0 * c or W_mm <= 2.0 * c:
        return _square()

    # Polygon with chamfers at:
    #  - LR corner (L,0) => (L-c,0) -> (L,c)
    #  - UL corner (0,W) => (0,W-c) -> (c,W)
    pts = [
        (0.0, 0.0),
        (L_mm - c, 0.0),
        (L_mm, c),
        (L_mm, W_mm),
        (c, W_mm),
        (0.0, W_mm - c),
    ]

    solid = (
        cq.Workplane("XY")
        .polyline(pts)
        .close()
        .extrude(T_mm)
        .translate((0, 0, z))
    )
    return solid


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

        # IMPORTANT: per-layer STEP must only use per-layer cavities.
        cavities = list(layer.cavities or [])

        # Compute effective depth used for ordering:
        # - shallow first
        # - deep last
        # - through-cuts treated as "deepest"
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
            # Determine cut depth:
            # - Through-cut if requested depth >= thickness
            # - Else legacy blind pocket clamp
            try:
                req_mm = float(cav.depthIn) * INCH_TO_MM
            except Exception:
                continue

            if req_mm >= T_mm:
                cav_D = T_mm + THROUGH_CUT_EPS_MM
            else:
                cav_D = min(req_mm, T_mm * DEPTH_CLAMP_RATIO)

            shape = (cav.shape or "rect").strip().lower()

            corner_r_in = _safe_pos(cav.cornerRadiusIn) or _safe_pos(cav.corner_radius_in) or None

            z_top = z_bottom + T_mm
            z_cut = z_top - cav_D

            try:
                if shape == "circle":
                    dia_in = cav.diameterIn or min(cav.lengthIn, cav.widthIn)
                    dia_mm = float(dia_in) * INCH_TO_MM
                    r_mm = dia_mm / 2.0

                    # Convert editor top-left normalized coords -> CAD bottom-left coords
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

                else:
                    cav_L = float(cav.lengthIn) * INCH_TO_MM
                    cav_W = float(cav.widthIn) * INCH_TO_MM

                    if cav_L >= L_mm or cav_W >= W_mm:
