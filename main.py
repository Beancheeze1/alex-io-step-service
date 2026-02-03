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
            if max_r > 0:
                r = max(0.0, min(r, max_r))
            else:
                r = 0.0

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
                        continue

                    x_left = cav.x * L_mm
                    y_top_cad = W_mm * (1.0 - cav.y) - cav_W

                    x_left = max(0.0, min(L_mm - cav_L, x_left))
                    y_top_cad = max(0.0, min(W_mm - cav_W, y_top_cad))

                    r_mm = 0.0
                    if corner_r_in is not None and corner_r_in > 0:
                        r_mm = float(corner_r_in) * INCH_TO_MM
                        max_r = (min(cav_L, cav_W) / 2.0) - FILLET_EPS_MM
                        if max_r <= 0:
                            r_mm = 0.0
                        else:
                            r_mm = max(0.0, min(r_mm, max_r))

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


def _convex_hull_2d(points):
    """
    Monotonic chain convex hull.
    points: list of (x,y)
    returns hull list in CCW order without repeating first point
    """
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def stl_to_faces_json(stl_bytes: bytes):
    """
    Forge-equivalent STL → faces_json extraction.

    Key fix:
    - Ensure all returned numbers are plain Python float (no numpy.float32),
      so FastAPI can JSON-encode the response.
    """
    from collections import defaultdict

    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as tmp:
        stl_path = tmp.name
        tmp.write(stl_bytes)

    try:
        m = stlmesh.Mesh.from_file(stl_path)
        tris = m.vectors  # (n, 3, 3) numpy floats

        # 1) Upward-facing triangles
        top_tris = []
        for tri in tris:
            v1, v2, v3 = tri
            ux, uy, uz = v2 - v1
            vx, vy, vz = v3 - v1
            nz = ux * vy - uy * vx  # z-component of cross product
            if float(nz) > 0.0:
                top_tris.append(tri)

        if not top_tris:
            raise ValueError("No upward-facing triangles found")

        # 2) Group by Z plane (coplanar-ish)
        planes = defaultdict(list)
        for tri in top_tris:
            z = float((tri[0][2] + tri[1][2] + tri[2][2]) / 3.0)
            planes[round(z, 4)].append(tri)

        # 3) Choose largest plane by projected XY area
        def tri_area(t):
            a, b, c = t
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            cx, cy = float(c[0]), float(c[1])
            return abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) * 0.5

        plane_tris = max(planes.values(), key=lambda lst: sum(tri_area(t) for t in lst))

        # 4) Count edges; boundary edges appear once
        edge_count = defaultdict(int)

        def edge_key(a, b):
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            p1 = (round(ax, 5), round(ay, 5))
            p2 = (round(bx, 5), round(by, 5))
            return tuple(sorted((p1, p2)))

        for tri in plane_tris:
            pts = [
                (tri[0][0], tri[0][1]),
                (tri[1][0], tri[1][1]),
                (tri[2][0], tri[2][1]),
            ]
            for i in range(3):
                e = edge_key(pts[i], pts[(i + 1) % 3])
                edge_count[e] += 1

        boundary_edges = [e for e, c in edge_count.items() if c == 1]
        if not boundary_edges:
            raise ValueError("No boundary edges found on selected top plane")

        # 5) Assemble loops from boundary edge graph
        from collections import defaultdict as _dd

        adj = _dd(list)
        for a, b in boundary_edges:
            adj[a].append(b)
            adj[b].append(a)

        used = set()

        def edge_id(p, q):
            return tuple(sorted((p, q)))

        loops = []

        for start in list(adj.keys()):
            for nxt in adj[start]:
                eid = edge_id(start, nxt)
                if eid in used:
                    continue

                loop = [start]
                prev = None
                cur = start
                neighbor = nxt

                while True:
                    used.add(edge_id(cur, neighbor))
                    loop.append(neighbor)

                    prev, cur = cur, neighbor
                    nbs = adj[cur]

                    cand = None
                    for nn in nbs:
                        if nn == prev:
                            continue
                        if edge_id(cur, nn) not in used:
                            cand = nn
                            break

                    if cand is None:
                        if cur == start:
                            break
                        for nn in nbs:
                            if nn != prev:
                                cand = nn
                                break

                    if cand is None:
                        break

                    neighbor = cand
                    if neighbor == start:
                        used.add(edge_id(cur, neighbor))
                        loop.append(start)
                        break

                if len(loop) >= 4 and loop[0] == loop[-1]:
                    loops.append(loop)

        if not loops:
            raise ValueError("Failed to assemble loops from boundary edges")

        # 6) Scale detection + shift to (0,0)
        xs = [p[0] for loop in loops for p in loop]
        ys = [p[1] for loop in loops for p in loop]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        span = max(max_x - min_x, max_y - min_y)

        assume_mm = float(span) > 120.0
        scale = (1.0 / INCH_TO_MM) if assume_mm else 1.0

        out_loops = []
        for idx, loop in enumerate(loops):
            pts = [{"x": float((p[0] - min_x) * scale), "y": float((p[1] - min_y) * scale)} for p in loop]
            out_loops.append({"idx": idx, "closed": True, "points": pts})

        # 7) Largest area loop is outer
        def poly_area(pts):
            area2 = 0.0
            for i in range(len(pts) - 1):
                area2 += pts[i]["x"] * pts[i + 1]["y"] - pts[i + 1]["x"] * pts[i]["y"]
            return abs(area2) * 0.5

        outer_idx = max(range(len(out_loops)), key=lambda i: poly_area(out_loops[i]["points"]))

        return {"units": "in", "outerLoopIndex": int(outer_idx), "loops": out_loops}

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
        solid = build_cad_from_layout(payload.layout)
        step_text = export_step_text(solid)
    except Exception as exc:
        print("[STEP-SVC] Geometry error:", repr(exc))
        raise HTTPException(
            status_code=400,
            detail=f"Failed to build STEP geometry: {exc}",
        )

    if not step_text.strip():
        raise HTTPException(500, "STEP export produced empty text")

    return {
        "ok": True,
        "step": step_text,
        "quoteNo": payload.quoteNo,
        "materialLegend": payload.materialLegend,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
