# step_service/main.py
#
# SAFE GEOMETRY STEP microservice for Alex-IO foam layouts.
# Guarantees at least one valid solid is exported.
#
# FIX:
# - Circle cavities supported (cylindrical cuts)
# - Coordinate system alignment: editor uses top-left origin (y down),
#   CAD uses bottom-left origin (y up). Flip Y when placing cavities.

from typing import List, Optional
import os
import tempfile

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import cadquery as cq

INCH_TO_MM = 25.4
DEPTH_CLAMP_RATIO = 0.95  # never cut full thickness


class Cavity(BaseModel):
    lengthIn: float
    widthIn: float
    depthIn: float
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)

    shape: Optional[str] = None           # "rect" | "circle"
    diameterIn: Optional[float] = None    # for circle cavities

    @validator("lengthIn", "widthIn", "depthIn")
    def positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Dimension must be > 0")
        return v


class FoamLayer(BaseModel):
    thicknessIn: float
    label: Optional[str] = None
    cavities: Optional[List[Cavity]] = None

    @validator("thicknessIn")
    def positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Layer thickness must be > 0")
        return v


class Block(BaseModel):
    lengthIn: float
    widthIn: float
    thicknessIn: float

    @validator("lengthIn", "widthIn", "thicknessIn")
    def positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Block dimensions must be > 0")
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


def build_layer_block(L, W, T, z):
    return (
        cq.Workplane("XY")
        .box(L, W, T, centered=(False, False, False))
        .translate((0, 0, z))
    )


def build_cad_from_layout(layout: Layout) -> cq.Workplane:
    L_mm = layout.block.lengthIn * INCH_TO_MM
    W_mm = layout.block.widthIn * INCH_TO_MM

    z_bottom = 0.0
    valid_solids: List[cq.Workplane] = []

    for idx, layer in enumerate(layout.stack):
        T_mm = layer.thicknessIn * INCH_TO_MM
        if T_mm <= 0:
            continue

        base = build_layer_block(L_mm, W_mm, T_mm, z_bottom)
        working = base

        cavities = list(layer.cavities or [])
        if idx == 0 and layout.cavities:
            cavities.extend(layout.cavities)

        for cav in cavities:
            cav_D = min(cav.depthIn * INCH_TO_MM, T_mm * DEPTH_CLAMP_RATIO)

            shape = (cav.shape or "rect").lower()

            z_top = z_bottom + T_mm
            z_cut = z_top - cav_D

            try:
                if shape == "circle":
                    dia_in = cav.diameterIn or min(cav.lengthIn, cav.widthIn)
                    dia_mm = dia_in * INCH_TO_MM
                    r_mm = dia_mm / 2.0

                    # Convert editor top-left normalized coords -> CAD bottom-left coords
                    x_left = cav.x * L_mm
                    y_top_svg = cav.y * W_mm
                    y_top_cad = W_mm * (1.0 - cav.y) - (2.0 * r_mm)

                    # Clamp inside
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
                    cav_L = cav.lengthIn * INCH_TO_MM
                    cav_W = cav.widthIn * INCH_TO_MM

                    if cav_L >= L_mm or cav_W >= W_mm:
                        continue

                    x_left = cav.x * L_mm
                    y_top_svg = cav.y * W_mm

                    # Flip Y: CAD bottom-left origin
                    y_top_cad = W_mm * (1.0 - cav.y) - cav_W

                    x_left = max(0.0, min(L_mm - cav_L, x_left))
                    y_top_cad = max(0.0, min(W_mm - cav_W, y_top_cad))

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
