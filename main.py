# step_service/main.py
#
# Minimal STEP microservice for Alex-IO foam layouts.
#
# - Uses CadQuery (OpenCascade) to build robust solids.
# - One solid per foam layer (stacked in Z, inches → mm).
# - Each cavity is cut out of its layer via boolean difference.
# - Exports a single STEP file and returns its text in JSON.
#
# Expected to be deployed separately from the Next.js app
# (e.g. on Render, Fly.io, Railway, etc.).
#
# ENV:
#   PORT (optional, default 8000)

from typing import List, Optional
import os
import tempfile

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

import cadquery as cq  # Make sure cadquery is installed in this environment.

INCH_TO_MM = 25.4


class Cavity(BaseModel):
    lengthIn: float
    widthIn: float
    depthIn: float
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)

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
    cavities: Optional[List[Cavity]] = None  # legacy, treated as extra cavities on top layer


class StepRequest(BaseModel):
    layout: Layout
    quoteNo: str
    materialLegend: Optional[str] = None


app = FastAPI(title="Alex-IO STEP microservice")


def build_cad_from_layout(layout: Layout) -> cq.Workplane:
    """Build a CadQuery solid from the foam layout."""
    L_block_mm = layout.block.lengthIn * INCH_TO_MM
    W_block_mm = layout.block.widthIn * INCH_TO_MM

    z_bottom_mm = 0.0
    layer_solids: List[cq.Workplane] = []

    for idx, layer in enumerate(layout.stack):
        thickness_mm = layer.thicknessIn * INCH_TO_MM
        if thickness_mm <= 0:
            continue

        # Base foam layer: rectangular prism, corner at (0,0,z_bottom_mm).
        layer_solid = (
            cq.Workplane("XY")
            .box(L_block_mm, W_block_mm, thickness_mm, centered=(False, False, False))
            .translate((0.0, 0.0, z_bottom_mm))
        )

        # Gather cavities: layer-specific + legacy top-level (only on first layer).
        cavities: List[Cavity] = list(layer.cavities or [])
        if idx == 0 and layout.cavities:
            cavities.extend(layout.cavities)

        for cav in cavities:
            # Clamp depth to layer thickness to avoid over-cutting.
            depth_mm = min(cav.depthIn * INCH_TO_MM, thickness_mm)

            # Normalised position from layout: 0..1 across block length/width.
            left_mm = layout.block.lengthIn * cav.x * INCH_TO_MM
            top_mm = layout.block.widthIn * cav.y * INCH_TO_MM

            cav_L_mm = cav.lengthIn * INCH_TO_MM
            cav_W_mm = cav.widthIn * INCH_TO_MM

            # Cavity is a box that starts at (left, top, z_top - depth)
            # and cuts downward into the layer.
            z_layer_top_mm = z_bottom_mm + thickness_mm
            z_cav_bottom_mm = z_layer_top_mm - depth_mm

            cavity_solid = (
                cq.Workplane("XY")
                .box(cav_L_mm, cav_W_mm, depth_mm, centered=(False, False, False))
                .translate((left_mm, top_mm, z_cav_bottom_mm))
            )

            layer_solid = layer_solid.cut(cavity_solid)

        layer_solids.append(layer_solid)
        z_bottom_mm += thickness_mm

    if not layer_solids:
        raise ValueError("No valid layer geometry built from layout.")

    # Union all layer solids into one compound.
    solid = layer_solids[0]
    for other in layer_solids[1:]:
        solid = solid.union(other)

    return solid


def export_step_text(solid: cq.Workplane) -> str:
    """Export the given solid to STEP and return it as a UTF-8 string."""
    # CadQuery's exporters work with file paths; use a temporary file.
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cq.exporters.export(solid, tmp_path)
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    # STEP is ASCII/UTF-8 text.
    return data.decode("utf-8", errors="ignore")


@app.post("/step-from-layout")
async def step_from_layout(payload: StepRequest):
    try:
        solid = build_cad_from_layout(payload.layout)
        step_text = export_step_text(solid)
    except Exception as exc:
        # Log full error server-side, but return a friendly message.
        print("[STEP-SVC] Error building STEP:", repr(exc))
        raise HTTPException(
            status_code=400,
            detail=f"Failed to build STEP geometry: {exc}",
        )

    if not step_text.strip():
        raise HTTPException(status_code=500, detail="STEP export produced empty text")

    return {
        "ok": True,
        "step": step_text,
        "quoteNo": payload.quoteNo,
        "materialLegend": payload.materialLegend,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
