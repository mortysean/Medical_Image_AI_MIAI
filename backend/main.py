# -*- coding: utf-8 -*-
# /home/seanhuang/MIAI/backend/main.py

from __future__ import annotations

import os, io, uuid, base64, traceback, textwrap
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.utils.predict import predict_mask_with_meta

# ========== OpenAI ==========
from openai import OpenAI

MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None
if not api_key:
    print("⚠️  WARNING: OPENAI_API_KEY not set, /api/vision will not work.")

# ========== Paths ==========
BACKEND_DIR: Path = Path(__file__).resolve().parent
STATIC_DIR:  Path = (BACKEND_DIR / "static").resolve()
REPORT_DIR:  Path = (STATIC_DIR / "reports").resolve()

STATIC_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ========== FastAPI ==========
app = FastAPI(title="Breast Cancer Segmentation Backend")

# 静态目录挂载到 /static —— 前端可以直接 GET /static/xxx
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# CORS（开发期放开，生产建议改为白名单）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True}

# ------------------------------------------------------------------------------
# 1) 本地分割
# ------------------------------------------------------------------------------
@app.post("/api/predict/")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        result = predict_mask_with_meta(io.BytesIO(contents))
        return result
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

# ------------------------------------------------------------------------------
# 2) 生成 PDF 报告（返回可下载 URL）
# ------------------------------------------------------------------------------
class VisionReq(BaseModel):
    image: str                       # dataURL: data:image/png;base64,...
    mask: Optional[str] = None       # 可选 dataURL
    accuracy: Optional[float] = None
    lesions: Optional[List[dict]] = []

# --- ReportLab 版式 ---
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import utils as rl_utils
from reportlab.lib.colors import HexColor

PAGE_W, PAGE_H = A4
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 40, 40, 60, 50
COL_GAP = 20

def _draw_header_footer(c: rl_canvas.Canvas, title: str, page_no: int):
    c.setStrokeColor(HexColor("#374151"))
    c.line(MARGIN_L, PAGE_H - MARGIN_T + 10, PAGE_W - MARGIN_R, PAGE_H - MARGIN_T + 10)
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(HexColor("#111827"))
    c.drawString(MARGIN_L, PAGE_H - MARGIN_T + 20, title)
    c.setFont("Helvetica", 9)
    c.setFillColor(HexColor("#374151"))
    c.drawRightString(PAGE_W - MARGIN_R, PAGE_H - MARGIN_T + 20,
                      datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    c.setStrokeColor(HexColor("#374151"))
    c.line(MARGIN_L, MARGIN_B - 12, PAGE_W - MARGIN_R, MARGIN_B - 12)
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(HexColor("#374151"))
    c.drawCentredString(PAGE_W / 2, MARGIN_B - 26,
        f"This report is AI-generated for research purposes only. — Page {page_no}")

def _new_page(c: rl_canvas.Canvas, title: str, page_no: int):
    c.showPage()
    _draw_header_footer(c, title, page_no)

def _decode_data_url(data_url: Optional[str]) -> Optional[ImageReader]:
    if not data_url or "," not in data_url:
        return None
    _, b64 = data_url.split(",", 1)
    try:
        return ImageReader(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        return None

def _fit_draw_image(c: rl_canvas.Canvas, img_reader: ImageReader, x, y, box_w, box_h):
    # ImageReader.getSize() 更稳（避免依赖 ._image）
    iw, ih = img_reader.getSize()
    scale = min(box_w / iw, box_h / ih)
    dw, dh = iw * scale, ih * scale
    c.drawImage(img_reader, x + (box_w - dw) / 2, y + (box_h - dh) / 2,
                width=dw, height=dh, preserveAspectRatio=True, mask='auto')

def _draw_section_title(c: rl_canvas.Canvas, text: str, y: float):
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(HexColor("#111827"))
    c.drawString(MARGIN_L, y, text)
    c.setStrokeColor(HexColor("#4F46E5"))
    c.line(MARGIN_L, y - 6, PAGE_W - MARGIN_R, y - 6)
    return y - 18

def _draw_paragraph(c: rl_canvas.Canvas, text: str, y: float,
                    font="Helvetica", size=11, leading=15):
    if not text:
        return y
    max_chars = 100
    c.setFont(font, size)
    c.setFillColor(HexColor("#111827"))
    for raw in text.splitlines():
        wrapped = textwrap.wrap(raw.strip(), width=max_chars) or [""]
        for line in wrapped:
            if y < MARGIN_B + 40:
                return None  # 触发翻页
            c.drawString(MARGIN_L, y, line)
            y -= leading
    return y

def generate_pdf_report(report_text: str,
                        orig_b64: Optional[str],
                        mask_b64: Optional[str],
                        out_path: Path,
                        accuracy: Optional[float] = None,
                        lesions: Optional[List[dict]] = None):
    c = rl_canvas.Canvas(str(out_path), pagesize=A4)
    title = "Breast Cancer Segmentation Report"

    page_no = 1
    _draw_header_footer(c, title, page_no)

    # 封面
    y = PAGE_H - MARGIN_T - 10
    c.setFont("Helvetica-Bold", 22)
    c.setFillColor(HexColor("#111827"))
    c.drawCentredString(PAGE_W/2, y, title)
    y -= 24
    c.setFont("Helvetica", 12)
    c.setFillColor(HexColor("#374151"))
    c.drawCentredString(PAGE_W/2, y, "Generated by MIAI System")
    y -= 30

    # Summary
    if (accuracy is not None) or (lesions and len(lesions) > 0):
        y = _draw_section_title(c, "Model Summary", y)
        lines = []
        if accuracy is not None:
            lines.append(f"Reported accuracy: {accuracy:.4f}")
        if lesions:
            for i, l in enumerate(lesions, 1):
                lines.append(
                    f"Lesion {i}: (x={l.get('x')}, y={l.get('y')}), "
                    f"size={l.get('width')}×{l.get('height')}, area={l.get('area')}"
                )
        para = "\n".join(lines)
        ny = _draw_paragraph(c, para, y)
        if ny is None:
            page_no += 1
            _new_page(c, title, page_no)
            y = PAGE_H - MARGIN_T
            y = _draw_section_title(c, "Model Summary (cont.)", y)
            y = _draw_paragraph(c, para, y) or (PAGE_H - MARGIN_T - 100)
        else:
            y = ny
        y -= 10

    # 原图 vs Mask
    y = _draw_section_title(c, "Original vs Segmentation", y)
    block_h = 240
    col_w = (PAGE_W - MARGIN_L - MARGIN_R - COL_GAP) / 2
    left_x = MARGIN_L
    right_x = MARGIN_L + col_w + COL_GAP
    img_y = y - block_h

    orig_img = _decode_data_url(orig_b64)
    mask_img = _decode_data_url(mask_b64)

    if orig_img:
        _fit_draw_image(c, orig_img, left_x, img_y, col_w, block_h)
        c.setFont("Helvetica", 10)
        c.setFillColor(HexColor("#374151"))
        c.drawCentredString(left_x + col_w/2, img_y - 14, "Original Image")

    if mask_img:
        _fit_draw_image(c, mask_img, right_x, img_y, col_w, block_h)
        c.setFont("Helvetica", 10)
        c.setFillColor(HexColor("#374151"))
        c.drawCentredString(right_x + col_w/2, img_y - 14, "Segmentation Mask")

    y = img_y - 28
    if y < MARGIN_B + 120:
        page_no += 1
        _new_page(c, title, page_no)
        y = PAGE_H - MARGIN_T

    # GPT Analysis
    y = _draw_section_title(c, "GPT Analysis", y)
    remaining = report_text or "(empty)"
    while True:
        ny = _draw_paragraph(c, remaining, y)
        if ny is not None:
            y = ny
            break
        page_no += 1
        _new_page(c, title, page_no)
        y = PAGE_H - MARGIN_T
        ny = _draw_paragraph(c, remaining, y)
        if ny is None:
            continue
        else:
            y = ny
            break

    c.save()

@app.post("/api/vision")
async def vision(req: VisionReq):
    if not client:
        raise HTTPException(500, "OPENAI_API_KEY not set")
    if not req.image:
        raise HTTPException(400, "missing image")

    try:
        prompt_text = (
            "You are an imaging analysis assistant.\n"
            "Compare the ultrasound Original image and its Segmentation Mask.\n"
            "Write a concise, professional, non-diagnostic report with short sections and bullet points.\n"
            "Focus on: overall observations, mask alignment/coverage, artifacts, and a brief summary."
        )

        openai_input = [{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_image", "image_url": req.image},
            ],
        }]
        if req.mask:
            openai_input[0]["content"].append(
                {"type": "input_image", "image_url": req.mask}
            )

        resp = client.responses.create(
            model=MODEL,
            input=openai_input,
            temperature=0.2,
            max_output_tokens=800,
        )
        report_text = (resp.output_text or "").strip()
        if not report_text:
            raise RuntimeError("Empty GPT response")

        # 生成 PDF 到 STATIC/reports 下
        filename = f"report_{uuid.uuid4().hex[:8]}.pdf"
        out_path: Path = REPORT_DIR / filename
        generate_pdf_report(
            report_text=report_text,
            orig_b64=req.image,
            mask_b64=req.mask,
            out_path=out_path,
            accuracy=req.accuracy,
            lesions=req.lesions,
        )

        # 返回可直接 GET 的 URL（FastAPI 已挂载 /static）
        return {"report_url": f"/static/reports/{filename}"}

    except Exception as e:
        print("OpenAI /api/vision error >>>")
        print(traceback.format_exc())
        raise HTTPException(500, f"OpenAI error: {e}")

# ------------------------------------------------------------------------------
# 3) 备用下载（不走 /static，也能下）
# ------------------------------------------------------------------------------
@app.get("/download/{filename}")
async def download_file(filename: str):
    fpath = (REPORT_DIR / filename).resolve()
    if not fpath.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(str(fpath), media_type="application/pdf", filename=filename)

# ------------------------------------------------------------------------------
# 4) Debug：看静态目录里有哪些报告（便于排查 404）
# ------------------------------------------------------------------------------
@app.get("/__debug/reports_ls")
def reports_ls():
    files = sorted(p.name for p in REPORT_DIR.glob("*.pdf"))
    return {"dir": str(REPORT_DIR), "files": files}
