#!/usr/bin/env python3
"""Generate thesis redraw PNG figures using only Python standard library.

The local environment used for evidence work may not have matplotlib, pillow,
numpy, or pandas. This script writes simple RGB PNGs directly and uses only
existing small CSV/JSON/MD evidence files. It never reads or modifies raw data,
processed data, Windows results, TFRecords, checkpoints, or remote outputs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "thesis_evidence"
OUT = ROOT / "docs" / "thesis_figures_redraw"

W, H = 1600, 1000
BG = (248, 250, 252)
INK = (31, 41, 55)
MUTED = (100, 116, 139)
GRID = (203, 213, 225)
BLUE = (37, 99, 235)
TEAL = (15, 118, 110)
AMBER = (217, 119, 6)
RED = (220, 38, 38)
GREEN = (22, 163, 74)
PURPLE = (124, 58, 237)
PINK = (219, 39, 119)
SLATE = (71, 85, 105)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


FONT = {
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "C": ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01111", "10000", "10000", "10011", "10001", "10001", "01111"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "J": ["00111", "00010", "00010", "00010", "10010", "10010", "01100"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "Q": ["01110", "10001", "10001", "10001", "10101", "10010", "01101"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
    "Z": ["11111", "00001", "00010", "00100", "01000", "10000", "11111"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01111", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "11110"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    "%": ["11001", "11010", "00010", "00100", "01000", "01011", "10011"],
    "<": ["00010", "00100", "01000", "10000", "01000", "00100", "00010"],
    ">": ["01000", "00100", "00010", "00001", "00010", "00100", "01000"],
    "=": ["00000", "11111", "00000", "11111", "00000", "00000", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    ",": ["00000", "00000", "00000", "00000", "01100", "00100", "01000"],
    "*": ["00000", "10101", "01110", "11111", "01110", "10101", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


class Canvas:
    def __init__(self, w: int = W, h: int = H, bg=BG):
        self.w = w
        self.h = h
        self.px = bytearray(bg * (w * h))

    def set(self, x: int, y: int, color):
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            self.px[i : i + 3] = bytes(color)

    def rect(self, x0, y0, x1, y1, color, fill=True, width=1):
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        if fill:
            for y in range(max(0, y0), min(self.h, y1 + 1)):
                for x in range(max(0, x0), min(self.w, x1 + 1)):
                    self.set(x, y, color)
        else:
            for i in range(width):
                self.line(x0, y0 + i, x1, y0 + i, color)
                self.line(x0, y1 - i, x1, y1 - i, color)
                self.line(x0 + i, y0, x0 + i, y1, color)
                self.line(x1 - i, y0, x1 - i, y1, color)

    def line(self, x0, y0, x1, y1, color, width=1):
        x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            r = max(0, width // 2)
            for yy in range(y0 - r, y0 + r + 1):
                for xx in range(x0 - r, x0 + r + 1):
                    self.set(xx, yy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def circle(self, cx, cy, r, color, fill=True, width=1):
        cx, cy, r = int(cx), int(cy), int(r)
        if fill:
            for y in range(cy - r, cy + r + 1):
                for x in range(cx - r, cx + r + 1):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                        self.set(x, y, color)
        else:
            for a in range(360):
                rad = math.radians(a)
                x = int(cx + math.cos(rad) * r)
                y = int(cy + math.sin(rad) * r)
                for k in range(width):
                    self.set(x + k, y, color)

    def arrow(self, x0, y0, x1, y1, color=INK, width=3):
        self.line(x0, y0, x1, y1, color, width)
        ang = math.atan2(y1 - y0, x1 - x0)
        for delta in (math.pi * 0.8, -math.pi * 0.8):
            x2 = x1 + math.cos(ang + delta) * 24
            y2 = y1 + math.sin(ang + delta) * 24
            self.line(x1, y1, x2, y2, color, width)

    def text(self, x, y, text, color=INK, scale=3, max_width=None):
        text = str(text).replace("\\n", "\n").upper()
        cx = int(x)
        cy = int(y)
        line_h = 8 * scale + scale
        lines = []
        for paragraph in text.split("\n"):
            words = paragraph.split(" ")
            if max_width:
                cur = ""
                for word in words:
                    if self.text_width(word, scale) > max_width and not cur:
                        chunk = max(1, int(max_width // (6 * scale)))
                        for start in range(0, len(word), chunk):
                            lines.append(word[start : start + chunk])
                        continue
                    test = (cur + " " + word).strip()
                    if self.text_width(test, scale) <= max_width or not cur:
                        cur = test
                    else:
                        lines.append(cur)
                        cur = word
                if cur:
                    lines.append(cur)
            else:
                lines.append(paragraph)
        for li, line in enumerate(lines):
            xx = cx
            yy = cy + li * line_h
            for ch in line:
                pat = FONT.get(ch, FONT[" "])
                for py, row in enumerate(pat):
                    for px, bit in enumerate(row):
                        if bit == "1":
                            self.rect(xx + px * scale, yy + py * scale, xx + (px + 1) * scale - 1, yy + (py + 1) * scale - 1, color)
                xx += 6 * scale

    @staticmethod
    def text_width(text, scale=3):
        return len(str(text)) * 6 * scale

    def title(self, title, subtitle=None):
        self.text(70, 52, title, INK, 5, max_width=1450)
        if subtitle:
            self.text(72, 112, subtitle, MUTED, 3, max_width=1450)

    def save_png(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray()
        for y in range(self.h):
            raw.append(0)
            start = y * self.w * 3
            raw.extend(self.px[start : start + self.w * 3])
        def chunk(tag, data):
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        png = b"".join([
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", self.w, self.h, 8, 2, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
            chunk(b"IEND", b""),
        ])
        path.write_bytes(png)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(v, default=None):
    if v is None or v == "":
        return default
    try:
        return float(v)
    except Exception:
        return default


def draw_box(c: Canvas, x, y, w, h, label, color=BLUE, fill=(239, 246, 255), scale=3):
    c.rect(x, y, x + w, y + h, fill)
    c.rect(x, y, x + w, y + h, color, fill=False, width=3)
    c.text(x + 22, y + 24, label, color, scale, max_width=w - 44)


def heat_color(v, vmin, vmax):
    t = 0 if vmax == vmin else max(0, min(1, (v - vmin) / (vmax - vmin)))
    # light blue -> amber -> red
    if t < 0.5:
        a = t / 0.5
        c0, c1 = (219, 234, 254), (252, 211, 77)
    else:
        a = (t - 0.5) / 0.5
        c0, c1 = (252, 211, 77), (220, 38, 38)
    return tuple(int(c0[i] * (1 - a) + c1[i] * a) for i in range(3))


def draw_heatmap(c, x, y, w, h, rows=18, cols=28, mode="zc"):
    vals = []
    for r in range(rows):
        row = []
        for cc in range(cols):
            blob = math.exp(-((r - rows * 0.45) ** 2 / 28 + (cc - cols * 0.62) ** 2 / 40))
            streak = math.exp(-((cc - cols * 0.25) ** 2 / 10)) * (0.4 + 0.6 * math.sin(r * 0.65) ** 2)
            v = blob + 0.65 * streak
            if mode == "mask":
                v = 1.0 if v > 0.48 else 0.0
            elif mode == "severity":
                v = max(0.0, v - 0.22)
            row.append(v)
        vals.append(row)
    cell_w = w / cols
    cell_h = h / rows
    flat = [v for row in vals for v in row]
    vmin, vmax = min(flat), max(flat)
    for r, row in enumerate(vals):
        for cc, v in enumerate(row):
            if mode == "mask":
                col = RED if v > 0 else (226, 232, 240)
            else:
                col = heat_color(v, vmin, vmax)
            c.rect(x + cc * cell_w, y + r * cell_h, x + (cc + 1) * cell_w - 1, y + (r + 1) * cell_h - 1, col)
    c.rect(x, y, x + w, y + h, INK, fill=False, width=2)
    return vals


def chart_area(c, x, y, w, h, title, xlab="", ylab=""):
    c.rect(x, y, x + w, y + h, WHITE)
    c.rect(x, y, x + w, y + h, GRID, fill=False, width=2)
    for i in range(1, 5):
        yy = y + h - i * h / 5
        c.line(x, yy, x + w, yy, (226, 232, 240), 1)
    c.text(x, y - 45, title, INK, 3, max_width=w)
    if xlab:
        c.text(x + w // 2 - 90, y + h + 26, xlab, MUTED, 2)
    if ylab:
        c.text(x - 2, y - 24, ylab, MUTED, 2)


def map_pt(x, y, x0, x1, y0, y1, left, top, w, h):
    px = left + (x - x0) / (x1 - x0) * w if x1 != x0 else left
    py = top + h - (y - y0) / (y1 - y0) * h if y1 != y0 else top + h
    return px, py


def draw_line_series(c, points, area, xr, yr, color, width=3):
    left, top, w, h = area
    prev = None
    for x, y in points:
        px, py = map_pt(x, y, xr[0], xr[1], yr[0], yr[1], left, top, w, h)
        if prev:
            c.line(prev[0], prev[1], px, py, color, width)
        prev = (px, py)


def draw_scatter(c, points, area, xr, yr, color, r=4):
    left, top, w, h = area
    for x, y in points:
        px, py = map_pt(x, y, xr[0], xr[1], yr[0], yr[1], left, top, w, h)
        c.circle(px, py, r, color)


def draw_bars(c, labels, values, area, yr, colors):
    left, top, w, h = area
    n = len(values)
    bw = w / max(1, n) * 0.62
    for i, v in enumerate(values):
        x = left + (i + 0.19) * w / n
        py = top + h - (v - yr[0]) / (yr[1] - yr[0]) * h
        zero = top + h - (0 - yr[0]) / (yr[1] - yr[0]) * h
        y0, y1 = min(py, zero), max(py, zero)
        c.rect(x, y0, x + bw, y1, colors[i % len(colors)])
        c.text(x, top + h + 24, labels[i][:10], INK, 2, max_width=int(w / n))
        c.text(x, y0 - 26 if v >= 0 else y1 + 8, f"{v:.3f}", INK, 2)


def fig_01():
    c = Canvas()
    c.title("XSI CAST AZIMUTH MISMATCH", "SCHEMATIC ONLY  -  NO PERFORMANCE CLAIM")
    draw_box(c, 120, 210, 360, 560, "XSI SONIC\\nRECEIVER ARRAY\\nCWT INPUT", BLUE, (239, 246, 255), 3)
    draw_box(c, 1120, 210, 360, 560, "CAST ZC IMAGE\\nAZIMUTH MAP", RED, (254, 242, 242), 3)
    # depth tracks
    for y in range(285, 720, 62):
        c.line(190, y, 410, y + 22 * math.sin(y / 53), TEAL, 2)
        c.line(1190, y, 1410, y, SLATE, 1)
    c.arrow(520, 330, 1080, 330, INK, 3)
    c.text(670, 280, "DEPTH ALIGNMENT", INK, 3)
    c.arrow(1080, 600, 520, 600, RED, 3)
    c.text(620, 555, "AZIMUTH OFFSET UNKNOWN", RED, 3)
    # polar rings
    for r in (70, 110, 150):
        c.circle(800, 510, r, GRID, fill=False, width=2)
    c.line(800, 510, 930, 455, BLUE, 5)
    c.line(800, 510, 715, 390, RED, 5)
    c.text(705, 690, "DIRECT POINTWISE\\nAZIMUTH SUPERVISION\\nIS UNRELIABLE", INK, 3, max_width=360)
    c.text(80, 875, "THESIS RESPONSE: USE WEAK LABELS  -  1D PERCENTAGE OR FFT MAGNITUDE", MUTED, 3)
    c.save_png(OUT / "fig_01_xsi_cast_azimuth_mismatch_schematic.png")


def fig_02():
    c = Canvas()
    c.title("DATA AND EVIDENCE PIPELINE", "FINAL WRITING PACK  -  RESULTS DIRECTORY READ ONLY")
    draw_box(c, 80, 230, 260, 120, "XSI WAVEFORMS", BLUE)
    draw_box(c, 430, 230, 260, 120, "CWT\\n150 X 400 X 8", TEAL, (240, 253, 250))
    draw_box(c, 790, 230, 280, 120, "EXPLICIT\\nTFRECORD SPLIT", PURPLE, (245, 243, 255))
    draw_box(c, 1180, 230, 300, 120, "EFFICIENTNET\\nREGRESSION", BLUE)
    for x0, x1 in [(340, 430), (690, 790), (1070, 1180)]:
        c.arrow(x0, 290, x1 - 10, 290, INK)
    draw_box(c, 80, 560, 260, 120, "CAST ZC", RED, (254, 242, 242))
    draw_box(c, 430, 500, 280, 100, "SEVERITY\\nMAX(0, 2.5-ZC)", AMBER, (255, 251, 235))
    draw_box(c, 430, 640, 280, 100, "CHANNELING\\nMASK ZC < 2.5", RED, (254, 242, 242))
    draw_box(c, 820, 500, 300, 100, "FFT MAGNITUDE\\nEXP-008 MAIN", TEAL, (240, 253, 250))
    draw_box(c, 820, 640, 300, 100, "1D PERCENTAGE\\nEXP-007 FALLBACK", AMBER, (255, 251, 235))
    c.arrow(340, 620, 430, 550, INK)
    c.arrow(340, 620, 430, 690, INK)
    c.arrow(710, 550, 820, 550, INK)
    c.arrow(710, 690, 820, 690, INK)
    c.arrow(1120, 550, 1270, 350, INK)
    c.arrow(1120, 690, 1310, 350, INK)
    c.text(790, 820, "FINAL REPORT: EXP-008 MAINLINE, EXP-007 FALLBACK, EXP-006 RANDOM-SPLIT BASELINE", INK, 3, max_width=700)
    c.save_png(OUT / "fig_02_data_pipeline.png")


def fig_03():
    c = Canvas()
    c.title("1D PERCENTAGE LABEL CONSTRUCTION", "METHOD SCHEMATIC  -  NOT NUMERIC RESULT")
    xs = [90, 470, 850, 1210]
    labels = ["CAST ZC\\nSLICE", "MASK\\nZC < 2.5", "AZIMUTH\\nMEAN %", "70-POINT\\nPROFILE"]
    for i, x in enumerate(xs):
        c.text(x, 190, labels[i], INK, 3, max_width=260)
    draw_heatmap(c, 80, 300, 280, 360, mode="zc")
    c.arrow(380, 480, 450, 480)
    draw_heatmap(c, 470, 300, 280, 360, mode="mask")
    c.arrow(770, 480, 840, 480)
    # percentage bars
    vals = []
    for i in range(28):
        v = 10 + 60 * math.exp(-((i - 12) ** 2) / 45) + 18 * math.sin(i * 0.7)
        vals.append(max(0, min(100, v)))
    for i, v in enumerate(vals):
        y = 660 - v * 3.2
        c.rect(860 + i * 10, y, 866 + i * 10, 660, TEAL)
    c.rect(850, 300, 300 + 850, 660, INK, fill=False, width=2)
    c.text(880, 700, "PERCENT BY DEPTH", MUTED, 2)
    c.arrow(1160, 480, 1210, 480)
    pts = []
    for i in range(70):
        v = 12 + 50 * math.exp(-((i - 28) ** 2) / 180) + 10 * math.sin(i * 0.35)
        pts.append((i, max(0, min(100, v))))
    chart_area(c, 1210, 300, 300, 360, "PROFILE", "INDEX", "%")
    draw_line_series(c, pts, (1210, 300, 300, 360), (0, 69), (0, 100), RED, 3)
    c.text(105, 825, "FORMULA EVIDENCE: OLD EXP-007 CREATE_TFRECORDS CODE  -  LABEL = AZIMUTH MEAN OF ZC < 2.5", MUTED, 3, max_width=1390)
    c.save_png(OUT / "fig_03_percentage_label_construction.png")


def fig_04():
    c = Canvas()
    c.title("FFT SEVERITY MAGNITUDE LABEL", "EXP-008 MAIN LABEL ROUTE  -  METHOD SCHEMATIC")
    draw_heatmap(c, 95, 300, 260, 330, mode="zc")
    c.text(110, 235, "CAST ZC", INK, 3)
    c.arrow(370, 465, 450, 465)
    draw_heatmap(c, 470, 300, 260, 330, mode="severity")
    c.text(465, 225, "SEVERITY =\\nMAX(0, 2.5-ZC)", INK, 3)
    c.arrow(750, 465, 850, 465)
    # FFT bars by coefficient
    vals = [0.75 * math.exp(-k / 13) + 0.08 * math.sin(k * 0.9) for k in range(30)]
    c.rect(870, 300, 300, 330, WHITE)
    c.rect(870, 300, 300, 330, INK, fill=False, width=2)
    for k, v in enumerate(vals):
        h = max(8, v * 250)
        col = TEAL if k < 6 else (AMBER if k < 15 else BLUE)
        c.rect(880 + k * 9, 610 - h, 886 + k * 9, 610, col)
    c.text(865, 225, "AZIMUTH FFT\\nMAGNITUDE", INK, 3)
    c.arrow(1190, 465, 1270, 465)
    c.rect(1285, 300, 230, 330, WHITE)
    c.rect(1285, 300, 230, 330, INK, fill=False, width=2)
    for d in range(28):
        for k in range(18):
            v = math.exp(-k / 7) * (0.3 + 0.7 * math.exp(-((d - 12) ** 2) / 80))
            c.rect(1290 + k * 12, 305 + d * 11, 1300 + k * 12, 314 + d * 11, heat_color(v, 0, 1))
    c.text(1280, 225, "70 X 30\\nTARGET MAP", INK, 3)
    c.text(105, 815, "PHASE IS DISCARDED FOR ROTATION-INVARIANT WEAK SUPERVISION", MUTED, 3, max_width=1390)
    c.save_png(OUT / "fig_04_fft_severity_label_construction.png")


def fig_05():
    c = Canvas()
    c.title("CWT EFFICIENTNET REGRESSION ARCHITECTURE", "CODE-DERIVED SCHEMATIC  -  NOT A PERFORMANCE FIGURE")
    boxes = [
        (70, 335, 210, 150, "CWT INPUT\\n150 X 400 X 8", TEAL),
        (335, 335, 210, 150, "1 X 1 CONV\\n8 TO 3 CH", BLUE),
        (600, 315, 260, 190, "EFFICIENTNETV2B0\\nBACKBONE", PURPLE),
        (915, 335, 190, 150, "GLOBAL AVG\\nPOOL", AMBER),
        (1160, 335, 170, 150, "DROPOUT", SLATE),
        (1380, 335, 170, 150, "DENSE\\nOUTPUT", RED),
    ]
    for x, y, w, h, label, col in boxes:
        draw_box(c, x, y, w, h, label, col, (255, 255, 255), 3)
    for (x, y, w, h, *_), (x2, y2, *_rest) in zip(boxes[:-1], boxes[1:]):
        c.arrow(x + w + 10, y + h / 2, x2 - 12, y2 + _rest[1] / 2 if False else y2 + 75, INK, 3)
    c.arrow(280, 410, 335, 410)
    c.arrow(545, 410, 600, 410)
    c.arrow(860, 410, 915, 410)
    c.arrow(1105, 410, 1160, 410)
    c.arrow(1330, 410, 1380, 410)
    draw_box(c, 360, 650, 350, 130, "EXP-008 OUTPUT\\nFFT SEVERITY 70 X 30", TEAL, (240, 253, 250), 3)
    draw_box(c, 870, 650, 350, 130, "EXP-007 OUTPUT\\n1D PERCENTAGE 70", AMBER, (255, 251, 235), 3)
    c.line(1465, 485, 535, 650, GRID, 2)
    c.line(1465, 485, 1045, 650, GRID, 2)
    c.text(120, 850, "TRAINING USES EXPLICIT TRAIN / VAL / TEST TFRECORDS  -  NO VALIDATION_SPLIT", MUTED, 3, max_width=1300)
    c.save_png(OUT / "fig_05_cwt_efficientnet_regression_architecture.png")


def fig_06():
    rows = read_csv(EVIDENCE / "remote_exp008_depth_blocked_train" / "training_history.csv")
    epochs = [int(r["epoch"]) for r in rows]
    train_mae = [num(r["mae"]) for r in rows]
    val_mae = [num(r["val_mae"]) for r in rows]
    train_loss = [num(r["loss"]) for r in rows]
    val_loss = [num(r["val_loss"]) for r in rows]
    c = Canvas()
    c.title("EXP-008 DEPTH-HELDOUT TRAINING CURVE", "SINGLE-WELL ARRAY_03  -  EARLY OVERFITTING OBSERVED")
    area1 = (125, 250, 620, 520)
    area2 = (865, 250, 620, 520)
    chart_area(c, *area1, "MAE", "EPOCH", "MAE")
    chart_area(c, *area2, "LOSS", "EPOCH", "LOSS")
    draw_line_series(c, list(zip(epochs, train_mae)), area1, (1, max(epochs)), (0, max(train_mae + val_mae) * 1.1), BLUE, 4)
    draw_line_series(c, list(zip(epochs, val_mae)), area1, (1, max(epochs)), (0, max(train_mae + val_mae) * 1.1), RED, 4)
    draw_line_series(c, list(zip(epochs, train_loss)), area2, (1, max(epochs)), (0, max(train_loss + val_loss) * 1.1), BLUE, 4)
    draw_line_series(c, list(zip(epochs, val_loss)), area2, (1, max(epochs)), (0, max(train_loss + val_loss) * 1.1), RED, 4)
    c.text(160, 820, "BLUE TRAIN   RED VALIDATION   EARLY STOP AT EPOCH 12 RESTORED BEST EPOCH 2", INK, 3, max_width=1250)
    c.save_png(OUT / "fig_06_exp008_training_curve.png")


def fig_07():
    rows = read_csv(EVIDENCE / "remote_exp008_depth_blocked_train" / "prediction_summary.csv")
    pts = [(num(r["true_mean_integrated_severity"]), num(r["pred_mean_integrated_severity"])) for r in rows]
    pts = [(x, y) for x, y in pts if x is not None and y is not None]
    mx = max(max(x for x, _ in pts), max(y for _, y in pts)) * 1.05
    c = Canvas()
    c.title("EXP-008 PREDICTION SCATTER", "TEST SPLIT 423 SAMPLES  -  SINGLE-WELL DEPTH-HELDOUT")
    area = (170, 220, 980, 660)
    chart_area(c, *area, "TRUE VS PREDICTED INTEGRATED SEVERITY", "TRUE", "PRED")
    draw_scatter(c, pts, area, (0, mx), (0, mx), BLUE, 4)
    draw_line_series(c, [(0, 0), (mx, mx)], area, (0, mx), (0, mx), RED, 3)
    c.text(1210, 290, "TEST METRICS", INK, 3)
    c.text(1210, 350, "MAE 0.079", BLUE, 3)
    c.text(1210, 410, "RMSE 0.277", BLUE, 3)
    c.text(1210, 470, "R2 0.107", BLUE, 3)
    c.text(1210, 530, "SPEARMAN 0.481", BLUE, 3)
    c.text(1210, 650, "NOT MULTI-WELL\\nGENERALIZATION", RED, 3, max_width=330)
    c.save_png(OUT / "fig_07_exp008_prediction_scatter.png")


def baseline_values(path, metrics=("overall_mae", "overall_rmse", "overall_r2"), run=None):
    rows = read_csv(path)
    values = {}
    for r in rows:
        if run and r.get("run") != run:
            continue
        pred = r.get("predictor") or r.get("comparator")
        if not pred:
            continue
        values.setdefault(pred, {})
        for m in metrics:
            if m in r:
                values[pred][m] = num(r[m])
            elif r.get("metric_name") == m:
                values[pred][m] = num(r.get("metric_value"))
    return values


def grouped_baseline_chart(c, data, order, title_note):
    metrics = ["overall_mae", "overall_rmse", "overall_r2"]
    metric_titles = ["MAE", "RMSE", "R2"]
    colors = [BLUE, RED, AMBER, TEAL]
    for j, m in enumerate(metrics):
        area = (110 + j * 500, 300, 380, 420)
        vals = [data[o].get(m, 0) for o in order]
        lo = min(vals + [0])
        hi = max(vals + [0])
        pad = (hi - lo) * 0.18 if hi != lo else 1
        chart_area(c, *area, metric_titles[j], "", "")
        draw_bars(c, [o.replace("train_", "tr_") for o in order], vals, area, (lo - pad, hi + pad), colors)
    c.text(110, 815, title_note, MUTED, 3, max_width=1360)


def fig_08():
    data = baseline_values(EVIDENCE / "exp008_depthheldout_baseline_comparison.csv")
    order = ["model", "zero", "train_mean", "train_median"]
    c = Canvas()
    c.title("EXP-008 BASELINE COMPARISON", "MODEL IS BETTER BY RMSE/R2, BUT ZERO HAS LOWER MAE")
    grouped_baseline_chart(c, data, order, "REPORT MAE, RMSE, AND R2 TOGETHER.  SINGLE-WELL DEPTH-HELDOUT ONLY.")
    c.save_png(OUT / "fig_08_exp008_baseline_comparison.png")


def fig_09():
    rows = [r for r in read_csv(EVIDENCE / "exp008_depthheldout_error_structure.csv") if r["analysis_type"] == "per_fft_coefficient"]
    pts_model = [(int(r["index"]), num(r["model_mae"])) for r in rows]
    pts_zero = [(int(r["index"]), num(r["zero_mae"])) for r in rows]
    ymax = max(v for _, v in pts_model) * 1.2
    c = Canvas()
    c.title("EXP-008 PER-FFT-COEFFICIENT ERROR", "LOW-FREQUENCY COEFFICIENTS HAVE LARGER ABSOLUTE ERROR")
    area = (150, 230, 1200, 600)
    chart_area(c, *area, "MAE BY FFT COEFFICIENT", "K", "MAE")
    draw_line_series(c, pts_model, area, (0, 29), (0, ymax), BLUE, 4)
    draw_line_series(c, pts_zero, area, (0, 29), (0, ymax), RED, 3)
    c.text(1380, 310, "BLUE MODEL", BLUE, 3)
    c.text(1380, 370, "RED ZERO", RED, 3)
    c.text(1380, 500, "K=0..5\\nLOW FREQ\\nARE HARDEST", INK, 3, max_width=180)
    c.save_png(OUT / "fig_09_exp008_per_fft_coefficient_error.png")


def fig_10_missing():
    path = OUT / "fig_10_exp007_prediction_scatter_missing.md"
    path.write_text(
        "# fig_10_exp007_prediction_scatter.png not generated\n\n"
        "Reason: local evidence contains EXP-007 aggregate metrics and severity-group metrics, "
        "but no per-sample `prediction_summary.csv` or prediction-vs-truth arrays for train_v002. "
        "Generating a scatter plot would require copying the remote small prediction artifact or "
        "rerunning evaluation, which is outside this redraw-only task.\n\n"
        "Status: `missing_data_no_plot`.\n",
        encoding="utf-8",
    )


def fig_11():
    data = baseline_values(EVIDENCE / "exp007_depthheldout_baseline_comparison.csv", metrics=("mae", "rmse", "r2"), run="train_v002")
    # normalize metric names for grouped chart helper
    converted = {k: {"overall_mae": v.get("mae"), "overall_rmse": v.get("rmse"), "overall_r2": v.get("r2")} for k, v in data.items()}
    order = ["model", "zero", "train_mean", "train_median"]
    c = Canvas()
    c.title("EXP-007 BASELINE COMPARISON", "FALLBACK RESULT  -  ZERO HAS LOWER MAE")
    grouped_baseline_chart(c, converted, order, "EXP-007 BEATS TRAIN-BASED BASELINES BY MAE/RMSE/R2, BUT NOT ZERO BY MAE.")
    c.save_png(OUT / "fig_11_exp007_baseline_comparison.png")


def fig_12():
    rows = read_csv(EVIDENCE / "remote_exp008_depth_blocked_train" / "prediction_summary.csv")
    parsed = []
    for r in rows:
        true = num(r["true_mean_integrated_severity"])
        pred = num(r["pred_mean_integrated_severity"])
        depth = num(r["depth_ft"])
        if true is not None and pred is not None and depth is not None:
            parsed.append((depth, true, pred))
    high = [(d, t, p) for d, t, p in parsed if t >= 5.0]
    if not high:
        (OUT / "fig_12_limitation_high_severity_underestimation_missing.md").write_text(
            "# fig_12 not generated\n\nNo samples with true integrated severity >= 5.0 were found in local prediction_summary.csv.\n",
            encoding="utf-8",
        )
        return
    d0, d1 = min(d for d, _, _ in high), max(d for d, _, _ in high)
    window = [(d, t, p) for d, t, p in parsed if d0 - 3 <= d <= d1 + 3]
    ymax = max(max(t, p) for _, t, p in window) * 1.1
    c = Canvas()
    c.title("HIGH-SEVERITY UNDERESTIMATION", "EXP-008 TEST DEPTH SEGMENT  -  LIMITATION EVIDENCE")
    area = (150, 230, 1120, 600)
    chart_area(c, *area, "TRUE AND PREDICTED INTEGRATED SEVERITY", "DEPTH FT", "SEVERITY")
    draw_line_series(c, [(d, t) for d, t, _ in window], area, (d0 - 3, d1 + 3), (0, ymax), RED, 4)
    draw_line_series(c, [(d, p) for d, _, p in window], area, (d0 - 3, d1 + 3), (0, ymax), BLUE, 4)
    c.text(1320, 330, "RED TRUE", RED, 3)
    c.text(1320, 390, "BLUE PRED", BLUE, 3)
    c.text(1320, 520, "HIGH TRUE\\nPEAKS\\nLOW PRED", INK, 3, max_width=220)
    c.save_png(OUT / "fig_12_limitation_high_severity_underestimation.png")


def write_docs():
    captions = """# Thesis Figure Captions

Generated: 2026-07-08. Figures are derived only from existing `docs/thesis_evidence` files and archived small artifacts. No raw data, processed data, Windows results, TFRecord, checkpoint, or large NPZ file was copied or modified.

| figure | status | 中文图题 | 图注草稿 |
| --- | --- | --- | --- |
| fig_01_xsi_cast_azimuth_mismatch_schematic.png | generated | XSI 与 CAST 方位失配问题示意图 | 示意 XSI 声波接收器阵列与 CAST Zc 方位图之间存在未知方位偏移，直接点对点方位监督不可靠，因此本文采用 1D percentage 与 FFT magnitude 弱监督标签路线。该图为方法示意，不含性能结论。 |
| fig_02_data_pipeline.png | generated | XSI-CWT 与 CAST 弱标签数据构建流程 | 展示从 XSI 波形到 CWT 输入、从 CAST Zc 到 severity/percentage/FFT 标签、再到显式 depth-heldout TFRecord 与 EfficientNet 回归模型的整体流程。 |
| fig_03_percentage_label_construction.png | generated | 一维窜槽百分比标签构造流程 | 根据 EXP-007 代码证据，将 CAST Zc 以 2.5 为阈值生成窜槽掩膜，再沿方位求平均得到深度方向 percentage profile。该图为标签构造示意，不使用图中数值作为实验结果。 |
| fig_04_fft_severity_label_construction.png | generated | FFT severity magnitude 标签构造流程 | 根据 EXP-008 方法路线，将 CAST Zc 转换为 severity=max(0,2.5-Zc)，再沿方位维计算 FFT magnitude，丢弃相位以降低对方位匹配的依赖。 |
| fig_05_cwt_efficientnet_regression_architecture.png | generated | CWT-EfficientNet 回归模型结构示意 | 展示 150x400x8 CWT 输入经过 1x1 通道适配、EfficientNetV2B0 backbone、全局池化、dropout 与 Dense 回归头输出 EXP-008 70x30 或 EXP-007 70 维标签。 |
| fig_06_exp008_training_curve.png | generated | EXP-008 depth-heldout 训练曲线 | 基于 `remote_exp008_depth_blocked_train/training_history.csv` 重画训练/验证 loss 与 MAE，显示 train_v001 在单井 depth-heldout 设置下早期过拟合并由 EarlyStopping 恢复较早 epoch 权重。 |
| fig_07_exp008_prediction_scatter.png | generated | EXP-008 depth-heldout 测试预测散点图 | 基于 `prediction_summary.csv` 展示测试集 integrated severity 的预测-真值关系。结果只代表 `array_03` 单井 depth-heldout，不代表多井泛化。 |
| fig_08_exp008_baseline_comparison.png | generated | EXP-008 与简单基线对比 | 基于 `exp008_depthheldout_baseline_comparison.csv` 展示模型、zero、train-mean、train-median 的 MAE/RMSE/R2。图注必须说明模型未优于 zero baseline 的 MAE。 |
| fig_09_exp008_per_fft_coefficient_error.png | generated | EXP-008 按 FFT 系数的误差分布 | 基于 `exp008_depthheldout_error_structure.csv` 展示不同 FFT 系数的 MAE，低频系数绝对误差更高，是主要 limitation。 |
| fig_10_exp007_prediction_scatter.png | missing_data_no_plot | EXP-007 depth-heldout 测试预测散点图 | 本地图件包未生成该图，因为缺少 EXP-007 train_v002 逐样本 prediction summary 或预测数组。见 `fig_10_exp007_prediction_scatter_missing.md`。 |
| fig_11_exp007_baseline_comparison.png | generated | EXP-007 fallback 与简单基线对比 | 基于 `exp007_depthheldout_baseline_comparison.csv` 展示 EXP-007 train_v002 与 zero/train-mean/train-median 的 MAE/RMSE/R2。EXP-007 优于 train-based baselines，但未优于 zero baseline 的 MAE。 |
| fig_12_limitation_high_severity_underestimation.png | generated | EXP-008 高严重度样本低估现象 | 基于 EXP-008 测试集 prediction summary 中高 integrated severity 深度段绘制真值与预测趋势，展示高严重度峰值被模型低估，是本文主要限制之一。 |
"""
    (OUT / "figure_captions.md").write_text(captions, encoding="utf-8")

    plan = """# Thesis Figure Insert Plan

Generated: 2026-07-08.

| chapter | figure | placement | note |
| --- | --- | --- | --- |
| 第2章 数据与问题定义 | fig_01_xsi_cast_azimuth_mismatch_schematic.png | 方位失配问题介绍后 | 方法示意，不含性能。 |
| 第2章 数据构建 | fig_02_data_pipeline.png | 数据构建流程小节 | 连接 XSI/CWT 与 CAST 标签。 |
| 第3章 标签构造 | fig_03_percentage_label_construction.png | 1D percentage fallback 标签小节 | EXP-007 标签路线。 |
| 第3章 标签构造 | fig_04_fft_severity_label_construction.png | FFT severity 主线标签小节 | EXP-008 方法创新核心图。 |
| 第4章 模型方法 | fig_05_cwt_efficientnet_regression_architecture.png | 模型结构小节 | Code-derived schematic. |
| 第5章 实验结果 | fig_06_exp008_training_curve.png | EXP-008 depth-heldout 训练过程 | 强调 early overfitting。 |
| 第5章 实验结果 | fig_07_exp008_prediction_scatter.png | EXP-008 depth-heldout 测试结果 | 单井 depth-heldout。 |
| 第5章 实验结果 | fig_08_exp008_baseline_comparison.png | EXP-008 基线对比 | 必须同时写 MAE caveat。 |
| 第5章 实验结果 / 第7章讨论 | fig_09_exp008_per_fft_coefficient_error.png | 误差结构分析 | 低频 FFT 系数误差。 |
| 第5章 fallback 对照 | fig_10_exp007_prediction_scatter_missing.md | 不插入 PNG | 等远程逐样本预测小文件复制后再补。 |
| 第5章 fallback 对照 | fig_11_exp007_baseline_comparison.png | EXP-007 fallback 基线表旁 | 不作为主线。 |
| 第7章 讨论 | fig_12_limitation_high_severity_underestimation.png | 高严重度低估限制 | 不夸大为工程泛化。 |
"""
    (OUT / "figure_insert_plan.md").write_text(plan, encoding="utf-8")


def copy_existing_trace_files():
    (OUT / "source_trace").mkdir(parents=True, exist_ok=True)
    for rel in [
        "final_thesis_metrics_table.csv",
        "exp008_depthheldout_baseline_comparison.csv",
        "exp008_depthheldout_error_structure.csv",
        "exp007_depthheldout_baseline_comparison.csv",
        "remote_exp008_depth_blocked_train/training_history.csv",
        "remote_exp008_depth_blocked_train/prediction_summary.csv",
        "remote_exp007_depth_blocked_train/severity_group_metrics.csv",
    ]:
        src = EVIDENCE / rel
        if src.exists() and src.stat().st_size < 5_000_000:
            dst = OUT / "source_trace" / rel.replace("/", "__")
            shutil.copyfile(src, dst)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    copy_existing_trace_files()
    fig_01()
    fig_02()
    fig_03()
    fig_04()
    fig_05()
    fig_06()
    fig_07()
    fig_08()
    fig_09()
    fig_10_missing()
    fig_11()
    fig_12()
    write_docs()
    print(f"generated figure redraw pack under {OUT}")


if __name__ == "__main__":
    main()
