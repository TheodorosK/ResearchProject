"""Minimal dependency-free SVG charts (line with confidence whiskers, stacked bars, heatmap)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

PALETTE = ["#1b6ca8", "#d95f02", "#2a9d3f", "#7b3294", "#b8860b", "#c0392b", "#555555"]
W, H, ML, MR, MT, MB = 760, 440, 70, 20, 44, 58


def _svg(inner: str, title: str, w: int = W, h: int = H) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" font-family="Helvetica,Arial,sans-serif">'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/><text x="{w / 2}" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#222">{escape(title)}</text>'
            f'{inner}</svg>')


def _ticks(lo: float, hi: float, log: bool) -> List[float]:
    if log:
        a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
        t = [10.0 ** e for e in range(a, b + 1)]
        mids = [m * 10.0 ** e for e in range(a, b) for m in (2, 5)]
        return sorted(t + [m for m in mids if lo <= m <= hi])
    step = (hi - lo) / 5 or 1
    return [lo + i * step for i in range(6)]


def line_chart(title: str, series: Dict[str, List[Tuple[float, float, Optional[float], Optional[float]]]], xlabel: str, ylabel: str,
               logx: bool = True, logy: bool = True, ref: Optional[float] = 1.0) -> str:
    pts = [p for s in series.values() for p in s]
    if not pts:
        return _svg('<text x="380" y="220" text-anchor="middle" fill="#666">no data</text>', title)
    xs = [p[0] for p in pts]
    ys = [v for p in pts for v in (p[1], p[2], p[3], ref) if v]
    xlo, xhi = min(xs) * (0.8 if logx else 1), max(xs) * (1.25 if logx else 1)
    ylo, yhi = min(ys) * (0.8 if logy else 1), max(ys) * (1.25 if logy else 1)
    fx = (lambda v: math.log10(v)) if logx else (lambda v: v)
    fy = (lambda v: math.log10(v)) if logy else (lambda v: v)
    sx = lambda v: ML + (fx(v) - fx(xlo)) / ((fx(xhi) - fx(xlo)) or 1) * (W - ML - MR)
    sy = lambda v: H - MB - (fy(v) - fy(ylo)) / ((fy(yhi) - fy(ylo)) or 1) * (H - MT - MB)
    out = [f'<rect x="{ML}" y="{MT}" width="{W - ML - MR}" height="{H - MT - MB}" fill="none" stroke="#999"/>']
    for t in _ticks(ylo, yhi, logy):
        if ylo <= t <= yhi:
            out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="#e3e3e3"/><text x="{ML - 6}" y="{sy(t) + 4:.1f}" text-anchor="end" font-size="11" fill="#444">{t:g}</text>')
    for t in _ticks(xlo, xhi, logx):
        if xlo <= t <= xhi:
            out.append(f'<line y1="{MT}" y2="{H - MB}" x1="{sx(t):.1f}" x2="{sx(t):.1f}" stroke="#eee"/><text x="{sx(t):.1f}" y="{H - MB + 16}" text-anchor="middle" font-size="11" fill="#444">{t:g}</text>')
    if ref is not None and ylo <= ref <= yhi:
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{sy(ref):.1f}" y2="{sy(ref):.1f}" stroke="#c0392b" stroke-dasharray="6 4"/>'
                   f'<text x="{ML + 6}" y="{sy(ref) + 14:.1f}" text-anchor="start" font-size="11" fill="#c0392b">parity (GPU = CPU)</text>')
    for i, (name, s) in enumerate(series.items()):
        col = PALETTE[i % len(PALETTE)]
        s = sorted(s)
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="2" points="' + " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in s) + '"/>')
        for p in s:
            out.append(f'<circle cx="{sx(p[0]):.1f}" cy="{sy(p[1]):.1f}" r="3.5" fill="{col}"/>')
            if p[2] and p[3]:
                x = sx(p[0])
                out.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(p[2]):.1f}" y2="{sy(p[3]):.1f}" stroke="{col}" stroke-width="1.5"/>'
                           f'<line x1="{x - 4:.1f}" x2="{x + 4:.1f}" y1="{sy(p[2]):.1f}" y2="{sy(p[2]):.1f}" stroke="{col}"/>'
                           f'<line x1="{x - 4:.1f}" x2="{x + 4:.1f}" y1="{sy(p[3]):.1f}" y2="{sy(p[3]):.1f}" stroke="{col}"/>')
        out.append(f'<rect x="{ML + 10 + 130 * (i % 4)}" y="{MT + 8 + 16 * (i // 4)}" width="10" height="10" fill="{col}"/>'
                   f'<text x="{ML + 24 + 130 * (i % 4)}" y="{MT + 17 + 16 * (i // 4)}" font-size="11" fill="#333">{escape(name)}</text>')
    out.append(f'<text x="{(ML + W - MR) / 2}" y="{H - 14}" text-anchor="middle" font-size="12" fill="#333">{escape(xlabel)}</text>'
               f'<text transform="translate(16,{(MT + H - MB) / 2}) rotate(-90)" text-anchor="middle" font-size="12" fill="#333">{escape(ylabel)}</text>')
    return _svg("".join(out), title)


def stacked_bars(title: str, groups: Sequence[str], components: Dict[str, Sequence[float]], ylabel: str = "share of wall time") -> str:
    if not groups:
        return _svg('<text x="380" y="220" text-anchor="middle" fill="#666">no data</text>', title)
    bw = (W - ML - MR) / len(groups)
    sy = lambda v: H - MB - v * (H - MT - MB)
    out = [f'<rect x="{ML}" y="{MT}" width="{W - ML - MR}" height="{H - MT - MB}" fill="none" stroke="#999"/>']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{ML}" x2="{W - MR}" y1="{sy(t):.1f}" y2="{sy(t):.1f}" stroke="#e3e3e3"/><text x="{ML - 6}" y="{sy(t) + 4:.1f}" text-anchor="end" font-size="11">{t:.0%}</text>')
    for gi, g in enumerate(groups):
        acc = 0.0
        for ci, (cname, vals) in enumerate(components.items()):
            v = max(0.0, min(vals[gi], 1.0 - acc))
            out.append(f'<rect x="{ML + gi * bw + bw * 0.15:.1f}" y="{sy(acc + v):.1f}" width="{bw * 0.7:.1f}" height="{max(sy(acc) - sy(acc + v), 0):.1f}" fill="{PALETTE[ci % len(PALETTE)]}"/>')
            acc += v
        out.append(f'<text x="{ML + gi * bw + bw / 2:.1f}" y="{H - MB + 16}" text-anchor="middle" font-size="11" fill="#333">{escape(g)}</text>')
    for ci, cname in enumerate(components):
        out.append(f'<rect x="{ML + 10 + 120 * ci}" y="{H - 26}" width="10" height="10" fill="{PALETTE[ci % len(PALETTE)]}"/><text x="{ML + 24 + 120 * ci}" y="{H - 17}" font-size="11">{escape(cname)}</text>')
    out.append(f'<text transform="translate(16,{(MT + H - MB) / 2}) rotate(-90)" text-anchor="middle" font-size="12">{escape(ylabel)}</text>')
    return _svg("".join(out), title)


def heatmap(title: str, rows: Sequence[str], cols: Sequence[str], values: Sequence[Sequence[Optional[float]]], center: float = 1.0) -> str:
    """Ratio heatmap on a log colour scale diverging around ``center`` (blue < 1 = CPU faster, orange > 1 = GPU faster)."""
    cw, ch = 110, 54
    w, h = ML + 40 + cw * len(cols) + 30, MT + 30 + ch * len(rows) + 50
    out = []
    for ri, r in enumerate(rows):
        out.append(f'<text x="{ML + 30}" y="{MT + 30 + ri * ch + ch / 2 + 4}" text-anchor="end" font-size="12">{escape(r)}</text>')
        for ci, v in enumerate(values[ri]):
            x, y = ML + 40 + ci * cw, MT + 30 + ri * ch
            if v is None:
                out.append(f'<rect x="{x}" y="{y}" width="{cw - 2}" height="{ch - 2}" fill="#eee"/><text x="{x + cw / 2}" y="{y + ch / 2 + 4}" text-anchor="middle" font-size="11" fill="#888">n/a</text>')
                continue
            z = max(-1.0, min(1.0, math.log10(v / center) / 1.0))
            col = f"rgb({int(255 - 120 * (-z))},{int(255 - 70 * (-z))},255)" if z < 0 else f"rgb(255,{int(255 - 110 * z)},{int(255 - 200 * z)})"
            out.append(f'<rect x="{x}" y="{y}" width="{cw - 2}" height="{ch - 2}" fill="{col}" stroke="#fff"/><text x="{x + cw / 2}" y="{y + ch / 2 + 4}" text-anchor="middle" font-size="13" fill="#111">{v:.2f}x</text>')
    for ci, c in enumerate(cols):
        out.append(f'<text x="{ML + 40 + ci * cw + cw / 2}" y="{MT + 22}" text-anchor="middle" font-size="12">{escape(c)}</text>')
    out.append(f'<text x="{w / 2}" y="{h - 14}" text-anchor="middle" font-size="11" fill="#555">GPU/CPU throughput: blue &lt; 1 (CPU faster), white = 1, orange &gt; 1 (GPU faster)</text>')
    return _svg("".join(out), title, w, h)
