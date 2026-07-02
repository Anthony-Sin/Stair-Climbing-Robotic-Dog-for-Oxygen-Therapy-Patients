"""Text-composite HUD widgets built on the shape + text primitives.

Group titles, label/value rows, the bracket-cornered data + focal tiles, status
pills / bars / dots, the cyberpunk decode reveal and caption-over-value stacks,
button glyphs and section labels.  These compose the low-level linework in
:mod:`core.hud.hud_shapes` with the deferred :class:`~core.hud.hud_text.TextLayer`.
"""
import cv2
import numpy as np
from typing import List, Sequence, Tuple

from core.hud.hud_theme import (
    BG_BASE, BG_PANEL, HAIRLINE, TEXT, DIM, ACCENT, ALERT, ACCENT_DK, INK, FG,
    bgr,
)
from core.hud.hud_text import TextLayer, _text_width
from core.hud.hud_shapes import (
    panel, texture_fill, sweep_line, fill_region, _bevel_pts, _fill_poly_alpha,
)


# ───────────────────────────── Group title / tick ─────────────────────────────
def group_title(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, title: str,
                *, right: bool = False, size: int = 13, color=ACCENT) -> None:
    """Accent group header with a single open corner tick (top-left, or mirrored
    top-right).  Tracked uppercase, no box."""
    leg = 9
    c = bgr(color)
    if right:
        cv2.line(frame_bgr, (x, y), (x - leg, y), c, 2, cv2.LINE_AA)
        cv2.line(frame_bgr, (x, y), (x, y + leg), c, 2, cv2.LINE_AA)
        layer.add(x - leg - 6, y - 3, title.upper(), color, size=size, bold=True,
                  anchor="rt", tracking=1.0)
    else:
        cv2.line(frame_bgr, (x, y), (x + leg, y), c, 2, cv2.LINE_AA)
        cv2.line(frame_bgr, (x, y), (x, y + leg), c, 2, cv2.LINE_AA)
        layer.add(x + leg + 6, y - 3, title.upper(), color, size=size, bold=True,
                  anchor="lt", tracking=1.0)


def kv_row(layer: TextLayer, x_label: int, x_value: int, y: int, label: str,
           value: str, value_color=TEXT, *, label_color=DIM, size: int = 12,
           value_bold: bool = False, right: bool = False) -> None:
    """One label/value row: label (dim) left, value (text/accent) at the value column."""
    layer.add(x_label, y, label.upper(), label_color, size=size, anchor="lt")
    layer.add(x_value, y, value, value_color, size=size, bold=value_bold,
              anchor="rt" if right else "lt")


# ─────────────────────────────── Data tile (§2.4) ─────────────────────────────
def data_tile(frame_bgr: np.ndarray, layer: TextLayer, x0: int, y0: int, x1: int, y1: int,
              title: str, code_lines: Sequence[str],
              rows: Sequence[Tuple[str, str, Tuple[int, int, int]]], *,
              right: bool = False, accent: Tuple[int, int, int] = ACCENT,
              size: int = 12, progress: float = 1.0, seed: int = 0) -> None:
    """Bracket-cornered data tile: a stacked hex code-block on the left, an accent
    title + label/value rows on the right (§2.4 'data tile').  ``right`` mirrors it
    (code block on the right) for the right-hand status rail.  ``progress`` < 1
    drives the per-value decode-in animation and a load sweep.
    """
    panel(frame_bgr, x0, y0, x1, y1, accent=accent, corners=("tl",) if right else ("tr",))
    code_w = max(_text_width(c, size - 1) for c in code_lines) if code_lines else 0
    code_w = max(code_w, _text_width("0000", size - 1))
    pad = 8

    if not right:
        code_x = x0 + pad
        div_x = code_x + code_w + pad
        title_x, val_edge = div_x + pad, x1 - pad
        title_anchor_x, title_right = title_x, False
    else:
        code_x = x1 - pad - code_w
        div_x = code_x - pad
        title_x, val_edge = x0 + pad, div_x - pad
        title_anchor_x, title_right = x1 - pad, True

    # title spans the top; code block + rows start beneath it (no collision)
    group_title(frame_bgr, layer, title_anchor_x, y0 + 8, title, right=title_right, size=size + 1)
    body_y = y0 + 30
    cv2.line(frame_bgr, (div_x, body_y - 4), (div_x, y1 - 8), bgr(HAIRLINE), 1, cv2.LINE_AA)

    # hex code block (cyan, monospace, stacked)
    for i, code in enumerate(code_lines):
        layer.add(code_x, body_y + i * (size + 4), code, accent, size=size - 1,
                  bold=True, anchor="lt", tracking=1.0, mono=True)

    # rows (labels static, values decode-in)
    ry = body_y
    for lbl, val, col in rows:
        layer.add(title_x, ry, lbl, DIM, size=size, anchor="lt")
        decode_text(layer, val_edge, ry, val, col, progress=progress, size=size,
                    anchor="rt", seed=seed)
        ry += size + 6

    if progress < 1.0:
        sweep_line(frame_bgr, x0 + 1, body_y - 2, x1 - 1, y1 - 4, progress, color=accent)


# ────────────────────────────── Status pill ───────────────────────────────────
def status_pill(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, text: str,
                state: str = "scanning", *, size: int = 12) -> int:
    """Sharp-cornered (radius 0) lock/mode chip.

      ``locked``   → cyan fill,   filled ▣, ink text
      ``scanning`` → hairline fill, hollow ◌ ring (dim), text
      ``lost``     → red fill,     ✗ mark,   white bold text
    Returns the chip's right edge.
    """
    text = text.upper()
    pad = max(7, size // 2)
    gap = 6
    mark_box = size + 2
    char_w = _text_width("M", size, True)
    text_w = len(text) * char_w
    x1 = x + pad + mark_box + gap + text_w + pad
    h = size + 10
    y1 = y + h
    cy = (y + y1) // 2

    fills = {"locked": ACCENT, "lost": ALERT, "scanning": HAIRLINE}
    fill = fills.get(state, HAIRLINE)
    txt_col = INK if state == "locked" else (FG if state == "lost" else TEXT)
    mark_col = txt_col if state != "scanning" else DIM
    fill_region(frame_bgr, x, y, x1, y1, fill, 0.92 if state != "scanning" else 0.7)
    cv2.rectangle(frame_bgr, (x, y), (x1, y1), bgr(ACCENT if state == "scanning" else fill), 1, cv2.LINE_AA)

    mcx, ms = x + pad + mark_box // 2, max(4, size // 3)
    if state == "locked":
        cv2.rectangle(frame_bgr, (mcx - ms, cy - ms), (mcx + ms, cy + ms), bgr(mark_col), -1, cv2.LINE_AA)
    elif state == "lost":
        cv2.line(frame_bgr, (mcx - ms, cy - ms), (mcx + ms, cy + ms), bgr(mark_col), 2, cv2.LINE_AA)
        cv2.line(frame_bgr, (mcx - ms, cy + ms), (mcx + ms, cy - ms), bgr(mark_col), 2, cv2.LINE_AA)
    else:
        cv2.circle(frame_bgr, (mcx, cy), ms, bgr(mark_col), 1, cv2.LINE_AA)

    layer.add(x + pad + mark_box + gap, cy, text, txt_col, size=size, bold=True,
              anchor="lm", outline=False)
    return x1


# ───────────────────────────── Focal data tile ────────────────────────────────
def focal_tag(frame_bgr: np.ndarray, layer: TextLayer, cx: int, top: int,
              segments: List[Tuple[str, Tuple[int, int, int]]], *,
              accent: Tuple[int, int, int] = ACCENT, size: int = 14,
              pad_x: int = 14, pad_y: int = 7, sep: str = " / ",
              sep_color: Tuple[int, int, int] = DIM,
              progress: float = 1.0, seed: int = 0) -> int:
    """A centred bracket-cornered focal tile (§2.4 'bracket frame' / 'data tile').

    Renders ``LABEL / VALUE`` (each ``(text, colour)`` segment laid out in true
    monospace) on a near-black scrim with a 1px hairline border + open ``accent``
    corner brackets — the at-a-glance designator for a tracked subject.  Centred on
    ``cx`` with its top edge at ``top``; returns the tile's bottom ``y`` so callers
    can stack several down the middle of the frame.
    """
    laid: List[Tuple[str, Tuple[int, int, int], int]] = []
    for i, (text, col) in enumerate(segments):
        if i:
            laid.append((sep, sep_color, _text_width(sep, size, mono=True)))
        laid.append((text, col, _text_width(text, size, mono=True)))
    text_w = sum(w for _, _, w in laid)

    box_w = text_w + 2 * pad_x
    box_h = size + 2 * pad_y
    x0 = int(round(cx - box_w / 2.0))
    x1 = x0 + box_w
    y0 = int(top)
    y1 = y0 + box_h

    # Angled (non-square) tile: bevel the top-right + bottom-left corners.
    cut = max(7, box_h // 3)
    pts = np.array(_bevel_pts(x0, y0, x1, y1, cut, ("tr", "bl")), np.int32)
    _fill_poly_alpha(frame_bgr, pts, BG_BASE, 0.74)
    texture_fill(frame_bgr, x0 + 2, y0 + 2, x1 - 2, y1 - 2, alpha=0.4)
    cv2.polylines(frame_bgr, [pts], True, bgr(HAIRLINE), 1, cv2.LINE_AA)
    leg = max(9, box_h // 3)
    ac = bgr(accent)
    for bx, by, sx, sy in ((x0, y0, 1, 1), (x1, y1, -1, -1)):    # brackets on square corners
        cv2.line(frame_bgr, (bx, by), (bx + sx * leg, by), ac, 2, cv2.LINE_AA)
        cv2.line(frame_bgr, (bx, by), (bx, by + sy * leg), ac, 2, cv2.LINE_AA)
    cv2.line(frame_bgr, (x1 - cut, y0), (x1, y0 + cut), ac, 2, cv2.LINE_AA)   # accent bevels
    cv2.line(frame_bgr, (x0, y1 - cut), (x0 + cut, y1), ac, 2, cv2.LINE_AA)

    if progress < 1.0:
        sweep_line(frame_bgr, x0 + 1, y0 + 1, x1 - 1, y1 - 1, progress, color=accent)
    cyc = (y0 + y1) // 2
    cursor = x0 + pad_x
    for text, col, w in laid:
        decode_text(layer, cursor, cyc, text, col, progress=progress, size=size,
                    bold=True, anchor="lm", seed=seed)
        cursor += w
    return y1


# ─────────────────── Cyberpunk decode / load-in display kit ────────────────────
_SCRAMBLE = "ABCDEF0123456789#%&*<>/\\=+|:"


def decode_text(layer: TextLayer, x: int, y: int, text, color=TEXT, *,
                progress: float = 1.0, size: int = 13, bold: bool = False,
                anchor: str = "lt", seed: int = 0,
                scramble_color: Tuple[int, int, int] = ACCENT_DK) -> None:
    """Cyberpunk 'decrypting' reveal: a growing real prefix + a per-frame scrambled
    tail.  Monospace keeps the width stable so nothing reflows; ``progress`` >= 1
    renders the plain string (cheap)."""
    text = str(text)
    n = len(text)
    if n == 0:
        return
    if progress >= 1.0:
        layer.add(x, y, text, color, size=size, bold=bold, anchor=anchor, mono=True)
        return
    cw = _text_width("M", size, bold, mono=True)
    va = anchor[1] if len(anchor) > 1 else "t"
    if anchor and anchor[0] == "r":
        x0 = x - n * cw
    elif anchor and anchor[0] == "m":
        x0 = x - (n * cw) // 2
    else:
        x0 = x
    k = int(max(0.0, min(1.0, progress)) * n)
    revealed, rem = text[:k], text[k:]
    scr = "".join(" " if c == " " else _SCRAMBLE[(seed * 13 + (k + i) * 7 + ord(c)) % len(_SCRAMBLE)]
                  for i, c in enumerate(rem))
    if revealed:
        layer.add(x0, y, revealed, color, size=size, bold=bold, anchor="l" + va, mono=True)
    layer.add(x0 + len(revealed) * cw, y, scr, scramble_color, size=size, bold=bold,
              anchor="l" + va, mono=True)


def label_stack(layer: TextLayer, x: int, y: int, caption, value, *,
                value_color=TEXT, caption_color=DIM, caption_size: int = 10,
                value_size: int = 16, bold: bool = True, right: bool = False,
                progress: float = 1.0, seed: int = 0) -> int:
    """Cyberpunk caption-over-value: a tiny tracked dim caption with a larger value
    beneath (the 'INPUT DATA → ENEMY #1' motif).  Returns the y below the value."""
    anc = "r" if right else "l"
    layer.add(x, y, str(caption).upper(), caption_color, size=caption_size,
              anchor=anc + "t", tracking=1.5)
    vy = y + caption_size + 5
    decode_text(layer, x, vy, value, value_color, progress=progress, size=value_size,
                bold=bold, anchor=anc + "t", seed=seed)
    return vy + value_size + 7


def section_label(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, w: int,
                  text, *, color=DIM) -> None:
    """Tiny dim section header with a hairline rule trailing to the right
    (the 'DMG TYPE / RESISTANCES' divider motif)."""
    t = str(text).upper()
    layer.add(x, y, t, color, size=10, anchor="lt", tracking=1.5)
    tw = _text_width(t, 10) + 14
    if x + tw < x + w:
        cv2.line(frame_bgr, (x + tw, y + 5), (x + w, y + 5), bgr(HAIRLINE), 1, cv2.LINE_AA)


def status_bar(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, w: int,
               text, state: str = "tracking", *, progress: float = 1.0,
               size: int = 12) -> int:
    """Threat-style bar: a filled ``!`` marker square + a label bar whose fill grows
    to ``progress``.  locked→accent, lost→alert, else accent.  Returns bottom y."""
    h = size + 12
    col = {"locked": ACCENT, "lost": ALERT}.get(state, ACCENT)
    mb = h
    fill_region(frame_bgr, x, y, x + mb, y + h, col, 0.92)
    layer.add(x + mb // 2, y + h // 2, "!", INK, size=size, bold=True, anchor="mm", outline=False)
    bx0, bx1 = x + mb + 3, x + w
    fill_region(frame_bgr, bx0, y, bx1, y + h, BG_PANEL, 0.7)
    fw = int((bx1 - bx0) * max(0.0, min(1.0, progress)))
    if fw > 0:
        fill_region(frame_bgr, bx0, y, bx0 + fw, y + h, col, 0.22)
    cv2.rectangle(frame_bgr, (bx0, y), (bx1, y + h), bgr(col), 1, cv2.LINE_AA)
    layer.add(bx0 + 8, y + h // 2, str(text).upper(), TEXT, size=size, bold=True, anchor="lm")
    return y + h


def button_glyph(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, key, label,
                 *, size: int = 12, accent: Tuple[int, int, int] = ACCENT) -> int:
    """Circled key glyph + label (a command-list item).  Returns the right-edge x."""
    r = size // 2 + 3
    cyc = y + r
    cv2.circle(frame_bgr, (x + r, cyc), r, bgr(accent), 1, cv2.LINE_AA)
    layer.add(x + r, cyc, str(key)[:1].upper(), accent, size=size - 2, bold=True,
              anchor="mm", outline=False)
    lx = x + 2 * r + 7
    layer.add(lx, cyc, str(label), TEXT, size=size, anchor="lm")
    return lx + _text_width(str(label), size)


def status_dots(frame_bgr: np.ndarray, layer: TextLayer, x: int, y: int, w: int,
                items: Sequence[Tuple[str, bool]], *, size: int = 11) -> None:
    """A clean labelled status-pip row (filled cyan dot = ok, hollow dim = off) —
    replaces the old warning-triangle triad.  Evenly distributes ``items`` across
    ``w``."""
    n = max(1, len(items))
    step = w // n
    for i, (lbl, ok) in enumerate(items):
        cxp = x + i * step + 5
        cyp = y + size // 2
        if ok:
            cv2.circle(frame_bgr, (cxp, cyp), max(3, size // 3), bgr(ACCENT), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame_bgr, (cxp, cyp), max(3, size // 3), bgr(DIM), 1, cv2.LINE_AA)
        layer.add(cxp + size, cyp, str(lbl).upper(), TEXT if ok else DIM, size=size - 1,
                  anchor="lm")
