"""Montage rendering for the stair-sweep presenter: the pure filtergraph builder, the
speed-normalization math, and the ffmpeg orchestration for the 2x3 grid + standalone clips.

Split out of ``sweep_present.py`` (single-responsibility). ``build_filtergraph`` and
``_compute_speeds`` are pure (unit-tested without ffmpeg); ``render_montage``/``render_clip``
shell out to ffmpeg.
"""
import os
import shutil
import subprocess

from sweep_constants import (
    CANVAS_W, CANVAS_H, COLS, CELL_W, CELL_H, CAPTION_BAR_H,
)
from sweep_helpers import log, have, fmt_time, _write_caption


# ---------------------------------------------------------------------------
# montage (pure filtergraph builder + ffmpeg orchestration)
# ---------------------------------------------------------------------------
def build_filtergraph(tiles, duration, fps=30, cell_w=CELL_W, cell_h=CELL_H,
                      canvas_w=CANVAS_W, canvas_h=CANVAS_H):
    """Compose the -filter_complex string for the montage. PURE: paths in, string out.

    Each tile is a dict with:
      kind:        "video" | "image" | "placeholder"
      input_index: ffmpeg -i index for video/image (None for placeholder)
      x, y:        top-left of the cell on the canvas
      m:           setpts multiplier (video only; <1 speeds up)
      pad:         seconds of cloned-tail padding (video only)
      fit:         "cover" | "letterbox" (video only)
      caption:     relative textfile name, or None
      best:        bool -- draw the BEST badge/border
      font:        relative font filename for drawtext
    Font + caption paths are relative (ffmpeg is run with cwd=the montage workdir) so no
    Windows drive-colon ever reaches the filtergraph parser.
    """
    parts = [f"color=c=black:s={canvas_w}x{canvas_h}:r={fps}:d={duration:.3f}[base]"]
    labels = []
    for i, t in enumerate(tiles):
        chain = []
        src = ""
        if t["kind"] == "video":
            src = f"[{t['input_index']}:v]"
            chain.append(f"setpts=PTS*{t['m']:.6f}")
            if t.get("fit") == "letterbox":
                chain.append(f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease")
                chain.append(f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:black")
            else:
                chain.append(f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=increase")
                chain.append(f"crop={cell_w}:{cell_h}")
            if t.get("pad", 0) > 0.01:
                chain.append(f"tpad=stop_mode=clone:stop_duration={t['pad']:.3f}")
            chain.append(f"fps={fps}")
        elif t["kind"] == "image":
            src = f"[{t['input_index']}:v]"
            chain.append(f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=decrease")
            chain.append(f"pad={cell_w}:{cell_h}:(ow-iw)/2:(oh-ih)/2:white")
            chain.append(f"fps={fps}")
        else:  # placeholder
            chain.append(f"color=c=0x141414:s={cell_w}x{cell_h}:r={fps}:d={duration:.3f}")

        if t.get("best"):
            chain.append(f"drawbox=x=0:y=0:w={cell_w}:h={cell_h}:color=gold@0.95:t=8")
        if t.get("caption"):
            font = t.get("font", "font.ttf")
            y0 = cell_h - CAPTION_BAR_H
            chain.append(f"drawbox=x=0:y={y0}:w={cell_w}:h={CAPTION_BAR_H}:color=black@0.55:t=fill")
            chain.append(
                f"drawtext=fontfile={font}:textfile={t['caption']}:fontcolor=white:"
                f"fontsize=25:x=16:y={y0 + 7}:line_spacing=6"
            )
        if t.get("best"):
            font = t.get("font", "font.ttf")
            chain.append("drawbox=x=0:y=0:w=118:h=34:color=gold@0.95:t=fill")
            chain.append(f"drawtext=fontfile={font}:text=BEST:fontcolor=black:fontsize=24:x=20:y=4")

        out_lbl = f"c{i}"
        parts.append(f"{src}{','.join(chain)}[{out_lbl}]")
        labels.append(out_lbl)

    prev = "base"
    for i, t in enumerate(tiles):
        last = i == len(tiles) - 1
        out_lbl = "out" if last else f"o{i}"
        parts.append(f"[{prev}][{labels[i]}]overlay=x={t['x']}:y={t['y']}:shortest=0[{out_lbl}]")
        prev = out_lbl
    return ";".join(parts)


def _compute_speeds(durs, target, mode):
    """Return (multipliers, pads) per duration so tiles finish at `target` seconds."""
    valid = [d for d in durs if d]
    if not valid:
        return [None] * len(durs), [0.0] * len(durs)
    if mode == "uniform":
        ref = max(valid)
        ms = [(target / ref) if d else None for d in durs]
    else:  # sync: every clip finishes at target
        ms = [(target / d) if d else None for d in durs]
    pads = []
    for d, m in zip(durs, ms):
        sped = d * m if (d and m) else 0.0
        pads.append(max(0.0, target - sped))
    return ms, pads


def render_montage(eps, stats_card, out_path, montage_seconds, mode, fit):
    """Build the 2x3 grid montage with ffmpeg. Returns out_path or None."""
    if not have("ffmpeg"):
        log("WARNING: ffmpeg not found on PATH; skipping montage")
        return None
    work = os.path.join(os.path.dirname(out_path), "_montage")
    os.makedirs(work, exist_ok=True)

    # resolve a font into the workdir (relative reference dodges drive-colon escaping)
    font_rel = "font.ttf"
    font_src = next((p for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf",
                                 r"C:\Windows\Fonts\segoeui.ttf") if os.path.exists(p)), None)
    if font_src:
        shutil.copyfile(font_src, os.path.join(work, font_rel))
    else:
        log("WARNING: no system font found; captions may not render")

    # fixed 5 riser cells (ascending) + stats card in cell 5
    grid = list(eps[:5]) + [None] * (5 - len(eps))
    durs = [e["video_dur_s"] if (e and e.get("video")) else None for e in grid]
    ms, pads = _compute_speeds(durs, montage_seconds, mode)

    inputs = []   # ffmpeg -i file list, in input-index order
    tiles = []
    for i, e in enumerate(grid):
        x, y = (i % COLS) * CELL_W, (i // COLS) * CELL_H
        if e and e.get("video"):
            cap = os.path.join(work, f"cap{i}.txt")
            h = e.get("height")
            l1 = (f"{h:.3f} m   {e['label_short']}" if h is not None else e["label_short"])
            l2 = f"{e['verdict_short']}   |   {fmt_time(e.get('real_time_s'))}"
            _write_caption(cap, l1, l2)
            idx = len(inputs)
            inputs.append(os.path.abspath(e["video"]))
            tiles.append({"kind": "video", "input_index": idx, "x": x, "y": y,
                          "m": ms[i] or 1.0, "pad": pads[i], "fit": fit,
                          "caption": f"cap{i}.txt", "best": bool(e.get("_best")),
                          "font": font_rel})
        else:
            cap = os.path.join(work, f"cap{i}.txt")
            label = (f"{e['height']:.3f} m" if (e and e.get("height") is not None) else "(no run)")
            note = "no video recorded" if e else ""
            _write_caption(cap, *([label, note] if note else [label]))
            tiles.append({"kind": "placeholder", "input_index": None, "x": x, "y": y,
                          "caption": f"cap{i}.txt", "font": font_rel})

    # stats card -> cell 5
    sx, sy = (5 % COLS) * CELL_W, (5 // COLS) * CELL_H
    if stats_card and os.path.exists(stats_card):
        idx = len(inputs)
        inputs.append(os.path.abspath(stats_card))
        tiles.append({"kind": "image", "input_index": idx, "x": sx, "y": sy})
    else:
        tiles.append({"kind": "placeholder", "input_index": None, "x": sx, "y": sy})

    fg = build_filtergraph(tiles, montage_seconds)

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    n_video = sum(1 for t in tiles if t["kind"] == "video")
    for p in inputs[:n_video]:
        cmd += ["-i", p]
    for t in tiles:           # image inputs (stats card) need looping for the full duration
        if t["kind"] == "image":
            cmd += ["-loop", "1", "-t", f"{montage_seconds:.3f}", "-i", inputs[t["input_index"]]]
    cmd += ["-filter_complex", fg, "-map", "[out]", "-r", "30", "-t", f"{montage_seconds:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
            "-movflags", "+faststart", os.path.abspath(out_path)]

    log(f"rendering montage ({n_video} clips + stats card) -> {out_path}")
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"ERROR: ffmpeg montage failed (exit {r.returncode}):\n{r.stderr.strip()[:1500]}")
        return None
    return out_path


def render_clip(ep, out_path, montage_seconds, fit, work):
    """Render one riser's standalone labeled + speed-normalized clip."""
    if not (have("ffmpeg") and ep.get("video")):
        return None
    font_rel = "font.ttf"   # already copied by render_montage into a sibling work dir; ensure here too
    font_src = next((p for p in (r"C:\Windows\Fonts\arialbd.ttf", r"C:\Windows\Fonts\arial.ttf") if os.path.exists(p)), None)
    if font_src and not os.path.exists(os.path.join(work, font_rel)):
        shutil.copyfile(font_src, os.path.join(work, font_rel))
    d = ep["video_dur_s"]
    m = (montage_seconds / d) if d else 1.0
    pad = max(0.0, montage_seconds - (d * m if d else 0.0))
    h = ep.get("height")
    cap_name = f"clipcap_{ep['run_leaf']}.txt"
    l1 = (f"{h:.3f} m   {ep['label_short']}" if h is not None else ep["label_short"])
    l2 = f"{ep['verdict_short']}   |   {fmt_time(ep.get('real_time_s'))}"
    _write_caption(os.path.join(work, cap_name), l1, l2)
    tile = {"kind": "video", "input_index": 0, "x": 0, "y": 0, "m": m, "pad": pad,
            "fit": fit, "caption": cap_name, "best": False, "font": font_rel}
    fg = build_filtergraph([tile], montage_seconds, cell_w=1280, cell_h=720,
                           canvas_w=1280, canvas_h=720)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", os.path.abspath(ep["video"]),
           "-filter_complex", fg, "-map", "[out]", "-r", "30", "-t", f"{montage_seconds:.3f}",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
           "-movflags", "+faststart", os.path.abspath(out_path)]
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if r.returncode != 0:
        log(f"ERROR: clip {out_path} failed: {r.stderr.strip()[:600]}")
        return None
    return out_path
