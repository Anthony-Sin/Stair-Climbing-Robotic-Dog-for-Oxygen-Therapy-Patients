"""Contact-sheet image generation from run videos and verification PNGs.

Moved verbatim from analyze_run.py as part of a pure structural split.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from analyze_run_constants import (
    FRAMES_PER_VIDEO,
    LABEL_HEIGHT,
    THUMB_H,
    THUMB_W,
    VERIFICATION_PNGS,
    VIDEO_NAMES,
    _CV2_AVAILABLE,
    cv2,
    np,
)


# ---------------------------------------------------------------------------
# Contact-sheet image generation
# ---------------------------------------------------------------------------
def _label_frame(frame: np.ndarray, label: str) -> np.ndarray:
    """Add a dark banner with label text at top of frame."""
    h, w = frame.shape[:2]
    out = np.zeros((h + LABEL_HEIGHT, w, 3), dtype=np.uint8)
    out[:LABEL_HEIGHT, :] = (30, 30, 30)
    cv2.putText(
        out, label, (6, LABEL_HEIGHT - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA,
    )
    out[LABEL_HEIGHT:, :] = frame
    return out


def _thumb(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA)


def extract_video_frames(path: Path, n: int = FRAMES_PER_VIDEO) -> List[Tuple[np.ndarray, str]]:
    """Return (bgr_frame, label) tuples sampled evenly from a video."""
    if not path.exists():
        return []
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total < 1:
        cap.release()
        return []

    indices = []
    if n == 1:
        indices = [0]
    else:
        for i in range(n):
            idx = int(round(i * (total - 1) / (n - 1)))
            indices.append(max(0, min(total - 1, idx)))

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            pct = int(100 * idx / max(1, total - 1))
            frames.append((_thumb(frame), f"{path.name} [{pct}%]"))
    cap.release()
    return frames


def load_png_as_bgr(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    img = cv2.imread(str(path))
    return img


def build_contact_sheet(run_dir: Path) -> Optional[np.ndarray]:
    """Build a contact sheet from all cameras and verification PNGs."""
    if not _CV2_AVAILABLE:
        return None
    videos_dir = run_dir / "videos"
    reports_dir = run_dir / "reports"

    all_cells: List[np.ndarray] = []

    # -- Verification PNGs first (wide scene overview) --
    for png_name in VERIFICATION_PNGS:
        img = load_png_as_bgr(reports_dir / png_name)
        if img is not None:
            cell = _label_frame(_thumb(img), png_name)
            all_cells.append(cell)

    # -- Video frames for each camera --
    for vid_name in VIDEO_NAMES:
        vid_path = videos_dir / vid_name
        frames = extract_video_frames(vid_path, n=FRAMES_PER_VIDEO)
        for frame_bgr, label in frames:
            all_cells.append(_label_frame(frame_bgr, label))

    if not all_cells:
        return None

    # Normalise all cells to same height
    cell_h = THUMB_H + LABEL_HEIGHT
    cell_w = THUMB_W
    for i, cell in enumerate(all_cells):
        if cell.shape[0] != cell_h or cell.shape[1] != cell_w:
            all_cells[i] = cv2.resize(cell, (cell_w, cell_h))

    # Layout: 4 cells per row (1 row per video + verification row)
    COLS = 4
    while len(all_cells) % COLS:
        all_cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    rows = []
    for i in range(0, len(all_cells), COLS):
        rows.append(np.concatenate(all_cells[i:i + COLS], axis=1))
    sheet = np.concatenate(rows, axis=0)
    return sheet
