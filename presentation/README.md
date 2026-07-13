# TASH pitch — PowerPoint edition

A PowerPoint (`.pptx`) mirror of the live pitch website
(`src/tools/blueprint_viewer/`), for offline / projector presenting.

## Files

| File | Use |
|------|-----|
| **`TASH_pitch.pptx`** | **Primary deck — present with this one.** 10 slides, embedded playable videos, robot shown as a high-res render. Opens reliably in PowerPoint (Windows/Mac), Keynote, and Google Slides. |
| `TASH_pitch_3D.pptx` | Same deck, but the Architecture slide carries the robot as a **native, rotatable 3D model** (`.glb`) instead of a render. Experimental — see note below. |
| `build/` | The generator (`build.py`), the 3D injector (`build_3d.py`), the preview renderer, and `content.json` (slide text pulled from the website) — so the deck can be regenerated. |

## What carried over from the website

- All 10 sections: Cover · Problem · Architecture · Walking · Climbing · Live demo · Training · Height sweep · Challenges · Potential.
- The two-column "sheet" design, palette and type (Segoe UI / Consolas — the site's fonts).
- The **SVG diagrams and charts** (docker/control-loop diagrams, training + sweep charts), rendered at high DPI so they stay crisp.
- The Isaac Sim **render photos** and the **3D scenes** (annotated robot, topple, living-room).
- The **demo videos**, embedded and playable in the slideshow: the follow HUD, the six rising-riser climb clips, the flat-ground walk, the climb rollout, and the fall clip.

## Videos

Each video is a real `.mp4` (H.264) embedded in the file, with a branded poster
frame. They play on click in Slide Show mode. Present in **PowerPoint,
Keynote, Edge, or Chrome/Google Slides** — all decode H.264. (A stripped
codec-less browser will show the poster but not play; a normal presentation
machine is fine.)

## Native 3D robot

The website's live WebGL/Three.js scroll interaction can't exist inside a
`.pptx` — but PowerPoint *does* support native 3D models. Two ways to get it:

1. **`TASH_pitch_3D.pptx`** already embeds `robot.glb` on the Architecture
   slide as a native 3D object (drag to rotate; add a *Morph* transition or a
   *3D Scene → Turntable* animation for motion). It includes an image fallback,
   so it still opens and shows the robot in readers without 3D support. This is
   **experimental** — it was validated structurally but not opened in desktop
   PowerPoint from the build environment. If it ever shows a repair prompt, use
   the primary deck instead.
2. **One-click, guaranteed:** in `TASH_pitch.pptx`, go to the Architecture
   slide → **Insert ▸ 3D Models ▸ This Device** and pick
   `../src/tools/blueprint_viewer/models/robot.glb`. PowerPoint imports it as a
   fully native, rotatable/animatable 3D object.

## Regenerating

```sh
cd presentation/build
python3 build.py        # -> TASH_pitch.pptx   (needs the captured assets/ + posters/)
python3 build_3d.py     # -> TASH_pitch_3D.pptx (injects the native 3D model)
```

The image assets (diagrams/charts/scenes) are high-DPI screenshots captured
from the running website; `build.py` references them from `assets/`. The
website's `models/robot.glb` and `assets/clips/*.mp4` are used directly.
