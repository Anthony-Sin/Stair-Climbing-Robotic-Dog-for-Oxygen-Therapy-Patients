# Blueprint viewer

A no-build-step Three.js "blueprint" viewer for the stair-climbing robot dog:
loads `models/robot.glb`, strips its materials down to a flat blueprint
palette, draws crisp technical-pen outlines with a custom post-processing
edge pass, and lets you scrub the "follow" / "climb" animation clips by
hand with a real `<input type="range">`.

## Running it

### Option A — Claude Code launch config

```
.claude/launch.json
```

already defines a `blueprint-viewer` configuration (`python -m http.server
8741 -d src/tools/blueprint_viewer`). Launch it from the Claude Code preview
tooling, or run the equivalent command yourself:

### Option B — plain `python -m http.server`

```powershell
cd src/tools/blueprint_viewer
python -m http.server 8741
```

then open `http://localhost:8741/`. Any static file server works — the app
is plain ES modules with an import map, no bundler/build step required.
Do NOT open `index.html` directly via `file://` — GLTFLoader's fetch of
`./models/robot.glb` and the ES module imports both require an HTTP origin.

## Regenerating the model

`models/robot.glb` is produced by the `pipeline/` scripts (owned/maintained
separately from this viewer — see `pipeline/` for the baking scripts and
`pipeline/mesh_cache/` for the source `.dae` meshes). Typically:

```powershell
python src/tools/blueprint_viewer/pipeline/bake_gltf.py
```

If `models/robot.glb` is missing or fails to load (404, parse error), the
viewer automatically falls back to an in-code procedural placeholder robot
(boxes-and-legs quadruped + two keyframed clips named `follow`/`climb`) so
every UI control stays fully functional without the real asset. A small
"placeholder model — run pipeline/bake_gltf.py" chip appears on screen in
that case.

## File map

```
index.html                    Page shell, import map, DOM structure for the UI overlay
styles.css                    Blueprint/anime.js-hero aesthetic, light+dark themes, responsive layout
js/main.js                    Scene/camera/renderer setup, model loading, scrub/phase/playback logic,
                               follow-cam, resize handling, window.__viewer debug API
js/palette.js                 Single source of truth for the light/dark color palette (DOM + WebGL)
js/BlueprintEdgesPass.js      Custom EffectComposer Pass: normal+depth capture -> edge-detection shader
js/PlaceholderRobot.js        Procedural placeholder robot + "follow"/"climb" AnimationClip builders
js/PartLabels.js              SVG leader-line part-callout overlay
vendor/                       Locally vendored three.js r170 (three.module.js + examples/jsm addons)
models/                       robot.glb (+ .meta.json sidecar) — NOT owned by this viewer, see above
pipeline/                     Model-baking scripts — NOT owned by this viewer
```

## Debug / automation API

`window.__viewer` is available in the console for scripted verification:

```js
await window.__viewer.ready;             // resolves once the model (real or placeholder) has loaded
window.__viewer.scrub(50);               // set the scrubber to 50%
window.__viewer.setPhase('climb');       // switch phase ('follow' | 'climb')
window.__viewer.getState();              // { phase, timeSec, duration, pct, usingPlaceholder, theme }
window.__viewer.renderFrame();           // manually render one frame without waiting for rAF
                                          // (rAF is throttled to ~0 on a backgrounded/hidden tab —
                                          // this is the escape hatch for headless/automated capture)
window.__viewer.getNodeWorldPosition('patient_root'); // world position of a named scene node, or null
window.__viewer.getCameraState();        // { position, target, near, far }
```

## Tuning knobs

### Edge pass (`js/main.js`, where `BlueprintEdgesPass` is constructed)

| Option            | Default | Effect                                                                 |
|--------------------|---------|-------------------------------------------------------------------------|
| `normalThreshold`  | `0.4`   | Lower = more sensitive to shallow creases/interior detail lines.       |
| `depthThreshold`   | `0.025` | Lower = more sensitive to depth-only (silhouette/occlusion) edges. **Do not drop this much below ~0.01** — a real depth texture has enough quantization noise at these camera distances that a too-tight threshold makes whole flat faces (especially shallow-angle surfaces like stair treads) flicker as false edges instead of producing thin lines. This was empirically tuned; see the comment above the constructor call for the full story. |
| `thickness`        | `1.2`   | Edge line thickness in pixels (resolution-aware — same visual thickness regardless of canvas size/DPR). |

`uOpacity` (edge-ink mix strength) and `uInkColor` (set via the palette, not
directly) are also available on `edgesPass.uniforms` if further tuning is
needed.

### Camera fit (`js/main.js`, `fitCameraToObject`)

* `offsetMultiplier` (default `2.4`, passed as the function's second
  argument) controls how tightly the initial camera frames the robot's
  bounding box at load. It was widened from an initial `1.6` because a
  tight fit combined with a human-scale patient figure standing near the
  robot in the baked scene made the patient loom uncomfortably close/large
  in the default view — a larger multiplier gives breathing room without
  changing what's being fit (still strictly the `robot_base` subtree, per
  spec). Users can always re-frame further with OrbitControls (scroll to
  zoom, drag to orbit).
* `camera.near`/`camera.far` are derived from `controls.minDistance` /
  `controls.maxDistance` (currently `0.6` / `15`) plus the object's own
  size — NOT from `offsetMultiplier` — specifically to keep the near:far
  ratio low enough that the depth buffer retains enough precision for the
  edge pass's depth-discontinuity term (see the code comment for the
  numbers; a too-wide ratio was the root cause of the depth-threshold issue
  above).

### Palette (`js/palette.js`)

Both the light and dark palettes live in a single `PALETTES` object shared
by the DOM (via CSS custom properties) and the WebGL scene (background /
material colors / edge-pass ink color), so editing a color there updates
both consistently. `oxygenTankColor` and `patientColor` give those two
named subtrees a slightly different tint from the shared body material —
edit `TINTED_NODE_NAMES` in `js/main.js` to add more tinted subtrees.

## Vendoring notes

Three.js r170 (`0.170.0`) and its `examples/jsm` addons used by this app
(`OrbitControls`, `GLTFLoader`, the postprocessing chain, `FXAAShader`,
`BufferGeometryUtils`, and everything those transitively import) are
vendored locally under `vendor/`, fetched from the pinned jsDelivr CDN
build and verified dependency-complete (no further relative imports were
missed). The import map in `index.html` points at these local files. If
vendoring ever needs to be redone or the version bumped, re-resolve the
full transitive closure of relative imports starting from the addon entry
points listed above — several addon files import from sibling addon files
(e.g. `EffectComposer.js` imports `ShaderPass.js`, `MaskPass.js`,
`CopyShader.js`), not just from `three` itself.

If local vendoring ever proves unreliable in a given environment, the
import map can be repointed at the CDN directly:

```html
<script type="importmap">
{
  "imports": {
    "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
    "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/"
  }
}
</script>
```
