# Live pitch website

The interactive pitch site (the `src/tools/blueprint_viewer/` Three.js
scroll-deck) is published live via GitHub Pages.

- **Live URL:** https://anthony-sin.github.io/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients-Website/
- **Source of truth:** `src/tools/blueprint_viewer/` in this (private) repo.
- **Hosting repo:** the public
  [`Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients-Website`](https://github.com/Anthony-Sin/Stair-Climbing-Robotic-Dog-for-Oxygen-Therapy-Patients-Website)
  repo holds a deploy copy of the viewer at its root so this research repo can
  stay private. A GitHub Actions workflow there (`.github/workflows/deploy-pages.yml`)
  enables Pages and publishes the root on every push to `main`.

## Updating the live site

The public repo is a static copy of the viewer. To push viewer changes live,
sync the contents of `src/tools/blueprint_viewer/` (minus the dev-only
`serve.py`, `pipeline/`, `audit/`, `diag/`, `AGENTS.md`, `IK_OVERHAUL_SPEC.md`)
into the hosting repo's root, keeping its `.nojekyll` and
`.github/workflows/deploy-pages.yml`, then commit and push to `main`. Pages
redeploys automatically.

## Notes

- The site is a no-build-step static bundle: plain ES modules with an import
  map, Three.js r170 vendored under `vendor/`. Any static host works; it must
  be served over HTTP (not `file://`) because it fetches `models/*.glb`.
- `assets/renders/*.png` (the three problem-slide photos) are required by the
  site but were historically dropped by the global `*.png` rule in
  `.gitignore`; a negation now keeps them tracked. The "Team" slide and its
  personal photo remain intentionally excluded.
