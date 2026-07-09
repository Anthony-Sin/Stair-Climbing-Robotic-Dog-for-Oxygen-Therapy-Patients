// palette.js
//
// Single source of truth for the light/dark color scheme.
// The same object is used to:
//   1. drive the CSS custom properties on <html> (DOM overlay chrome), and
//   2. drive the three.js scene.background / material colors / edge-pass
//      uInkColor uniform,
// so the DOM and the WebGL canvas never fall out of sync when the theme
// toggles.
//
// 2026-07-10: repainted from the monochrome "blueprint" scheme to a toon
// (cel-shaded) palette that mirrors the Isaac Sim set's real material colors
// -- oak-wood stairs, dark-iron handrails, near-white O2 tank on a dark
// cradle, vibrant blue tile floor (see src/sim/isaac/isaac_env.py's
// WOOD_BASE/RAIL_COLOR/floor_color and src/sim/isaac/env/o2_tank.py's
// tank/holder DisplayColor) -- with the robot itself repainted whitish per
// explicit user direction (the real Go2 is black; this viewer's robot is
// NOT trying to match that). These are STYLIZED/toon restatements of those
// sim colors (brighter, more saturated) rather than exact PBR values, so
// they read clearly under MeshToonMaterial's hard-banded shading -- the raw
// sim oak (0.30, 0.19, 0.08) in particular renders almost black-brown once
// quantized into toon bands. Object colors are intentionally the SAME across
// both themes (they represent the "real world" material, not UI chrome) --
// only sceneBackground/ink continue to differ per theme. patientColor is
// UNCHANGED from the prior scheme (explicitly out of scope -- the person/
// patient's color is not part of this repaint).
export const PALETTES = {
	light: {
		name: 'light',
		// DOM
		backdrop: '#1b1b18',
		sheet: '#d6d2ca',
		sheetBorder: 'rgba(0,0,0,.25)',
		ink: '#2f2c28',
		headline: '#262320',
		// three.js
		sceneBackground: 0xd6d2ca,
		// Subtle vertical backdrop gradient (top -> bottom), replacing the flat
		// grey with a soft "studio set" falloff for a bit more depth/game-vibe.
		// Endpoints stay close to sceneBackground so the canvas blends into the
		// surrounding DOM sheet at its rounded corners.
		bgGradientTop: 0xe3e0d9,
		bgGradientBottom: 0xc7c2b8,
		materialColor: 0xcbc6bb, // unused fallback now that every real subtree is tagged (see TINTED_NODE_NAMES) -- kept as an inert default for any untagged mesh
		robotColor: 0xe7e3d9, // toon whitish body/legs (per explicit user direction, not the real Go2's black)
		oxygenTankColor: 0x8fc4cf, // clean medical teal-cyan -- reads as an oxygen concentrator and pops off the silver robot/white body instead of blending into them
		cradleRailsColor: 0x333333, // dark holder/cradle -- matches o2_tank.py's holder_geom DisplayColor (0.2,0.2,0.2)
		stairsColor: 0xc07d3c, // warm toon oak -- richer/brighter than the raw sim WOOD_BASE (0.30,0.19,0.08) so the large flat stair faces read as stylized honey-oak, not flat cardboard-brown
		handrailColor: 0x332e29, // dark iron rails -- toon restatement of isaac_env.py's RAIL_COLOR (0.22,0.20,0.18)
		groundColor: 0x3fa3ee, // vivid toon tile blue -- baked into the procedural tile texture (see makeGroundTileTexture), a touch more saturated than the sim's floor_color (0.32,0.60,0.84) for a game-floor pop
		groundGroutColor: 0x2d5f8a, // blue-toned grout so the grid reads as part of the tile field (a cool tile-on-tile seam) rather than a dark charcoal line cutting across it
		patientColor: 0xb0552f, // warm terracotta accent -- the patient is the one
		// thing in the scene allowed actual color (per design direction), so it
		// reads as a person against the otherwise monochrome blueprint robot/set.
		// UNCHANGED by the 2026-07-10 toon repaint.
		logoColor: 0x2b2723, // dark charcoal text -- the old white-on-black logo would vanish against the new whitish body
		inkColorGl: 0x1a1714, // near-black ink for a bold anime outline (darker than the old 0x2f2c28)
	},
	dark: {
		name: 'dark',
		// DOM
		backdrop: '#0f0f0d',
		sheet: '#2a2925',
		sheetBorder: 'rgba(0,0,0,.35)',
		ink: '#d8d4cc',
		headline: '#e8e4dc',
		// three.js
		sceneBackground: 0x2a2925,
		bgGradientTop: 0x34302a,
		bgGradientBottom: 0x201e1b,
		materialColor: 0x3a3934, // unused fallback, see light theme's comment
		robotColor: 0xe7e3d9,
		oxygenTankColor: 0x8fc4cf, // clean medical teal-cyan (same as light) so the payload reads as oxygen equipment
		cradleRailsColor: 0x333333,
		stairsColor: 0xc07d3c,
		handrailColor: 0x332e29,
		groundColor: 0x3fa3ee,
		groundGroutColor: 0x2d5f8a,
		patientColor: 0xc96a3f, // brighter terracotta so the accent still reads
		// against the darker theme's paper tone (same hue as light, lifted value).
		// UNCHANGED by the 2026-07-10 toon repaint.
		logoColor: 0x2b2723,
		// Bold BLACK outline in dark theme too (was light 0xd8d4cc, which inverted
		// the ink to a pale line). The anime outline should read black regardless
		// of theme -- the robot/wood/floor are light/colored enough that a black
		// outline stays legible against the dark backdrop.
		inkColorGl: 0x1a1714,
	},
};

/** Convert a 0xRRGGBB int to a "#rrggbb" CSS hex string. */
export function glColorToCss(hex) {
	return '#' + hex.toString(16).padStart(6, '0');
}

/**
 * Apply a palette to the document: sets CSS custom properties on
 * document.documentElement so styles.css can reference var(--ink) etc.
 */
export function applyPaletteToDom(palette) {
	const root = document.documentElement;
	root.style.setProperty('--backdrop', palette.backdrop);
	root.style.setProperty('--sheet', palette.sheet);
	root.style.setProperty('--sheet-border', palette.sheetBorder);
	root.style.setProperty('--ink', palette.ink);
	root.style.setProperty('--headline', palette.headline);
	root.setAttribute('data-theme', palette.name);
}
