// palette.js
//
// Single source of truth for the light/dark "blueprint" color scheme.
// The same object is used to:
//   1. drive the CSS custom properties on <html> (DOM overlay chrome), and
//   2. drive the three.js scene.background / material colors / edge-pass
//      uInkColor uniform,
// so the DOM and the WebGL canvas never fall out of sync when the theme
// toggles.

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
		materialColor: 0xdad6ce,
		oxygenTankColor: 0xe4e0d8, // "a touch lighter"
		patientColor: 0x000000, // toon black -- the patient is the one
		// thing in the scene allowed actual color (per design direction), so it
		// reads as a person against the otherwise monochrome blueprint robot/set.
		inkColorGl: 0x2f2c28,
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
		materialColor: 0x3a3934,
		oxygenTankColor: 0x46453f,
		patientColor: 0x000000, // toon black so the accent still reads
		// against the darker theme's paper tone (same hue as light, lifted value).
		inkColorGl: 0xd8d4cc,
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
