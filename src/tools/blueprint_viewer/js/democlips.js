// democlips.js
//
// The Live demo (#s-demo) is real Isaac Sim footage: a main OpenCV / YOLO
// "target-acquisition" HUD video (the dog following the patient) plus a left-
// corner rail of real climb clips at rising stair heights. This module:
//   - lazy-loads every clip (they ship preload="none" + data-src so nothing
//     downloads until the demo is actually reached), and
//   - auto-plays them when the Live demo scrolls into view, pausing when it
//     scrolls away so a stack of muted loops isn't decoding off-screen, and
//   - jumps to the Potential slide's 3D viewer (at the climb) when a clip tile
//     is clicked, so the strip doubles as a "show me this in 3D" control.
//
// Kept out of main.js on purpose: pure DOM/video wiring, no three.js dependency
// (it only touches the public window.__viewer API, never main.js internals).

const demo = document.getElementById( 's-demo' );

if ( demo ) {

	const videos = [ ...demo.querySelectorAll( 'video' ) ];
	let loaded = false;

	function ensureLoaded() {

		if ( loaded ) return;
		loaded = true;
		for ( const v of videos ) {

			if ( v && v.dataset.src && ! v.src ) { v.src = v.dataset.src; v.load(); }

		}

	}

	function playAll() {

		ensureLoaded();
		for ( const v of videos ) {

			if ( ! v ) continue;
			// The 0.175 m clip is a PARTIAL climb that auto-quit at ~10 s and loops —
			// at normal speed its early restart can read as a completed climb. Run it
			// visibly faster than the others so the "it never reaches the top, it just
			// resets" is obvious. (Reasserted each play; playbackRate resets on load.)
			if ( v.dataset.src && v.dataset.src.includes( '0p175' ) ) v.playbackRate = 1.9;
			const p = v.play();
			if ( p && p.catch ) p.catch( () => {} ); // autoplay policy may block until interaction; harmless

		}

	}

	function pauseAll() {
		for ( const v of videos ) {
			if ( v ) {
				v.pause();
				try {
					v.currentTime = 0;
				} catch ( e ) {}
			}
		}
	}

	// Auto play/pause with the demo's visibility. threshold 0.35 so the clips only
	// spin up once a good chunk of the demo is on screen (the user has arrived at
	// it), not while it's a sliver at the fold.
	new IntersectionObserver(
		( entries ) => { entries[ 0 ].isIntersecting ? playAll() : pauseAll(); },
		{ threshold: 0.35 },
	).observe( demo );

	// Click a clip -> jump to the Potential slide's 3D viewer at the matching phase
	// (follow for the flat-ground tile, climb for the stair tiles).
	// window.__viewer is set up by main.js once the model is ready.
	for ( const tile of demo.querySelectorAll( '.demo-clip' ) ) {

		tile.addEventListener( 'click', () => {

			const pot = document.getElementById( 's-potential' );
			if ( pot ) {
				setTimeout(() => {
					pot.scrollIntoView( { behavior: 'smooth', block: 'start' } );
				}, 5000);
			}
			const v = window.__viewer;
			if ( v && typeof v.setPhase === 'function' ) v.setPhase( tile.dataset.phase === 'follow' ? 'follow' : 'climb' );

		} );

	}

}
