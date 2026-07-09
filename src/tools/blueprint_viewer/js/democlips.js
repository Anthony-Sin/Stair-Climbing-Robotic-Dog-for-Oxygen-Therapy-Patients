// democlips.js
//
// The ACT-2 demo filmstrip: a row of real Isaac Sim climb clips (rising stair
// heights) pinned along the bottom bar. Requirements:
//   - they AUTO-PLAY when the demo scrolls into view (no click needed), and
//     pause + rewind when it scrolls away (so five muted loops aren't decoding
//     off-screen), and
//   - clicking a tile jumps the interactive 3D rollout to the climb phase, so
//     the strip doubles as a "show me this" control, not just decoration.
//
// Kept out of main.js on purpose: this is pure DOM/video wiring with no three.js
// dependency, and it only touches the public window.__viewer API (never main.js
// internals). The <video> elements ship with preload="none" + a data-src so the
// mp4s don't download until the demo is actually reached.

const strip = document.getElementById( 'demo-filmstrip' );
const viewerSection = document.getElementById( 'viewer-section' );

if ( strip && viewerSection ) {

	const tiles = [ ...strip.querySelectorAll( '.film-tile' ) ];
	const videos = tiles.map( ( t ) => t.querySelector( 'video' ) );
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
			const p = v.play();
			if ( p && p.catch ) p.catch( () => {} ); // autoplay policy may block until interaction; harmless

		}

	}

	function pauseAll() {

		for ( const v of videos ) if ( v ) v.pause();

	}

	// Auto play/pause with the demo's visibility. threshold 0.35 so the clips
	// only spin up once a good chunk of the demo is on screen (i.e. the user has
	// actually arrived at it), not while it's a sliver at the fold.
	new IntersectionObserver(
		( entries ) => { entries[ 0 ].isIntersecting ? playAll() : pauseAll(); },
		{ threshold: 0.35 },
	).observe( viewerSection );

	// Click a tile -> jump the 3D rollout to the climb (window.__viewer is set up
	// by main.js once the model is ready; guard for the pre-ready window).
	for ( const tile of tiles ) {

		tile.addEventListener( 'click', () => {

			const v = window.__viewer;
			if ( v && typeof v.setPhase === 'function' ) v.setPhase( 'climb' );
			viewerSection.scrollIntoView( { behavior: 'smooth', block: 'center' } );

		} );

	}

}
