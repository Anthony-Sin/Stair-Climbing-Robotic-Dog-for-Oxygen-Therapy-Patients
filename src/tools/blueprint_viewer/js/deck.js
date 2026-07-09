// deck.js
//
// Scroll orchestrator for the pitch deck. Owns everything EXCEPT the two WebGL
// scenes (hero.js = the shared Solution stage; main.js = the demo viewer):
//   - injects every section's HTML from content.js (one source of truth),
//   - builds the consistent fixed nav rail (scrolling also advances sections),
//   - on scroll, reveals each section (text effects), switches the pinned 3D
//     stage's policy as the Solution step-sections pass, and autoplays the demo
//     rollout only while the demo is on screen.

import { SECTIONS, POLICIES, DIAGRAMS, CONTENT, buildVisualHTML, buildCopyHTML } from './content.js';

const deck = document.getElementById( 'deck' );
const navDots = document.getElementById( 'nav-dots' );
const idToIdx = new Map( SECTIONS.map( ( s, i ) => [ s.id, i ] ) );
const isStageId = new Set( SECTIONS.filter( ( s ) => s.kind === 'stage' ).map( ( s ) => s.id ) );

// ---------------------------------------------------------------------------
// 1. Inject content-section HTML (Problem / Results×2 / Potential / Team)
// ---------------------------------------------------------------------------
for ( const [ id, c ] of Object.entries( CONTENT ) ) {

	const sec = document.getElementById( id );
	if ( ! sec ) continue;
	// Guarded: bespoke sections (e.g. Potential, whose visual is the 3D viewer)
	// may not carry every slot — skip whatever they omit instead of throwing.
	const eyebrowEl = sec.querySelector( '.slide-eyebrow' );
	const titleEl = sec.querySelector( '.slide-title' );
	const visualEl = sec.querySelector( '[data-visual]' );
	const copyEl = sec.querySelector( '[data-copy]' );
	if ( eyebrowEl ) eyebrowEl.textContent = c.group;
	if ( titleEl ) titleEl.textContent = c.title;
	if ( visualEl ) visualEl.innerHTML = buildVisualHTML( c.visual );
	if ( copyEl ) copyEl.innerHTML = buildCopyHTML( c.copy );

}

// ---------------------------------------------------------------------------
// 2. Inject the 3 Solution step-panels (they float on the right; the pinned
//    3D stage shows on the left). Each carries its own real Isaac clip.
// ---------------------------------------------------------------------------
function solPanelHTML( p ) {

	const clip = p.clip
		? `<figure class="sol-clip"><div class="sol-clip-frame"><video class="sol-clip-video" muted loop playsinline preload="none" data-src="${ p.clip.src }"></video></div><figcaption>${ p.clip.cap }</figcaption></figure>`
		: '';
	return `<span class="slide-eyebrow">Solution &amp; technology</span>
		<h2 class="slide-title">${ p.title }</h2>
		<div class="sol-diagram">${ DIAGRAMS[ p.id ] || '' }</div>
		${ clip }
		<p class="sol-body">${ p.body }</p>`;

}

for ( const step of document.querySelectorAll( '.sol-step' ) ) {

	const p = POLICIES[ step.dataset.policy ];
	if ( p ) step.querySelector( '.sol-panel' ).innerHTML = solPanelHTML( p );

}

// ---------------------------------------------------------------------------
// 3. Nav rail (consistent, fixed) + prev/next. Clicking scrolls to a section.
// ---------------------------------------------------------------------------
SECTIONS.forEach( ( s ) => {

	const b = document.createElement( 'button' );
	b.type = 'button'; b.className = 'nav-dot'; b.setAttribute( 'role', 'tab' );
	b.dataset.target = s.id; b.setAttribute( 'aria-label', s.label );
	b.innerHTML = `<span class="nav-dot-label">${ s.label }</span>`;
	b.addEventListener( 'click', () => scrollToSection( s.id ) );
	navDots.appendChild( b );

} );

function scrollToSection( id ) {

	const el = document.getElementById( id );
	if ( el ) el.scrollIntoView( { behavior: 'smooth', block: 'start' } );

}

let activeIdx = 0;

function setActiveDot( i ) {

	activeIdx = i;
	[ ...navDots.children ].forEach( ( d, j ) => d.classList.toggle( 'active', j === i ) );

}

document.getElementById( 'nav-prev' ).addEventListener( 'click', () => scrollToSection( SECTIONS[ Math.max( 0, activeIdx - 1 ) ].id ) );
document.getElementById( 'nav-next' ).addEventListener( 'click', () => scrollToSection( SECTIONS[ Math.min( SECTIONS.length - 1, activeIdx + 1 ) ].id ) );

// ---------------------------------------------------------------------------
// 3b. Wheel / keyboard section navigation — ONE gesture advances ONE section.
//     Native scroll-snap alone made the user "scroll a lot" to move down (a
//     trackpad flick barely nudged past a snap point and sprang back). We take
//     over the wheel so a single decisive flick jumps to the next/prev section,
//     while still letting a genuinely-overflowing inner region (long copy)
//     scroll on its own before nav takes over at its boundary.
// ---------------------------------------------------------------------------
let navLock = false;
let navReleaseTimer = null;

function navBy( dir ) {

	const i = Math.min( SECTIONS.length - 1, Math.max( 0, activeIdx + dir ) );
	if ( i !== activeIdx ) scrollToSection( SECTIONS[ i ].id );

}

// True when some ancestor of `node` can still scroll in the wheel's direction —
// so inner scroll wins over section-nav until it hits its own boundary.
function canScrollInside( node, deltaY ) {

	let el = node;
	while ( el && el !== deck && el.nodeType === 1 ) {

		const oy = getComputedStyle( el ).overflowY;
		if ( ( oy === 'auto' || oy === 'scroll' ) && el.scrollHeight > el.clientHeight + 1 ) {

			if ( deltaY > 0 && el.scrollTop + el.clientHeight < el.scrollHeight - 1 ) return true;
			if ( deltaY < 0 && el.scrollTop > 1 ) return true;

		}
		el = el.parentElement;

	}
	return false;

}

// Listen on WINDOW, not #deck: the deck-only listener meant a wheel over any
// fixed overlay (the nav rail, the pinned hero stage) — or before the deck had
// pointer focus — did nothing, so it felt like "I have to click the page first
// to scroll". Window-level catches every wheel regardless of what's under the
// cursor; canScrollInside still lets a genuinely-overflowing inner region scroll
// itself first.
window.addEventListener( 'wheel', ( e ) => {

	if ( Math.abs( e.deltaY ) < Math.abs( e.deltaX ) ) return; // ignore horizontal intent
	if ( canScrollInside( e.target, e.deltaY ) ) return;
	e.preventDefault();

	if ( ! navLock ) { navLock = true; navBy( e.deltaY > 0 ? 1 : -1 ); }
	// Reset the release on every wheel tick so trackpad momentum is swallowed:
	// the lock only lifts ~0.45 s after the LAST wheel event, not the first.
	if ( navReleaseTimer ) clearTimeout( navReleaseTimer );
	navReleaseTimer = setTimeout( () => { navLock = false; }, 450 );

}, { passive: false } );

window.addEventListener( 'keydown', ( e ) => {

	if ( e.target && /^(INPUT|TEXTAREA|SELECT)$/.test( e.target.tagName ) ) return;
	let dir = 0, jump = null;
	if ( e.key === 'ArrowDown' || e.key === 'PageDown' || e.key === ' ' ) dir = 1;
	else if ( e.key === 'ArrowUp' || e.key === 'PageUp' ) dir = -1;
	else if ( e.key === 'Home' ) jump = 0;
	else if ( e.key === 'End' ) jump = SECTIONS.length - 1;
	else return;

	e.preventDefault();
	if ( navLock ) return;
	navLock = true;
	if ( jump !== null ) scrollToSection( SECTIONS[ jump ].id ); else navBy( dir );
	if ( navReleaseTimer ) clearTimeout( navReleaseTimer );
	navReleaseTimer = setTimeout( () => { navLock = false; }, 650 );

} );

// ---------------------------------------------------------------------------
// 4. Clip lazy-load helpers (Solution step clips)
// ---------------------------------------------------------------------------
function playClipIn( step ) {

	const v = step.querySelector( '.sol-clip-video' );
	if ( ! v ) return;
	if ( v.dataset.src && ! v.src ) { v.src = v.dataset.src; }
	const p = v.play(); if ( p && p.catch ) p.catch( () => {} );

}
function pauseClipIn( step ) { const v = step.querySelector( '.sol-clip-video' ); if ( v ) v.pause(); }

// ---------------------------------------------------------------------------
// 5. Activation — the single "this section is now front-and-centre" handler.
//    Drives the nav dot, the pinned stage's policy, the Solution clips, and
//    the demo's autoplay. window.__hero / window.__viewer are set up async by
//    hero.js / main.js once their models load, so every call is guarded.
// ---------------------------------------------------------------------------
function onActivate( id ) {

	if ( idToIdx.has( id ) ) setActiveDot( idToIdx.get( id ) );

	const onStage = isStageId.has( id );
	const hero = window.__hero;
	if ( hero ) {

		if ( onStage ) { hero.showPolicy( document.getElementById( id ).dataset.policy ); hero.setStageVisible( true ); }
		else hero.setStageVisible( false );

	}

	// Solution clips: play the active step's, pause the others.
	for ( const step of document.querySelectorAll( '.sol-step' ) ) {

		if ( step.id === id ) playClipIn( step ); else pauseClipIn( step );

	}

	// The sweep "fall" clip autoplays only while the sweep is on screen (lazy-load
	// its data-src on first arrival).
	const sweepVid = document.querySelector( '#s-sweep video' );
	if ( sweepVid ) {

		if ( id === 's-sweep' ) {

			if ( sweepVid.dataset.src && ! sweepVid.src ) sweepVid.src = sweepVid.dataset.src;
			const pl = sweepVid.play(); if ( pl && pl.catch ) pl.catch( () => {} );

		} else sweepVid.pause();

	}

	// The interactive 3D viewer now lives on the Potential slide as an autoplaying
	// cinematic view: turn cinematic + autoplay ON while Potential is on screen,
	// and pause it otherwise so it isn't rendering a rollout off-screen.
	const viewer = window.__viewer;
	if ( viewer ) {

		const onViewer = ( id === 's-potential' );
		// Gate the heavy render loop FIRST (idle unless Potential is on screen), then
		// drive cinematic + autoplay. setActive(true) paints a frame immediately so
		// there's no blank flash on arrival.
		if ( typeof viewer.setActive === 'function' ) viewer.setActive( onViewer );
		if ( typeof viewer.setCinematic === 'function' ) viewer.setCinematic( onViewer );
		if ( typeof viewer.play === 'function' ) viewer.play( onViewer );

	}

}

// ---------------------------------------------------------------------------
// 6. Observers. `ioActive` picks the front-and-centre section (>=55% visible).
//    `ioReveal` fades each section's content in as it enters (text effects).
// ---------------------------------------------------------------------------
let currentActive = null;

const ioActive = new IntersectionObserver( ( entries ) => {

	for ( const e of entries ) {

		if ( e.isIntersecting && e.intersectionRatio >= 0.55 && e.target.id !== currentActive ) {

			currentActive = e.target.id;
			onActivate( currentActive );

		}

	}

}, { root: deck, threshold: [ 0.55 ] } );

const ioReveal = new IntersectionObserver( ( entries ) => {

	for ( const e of entries ) {

		if ( e.isIntersecting && e.intersectionRatio >= 0.25 ) e.target.classList.add( 'in' );

	}

}, { root: deck, threshold: [ 0.25 ] } );

for ( const s of SECTIONS ) {

	const el = document.getElementById( s.id );
	if ( ! el ) continue;
	ioActive.observe( el );
	ioReveal.observe( el );

}

// If hero.js / main.js finish loading AFTER the first activation fired, re-run
// it once each is ready so the stage/demo pick up the current section.
function reactivateWhenReady( flagName, promiseGetter ) {

	const p = promiseGetter();
	if ( p && p.then ) p.then( () => { if ( currentActive ) onActivate( currentActive ); } );

}
reactivateWhenReady( 'viewer', () => window.__viewer && window.__viewer.ready );
// hero has no ready promise; poll briefly until it exists, then reactivate once.
let heroTries = 0;
const heroWait = setInterval( () => {

	if ( window.__hero || heroTries++ > 60 ) { clearInterval( heroWait ); if ( window.__hero && currentActive ) onActivate( currentActive ); }

}, 250 );
