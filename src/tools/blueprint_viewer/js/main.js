// main.js
//
// Entry point for the blueprint viewer: scene/camera/renderer setup,
// GLTFLoader (with a fully-functional in-code placeholder fallback),
// the blueprint post-processing pipeline, phase/scrub UI wiring, the
// moving-robot follow-cam, part-label leader lines, and the
// window.__viewer debug/verification API.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { FXAAShader } from 'three/addons/shaders/FXAAShader.js';

import { PALETTES, applyPaletteToDom } from './palette.js';
import { BlueprintEdgesPass } from './BlueprintEdgesPass.js';
import { buildPlaceholderRobot } from './PlaceholderRobot.js';
import { PartLabels } from './PartLabels.js';

// ===========================================================================
// DOM references
// ===========================================================================

const canvasHost = document.getElementById( 'canvas-host' );
const scrubber = document.getElementById( 'scrubber' );
const timeReadout = document.getElementById( 'time-readout' );
const phaseFollowBtn = document.getElementById( 'phase-follow' );
const phaseClimbBtn = document.getElementById( 'phase-climb' );
const themeToggle = document.getElementById( 'theme-toggle' );
const trackingToggle = document.getElementById( 'tracking-toggle' );
const playToggle = document.getElementById( 'play-toggle' );
const modelWarning = document.getElementById( 'model-warning' );

// ===========================================================================
// Renderer / scene / camera
//
// NOTE: canvasHost.clientWidth/clientHeight can legitimately read 0 here if
// this module executes before the browser has committed a layout pass for
// a just-inserted host element (observed in practice, not hypothetical —
// it wedges the canvas at a permanent 0x0 via three's setSize(w,h) inline
// style, which a plain CSS width:100% rule cannot override). So: construct
// with a throwaway 1x1 size, then let the single handleResize() function
// (defined below, also used by ResizeObserver/window resize) perform the
// real initial sizing once as part of boot. One measurement path, used
// both at startup and on every subsequent resize, instead of two.
// ===========================================================================

const renderer = new THREE.WebGLRenderer( { antialias: true, alpha: false } );
renderer.setPixelRatio( Math.min( window.devicePixelRatio || 1, 2 ) );
renderer.setSize( 1, 1 );
canvasHost.insertBefore( renderer.domElement, canvasHost.firstChild );

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera( 45, 1, 0.05, 100 );
camera.position.set( 1.6, 1.2, 2.2 );

const controls = new OrbitControls( camera, renderer.domElement );
controls.enableDamping = true;
controls.minDistance = 0.6;
controls.maxDistance = 15;
controls.autoRotate = false;
controls.target.set( 0, 0.5, 0 );
controls.update();

// Lighting: hemisphere + soft directional, for faint tonal separation only
// (the look is line-art, not a shaded render — the edge pass carries the
// actual "drawing").
// Measured against the anime.js reference: fills must stay within ~10% of the
// paper tone (a near-black hemisphere ground bounce + 0.6 directional read as a
// clay render, with side faces dropping to ~25% brightness). Intensity 2.6 was
// calibrated by pixel-sampling a live render: with three's physical light units
// (hemisphere irradiance is divided by pi) it puts an upward #dad6ce face at
// ~sRGB 205 against the #d6d2ca (214) paper background.
const hemiLight = new THREE.HemisphereLight( 0xffffff, 0xd8d4cc, 2.6 );
scene.add( hemiLight );
const dirLight = new THREE.DirectionalLight( 0xffffff, 0.15 );
dirLight.position.set( 3, 5, 2 );
scene.add( dirLight );

// ===========================================================================
// Palette / theme
// ===========================================================================

let currentThemeName = localStorage.getItem( 'blueprint-viewer-theme' ) || 'light';
if ( currentThemeName !== 'light' && currentThemeName !== 'dark' ) currentThemeName = 'light';

function applyTheme( name ) {

	currentThemeName = name;
	const palette = PALETTES[ name ];

	applyPaletteToDom( palette );
	scene.background = new THREE.Color( palette.sceneBackground );

	if ( bodyMaterial ) bodyMaterial.color.set( palette.materialColor );
	if ( oxygenTankMaterial ) oxygenTankMaterial.color.set( palette.oxygenTankColor );
	if ( patientMaterial ) patientMaterial.color.set( palette.patientColor );

	if ( edgesPass ) edgesPass.setInkColor( palette.inkColorGl );
	if ( partLabels ) partLabels.setInkColor( palette.ink );

	themeToggle.textContent = name;
	themeToggle.setAttribute( 'aria-pressed', name === 'dark' ? 'true' : 'false' );

	localStorage.setItem( 'blueprint-viewer-theme', name );

}

// ===========================================================================
// Blueprint materials
//
// Traverse the loaded model, strip all textures/materials, and assign a
// flat MeshStandardMaterial (color-only, roughness 1 / metalness 0) so the
// edge pass is the only thing doing "shading". A few named subtrees get a
// slightly different tint per the design spec (oxygen tank lighter,
// patient darker) while sharing the same roughness/metalness treatment.
// ===========================================================================

function makeBlueprintMaterial( colorHex ) {

	return new THREE.MeshStandardMaterial( {
		color: colorHex,
		roughness: 1,
		metalness: 0,
	} );

}

let bodyMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].materialColor );
let oxygenTankMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].oxygenTankColor );
let patientMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].patientColor );

const TINTED_NODE_NAMES = {
	oxygen_tank: () => oxygenTankMaterial,
	patient_root: () => patientMaterial,
};

/**
 * Strip all textures/materials from a loaded model's meshes and replace
 * them with the flat blueprint palette, disposing the originals. Named
 * subtrees (oxygen tank, patient) get their slightly-tinted variant;
 * everything else gets the shared body material.
 */
function applyBlueprintMaterials( root ) {

	// Find tinted-subtree roots first so their descendants inherit the tint
	// even if the tint node itself isn't a Mesh.
	const tintedAncestors = [];
	root.traverse( ( node ) => {

		if ( TINTED_NODE_NAMES[ node.name ] ) {

			tintedAncestors.push( { node, materialFn: TINTED_NODE_NAMES[ node.name ] } );

		}

	} );

	function tintFor( mesh ) {

		for ( const { node, materialFn } of tintedAncestors ) {

			let p = mesh;
			while ( p ) {

				if ( p === node ) return materialFn();
				p = p.parent;

			}

		}

		return bodyMaterial;

	}

	root.traverse( ( node ) => {

		if ( ! node.isMesh ) return;

		const oldMaterials = Array.isArray( node.material ) ? node.material : [ node.material ];
		for ( const mat of oldMaterials ) {

			if ( ! mat ) continue;
			for ( const key of [ 'map', 'normalMap', 'roughnessMap', 'metalnessMap', 'aoMap', 'emissiveMap', 'alphaMap' ] ) {

				if ( mat[ key ] ) mat[ key ].dispose();

			}
			mat.dispose();

		}

		node.material = tintFor( node );
		node.castShadow = false;
		node.receiveShadow = false;

	} );

}

// ===========================================================================
// Post-processing: RenderPass -> BlueprintEdgesPass -> FXAA -> OutputPass
// ===========================================================================

const composer = new EffectComposer( renderer );

const renderPass = new RenderPass( scene, camera );
composer.addPass( renderPass );

const edgesPass = new BlueprintEdgesPass( scene, camera, {
	inkColor: PALETTES[ currentThemeName ].inkColorGl,
	// 0.4 was tuned on the primitive-built robot; the real Isaac Go2 mesh
	// (324k tris of sculpted surface detail) saturates into dark speckle at
	// viewing distance with it. 0.55 was chosen by A/B captures at the
	// climb-summit wide shot: distance noise gone, close-up creases (logo,
	// panel lines) intact. 0.7 starts erasing leg interior definition.
	normalThreshold: 0.55,
	// NOTE: this was originally 0.0025 and looked correct in code review,
	// but empirically (see debug captures during development) it was WAY
	// too tight for a real depth texture's quantization noise at these
	// distances — entire flat faces (especially the shallow-angle stair
	// treads) flickered white/black as false "edges" instead of getting
	// thin silhouette lines. 0.025 (10x looser) was verified to produce
	// clean, thin, silhouette/occlusion-only depth edges with no
	// checkering, while normalThreshold independently and correctly
	// covers interior creases (see BlueprintEdgesPass.js class doc).
	depthThreshold: 0.025,
	thickness: 1.2,
} );
composer.addPass( edgesPass );

const fxaaPass = new ShaderPass( FXAAShader );
composer.addPass( fxaaPass );

const outputPass = new OutputPass();
composer.addPass( outputPass );

function updateFxaaResolution() {

	const pixelRatio = renderer.getPixelRatio();
	fxaaPass.material.uniforms[ 'resolution' ].value.set(
		1 / ( canvasHost.clientWidth * pixelRatio ),
		1 / ( canvasHost.clientHeight * pixelRatio ),
	);

}

// ===========================================================================
// Part labels overlay
// ===========================================================================

const partLabels = new PartLabels( canvasHost, camera );
partLabels.setInkColor( PALETTES[ currentThemeName ].ink );

// ===========================================================================
// Model state (shared across load/placeholder/phase-switch/scrub)
// ===========================================================================

/** @type {THREE.Object3D | null} */
let modelRoot = null;
/** @type {THREE.Object3D | null} */
let robotBase = null;
/** @type {THREE.AnimationMixer | null} */
let mixer = null;
/** @type {Map<string, THREE.AnimationAction>} */
let phaseActions = new Map(); // phaseName -> action
/** @type {Map<string, THREE.AnimationClip>} */
let phaseClips = new Map();
let currentPhase = 'follow';
let usingPlaceholder = false;

// Follow-cam bookkeeping: last known robot_base world position, used to
// translate the camera by the same delta the target moves each frame.
const _lastBaseWorldPos = new THREE.Vector3();
const _curBaseWorldPos = new THREE.Vector3();
const _baseDelta = new THREE.Vector3();
let trackingEnabled = true;
let hasLastBasePos = false;

// Playback (optional "play" chip) state — see PLAYBACK section below.
let isPlaying = false;
let lastPlaybackTimestamp = 0;

// ===========================================================================
// Camera fit — frame the robot_base subtree's bbox at t=0 on load
// ===========================================================================

function fitCameraToObject( object3d, offsetMultiplier = 2.4 ) {

	const box = new THREE.Box3().setFromObject( object3d );
	if ( box.isEmpty() ) return;

	const size = box.getSize( new THREE.Vector3() );
	const center = box.getCenter( new THREE.Vector3() );

	const maxDim = Math.max( size.x, size.y, size.z ) || 1;
	const fitDistance = ( maxDim * offsetMultiplier ) / Math.tan( ( camera.fov * Math.PI ) / 360 );

	const direction = new THREE.Vector3( 0.7, 0.45, 1 ).normalize();
	camera.position.copy( center ).addScaledVector( direction, fitDistance );

	// near/far: sized from the OrbitControls zoom range + object extent, NOT
	// from fitDistance*{tiny,huge} multipliers. The previous version derived
	// near=fitDistance/100, far=fitDistance*50, which for a ~1m robot gave a
	// near:far ratio of ~5000:1 — with a standard (non-logarithmic) depth
	// buffer that crushes almost all depth precision into the first few
	// percent of that range, leaving the actual geometry (which sits right
	// where the robot is, a few meters out) with barely any distinguishable
	// depth values between neighbouring pixels on the SAME flat face. That
	// silently broke BlueprintEdgesPass's depth-discontinuity term (entire
	// faces flickered as "edges" from raw depth-texture quantization noise,
	// see git history / incident notes for the debug captures that isolated
	// this). Keeping near:far comfortably under ~1000:1 here is what fixes
	// it — this is a real, load-bearing constraint of the edge pass, not
	// just camera-fit tuning.
	camera.near = Math.max( 0.01, controls.minDistance * 0.5 );
	camera.far = controls.maxDistance + maxDim * 4;
	camera.updateProjectionMatrix();

	controls.target.copy( center );
	controls.update();

}

// ===========================================================================
// Phase / action wiring
//
// Both "follow" and "climb" AnimationActions are started with .play() and
// immediately paused, then left paused permanently. Switching phases is
// just swapping action WEIGHTS (active=1, inactive=0) — never calling
// .stop()/.play() again — so switching is instant and glitch-free, and
// the scrub logic below (which sets .time directly) keeps working
// uniformly for whichever action is currently active.
// ===========================================================================

function setupActionsFromClips( clips ) {

	phaseActions.clear();
	phaseClips.clear();

	// Map clips by NAME ("follow", "climb"); fall back to clips[0]/clips[1]
	// positionally if names don't match.
	let followClip = clips.find( ( c ) => c.name === 'follow' );
	let climbClip = clips.find( ( c ) => c.name === 'climb' );

	if ( ! followClip && clips[ 0 ] ) followClip = clips[ 0 ];
	if ( ! climbClip && clips[ 1 ] ) climbClip = clips[ 1 ];
	if ( ! climbClip && clips[ 0 ] && clips[ 0 ] !== followClip ) climbClip = clips[ 0 ];

	if ( followClip ) {

		phaseClips.set( 'follow', followClip );
		const action = mixer.clipAction( followClip );
		action.play();
		action.paused = true;
		action.weight = 1;
		action.enabled = true;
		phaseActions.set( 'follow', action );

	}

	if ( climbClip ) {

		phaseClips.set( 'climb', climbClip );
		const action = mixer.clipAction( climbClip );
		action.play();
		action.paused = true;
		action.weight = 0;
		action.enabled = true;
		phaseActions.set( 'climb', action );

	}

}

function setPhase( phaseName, { resetSlider = true } = {} ) {

	if ( ! phaseActions.has( phaseName ) ) return;

	currentPhase = phaseName;

	for ( const [ name, action ] of phaseActions ) {

		action.weight = name === phaseName ? 1 : 0;

	}

	phaseFollowBtn.setAttribute( 'aria-pressed', phaseName === 'follow' ? 'true' : 'false' );
	phaseClimbBtn.setAttribute( 'aria-pressed', phaseName === 'climb' ? 'true' : 'false' );

	if ( resetSlider ) {

		const action = phaseActions.get( phaseName );
		action.time = 0;
		scrubber.value = '0';
		if ( mixer ) mixer.update( 0 );

	}

	updateTimeReadout();

}

// ===========================================================================
// *** THE CRUCIAL SCRUBBING LOGIC ***
//
// This is the point of the whole deliverable, so it's commented heavily:
//
// Every phase action is .play()'d once at setup time and then immediately
// .paused = true FOREVER. The render loop below never advances playback
// with clock.getDelta() — mixer.update(dt) is NEVER called with a nonzero
// dt from the rAF loop while scrub-driven. Instead:
//
//   1. The <input type="range"> fires an 'input' event with a 0..100 value.
//   2. We map that value to a TIME on the currently-active clip:
//        activeAction.time = (v / 100) * clip.duration
//   3. We call mixer.update(0) — passing a delta of ZERO. AnimationMixer's
//      internal accumulation still re-evaluates every active action's pose
//      AT ITS CURRENT .time and writes it to the scene graph, but because
//      the delta is 0, no action's .time is advanced by the update call
//      itself. This forces a pose re-evaluation at the new scrub time
//      without "playing" anything.
//
// The net effect: dragging the slider teleports the skeleton/robot_base to
// an arbitrary point in the clip, instantly and deterministically, with no
// dependency on frame rate or wall-clock time. This is what lets scrubbing
// feel like moving a physical film reel rather than fast-forwarding a
// video.
//
// The optional "play" chip (see PLAYBACK section) reuses this exact same
// primitive: it computes a new .time from a rAF timestamp delta itself,
// then calls mixer.update(0) — it never lets the mixer do the time
// advancement. This keeps ONE authoritative path for "what pose is shown",
// whether you're dragging or playing.
// ===========================================================================

function scrubToPercent( pct ) {

	pct = THREE.MathUtils.clamp( pct, 0, 100 );

	const action = phaseActions.get( currentPhase );
	if ( ! action || ! mixer ) return;

	const clip = phaseClips.get( currentPhase );
	const t = ( pct / 100 ) * clip.duration;

	action.time = t;      // (1)+(2) above: set the authoritative scrub time
	mixer.update( 0 );    // (3) above: force pose re-evaluation, advance nothing

	scrubber.value = String( pct );
	updateTimeReadout();

}

function updateTimeReadout() {

	const action = phaseActions.get( currentPhase );
	const clip = phaseClips.get( currentPhase );
	if ( ! action || ! clip ) return;

	const t = action.time.toFixed( 2 ).padStart( 5, '0' );
	const total = clip.duration.toFixed( 2 ).padStart( 5, '0' );
	timeReadout.textContent = `t ${t} / ${total} s · ${currentPhase}`;

}

scrubber.addEventListener( 'input', () => {

	// Slider drag while playing pauses playback (scrub is authoritative).
	if ( isPlaying ) setPlaying( false );

	scrubToPercent( parseFloat( scrubber.value ) );

} );

// ===========================================================================
// Phase buttons
// ===========================================================================

phaseFollowBtn.addEventListener( 'click', () => setPhase( 'follow' ) );
phaseClimbBtn.addEventListener( 'click', () => setPhase( 'climb' ) );

// ===========================================================================
// Optional play/pause chip
//
// Advances activeAction.time itself from rAF timestamp deltas, then calls
// mixer.update(0) (never mixer.update(dt)) and syncs the slider each
// frame — see the big comment block above for why. Stops at clip end.
// ===========================================================================

function setPlaying( shouldPlay ) {

	isPlaying = shouldPlay;
	playToggle.textContent = shouldPlay ? 'pause' : 'play';
	playToggle.setAttribute( 'aria-pressed', shouldPlay ? 'true' : 'false' );
	lastPlaybackTimestamp = performance.now();

}

playToggle.addEventListener( 'click', () => setPlaying( ! isPlaying ) );

function stepPlayback( nowMs ) {

	if ( ! isPlaying ) return;

	const action = phaseActions.get( currentPhase );
	const clip = phaseClips.get( currentPhase );
	if ( ! action || ! clip || ! mixer ) return;

	const dtSec = Math.max( 0, ( nowMs - lastPlaybackTimestamp ) / 1000 );
	lastPlaybackTimestamp = nowMs;

	let nextTime = action.time + dtSec;
	if ( nextTime >= clip.duration ) {

		nextTime = clip.duration;
		setPlaying( false );

	}

	action.time = nextTime;
	mixer.update( 0 ); // scrub-authoritative: never mixer.update(dtSec)

	const pct = clip.duration > 0 ? ( nextTime / clip.duration ) * 100 : 0;
	scrubber.value = String( pct );
	updateTimeReadout();

}

// ===========================================================================
// Keyboard: Left/Right nudge slider +/-0.5
// ===========================================================================

window.addEventListener( 'keydown', ( ev ) => {

	if ( ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement ) return;

	if ( ev.key === 'ArrowLeft' || ev.key === 'ArrowRight' ) {

		if ( isPlaying ) setPlaying( false );

		const delta = ev.key === 'ArrowLeft' ? -0.5 : 0.5;
		const next = THREE.MathUtils.clamp( parseFloat( scrubber.value ) + delta, 0, 100 );
		scrubToPercent( next );
		ev.preventDefault();

	}

} );

// ===========================================================================
// Theme + tracking toggles
// ===========================================================================

themeToggle.addEventListener( 'click', () => {

	applyTheme( currentThemeName === 'light' ? 'dark' : 'light' );

} );

trackingToggle.addEventListener( 'click', () => {

	trackingEnabled = ! trackingEnabled;
	trackingToggle.textContent = `tracking · ${ trackingEnabled ? 'on' : 'off' }`;
	trackingToggle.setAttribute( 'aria-pressed', trackingEnabled ? 'true' : 'false' );
	if ( trackingEnabled ) hasLastBasePos = false; // resync delta baseline on re-enable

} );

// ===========================================================================
// Model loading: GLTFLoader with placeholder fallback
// ===========================================================================

function disposeModelRoot() {

	if ( ! modelRoot ) return;
	scene.remove( modelRoot );
	modelRoot.traverse( ( node ) => {

		if ( node.isMesh ) {

			node.geometry?.dispose();

		}

	} );
	modelRoot = null;

}

function finishModelSetup( root, clips, baseNode ) {

	disposeModelRoot();

	modelRoot = root;
	robotBase = baseNode || root.getObjectByName( 'robot_base' ) || root;

	applyBlueprintMaterials( root );
	scene.add( root );

	mixer = new THREE.AnimationMixer( root );
	setupActionsFromClips( clips );

	partLabels.setSceneRoot( root );

	setPhase( 'follow', { resetSlider: true } );

	hasLastBasePos = false;
	fitCameraToObject( robotBase );

}

function loadPlaceholder( reason ) {

	usingPlaceholder = true;
	modelWarning.hidden = false;

	if ( reason ) console.warn( '[blueprint-viewer] falling back to placeholder model:', reason );

	const { root, clips, robotBase: baseNode } = buildPlaceholderRobot( bodyMaterial );
	finishModelSetup( root, clips, baseNode );

}

function loadRealModel() {

	const loader = new GLTFLoader();

	return new Promise( ( resolve ) => {

		loader.load(
			'./models/robot.glb',
			( gltf ) => {

				usingPlaceholder = false;
				modelWarning.hidden = true;

				const root = gltf.scene || gltf.scenes[ 0 ];
				const baseNode = root.getObjectByName( 'robot_base' ) || root;
				finishModelSetup( root, gltf.animations || [], baseNode );

				resolve();

			},
			undefined,
			( error ) => {

				console.error( '[blueprint-viewer] GLTFLoader failed to load ./models/robot.glb:', error );
				loadPlaceholder( error?.message || 'load error' );
				resolve();

			},
		);

	} );

}

// ===========================================================================
// Resize handling
// ===========================================================================

function handleResize() {

	const width = canvasHost.clientWidth;
	const height = canvasHost.clientHeight;
	if ( width === 0 || height === 0 ) return;

	const pixelRatio = Math.min( window.devicePixelRatio || 1, 2 );

	renderer.setPixelRatio( pixelRatio );
	renderer.setSize( width, height );

	composer.setPixelRatio( pixelRatio );
	composer.setSize( width, height );

	camera.aspect = width / height;
	camera.updateProjectionMatrix();

	updateFxaaResolution();

}

const resizeObserver = new ResizeObserver( () => handleResize() );
resizeObserver.observe( canvasHost );
window.addEventListener( 'resize', handleResize );

// ===========================================================================
// Render loop
//
// Every frame: controls.update() (damping), follow-cam target/position
// lerp, label overlay update, composer.render(). The mixer is NEVER
// advanced here with a clock delta — see the scrubbing block above. The
// optional playback chip advances time itself (also via mixer.update(0)),
// independent of this rAF's own clock.
// ===========================================================================

const clock = new THREE.Clock();

function animate() {

	requestAnimationFrame( animate );
	renderFrame();

}

// The actual per-frame work, factored out of the rAF scheduling wrapper
// above so it can also be invoked directly (see window.__viewer.renderFrame
// below) — useful for automated/headless verification tooling where the
// page may be backgrounded and browsers throttle requestAnimationFrame to
// near-zero (rAF is intentionally suspended for hidden tabs; this gives a
// legitimate manual escape hatch without fighting that browser behavior).
function renderFrame() {

	const nowMs = performance.now();
	stepPlayback( nowMs );

	// Follow-cam: because the robot travels metres during a clip, lerp the
	// OrbitControls target toward the robot_base world position and
	// translate the camera by the SAME delta each frame — this orbits
	// around a moving target instead of re-framing/snapping.
	if ( trackingEnabled && robotBase ) {

		robotBase.getWorldPosition( _curBaseWorldPos );

		if ( ! hasLastBasePos ) {

			_lastBaseWorldPos.copy( _curBaseWorldPos );
			hasLastBasePos = true;

		}

		_baseDelta.subVectors( _curBaseWorldPos, _lastBaseWorldPos );

		if ( _baseDelta.lengthSq() > 0 ) {

			camera.position.add( _baseDelta );
			controls.target.add( _baseDelta );

		}

		// Gentle extra lerp toward the base so any accumulated drift (e.g.
		// after a phase switch resets time to 0) settles smoothly rather
		// than snapping.
		const lerpFactor = 1 - Math.pow( 0.001, clock.getDelta() || 0.016 );
		controls.target.lerp( _curBaseWorldPos, Math.min( 1, lerpFactor ) );

		_lastBaseWorldPos.copy( _curBaseWorldPos );

	} else {

		clock.getDelta(); // keep the clock's internal timer sane even when unused

	}

	controls.update();

	partLabels.update( canvasHost.clientWidth, canvasHost.clientHeight );

	composer.render();

}

// ===========================================================================
// Debug / verification API
// ===========================================================================

let resolveReady;
const readyPromise = new Promise( ( resolve ) => { resolveReady = resolve; } );

window.__viewer = {
	ready: readyPromise,
	scrub( pct ) {

		scrubToPercent( pct );

	},
	setPhase( name ) {

		setPhase( name );

	},
	getState() {

		const action = phaseActions.get( currentPhase );
		const clip = phaseClips.get( currentPhase );
		const timeSec = action ? action.time : 0;
		const duration = clip ? clip.duration : 0;

		return {
			phase: currentPhase,
			timeSec,
			duration,
			pct: duration > 0 ? ( timeSec / duration ) * 100 : 0,
			usingPlaceholder,
			theme: currentThemeName,
		};

	},
	/**
	 * Manually run one frame of the render loop (controls.update() + label
	 * update + composer.render()) without waiting for requestAnimationFrame.
	 * Not used by normal interactive operation — the rAF-driven animate()
	 * loop (started at boot) is what drives the app for a real user. This
	 * exists for automated/headless verification tooling, since browsers
	 * throttle rAF to near-zero on a backgrounded/hidden tab.
	 */
	renderFrame() {

		renderFrame();

	},
	/**
	 * Debug helper: world-space position of a named node in the currently
	 * loaded model (real or placeholder), or null if not found. Useful for
	 * diagnosing camera-framing / part-label issues without adding one-off
	 * instrumentation each time.
	 */
	getNodeWorldPosition( name ) {

		if ( ! modelRoot ) return null;
		const node = modelRoot.getObjectByName( name );
		if ( ! node ) return null;
		const pos = new THREE.Vector3();
		node.getWorldPosition( pos );
		return { x: pos.x, y: pos.y, z: pos.z };

	},
	getCameraState() {

		return {
			position: { x: camera.position.x, y: camera.position.y, z: camera.position.z },
			target: { x: controls.target.x, y: controls.target.y, z: controls.target.z },
			near: camera.near,
			far: camera.far,
		};

	},
	/**
	 * Live references for headless verification/calibration tooling only
	 * (e.g. tuning light intensities or edge-pass uniforms in-page without a
	 * reload cycle). Not a stable public API.
	 */
	_internals: { scene, camera, renderer, composer, hemiLight, dirLight, edgesPass },
};

// ===========================================================================
// Boot
// ===========================================================================

applyTheme( currentThemeName );

// Perform the real initial sizing now (see the NOTE on the renderer/camera
// construction above) — canvasHost should have a committed layout by the
// time this module's top-level code finishes running, but guard anyway:
// if it's somehow still 0x0, the ResizeObserver below will catch the next
// genuine size change, and handleResize() itself also calls
// updateFxaaResolution() so that stays in sync too.
handleResize();
updateFxaaResolution();

loadRealModel().then( () => {

	animate();
	resolveReady();

} ).catch( ( err ) => {

	// Should be unreachable (loadRealModel resolves on both success and
	// failure paths via loadPlaceholder), but guard anyway so a boot-time
	// exception never leaves the app fully dark.
	console.error( '[blueprint-viewer] unexpected boot error:', err );
	loadPlaceholder( 'unexpected boot error' );
	animate();
	resolveReady();

} );
