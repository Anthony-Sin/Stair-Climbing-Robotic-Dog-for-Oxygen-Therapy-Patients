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
import { FontLoader } from 'three/addons/loaders/FontLoader.js';
import { TextGeometry } from 'three/addons/geometries/TextGeometry.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import { PALETTES, applyPaletteToDom } from './palette.js';
import { BlueprintEdgesPass } from './BlueprintEdgesPass.js';
import { buildPlaceholderRobot } from './PlaceholderRobot.js';
import { PartLabels } from './PartLabels.js';
import { PatientHuman } from './PatientHuman.js';

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
const plumbToggle = document.getElementById( 'plumb-toggle' );
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

// Lighting: hemisphere (unquantized ambient fill) + directional (quantized
// into the toon gradient bands below) + a shader-injected fresnel rim — see
// the "Blueprint materials" section for how MeshToonMaterial splits these
// two lights into a smooth indirect term vs. a banded direct term.
// dirLight was raised from the old flat-material value (0.15) because the
// toon gradient map only bands the DIRECTIONAL contribution — at 0.15 it was
// swamped by hemiLight's ambient fill and no bands were visible at all.
// Rebalanced by pixel-sampling a live render so the lit face still lands
// close to the old ~sRGB 205 target against the #d6d2ca (214) paper
// background, but with visible shadow/mid/lit steps across the form.
const hemiLight = new THREE.HemisphereLight( 0xffffff, 0xd8d4cc, 1.4 );
scene.add( hemiLight );
const dirLight = new THREE.DirectionalLight( 0xffffff, 1.6 );
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
	if ( logoMaterial ) logoMaterial.color.set( 0xffffff );

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
// cel-shaded (toon) material so the ink edge pass isn't the ONLY thing
// giving the sculpted mesh (324k tris of rivets/seams/panel lines) a sense
// of form — on a near-shadeless flat fill, that detail read as pure line
// clutter ("wired") instead of a shaded surface. A few named subtrees get a
// slightly different tint per the design spec (oxygen tank lighter, patient
// darker) while sharing the same toon/rim treatment.
//
// MeshToonMaterial quantizes ONLY the directional-light contribution through
// `gradientMap` (a 3-texel NearestFilter lookup -> hard shadow/mid/lit
// bands); the hemisphere light stays a smooth ambient fill on top, same as
// real cel animation (banded key light + flat ambient). A fresnel rim term
// is injected via onBeforeCompile since three's toon material has no built-in
// rim light.
// ===========================================================================

const CEL_GRADIENT_MAP = makeToonGradientMap( [ 0.38, 0.72, 1.0 ] );

const RIM_COLOR = new THREE.Color( 0xffffff );
const RIM_POWER = 2.4;
const RIM_INTENSITY = 0.45;

/** Small NearestFilter 1D texture used as MeshToonMaterial's gradientMap: one texel per band, so lighting snaps between bands instead of a smooth ramp. */
function makeToonGradientMap( levels ) {

	const data = new Uint8Array( levels.length );
	for ( let i = 0; i < levels.length; i ++ ) data[ i ] = Math.round( THREE.MathUtils.clamp( levels[ i ], 0, 1 ) * 255 );

	const texture = new THREE.DataTexture( data, levels.length, 1, THREE.RedFormat );
	texture.minFilter = THREE.NearestFilter;
	texture.magFilter = THREE.NearestFilter;
	texture.wrapS = THREE.ClampToEdgeWrapping;
	texture.wrapT = THREE.ClampToEdgeWrapping;
	texture.generateMipmaps = false;
	texture.needsUpdate = true;
	return texture;

}

function makeBlueprintMaterial( colorHex ) {

	const material = new THREE.MeshToonMaterial( {
		color: colorHex,
		gradientMap: CEL_GRADIENT_MAP,
	} );

	// Fresnel rim light: `vViewPosition` (view-space) is already declared by
	// lights_toon_pars_fragment and `vNormal` (view-space) by
	// normal_pars_fragment, so both are in scope for the injected snippet
	// below without redeclaring them.
	material.onBeforeCompile = ( shader ) => {

		shader.uniforms.uRimColor = { value: RIM_COLOR };
		shader.uniforms.uRimPower = { value: RIM_POWER };
		shader.uniforms.uRimIntensity = { value: RIM_INTENSITY };

		shader.fragmentShader = shader.fragmentShader
			.replace(
				'#define TOON',
				'#define TOON\nuniform vec3 uRimColor;\nuniform float uRimPower;\nuniform float uRimIntensity;',
			)
			.replace(
				'vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;',
				'vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;\n' +
				'\tfloat rimFresnel = pow( 1.0 - max( dot( normalize( vNormal ), normalize( vViewPosition ) ), 0.0 ), uRimPower );\n' +
				'\toutgoingLight += rimFresnel * uRimIntensity * uRimColor;',
			);

	};

	return material;

}

let bodyMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].materialColor );
let oxygenTankMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].oxygenTankColor );
let patientMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].patientColor );
let logoMaterial = makeBlueprintMaterial( 0xffffff );

// "patient_root" is no longer a tint target here: it's a bare transform anchor with
// no mesh of its own (see scene_build.build_patient_node) -- the patient's visible
// geometry is the separately-loaded PatientHuman model below, tinted directly by
// PatientHuman.attachTo().
const TINTED_NODE_NAMES = {
	oxygen_tank: () => oxygenTankMaterial,
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
// Post-processing: RenderPass -> BlueprintEdgesPass -> OutputPass
//
// No FXAA: it sat after the ink pass and treated every crisp ink stroke as
// exactly the high-contrast "jaggy" it exists to blur — softening deliberate
// technical-pen lines into a faint grey smear (measured live: removing FXAA
// collapsed a faint/dashed seam's ambiguous mid-grey pixel count in a test
// region from ~1700 to ~30, most of it converting to solid ink). A
// photorealistic-AA pass fights a line-art aesthetic; the edge pass's own
// smoothstep already supplies the (now-tightened) anti-aliasing this style
// wants.
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

const outputPass = new OutputPass();
composer.addPass( outputPass );

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

// Patient: a real imported+rigged human model (see PatientHuman.js), loaded in
// parallel with robot.glb below and wired up once both are ready. Kicked off here
// (not inside loadRealModel) so the two loads race instead of serializing.
const patientHuman = new PatientHuman();
const patientHumanReady = patientHuman.load();

// robot.meta.json: this pipeline's own stair_spec/landing_far_x_m (see the file
// itself — start_x_m/step_height_m/step_depth_m/step_count/landing_depth_m), needed
// by patientHuman.buildGait() to build the procedural gait's terrain model (see
// PatientGait.buildTerrain). Fetched here (racing the GLB loads, same pattern as
// patientHumanReady above) rather than inside loadRealModel, so a slow/failed fetch
// doesn't serialize behind the (much larger) robot.glb download. On failure: loudly
// console.error and degrade exactly like an Xbot load failure (no patient gait built
// — patientHuman.buildGait() is simply never called below, so the human stays
// un-posed rather than silently falling back to some invented default staircase).
const robotMetaReady = fetch( './models/robot.meta.json' )
	.then( ( r ) => r.json() )
	.catch( ( error ) => {

		console.error( '[blueprint-viewer] failed to load ./models/robot.meta.json — patient gait will not be built:', error );
		return null;

	} );

// ===========================================================================
// Plumb line: a literal vertical (world-up) reference planted at the patient's
// own ground point, extending past head height -- added per user request after
// screenshots of the retargeted patient looked "unnatural" (forward lean / squat)
// but were hard to judge precisely from a single static camera angle. Mirrors the
// hand-drawn vertical line the user overlaid on their own reference screenshots:
// with this rendered IN the scene, any forward/backward lean of the torso/head
// relative to a true vertical is visible directly, without guessing from
// perspective. Off by default (toggle chip) -- purely a debug/verification aid,
// not part of the "real" render.
// ===========================================================================

// Kept in sync with anim_bake.PATIENT_HIP_HEIGHT_M (0.92) -- the patient_root
// node's own world height above the patient's ground is exactly that constant
// (see anim_bake.py's docstring: "the mannequin's hip ... sits at pos.z +
// PATIENT_HIP_HEIGHT_M"), so subtracting it from the root's world Y recovers the
// ground point directly under the patient without needing a separate terrain query.
const PATIENT_HIP_HEIGHT_M = 0.92;
const PLUMB_LINE_HEIGHT_M = 1.9; // a bit above PATIENT_HEAD_HEIGHT_M (1.63) with margin

// depthTest:false + a high renderOrder: the whole point is comparing the body's
// silhouette against a TRUE vertical, same as the user's own hand-drawn overlay on
// their reference screenshots -- an overlay drawn on top of a photo is never
// occluded by the subject, so a depth-tested 3D line (which mostly hides inside the
// torso volume it's meant to be compared against) defeats the purpose.
const plumbLineMaterial = new THREE.MeshBasicMaterial( { color: 0x2255ee, depthTest: false, depthWrite: false } );
const plumbLine = new THREE.Mesh( new THREE.CylinderGeometry( 0.006, 0.006, PLUMB_LINE_HEIGHT_M, 8 ), plumbLineMaterial );
plumbLine.name = 'plumb_line';
plumbLine.visible = false;
plumbLine.renderOrder = 999;
plumbLine.frustumCulled = false; // same reasoning as PatientHuman's skinned mesh: this mesh's own node never sits where it's drawn relative to anything culling would track sanely
scene.add( plumbLine );
let plumbLineEnabled = false;

const _plumbHipWorld = new THREE.Vector3();

/** Re-plant the plumb line at the patient's current ground point. No-op while disabled or before the patient model has attached. */
function updatePlumbLine() {

	if ( ! plumbLineEnabled || ! patientHuman._attached ) return;

	patientHuman._patientRootNode.getWorldPosition( _plumbHipWorld );
	const groundY = _plumbHipWorld.y - PATIENT_HIP_HEIGHT_M;
	plumbLine.position.set( _plumbHipWorld.x, groundY + PLUMB_LINE_HEIGHT_M / 2, _plumbHipWorld.z );

}

// ===========================================================================
// Brand label: REAL 3D text geometry, not embossed-mesh crease detection
//
// The real Isaac Go2 USD's "unitree" wordmark is sculpted directly into the
// single fused `base` mesh (no material/UV tag to isolate it) as a shallow
// relief -- too shallow and too coarsely tessellated for BlueprintEdgesPass's
// normal-discontinuity edge detector to ever read as clean letterforms (its
// own dilate/erode "closing" pass exists specifically to bridge that gap and
// still wasn't enough; a flat SVG callout was tried next and rejected --
// the brand needs to actually be IN the render, not a UI tag floating over
// it). Fix: author a completely separate, genuinely sharp-edged text mesh
// (TextGeometry over a vendored typeface) and sit it on the body like a
// raised emblem. A flat extrusion's 90-degree side-wall/top-face normal
// break is exactly the strong, continuous discontinuity the edge pass is
// built for -- unlike the original scan's smoothly-blended organic relief,
// this WILL ink as solid, legible strokes at any camera distance.
// ===========================================================================

const LOGO_TEXT = 'unitree';
const LOGO_LETTER_HEIGHT = 0.02; // m, cap height
const LOGO_DEPTH = 0.003; // m, shallow raised-emblem extrusion
// Local-frame (base link: Z-up, X-forward -- see usd_mesh.py) placement on
// the real mesh's flat top-rear deck, hand-measured off the baked robot.glb
// (`robot_base` primitive) by clustering vertices near the surface's global
// z-max: the deck is a ~0.13x0.10 m flat plateau spanning x in
// [0.121, 0.252], y in [-0.052, 0.051], topping out at z ~= 0.089. Biased
// toward the low-x (rear) end of that range -- at the default 0.025 cap
// height the word's high-x end visibly wrapped onto the neck's curved
// surface (verified live via a bird's-eye + 3/4 preview render).
const LOGO_LOCAL_POSITION = new THREE.Vector3( 0.17, 0, 0.0905 );

function loadLogoFont() {

	return fetch( './vendor/fonts/helvetiker_bold.typeface.json' )
		.then( ( res ) => res.json() )
		.then( ( json ) => new FontLoader().parse( json ) )
		.catch( ( error ) => {

			console.warn( '[blueprint-viewer] logo font failed to load, skipping brand label:', error );
			return null;

		} );

}

const logoFontReady = loadLogoFont();

function buildLogoMesh( font ) {

	const geometry = new TextGeometry( LOGO_TEXT, {
		font,
		size: LOGO_LETTER_HEIGHT,
		depth: LOGO_DEPTH,
		curveSegments: 6,
		bevelEnabled: false,
	} );
	geometry.center();

	const mesh = new THREE.Mesh( geometry, logoMaterial );
	mesh.name = 'logo_label';
	mesh.position.copy( LOGO_LOCAL_POSITION );
	return mesh;

}

/**
 * Attach (or replace) the real 3D brand-label mesh under the given real-mesh
 * robot_base node. Only meaningful for the real GLB -- the placeholder robot
 * uses three's own Y-up/Z-forward convention and has no equivalent deck.
 */
function attachLogoLabel( baseNode, font ) {

	if ( ! font || ! baseNode ) return;

	const existing = baseNode.getObjectByName( 'logo_label' );
	if ( existing ) {

		existing.geometry?.dispose();
		existing.parent.remove( existing );

	}

	baseNode.add( buildLogoMesh( font ) );

}

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

	patientHuman.sync( phaseName, phaseActions.get( phaseName ).time );
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
	patientHuman.sync( currentPhase, t );

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
	patientHuman.sync( currentPhase, nextTime );

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

plumbToggle.addEventListener( 'click', () => {

	plumbLineEnabled = ! plumbLineEnabled;
	plumbLine.visible = plumbLineEnabled;
	plumbToggle.textContent = `plumb line · ${ plumbLineEnabled ? 'on' : 'off' }`;
	plumbToggle.setAttribute( 'aria-pressed', plumbLineEnabled ? 'true' : 'false' );
	if ( plumbLineEnabled ) updatePlumbLine();

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

				// Real mesh only (see attachLogoLabel doc comment) — races against
				// the GLTF load same as the patient human below.
				logoFontReady.then( ( font ) => attachLogoLabel( baseNode, font ) );

				// Wait for the (concurrently-loading) patient human model AND
				// robot.meta.json too, so the first rendered frame never shows the
				// robot without its patient — resolves either way (PatientHuman.
				// load() catches its own errors and just leaves .ready false,
				// degrading to "no patient shown"; robotMetaReady catches its own
				// fetch error and resolves null, degrading to "no patient gait
				// built" — see robotMetaReady's own comment).
				Promise.all( [ patientHumanReady, robotMetaReady ] ).then( ( [ , meta ] ) => {

					const isaacWorldNode = root.getObjectByName( 'isaac_world' );
					const patientRootNode = root.getObjectByName( 'patient_root' );
					if ( isaacWorldNode && patientRootNode ) {

						patientHuman.attachTo( isaacWorldNode, patientRootNode, patientMaterial );

						if ( meta ) {

							// phaseClips (module-level Map, populated by
							// setupActionsFromClips inside the finishModelSetup call
							// above, which already ran synchronously before this
							// async continuation) — buildGait needs the RAW
							// THREE.AnimationClip objects (to read patient_root's own
							// position/quaternion KeyframeTracks), not the
							// AnimationAction wrappers phaseActions holds.
							patientHuman.buildGait(
								{ follow: phaseClips.get( 'follow' ), climb: phaseClips.get( 'climb' ) },
								meta.stair_spec, meta.landing_far_x_m,
							);

						}

						patientHuman.sync( currentPhase, phaseActions.get( currentPhase )?.time ?? 0 );

					}

					resolve();

				} );

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

	updatePlumbLine();

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
	 * Patient-gait acceptance-bar diagnostic: sweeps BOTH phase clips at `dt`,
	 * driving the REAL path (action.time + mixer.update(0) + patientHuman.sync(...),
	 * exactly like scrubToPercent — no shortcuts that could diverge from what a user
	 * actually sees), reads REAL bone world positions, computes the metrics the
	 * orchestrator's acceptance bars check, and restores the viewer to whatever
	 * phase/time/slider it was at before this call ran (this is a read-only
	 * diagnostic, not a mode switch — a caller scrubbing afterward should see no
	 * trace this ran).
	 */
	patientDiag( { dt = 0.05 } = {} ) {

		if ( ! patientHuman._attached || ! patientHuman._schedules || ! modelRoot ) {

			return { perClip: {}, violations: [], ikSelfCheck: patientHuman.ikSelfCheckFailed, error: 'patient not ready' };

		}

		const isaacWorldNode = modelRoot.getObjectByName( 'isaac_world' );
		if ( ! isaacWorldNode ) return { perClip: {}, violations: [], ikSelfCheck: patientHuman.ikSelfCheckFailed, error: 'isaac_world node not found' };

		// Save prior state (phase, per-phase action times, scrubber value) to restore
		// after the sweep.
		const priorPhase = currentPhase;
		const priorTimes = new Map();
		for ( const [ name, action ] of phaseActions ) priorTimes.set( name, action.time );
		const priorScrubberValue = scrubber.value;

		const bones = patientHuman._bones;
		const violations = [];
		const perClip = {};

		const _tmpWorld = new THREE.Vector3();
		const _tmpLocal = new THREE.Vector3();

		/** getWorldPosition() then convert into isaac_world's own LOCAL frame (AGENTS.md incident #5's diagnostic pitfall: raw scene-space coordinates under isaac_world have already been rotated -90deg about X (Z-up -> Y-up), so comparing scene-space .z directly against this pipeline's native Z-up convention is apples-to-oranges). Returns a plain {x,y,z} in P-frame (isaac_world-local) meters. */
		function worldToPframe( bone ) {

			bone.getWorldPosition( _tmpWorld );
			isaacWorldNode.worldToLocal( _tmpLocal.copy( _tmpWorld ) );
			return { x: _tmpLocal.x, y: _tmpLocal.y, z: _tmpLocal.z };

		}

		function pushViolation( clip, t, metric, value ) {

			violations.push( { clip, t, metric, value } );

		}

		for ( const clipName of [ 'follow', 'climb' ] ) {

			const clip = phaseClips.get( clipName );
			const action = phaseActions.get( clipName );
			const schedule = patientHuman._schedules[ clipName ];
			if ( ! clip || ! action || ! schedule ) continue;

			// setPhase (not just setting action.time) is REQUIRED here: every
			// phase's AnimationAction is always .play()'d/paused (see
			// setupActionsFromClips's own comment), with weight=1 for the ACTIVE
			// phase and weight=0 for the inactive one — three.js's own
			// AnimationMixer._updateWeight/AnimationAction._update never even
			// EVALUATES an action's interpolants when its weight is 0 (confirmed by
			// reading vendor/three.module.js's own AnimationAction._update: `if
			// (weight > 0) { ...evaluate... }`), so merely setting climb.time while
			// climb's weight is still 0 (follow active) would silently have ZERO
			// effect on patient_root's actual transform. setPhase makes this
			// clipName's action the weight=1 one before the sweep below sets its time.
			setPhase( clipName, { resetSlider: false } );

			const duration = clip.duration;
			const terrain = patientHuman._terrain;

			let maxPenetration = 0; // terrain.heightAt(toe.x) - toe.z, clamped to >=0 (positive = penetrating)
			let minSoleClearance = Infinity; // toe.z - terrain.heightAt(toe.x), can go negative (penetration)
			let plantedDriftMax = 0;
			let idleFootMotionMax = 0;
			let fkErrorMax = 0;
			let maxToeStepM = 0;
			let minHipAboveTerrain = Infinity, maxHipAboveTerrain = - Infinity;
			const stanceKneeBendDegs = [];
			let maxKneeBendDeg = 0;

			let prevLeftToe = null, prevRightToe = null;
			let plantedAnchorLeft = null, plantedAnchorRight = null; // {x,y} the CURRENT stance run started at, for plantedDriftMax
			let wasLeftPlanted = null, wasRightPlanted = null;

			for ( let t = 0; t <= duration + 1e-9; t += dt ) {

				const tt = Math.min( t, duration );

				action.time = tt;
				mixer.update( 0 );
				patientHuman.sync( clipName, tt );

				const leftToe = worldToPframe( bones.leftToeBase );
				const rightToe = worldToPframe( bones.rightToeBase );
				const leftFootP = worldToPframe( bones.leftFoot );
				const rightFootP = worldToPframe( bones.rightFoot );

				for ( const [ toe, footName ] of [ [ leftToe, 'leftToe' ], [ rightToe, 'rightToe' ] ] ) {

					const th = terrain.heightAt( toe.x );
					const penetration = th - toe.z; // positive = below terrain (bad)
					const clearance = toe.z - th;
					maxPenetration = Math.max( maxPenetration, penetration );
					minSoleClearance = Math.min( minSoleClearance, clearance );
					if ( penetration > 0.005 && violations.length < 40 ) pushViolation( clipName, tt, `penetration.${footName}`, penetration );

				}

				// plantedDriftMax: horizontal drift of a foot bone WHILE it stays
				// planted (per PatientGait's own pose.leftFoot.planted flag from the
				// most recent sync() — captured in patientHuman._lastSync).
				const ls = patientHuman._lastSync;
				if ( ls ) {

					if ( ls.leftPlanted ) {

						if ( wasLeftPlanted && plantedAnchorLeft ) {

							const d = Math.hypot( leftFootP.x - plantedAnchorLeft.x, leftFootP.y - plantedAnchorLeft.y );
							plantedDriftMax = Math.max( plantedDriftMax, d );

						} else {

							plantedAnchorLeft = { x: leftFootP.x, y: leftFootP.y };

						}

					} else plantedAnchorLeft = null;
					wasLeftPlanted = ls.leftPlanted;

					if ( ls.rightPlanted ) {

						if ( wasRightPlanted && plantedAnchorRight ) {

							const d = Math.hypot( rightFootP.x - plantedAnchorRight.x, rightFootP.y - plantedAnchorRight.y );
							plantedDriftMax = Math.max( plantedDriftMax, d );

						} else {

							plantedAnchorRight = { x: rightFootP.x, y: rightFootP.y };

						}

					} else plantedAnchorRight = null;
					wasRightPlanted = ls.rightPlanted;

					if ( plantedDriftMax > 0.01 && violations.length < 40 ) pushViolation( clipName, tt, 'plantedDrift', plantedDriftMax );

					// fkErrorMax: achieved Foot bone (ankle) P-frame position vs the
					// IK target sync() just solved for.
					const leftAnkleErr = Math.hypot(
						leftFootP.x - ls.leftAnkleTargetWorld.x, leftFootP.y - ls.leftAnkleTargetWorld.y, leftFootP.z - ls.leftAnkleTargetWorld.z,
					);
					const rightAnkleErr = Math.hypot(
						rightFootP.x - ls.rightAnkleTargetWorld.x, rightFootP.y - ls.rightAnkleTargetWorld.y, rightFootP.z - ls.rightAnkleTargetWorld.z,
					);
					fkErrorMax = Math.max( fkErrorMax, leftAnkleErr, rightAnkleErr );
					if ( Math.max( leftAnkleErr, rightAnkleErr ) > 0.012 && violations.length < 40 ) pushViolation( clipName, tt, 'fkError', Math.max( leftAnkleErr, rightAnkleErr ) );

					// kneeBendDeg
					stanceKneeBendDegs.push( ls.leftPlanted ? ls.leftKneeBendDeg : null );
					stanceKneeBendDegs.push( ls.rightPlanted ? ls.rightKneeBendDeg : null );
					maxKneeBendDeg = Math.max( maxKneeBendDeg, ls.leftKneeBendDeg, ls.rightKneeBendDeg );

					// idleFootMotionMax: max per-sample foot displacement while root
					// speed < 0.02 m/s.
					if ( prevLeftToe && ls.speed < 0.02 ) {

						const mL = Math.hypot( leftToe.x - prevLeftToe.x, leftToe.y - prevLeftToe.y, leftToe.z - prevLeftToe.z );
						const mR = Math.hypot( rightToe.x - prevRightToe.x, rightToe.y - prevRightToe.y, rightToe.z - prevRightToe.z );
						const m = Math.max( mL, mR );
						idleFootMotionMax = Math.max( idleFootMotionMax, m );
						if ( m > 0.002 && violations.length < 40 ) pushViolation( clipName, tt, 'idleFootMotion', m );

					}

				}

				if ( prevLeftToe ) {

					const stepL = Math.hypot( leftToe.x - prevLeftToe.x, leftToe.y - prevLeftToe.y, leftToe.z - prevLeftToe.z );
					const stepR = Math.hypot( rightToe.x - prevRightToe.x, rightToe.y - prevRightToe.y, rightToe.z - prevRightToe.z );
					maxToeStepM = Math.max( maxToeStepM, stepL, stepR );

				}
				prevLeftToe = leftToe; prevRightToe = rightToe;

				// hipHeightAboveTerrain: patient_root's own P-frame Z (world hip
				// height) minus terrain height under the root's own X.
				const rootLocal = worldToPframe( patientHuman._patientRootNode );
				const hipAbove = rootLocal.z - terrain.heightAt( rootLocal.x );
				minHipAboveTerrain = Math.min( minHipAboveTerrain, hipAbove );
				maxHipAboveTerrain = Math.max( maxHipAboveTerrain, hipAbove );

				if ( tt >= duration ) break;

			}

			const stanceVals = stanceKneeBendDegs.filter( ( v ) => v !== null ).sort( ( a, b ) => a - b );
			const stanceMedian = stanceVals.length ? stanceVals[ Math.floor( stanceVals.length / 2 ) ] : 0;

			perClip[ clipName ] = {
				minSoleClearance, maxPenetration: Math.max( 0, maxPenetration ),
				plantedDriftMax, idleFootMotionMax, fkErrorMax,
				kneeBendDeg: { stanceMedian, max: maxKneeBendDeg },
				maxToeStepM, hipHeightAboveTerrain: { min: minHipAboveTerrain, max: maxHipAboveTerrain },
			};

		}

		// Restore prior state.
		for ( const [ name, t ] of priorTimes ) {

			const action = phaseActions.get( name );
			if ( action ) action.time = t;

		}
		setPhase( priorPhase, { resetSlider: false } );
		mixer.update( 0 );
		patientHuman.sync( priorPhase, phaseActions.get( priorPhase )?.time ?? 0 );
		scrubber.value = priorScrubberValue;
		updateTimeReadout();

		return { perClip, violations: violations.slice( 0, 40 ), ikSelfCheck: patientHuman.ikSelfCheckFailed };

	},
	/**
	 * Live references for headless verification/calibration tooling only
	 * (e.g. tuning light intensities or edge-pass uniforms in-page without a
	 * reload cycle). Not a stable public API.
	 */
	_internals: {
		scene, camera, renderer, composer, hemiLight, dirLight, edgesPass, patientHuman, controls,
		get patientGait() { return { terrain: patientHuman._terrain, schedules: patientHuman._schedules, params: patientHuman._gaitParams }; },
	},
};

// ===========================================================================
// Boot
// ===========================================================================

applyTheme( currentThemeName );

// Perform the real initial sizing now (see the NOTE on the renderer/camera
// construction above) — canvasHost should have a committed layout by the
// time this module's top-level code finishes running, but guard anyway:
// if it's somehow still 0x0, the ResizeObserver below will catch the next
// genuine size change.
handleResize();

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
