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
const cinematicToggle = document.getElementById( 'cinematic-toggle' );
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

// Lighting: hemisphere (unquantized ambient fill) + a warm KEY directional
// light (quantized into the toon gradient bands below, and the only light
// that casts shadows) + a cool, dimmer, unshadowed FILL directional light
// from the opposite side + a shader-injected fresnel rim — see the
// "Blueprint materials" section for how MeshToonMaterial splits the two
// directional lights' combined contribution into banded direct terms, with
// hemiLight staying a smooth ambient term on top.
//
// 2026-07-10 lighting pass ("should look better than the sim version"):
// added the fill light and warm/cool color split (classic complementary
// key+fill toon grading -- a warm key against a cool fill/ambient reads far
// richer than a single flat white light) and enabled real-time shadows.
// dirLight was raised from the old flat-material value (0.15) because the
// toon gradient map only bands the DIRECTIONAL contribution — at 0.15 it was
// swamped by hemiLight's ambient fill and no bands were visible at all.
// Rebalanced by pixel-sampling a live render so the lit face still lands
// close to the old ~sRGB 205 target against the #d6d2ca (214) paper
// background, but with visible shadow/mid/lit steps across the form.
const hemiLight = new THREE.HemisphereLight( 0xffffff, 0xd8d4cc, 1.4 );
scene.add( hemiLight );
const dirLight = new THREE.DirectionalLight( 0xfff2df, 1.6 ); // warm key light
dirLight.position.set( 3, 5, 2 );
scene.add( dirLight );

const fillLight = new THREE.DirectionalLight( 0xb9d3ff, 0.55 ); // cool fill, opposite side, no shadow
fillLight.position.set( -3.5, 2.2, -2.4 );
scene.add( fillLight );

// ---------------------------------------------------------------------------
// Shadows: dirLight (the key light) casts; a tight, moving orthographic
// shadow frustum re-centers on the robot's current world position every
// frame (see renderFrame() below) so a fixed small mapSize still gets good
// texel density anywhere along the ~18 m follow+climb route, instead of
// needing one giant frustum covering the whole course at low resolution.
// PCFSoftShadowMap for a softer edge that sits better with the toon/ink look
// than a hard shadow-map edge would.
// ---------------------------------------------------------------------------
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

dirLight.castShadow = true;
dirLight.shadow.mapSize.set( 2048, 2048 );
dirLight.shadow.camera.near = 0.5;
dirLight.shadow.camera.far = 14;
dirLight.shadow.camera.left = -3.5;
dirLight.shadow.camera.right = 3.5;
dirLight.shadow.camera.top = 3.5;
dirLight.shadow.camera.bottom = -3.5;
dirLight.shadow.bias = -0.0015;
dirLight.shadow.normalBias = 0.02;
dirLight.shadow.camera.updateProjectionMatrix();
scene.add( dirLight.target );

// Fixed offset from the shadow-follow target to the key light (same vector as
// the light's initial position above) -- recomputed each frame relative to
// the robot's CURRENT position instead of the world origin, see renderFrame().
const DIR_LIGHT_OFFSET = new THREE.Vector3( 3, 5, 2 );
const _shadowFollowPos = new THREE.Vector3();

// ===========================================================================
// Palette / theme
// ===========================================================================

let currentThemeName = localStorage.getItem( 'blueprint-viewer-theme' ) || 'light';
if ( currentThemeName !== 'light' && currentThemeName !== 'dark' ) currentThemeName = 'light';

function applyTheme( name ) {

	currentThemeName = name;
	const palette = PALETTES[ name ];

	applyPaletteToDom( palette );
	if ( scene.background && scene.background.isTexture ) scene.background.dispose();
	scene.background = makeBackgroundGradient( palette.bgGradientTop, palette.bgGradientBottom );

	if ( bodyMaterial ) bodyMaterial.color.set( palette.materialColor );
	if ( oxygenTankMaterial ) oxygenTankMaterial.color.set( palette.oxygenTankColor );
	if ( cradleRailsMaterial ) cradleRailsMaterial.color.set( palette.cradleRailsColor );
	if ( stairsMaterial ) stairsMaterial.color.set( palette.stairsColor );
	if ( handrailMaterial ) handrailMaterial.color.set( palette.handrailColor );
	// groundMaterial.color deliberately NOT re-tinted here: its tile colors are baked
	// into groundMaterial.map (see makeGroundTileTexture) and the material's own
	// .color stays neutral white always (set once at creation) so the toon shading
	// modulates the texture's own colors instead of double-tinting them.
	if ( patientMaterial ) patientMaterial.color.set( palette.patientColor );
	if ( robotMaterial ) robotMaterial.color.set( palette.robotColor );
	if ( logoMaterial ) logoMaterial.color.set( palette.logoColor );

	if ( edgesPass ) edgesPass.setInkColor( palette.inkColorGl );

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

// Three deliberately-separated cel bands (deep shadow / mid / lit). Pulled a
// little darker/wider apart than the prior [0.38,0.72,1.0] so the banding is
// clearly visible as stylized anime shading -- the large flat faces (stair
// side wall, robot body) were previously reading as one near-flat tone with
// barely-perceptible steps. The lit band stays 1.0 so the calibrated lit-face
// brightness against the paper background is unchanged.
const CEL_GRADIENT_MAP = makeToonGradientMap( [ 0.30, 0.60, 1.0 ] );

// Flatter gradient reserved for the wood (stairs). The punchy 3-band CEL map
// above made the lit tread-tops and the shadowed risers/side-walls read as TWO
// distinct wood colors (a light tan vs a darker orange-brown) -- user wanted a
// single wood tone. Lifting the shadow/mid bands close to the lit band keeps
// the wood essentially one color with only a hint of form, while the per-step
// black outlines (from the edge pass) still define the staircase geometry.
const WOOD_GRADIENT_MAP = makeToonGradientMap( [ 0.86, 0.94, 1.0 ] );

const RIM_COLOR = new THREE.Color( 0xffffff );
const RIM_POWER = 2.2;
const RIM_INTENSITY = 0.6; // brighter fresnel edge-glow for anime "pop" along silhouettes

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

function makeBlueprintMaterial( colorHex, options = {} ) {

	const material = new THREE.MeshToonMaterial( {
		color: colorHex,
		gradientMap: options.gradientMap ?? CEL_GRADIENT_MAP,
	} );

	// Force a distinct compiled program for grain vs non-grain materials. For a
	// BUILT-IN material (shaderID 'toon'), three's program cache key ignores the
	// onBeforeCompile-modified source entirely and keys only on material params +
	// customProgramCacheKey() (defaults to '') -- so without this every toon
	// material collides on one cache key and reuses whichever program compiled
	// FIRST (the rim-only body material), silently dropping the grain injection
	// below on the stairs material. Keying on the grain flag gives the grain
	// material its own program.
	material.customProgramCacheKey = () => ( options.grainTexture ? 'bp-grain' : 'bp-plain' );

	// Fresnel rim light: `vViewPosition` (view-space) is already declared by
	// lights_toon_pars_fragment and `vNormal` (view-space) by
	// normal_pars_fragment, so both are in scope for the injected snippet
	// below without redeclaring them.
	//
	// options.grainTexture (optional): a neutral (mean ~1.0) grayscale
	// multiplier map applied TRIPLANAR from world position -- used to give the
	// wood a stylized grain so the big flat stair faces read as anime/game wood
	// planks instead of a dead-flat fill, with no per-mesh UVs needed (the baked
	// stairs mesh has none). Sampled on all three world planes and blended by
	// the world normal so vertical side walls, horizontal treads, and risers all
	// get grain without streak-smearing.
	material.onBeforeCompile = ( shader ) => {

		shader.uniforms.uRimColor = { value: RIM_COLOR };
		shader.uniforms.uRimPower = { value: RIM_POWER };
		shader.uniforms.uRimIntensity = { value: RIM_INTENSITY };

		if ( options.grainTexture ) {

			shader.uniforms.uGrain = { value: options.grainTexture };
			shader.uniforms.uGrainScale = { value: options.grainScale ?? 1.6 }; // texture repeats per world metre
			shader.uniforms.uGrainAmount = { value: options.grainAmount ?? 1.0 }; // 0 = off, 1 = full modulation

			shader.vertexShader = shader.vertexShader
				.replace( '#include <common>', '#include <common>\nvarying vec3 vGrainWorldPos;\nvarying vec3 vGrainWorldNrm;' )
				.replace( '#include <begin_vertex>', '#include <begin_vertex>\n\tvGrainWorldPos = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;' )
				.replace( '#include <beginnormal_vertex>', '#include <beginnormal_vertex>\n\tvGrainWorldNrm = mat3( modelMatrix ) * objectNormal;' );

		}

		shader.fragmentShader = shader.fragmentShader
			.replace(
				'#define TOON',
				'#define TOON\nuniform vec3 uRimColor;\nuniform float uRimPower;\nuniform float uRimIntensity;'
				+ ( options.grainTexture
					? '\nvarying vec3 vGrainWorldPos;\nvarying vec3 vGrainWorldNrm;\nuniform sampler2D uGrain;\nuniform float uGrainScale;\nuniform float uGrainAmount;'
					: '' ),
			)
			.replace(
				'vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;',
				'vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;\n' +
				'\tfloat rimFresnel = pow( 1.0 - max( dot( normalize( vNormal ), normalize( vViewPosition ) ), 0.0 ), uRimPower );\n' +
				'\toutgoingLight += rimFresnel * uRimIntensity * uRimColor;',
			);

		if ( options.grainTexture ) {

			// Multiply the base color by the triplanar grain right after
			// <color_fragment> populates diffuseColor (map * material color), so
			// the grain feeds through the toon banding + rim like the base tint.
			shader.fragmentShader = shader.fragmentShader.replace(
				'#include <color_fragment>',
				'#include <color_fragment>\n'
				+ '\tvec3 grnW = abs( normalize( vGrainWorldNrm ) );\n'
				+ '\tgrnW /= ( grnW.x + grnW.y + grnW.z + 1e-5 );\n'
				+ '\tfloat grain = texture2D( uGrain, vGrainWorldPos.zy * uGrainScale ).r * grnW.x\n'
				+ '\t            + texture2D( uGrain, vGrainWorldPos.xz * uGrainScale ).r * grnW.y\n'
				+ '\t            + texture2D( uGrain, vGrainWorldPos.xy * uGrainScale ).r * grnW.z;\n'
				+ '\tdiffuseColor.rgb *= mix( 1.0, grain, uGrainAmount );',
			);

		}

	};

	return material;

}

// ---------------------------------------------------------------------------
// Backdrop gradient: a tall 1px-wide CanvasTexture (top color -> bottom color)
// used as scene.background instead of a flat fill. Rendered by three as a
// screen-filling backdrop, so it reads as a soft vertical "studio" falloff
// behind the set -- a small, cheap depth/vibe cue over a dead-flat color.
// Endpoints are kept close to the theme's paper tone (see palette bgGradient*)
// so the canvas still blends into the surrounding DOM sheet at its corners.
// ---------------------------------------------------------------------------
function makeBackgroundGradient( topHex, bottomHex ) {

	const canvas = document.createElement( 'canvas' );
	canvas.width = 2;
	canvas.height = 256;
	const ctx = canvas.getContext( '2d' );

	const grad = ctx.createLinearGradient( 0, 0, 0, canvas.height );
	grad.addColorStop( 0, `#${ new THREE.Color( topHex ).getHexString() }` );
	grad.addColorStop( 1, `#${ new THREE.Color( bottomHex ).getHexString() }` );
	ctx.fillStyle = grad;
	ctx.fillRect( 0, 0, canvas.width, canvas.height );

	const texture = new THREE.CanvasTexture( canvas );
	texture.colorSpace = THREE.SRGBColorSpace;
	texture.minFilter = THREE.LinearFilter;
	texture.magFilter = THREE.LinearFilter;
	texture.generateMipmaps = false;
	return texture;

}

// ---------------------------------------------------------------------------
// Wood grain: a neutral (near-white, mean ~1.0) grayscale multiplier map of
// irregular horizontal streaks, sampled TRIPLANAR from world position (see
// makeBlueprintMaterial's grainTexture option). Only darkens (streaks dip below
// 1.0, base stays 1.0) so it modulates the toon wood color without lightening
// it. Stored as NoColorSpace data so the sampled .r is the raw multiplier.
// This is what turns the big flat stair side wall from a cardboard fill into a
// stylized wood surface without needing UVs on the baked stairs mesh.
// ---------------------------------------------------------------------------
function makeWoodGrainTexture() {

	const size = 256;
	const canvas = document.createElement( 'canvas' );
	canvas.width = canvas.height = size;
	const ctx = canvas.getContext( '2d' );
	const img = ctx.createImageData( size, size );
	const data = img.data;

	// Per-row streak base: layered irregular sines so grain lines are uneven,
	// with only the positive peaks darkening (mostly-light wood, occasional
	// darker grain line).
	const rowVal = new Float32Array( size );
	for ( let y = 0; y < size; y ++ ) {

		const yy = y / size;
		const s = 0.5 * Math.sin( yy * Math.PI * 2 * 7 + Math.sin( yy * Math.PI * 2 * 2 ) * 1.5 )
			+ 0.3 * Math.sin( yy * Math.PI * 2 * 17 + 1.3 )
			+ 0.2 * Math.sin( yy * Math.PI * 2 * 31 + 2.1 );
		const d = Math.max( 0, s );
		rowVal[ y ] = 1.0 - 0.16 * Math.pow( d, 1.5 );

	}

	for ( let y = 0; y < size; y ++ ) {

		for ( let x = 0; x < size; x ++ ) {

			// gentle along-grain waviness so streaks aren't perfectly straight
			const wy = y + Math.sin( ( x / size ) * Math.PI * 2 * 2 ) * 2.0;
			const yi = ( ( Math.round( wy ) % size ) + size ) % size;
			let v = rowVal[ yi ] - Math.random() * 0.03;
			v = Math.max( 0.78, Math.min( 1.0, v ) );

			const b = Math.round( v * 255 );
			const i = ( y * size + x ) * 4;
			data[ i ] = data[ i + 1 ] = data[ i + 2 ] = b;
			data[ i + 3 ] = 255;

		}

	}

	ctx.putImageData( img, 0, 0 );

	const texture = new THREE.CanvasTexture( canvas );
	texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
	texture.colorSpace = THREE.NoColorSpace;
	texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
	return texture;

}

// ---------------------------------------------------------------------------
// Ground tile texture: a procedural CanvasTexture (soft tile fill + a thin,
// muted grout border that tiles seamlessly edge-to-edge) rather than a flat
// fill color, per explicit user feedback that a solid-color ground read as
// "a full blue platform" rather than a floor. GROUND_TILE_SIZE_M matches
// isaac_env.py's own TILE_SIZE (0.60 m grout pitch) so the tiling reads at
// the same real-world scale as the sim's floor grid.
//
// 2026-07-10, revised same day: the first version's per-tile off-center
// radial highlight gradient was a mistake for a REPEATING texture -- baked
// into every single tile repeat, it read as a grid of bright blobs once
// tiled across the floor ("too bold... two big [blobs]", per direct user
// feedback), not a subtle sheen. Removed entirely. Also thinned the grout
// (5% of tile -> 1.8%) and pulled both the tile fill and the grout color
// toward EACH OTHER (see softTile/softGrout below) so the grid reads as a
// gentle seam rather than a stark, high-contrast checkerboard -- flat fully-
// saturated color fields next to near-black lines is what "bold/cartoonish"
// usually means; blending them toward a shared mid-tone is what "nice"
// usually means.
// ---------------------------------------------------------------------------
const GROUND_TILE_SIZE_M = 0.60;

function makeGroundTileTexture( tileColorHex, groutColorHex ) {

	const size = 256;
	const canvas = document.createElement( 'canvas' );
	canvas.width = canvas.height = size;
	const ctx = canvas.getContext( '2d' );

	const tileColor = new THREE.Color( tileColorHex );
	const groutColor = new THREE.Color( groutColorHex );

	// Soften both toward each other and toward white: less saturated fill, less
	// near-black grout -- a calmer, lower-contrast pairing than the raw palette
	// values (which are tuned for the flat-color robot/stairs/etc, not a large
	// repeating floor field where high contrast reads as busy/bold).
	const softTile = tileColor.clone().lerp( new THREE.Color( 0xffffff ), 0.30 );
	const softGrout = groutColor.clone().lerp( tileColor, 0.45 );

	ctx.fillStyle = `#${ softTile.getHexString() }`;
	ctx.fillRect( 0, 0, size, size );

	// Grout: a thin stroked border inset by half its own width, so adjacent tiles'
	// borders butt together into one continuous grid line once repeated. Drawn at
	// less than full opacity for a soft seam rather than a hard-edged line.
	const groutW = size * 0.018;
	ctx.globalAlpha = 0.75;
	ctx.strokeStyle = `#${ softGrout.getHexString() }`;
	ctx.lineWidth = groutW;
	ctx.strokeRect( groutW / 2, groutW / 2, size - groutW, size - groutW );
	ctx.globalAlpha = 1;

	const texture = new THREE.CanvasTexture( canvas );
	texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
	texture.colorSpace = THREE.SRGBColorSpace;
	texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
	return texture;

}

/**
 * Bake per-vertex tile UVs onto the "ground" mesh from its own local-space
 * (x, y) positions (already true world/route meters -- the "ground" SceneNode
 * has zero local translation/rotation, see scene_build.build_ground_node), so
 * RepeatWrapping tiles the texture at the physical GROUND_TILE_SIZE_M scale
 * with no per-bake Python step needed. geo.box() (pipeline/geometry.py)
 * emits no UV attribute at all, so this is the mesh's ONLY uv data --
 * harmless for any other mesh reusing the same box() builder since they have
 * no .map to sample it.
 */
function addGroundTileUVs( root ) {

	const groundMesh = root.getObjectByName( 'ground' );
	if ( ! groundMesh || ! groundMesh.isMesh ) return;

	const posAttr = groundMesh.geometry.getAttribute( 'position' );
	if ( ! posAttr ) return;

	const uvArray = new Float32Array( posAttr.count * 2 );
	for ( let i = 0; i < posAttr.count; i ++ ) {

		uvArray[ i * 2 ] = posAttr.getX( i ) / GROUND_TILE_SIZE_M;
		uvArray[ i * 2 + 1 ] = posAttr.getY( i ) / GROUND_TILE_SIZE_M;

	}

	groundMesh.geometry.setAttribute( 'uv', new THREE.BufferAttribute( uvArray, 2 ) );

}

let bodyMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].materialColor );
let oxygenTankMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].oxygenTankColor );
let cradleRailsMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].cradleRailsColor );
const woodGrainTexture = makeWoodGrainTexture();
let stairsMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].stairsColor, { grainTexture: woodGrainTexture, grainScale: 1.4, grainAmount: 0.8, gradientMap: WOOD_GRADIENT_MAP } );
let handrailMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].handrailColor );
let groundMaterial = makeBlueprintMaterial( 0xffffff ); // neutral -- tile colors live in .map, see makeGroundTileTexture
groundMaterial.map = makeGroundTileTexture( PALETTES[ currentThemeName ].groundColor, PALETTES[ currentThemeName ].groundGroutColor );
let patientMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].patientColor );
let robotMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].robotColor );
let logoMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].logoColor );

// "patient_root" is no longer a tint target here: it's a bare transform anchor with
// no mesh of its own (see scene_build.build_patient_node) -- the patient's visible
// geometry is the separately-loaded PatientHuman model below, tinted directly by
// PatientHuman.attachTo().
//
// "stairs" and "handrails" are separate top-level scene nodes (2026-07-10 pipeline
// change, see scene_build.build_handrails_node) specifically so the wood treads/
// landing and the iron rails can carry different toon colors -- they used to be one
// merged "stairs" mesh with a single material.
const TINTED_NODE_NAMES = {
	oxygen_tank: () => oxygenTankMaterial,
	cradle_rails: () => cradleRailsMaterial,
	stairs: () => stairsMaterial,
	handrails: () => handrailMaterial,
	ground: () => groundMaterial,
	robot_base: () => robotMaterial,
	FL_hip: () => robotMaterial,
	FR_hip: () => robotMaterial,
	RL_hip: () => robotMaterial,
	RR_hip: () => robotMaterial,
};

// Per-subtree shadow role (2026-07-10 lighting pass), looked up by the same
// nearest-tagged-ancestor walk as TINTED_NODE_NAMES above. Ground only
// receives (a razor-thin slab casting its own shadow is pointless); the
// robot/payload only cast (self-shadowing a 324k-tri mesh from one key light
// reads as noisy speckle, not form); stairs/handrails do both, so the
// staircase believably shadows itself and the ground below it. Anything
// untagged (falls back to bodyMaterial) defaults to both.
const SHADOW_ROLES = {
	ground: { cast: false, receive: true },
	stairs: { cast: true, receive: true },
	handrails: { cast: true, receive: true },
	oxygen_tank: { cast: true, receive: false },
	cradle_rails: { cast: true, receive: false },
	robot_base: { cast: true, receive: false },
	FL_hip: { cast: true, receive: false },
	FR_hip: { cast: true, receive: false },
	RL_hip: { cast: true, receive: false },
	RR_hip: { cast: true, receive: false },
};
const DEFAULT_SHADOW_ROLE = { cast: true, receive: true };

/**
 * Strip all textures/materials from a loaded model's meshes and replace
 * them with the flat toon palette, disposing the originals. Named subtrees
 * (oxygen tank, cradle rails, stairs, handrails, ground, patient) get their
 * own tint; everything else falls back to the shared body material.
 *
 * Walks UP from the mesh, testing each ancestor's OWN name against
 * TINTED_NODE_NAMES and returning on the FIRST (i.e. nearest/most specific)
 * match. This must be nearest-wins, not "first tagged name found by a
 * root-down traversal": a prior version pre-collected every tagged node via
 * root.traverse (which visits parents before children) and, for each mesh,
 * scanned that traversal-ordered list checking whether ANY of the mesh's
 * ancestors matched -- so a coarse ancestor tag discovered earlier (e.g.
 * "robot_base") always won over a more specific tag on one of its own
 * children (e.g. "oxygen_tank"/"cradle_rails", both direct children of
 * robot_base). That silently made the O2 tank and its cradle always render
 * as plain robot color; invisible under the old near-monochrome scheme
 * (both were dark), glaring once the two got genuinely different colors
 * (2026-07-10 toon repaint) -- confirmed live by reading each mesh's
 * resolved material.color in the running scene.
 */
function applyBlueprintMaterials( root ) {

	function tintFor( mesh ) {

		let p = mesh;
		while ( p ) {

			const materialFn = TINTED_NODE_NAMES[ p.name ];
			if ( materialFn ) return materialFn();
			p = p.parent;

		}

		return bodyMaterial;

	}

	function shadowRoleFor( mesh ) {

		let p = mesh;
		while ( p ) {

			const role = SHADOW_ROLES[ p.name ];
			if ( role ) return role;
			p = p.parent;

		}

		return DEFAULT_SHADOW_ROLE;

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
		const role = shadowRoleFor( node );
		node.castShadow = role.cast;
		node.receiveShadow = role.receive;

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

// Default (orbit-view) edge style, and the cinematic override applied by the
// cinematic toggle. Kept as named constants (not magic numbers scattered across
// the toggle) since both places must agree, and they were tuned together via
// __viewer.edgeCoverage. Cinematic keeps the FULL silhouette but RAISES the
// normal threshold so only the strongest structural creases ink (a few
// "robotic" lines -- leg-body joins, camera mount, major panel seams -- not the
// busy rivet mesh) and pushes the interior fade far out so those few lines
// survive at the pulled-back framing distance. See cinematicToggle handler.
const EDGE_NORMAL_THRESHOLD = 0.66;
const EDGE_INTERIOR_FADE_NEAR = 4.0;
const EDGE_INTERIOR_FADE_FAR = 10.0;
// 0.9 (vs the orbit view's 0.66): only the strong structural creases ink -- a
// FEW robotic lines (leg-body joins, the camera-mount box, major panel seams),
// not the busy rivet/seam mesh. Verified via __viewer.edgeCoverage: at 0.9 the
// robot's per-part silhouette stays ~90-100% covered (legs/feet/mount all keep
// their outline, unlike the earlier over-flattened "interior strength 0") while
// interior-line density on the body drops to ~12% -- a clean but still-machined
// read. See the cinematic toggle handler.
const CINE_NORMAL_THRESHOLD = 0.9;
const CINE_INTERIOR_FADE_NEAR = 6.0;
const CINE_INTERIOR_FADE_FAR = 32.0;

const edgesPass = new BlueprintEdgesPass( scene, camera, {
	inkColor: PALETTES[ currentThemeName ].inkColorGl,
	// 0.4 was tuned on the primitive-built robot; the real Isaac Go2 mesh
	// (324k tris of sculpted surface detail) saturates into dark speckle at
	// viewing distance with it. 0.55 kept close-up creases intact but showed
	// a dense pile of interior lines. Raised to 0.66 per user feedback ("I
	// don't want multiple lines"): only the STRONGER seams (logos, main body
	// panels, leg joints) ink, so the body reads as a FEW clean lines rather
	// than a mesh of them, at every distance -- fewer strong lines also can't
	// pile into a blob far away, which is why the fade band below can be
	// pushed out so those lines survive to normal viewing distance.
	normalThreshold: EDGE_NORMAL_THRESHOLD,
	// Interior-crease fade band (view-space metres). Pushed out from the
	// original 2.0/6.5 so the (now-sparser) body lines PERSIST at close/medium
	// viewing distance instead of the body going to bare outline the moment you
	// step back -- they only thin out once the robot is genuinely far.
	interiorFadeNear: EDGE_INTERIOR_FADE_NEAR,
	interiorFadeFar: EDGE_INTERIOR_FADE_FAR,
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
	// 1.2 -> 1.4: a bit bolder/thicker ink so the outline reads as a more
	// notable line (user: "make the black a bit more bold/bigger"). Kept modest
	// so nearby interior lines still don't fatten into each other.
	thickness: 1.4,
} );
composer.addPass( edgesPass );

const outputPass = new OutputPass();
composer.addPass( outputPass );

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

// ---------------------------------------------------------------------------
// Unified timeline: the "follow" and "climb" clips are concatenated into ONE
// continuous scrub timeline (they come from contiguous windows of the same
// Isaac recording, so the robot + patient are spatially continuous across the
// seam -- verified ~5-8 mm of drift at the join). `segments` is the ordered
// list [{ name, clip, action, start, duration }]; `totalDuration` is the sum;
// `globalTime` is the single authoritative playhead in [0, totalDuration].
// The two phase chips become jump-to-segment shortcuts (not mode switches),
// and the scrubber/play/readout all speak globalTime. See applyGlobalTime().
// ---------------------------------------------------------------------------
let segments = [];
let totalDuration = 0;
let globalTime = 0;

// Follow-cam bookkeeping: last known robot_base world position, used to
// translate the camera by the same delta the target moves each frame.
const _lastBaseWorldPos = new THREE.Vector3();
const _curBaseWorldPos = new THREE.Vector3();
const _baseDelta = new THREE.Vector3();
let trackingEnabled = true;
let hasLastBasePos = false;

// ---------------------------------------------------------------------------
// Cinematic two-subject follow-cam (opt-in via the "cinematic" chip; OFF by
// default). A distinct mode from the default robot-only orbit-follow above:
// when enabled it takes FULL control of the camera and keeps BOTH the robot
// and the patient framed in one shot -- aims at the point between them, pulls
// back just far enough that both fit with margin, and rides a slow side/above
// trailing angle with a gentle sway so it reads as a moving, "alive" camera
// rather than a locked orbit. Everything is critically damped (frame-rate-
// independent lerps) so the robot's stop-and-go pacing glides instead of
// jerking the frame.
//
// While active, OrbitControls is disabled and its update() is skipped (its
// update() otherwise reasserts the camera from its own spherical/target state
// every frame -- see blueprint-viewer memory); on the way out the control's
// target is re-synced so handing back to manual orbit doesn't snap.
// ---------------------------------------------------------------------------

let cinematicEnabled = false;
let cineNeedsInit = false; // snap the smoothed look-target on the first active frame
let cineTime = 0;          // seconds since this mode was last enabled, drives the sway

// Framing angle in three.js SCENE space. The -90deg-about-X isaac_world
// rotation maps the pipeline's Z-up/X-forward frame to three's Y-up, so here:
//   +X = travel / up-the-stairs direction, +Y = world up, +Z = the near side.
// The camera sits behind-side-above and looks forward/down at the pair, which
// shows the climbing profile and the stairs ahead while staying over open
// space -- a leading shot (camera ahead) risks clipping into the handrails
// during the climb.
const CINE_BASE_AZ = 118 * Math.PI / 180; // azimuth measured from +X in the XZ (ground) plane
const CINE_BASE_EL = 24 * Math.PI / 180;  // elevation above the ground plane
const CINE_SWAY_AZ = 9 * Math.PI / 180;   // slow left/right drift amplitude
const CINE_SWAY_EL = 4 * Math.PI / 180;   // slow rise/fall amplitude
const CINE_SWAY_AZ_PERIOD = 13;           // s, one full left-right sway
const CINE_SWAY_EL_PERIOD = 19;           // s, one full rise-fall sway

const CINE_SUBJECT_PAD = 0.95;   // half-a-body of extra framing radius so neither subject kisses the frame edge (m)
const CINE_FRAME_MARGIN = 1.16;  // >1 leaves breathing room around the pair
const CINE_MIN_DIST = 2.3;       // never dolly closer than this (m)
const CINE_MAX_DIST = 7.5;       // never drift further than this (m)
const CINE_TARGET_UP_BIAS = 0.15; // aim a touch above the base/hip midpoint so the pair sits mid-frame, not along the bottom (m)

// Frame-rate-independent smoothing bases for `1 - base^dt`: smaller = snappier.
// Position eases a touch floatier than the look-target so quick subject moves
// read as the camera gliding to catch up.
const CINE_POS_SMOOTH_BASE = 0.0030;
const CINE_TGT_SMOOTH_BASE = 0.0015;

const _cineRobotPos = new THREE.Vector3();
const _cinePatientPos = new THREE.Vector3();
const _cineTargetGoal = new THREE.Vector3();
const _cinePosGoal = new THREE.Vector3();
const _cineOffsetDir = new THREE.Vector3();
const _cineLookTarget = new THREE.Vector3(); // smoothed look-at point actually fed to camera.lookAt

/**
 * Drive the camera for one frame in cinematic mode. Frames the robot + patient
 * together, glides toward a swaying side/above trailing angle, and keeps
 * controls.target in sync for a snap-free handoff back to manual orbit.
 * Assumes `robotBase` is non-null (guarded by the caller).
 */
function updateCinematicCamera( dtSec ) {

	robotBase.getWorldPosition( _cineRobotPos );

	if ( patientHuman._attached && patientHuman._patientRootNode ) {

		patientHuman._patientRootNode.getWorldPosition( _cinePatientPos );

	} else {

		// Patient not attached yet (still loading): frame on the robot alone so
		// the mode still does something sane rather than aiming at the origin.
		_cinePatientPos.copy( _cineRobotPos );

	}

	// Look target: midpoint of the two subjects, nudged up a little so they sit
	// in the middle of frame rather than along the bottom edge.
	_cineTargetGoal.addVectors( _cineRobotPos, _cinePatientPos ).multiplyScalar( 0.5 );
	_cineTargetGoal.y += CINE_TARGET_UP_BIAS;

	if ( cineNeedsInit ) {

		// First active frame: snap the smoothed look-target onto the real one so
		// the camera doesn't swing in from wherever _cineLookTarget last sat
		// (the camera POSITION still glides in from its current spot -- a nice
		// reveal -- but the look direction locks onto the subjects immediately).
		_cineLookTarget.copy( _cineTargetGoal );
		cineNeedsInit = false;

	}

	// Distance: pull back just far enough that both subjects (plus a body-sized
	// pad) fit inside the vertical FOV, with margin; clamped so it never gets
	// uncomfortably close or drifts far away.
	const sep = _cineRobotPos.distanceTo( _cinePatientPos );
	const radius = 0.5 * sep + CINE_SUBJECT_PAD;
	const halfFov = THREE.MathUtils.degToRad( camera.fov ) * 0.5;
	let dist = ( radius / Math.tan( halfFov ) ) * CINE_FRAME_MARGIN;
	dist = THREE.MathUtils.clamp( dist, CINE_MIN_DIST, CINE_MAX_DIST );

	// Slow sway on the framing angle so the camera feels hand-held/alive rather
	// than mechanically locked. Two different periods (and a phase offset on the
	// elevation term) keep the motion from looking like a simple circle.
	cineTime += dtSec;
	const az = CINE_BASE_AZ + CINE_SWAY_AZ * Math.sin( cineTime * ( 2 * Math.PI / CINE_SWAY_AZ_PERIOD ) );
	const el = CINE_BASE_EL + CINE_SWAY_EL * Math.sin( cineTime * ( 2 * Math.PI / CINE_SWAY_EL_PERIOD ) + 1.3 );

	const cosEl = Math.cos( el );
	_cineOffsetDir.set( cosEl * Math.cos( az ), Math.sin( el ), cosEl * Math.sin( az ) );

	_cinePosGoal.copy( _cineTargetGoal ).addScaledVector( _cineOffsetDir, dist );

	const posLerp = Math.min( 1, 1 - Math.pow( CINE_POS_SMOOTH_BASE, dtSec ) );
	const tgtLerp = Math.min( 1, 1 - Math.pow( CINE_TGT_SMOOTH_BASE, dtSec ) );

	camera.position.lerp( _cinePosGoal, posLerp );
	_cineLookTarget.lerp( _cineTargetGoal, tgtLerp );

	camera.lookAt( _cineLookTarget );

	// Keep OrbitControls' target in sync so switching cinematic OFF resumes
	// manual orbit from exactly here, with no snap.
	controls.target.copy( _cineLookTarget );

}

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

function buildSideLogoMesh( font, textStr, isLeft ) {

	const group = new THREE.Group();
	group.name = isLeft ? 'logo_label_side_l' : 'logo_label_side_r';

	const size = 0.033; // both sides are the same physical height on the robot
	const chars = textStr.split('');
	const geometries = chars.map( char => new TextGeometry( char, {
		font,
		size,
		depth: 0.003,
		curveSegments: 6,
		bevelEnabled: false,
	} ) );

	const widths = geometries.map( geom => {

		geom.computeBoundingBox();
		const w = geom.boundingBox.max.x - geom.boundingBox.min.x;
		geom.center();
		return w;

	} );

	const kerning = size * 0.08;
	let totalWidth = 0;
	for ( let i = 0; i < chars.length; i ++ ) {

		totalWidth += widths[ i ];
		if ( i < chars.length - 1 ) totalWidth += kerning;

	}

	let currentX = -totalWidth / 2;
	const R = 1.25; // radius of curvature

	for ( let i = 0; i < chars.length; i ++ ) {

		const charWidth = widths[ i ];
		const x_local = currentX + charWidth / 2;
		currentX += charWidth + kerning;

		const mesh = new THREE.Mesh( geometries[ i ], logoMaterial );
		const y_offset = ( x_local * x_local ) / ( 2 * R );

		mesh.position.set( x_local, 0, -y_offset );
		mesh.rotation.set( 0, -x_local / R, 0 );

		group.add( mesh );

	}

	if ( isLeft ) {

		group.position.set( -0.007, 0.0962, 0.034 ); // center of Unitree on left side
		group.rotation.set( Math.PI / 2, Math.PI, 0 );

	} else {

		group.position.set( 0.008, -0.0962, 0.035 ); // center of Go2 on right side
		group.rotation.set( Math.PI / 2, 0, 0 );

	}

	return group;

}

/**
 * Attach (or replace) the real 3D brand-label mesh under the given real-mesh
 * robot_base node. Only meaningful for the real GLB -- the placeholder robot
 * uses three's own Y-up/Z-forward convention and has no equivalent deck.
 */
function attachLogoLabel( baseNode, font ) {

	if ( ! font || ! baseNode ) return;

	const toRemove = [];
	baseNode.traverse( ( child ) => {

		if ( child.name === 'logo_label' || child.name === 'logo_label_side_l' || child.name === 'logo_label_side_r' ) {

			toRemove.push( child );

		}

	} );
	for ( const child of toRemove ) {

		child.geometry?.dispose();
		child.parent.remove( child );

	}

	baseNode.add( buildLogoMesh( font ) );
	baseNode.add( buildSideLogoMesh( font, 'Unitree', true ) );
	baseNode.add( buildSideLogoMesh( font, 'Go2', false ) );

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

	// Build the unified timeline: concatenate whichever of follow/climb exist,
	// in that order, into one continuous playhead. Each segment records its
	// start offset on the global timeline so applyGlobalTime() can map a global
	// time back to (segment, local time). See the module-state comment above.
	segments = [];
	let acc = 0;
	for ( const name of [ 'follow', 'climb' ] ) {

		const clip = phaseClips.get( name );
		const action = phaseActions.get( name );
		if ( ! clip || ! action ) continue;
		segments.push( { name, clip, action, start: acc, duration: clip.duration } );
		acc += clip.duration;

	}
	totalDuration = acc;
	globalTime = 0;

}

/**
 * Resolve a global timeline position to its segment + local (within-clip) time.
 * The last segment whose start <= t wins; local time is clamped to that clip.
 */
function segmentAtGlobalTime( t ) {

	t = THREE.MathUtils.clamp( t, 0, totalDuration );
	let seg = segments[ 0 ] || null;
	for ( const s of segments ) if ( t >= s.start - 1e-9 ) seg = s;
	const local = seg ? THREE.MathUtils.clamp( t - seg.start, 0, seg.duration ) : 0;
	return { seg, local };

}

/**
 * Make `name`'s action the sole weighted (visible) one. Pure weight swap +
 * phase-chip highlight; no time/slider change. Split out from applyGlobalTime
 * so the patient-gait diagnostic (setPhase below, resetSlider:false) can
 * activate a clip's weight before driving its time directly.
 */
function setActivePhase( name ) {

	if ( ! phaseActions.has( name ) ) return;

	currentPhase = name;

	for ( const [ n, action ] of phaseActions ) action.weight = n === name ? 1 : 0;

	phaseFollowBtn.setAttribute( 'aria-pressed', name === 'follow' ? 'true' : 'false' );
	phaseClimbBtn.setAttribute( 'aria-pressed', name === 'climb' ? 'true' : 'false' );

}

/**
 * THE single authoritative "show this instant of the unified timeline" call.
 * Maps a global time to (segment, local), activates that segment, sets its
 * action.time, forces a zero-delta pose re-eval (see the scrubbing comment
 * block below), and syncs the patient at the SAME local time. Optionally
 * updates the slider position to match.
 */
function applyGlobalTime( t, { updateSlider = true } = {} ) {

	if ( ! mixer || segments.length === 0 ) return;

	globalTime = THREE.MathUtils.clamp( t, 0, totalDuration );

	const { seg, local } = segmentAtGlobalTime( globalTime );
	if ( ! seg ) return;

	setActivePhase( seg.name );
	seg.action.time = local;
	mixer.update( 0 );
	patientHuman.sync( seg.name, local );

	if ( updateSlider ) scrubber.value = String( totalDuration > 0 ? ( globalTime / totalDuration ) * 100 : 0 );
	updateTimeReadout();

}

/** Jump the unified playhead to the start of a named segment (phase-chip click). */
function jumpToSegment( name ) {

	const seg = segments.find( ( s ) => s.name === name );
	if ( ! seg ) return;
	if ( isPlaying ) setPlaying( false );
	applyGlobalTime( seg.start, { updateSlider: true } );

}

/**
 * Back-compat shim for the patient-gait diagnostic (window.__viewer.patientDiag),
 * which drives one clip's action.time directly and needs that clip weighted.
 * resetSlider:true re-homes the unified playhead to the segment start (matching
 * the old "reset to 0" semantics for that phase); resetSlider:false is a pure
 * weight swap that leaves the caller's own time/slider handling intact.
 */
function setPhase( phaseName, { resetSlider = true } = {} ) {

	if ( ! phaseActions.has( phaseName ) ) return;

	if ( resetSlider ) {

		jumpToSegment( phaseName );

	} else {

		setActivePhase( phaseName );

	}

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

	// pct now spans the WHOLE unified timeline (follow + climb), not one clip.
	// applyGlobalTime picks the right segment/local time and drives everything.
	applyGlobalTime( ( pct / 100 ) * totalDuration, { updateSlider: false } );

	scrubber.value = String( pct );

}

function updateTimeReadout() {

	if ( totalDuration <= 0 ) return;

	const t = globalTime.toFixed( 2 ).padStart( 5, '0' );
	const total = totalDuration.toFixed( 2 ).padStart( 5, '0' );
	timeReadout.textContent = `t ${t} / ${total} s · ${currentPhase}`;

}

scrubber.addEventListener( 'input', () => {

	// Slider drag while playing pauses playback (scrub is authoritative).
	if ( isPlaying ) setPlaying( false );

	scrubToPercent( parseFloat( scrubber.value ) );

} );

// ===========================================================================
// Phase buttons — now jump-to-segment shortcuts on the single unified timeline
// (follow starts at t=0, climb starts at the follow clip's end), not mode
// switches. The active chip is highlighted by applyGlobalTime as the playhead
// crosses the seam, so scrubbing/playing past the join re-lights the chips too.
// ===========================================================================

phaseFollowBtn.addEventListener( 'click', () => jumpToSegment( 'follow' ) );
phaseClimbBtn.addEventListener( 'click', () => jumpToSegment( 'climb' ) );

// ===========================================================================
// Optional play/pause chip
//
// Advances the GLOBAL playhead itself from rAF timestamp deltas, then routes
// through applyGlobalTime (mixer.update(0), never mixer.update(dt)) so the same
// single authoritative pose path drives dragging and playing alike. Plays
// straight through the follow->climb seam and stops at the end of the whole
// unified timeline.
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
	if ( ! mixer || segments.length === 0 ) return;

	const dtSec = Math.max( 0, ( nowMs - lastPlaybackTimestamp ) / 1000 );
	lastPlaybackTimestamp = nowMs;

	let nextTime = globalTime + dtSec;
	if ( nextTime >= totalDuration ) {

		nextTime = totalDuration;
		setPlaying( false );

	}

	applyGlobalTime( nextTime, { updateSlider: true } );

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

cinematicToggle.addEventListener( 'click', () => {

	cinematicEnabled = ! cinematicEnabled;
	cinematicToggle.textContent = `cinematic · ${ cinematicEnabled ? 'on' : 'off' }`;
	cinematicToggle.setAttribute( 'aria-pressed', cinematicEnabled ? 'true' : 'false' );

	// Cinematic gets a cleaner ink treatment: KEEP the full silhouette outline
	// (depth edges, always on) but RAISE the interior-crease threshold so only a
	// few strong structural "robotic" lines survive (leg-body joins, the camera
	// mount, major panel seams) instead of the busy rivet/seam mesh -- and push
	// the interior fade far out so those few lines don't wash away at the
	// pulled-back framing. Restores the orbit-view style on the way out. Interior
	// strength stays 1 in both (the earlier "strength 0" over-flattened the robot
	// -- it also killed the structural silhouettes the outline needs).
	edgesPass.setNormalThreshold( cinematicEnabled ? CINE_NORMAL_THRESHOLD : EDGE_NORMAL_THRESHOLD );
	edgesPass.setInteriorFade(
		cinematicEnabled ? CINE_INTERIOR_FADE_NEAR : EDGE_INTERIOR_FADE_NEAR,
		cinematicEnabled ? CINE_INTERIOR_FADE_FAR : EDGE_INTERIOR_FADE_FAR,
	);

	if ( cinematicEnabled ) {

		// Take full control of the camera. OrbitControls is disabled (so drags
		// don't fight the shot) and its update() is skipped in renderFrame while
		// active; cineNeedsInit snaps the look-target onto the subjects on the
		// first active frame so the shot doesn't swing in from the origin.
		cineTime = 0;
		cineNeedsInit = true;
		controls.enabled = false;

	} else {

		// Hand back to manual orbit from exactly where the cinematic cam left off,
		// then let the robot-only tracking follow-cam resync its delta baseline.
		controls.enabled = true;
		controls.target.copy( _cineLookTarget );
		hasLastBasePos = false;
		controls.update();

	}

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
	addGroundTileUVs( root );
	scene.add( root );

	mixer = new THREE.AnimationMixer( root );
	setupActionsFromClips( clips );

	applyGlobalTime( 0, { updateSlider: true } );

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

	// One authoritative clock delta per frame, shared by whichever camera mode
	// runs below (calling clock.getDelta() more than once per frame would split
	// the real elapsed time between the calls).
	const dtSec = clock.getDelta();

	// Cinematic mode takes precedence over the default robot-only follow: it
	// drives the camera fully (see updateCinematicCamera) and OrbitControls is
	// left disabled + its update() skipped this frame.
	const cinematicActive = cinematicEnabled && robotBase;

	if ( cinematicActive ) {

		updateCinematicCamera( dtSec || 0.016 );

	} else if ( trackingEnabled && robotBase ) {

		// Follow-cam: because the robot travels metres during a clip, lerp the
		// OrbitControls target toward the robot_base world position and
		// translate the camera by the SAME delta each frame — this orbits
		// around a moving target instead of re-framing/snapping.
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
		const lerpFactor = 1 - Math.pow( 0.001, dtSec || 0.016 );
		controls.target.lerp( _curBaseWorldPos, Math.min( 1, lerpFactor ) );

		_lastBaseWorldPos.copy( _curBaseWorldPos );

	}

	// Shadow-follow: recenter the key light's (tight, high-res) shadow frustum on
	// the robot's CURRENT world position every frame, independent of the
	// tracking-toggle above -- shadows should stay sharp near the action even when
	// the user has camera-tracking off and is orbiting freely. Same offset vector
	// as the light's own initial (3,5,2) position, so the light's direction (and
	// therefore shadow angle) never changes, only its world position does.
	if ( robotBase ) {

		robotBase.getWorldPosition( _shadowFollowPos );
		dirLight.target.position.copy( _shadowFollowPos );
		dirLight.position.copy( _shadowFollowPos ).add( DIR_LIGHT_OFFSET );

	}

	// Skip OrbitControls.update() while cinematic drives the camera directly:
	// its update() would reassert camera.position from its own spherical/target
	// state and stomp the shot we just set (blueprint-viewer memory).
	if ( ! cinematicActive ) controls.update();

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

		// timeSec/duration/pct now describe the UNIFIED timeline (follow+climb);
		// `phase` is which segment the playhead is currently in, and
		// `segmentTimeSec` is the local time within that segment's own clip.
		const action = phaseActions.get( currentPhase );

		return {
			phase: currentPhase,
			timeSec: globalTime,
			duration: totalDuration,
			pct: totalDuration > 0 ? ( globalTime / totalDuration ) * 100 : 0,
			segmentTimeSec: action ? action.time : 0,
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
	 * EDGE-COVERAGE PROBE (the "what's in the outline and what isn't" diagnostic).
	 *
	 * For the CURRENT camera/frame, measures per robot part how much of its
	 * on-screen silhouette boundary is actually being inked by the edge pass, and
	 * by which edge TYPE (depth/silhouette vs normal/interior crease), plus how
	 * dense the interior lines are. This is the objective signal used to tune the
	 * cinematic edge style (see the cinematic toggle): the goal is ~full boundary
	 * coverage on every part (legs/feet/body all outlined) with only a MODEST
	 * interior-line density (a few structural "robotic" lines, not a busy mesh).
	 *
	 * Method: (1) render one composer frame with the real materials to populate
	 * the edge mask, read it back (its .r=combined, .g=depth, .b=normal channels,
	 * see BlueprintEdgesPass); (2) re-render the scene with each mesh flat-colored
	 * by a per-PART id (unlit, tone-mapping off, into a linear RT so the id reads
	 * back exactly), read that back; (3) for each part, a pixel is a BOUNDARY
	 * pixel if any 4-neighbour belongs to a different part/background -- count how
	 * many boundary pixels have ink within `radius` px, split by edge type, and
	 * separately count interior (non-boundary) inked pixels. Read-only: restores
	 * every swapped material before returning.
	 *
	 * @returns per-part { areaPx, boundaryPx, silhouetteCovPct (any edge),
	 *   depthCovPct (depth edge only), interiorInkPct }.
	 */
	edgeCoverage( { edgeThresh = 0.35, radius = 1 } = {} ) {

		if ( ! modelRoot ) return { error: 'no model loaded' };

		const w = edgesPass._maskTarget.width;
		const h = edgesPass._maskTarget.height;

		// --- part bucketing: nearest named ancestor -> bucket ---
		const legLinks = [];
		for ( const q of [ 'FL', 'FR', 'RL', 'RR' ] ) for ( const seg of [ 'hip', 'thigh', 'calf', 'foot' ] ) legLinks.push( `${q}_${seg}` );
		const buckets = [ 'body', ...legLinks, 'payload', 'patient', 'structure', 'ground' ];
		const idOf = new Map( buckets.map( ( b, i ) => [ b, i + 1 ] ) );

		const bucketFor = ( mesh ) => {

			let p = mesh;
			while ( p ) {

				if ( legLinks.includes( p.name ) ) return p.name;
				if ( p.name === 'robot_base' ) return 'body';
				if ( p.name === 'oxygen_tank' || p.name === 'cradle_rails' ) return 'payload';
				if ( p.name === 'patient_human_anchor' ) return 'patient';
				if ( p.name === 'stairs' || p.name === 'handrails' ) return 'structure';
				if ( p.name === 'ground' ) return 'ground';
				p = p.parent;

			}
			return null;

		};

		// --- 1) mask (real materials) ---
		composer.render();
		const maskBuf = new Uint8Array( w * h * 4 );
		renderer.readRenderTargetPixels( edgesPass._maskTarget, 0, 0, w, h, maskBuf );

		// --- 2) per-part id render ---
		const idMats = new Map();
		const idMat = ( id ) => {

			if ( ! idMats.has( id ) ) {

				const m = new THREE.MeshBasicMaterial();
				m.toneMapped = false;
				m.color.setRGB( id / 255, 0, 0 ); // linear working space -> reads back as `id` in the R byte
				idMats.set( id, m );

			}
			return idMats.get( id );

		};

		const idRT = new THREE.WebGLRenderTarget( w, h, { minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter } );

		const restore = [];
		scene.traverse( ( n ) => {

			if ( ! n.isMesh ) return;
			restore.push( [ n, n.material ] );
			const b = bucketFor( n );
			n.material = idMat( b ? idOf.get( b ) : 0 );

		} );

		const prevBg = scene.background;
		const prevRT = renderer.getRenderTarget();
		const prevClear = new THREE.Color();
		renderer.getClearColor( prevClear );
		const prevAlpha = renderer.getClearAlpha();

		scene.background = null;
		renderer.setRenderTarget( idRT );
		renderer.setClearColor( 0x000000, 1 );
		renderer.clear( true, true, false );
		renderer.render( scene, camera );

		const idBuf = new Uint8Array( w * h * 4 );
		renderer.readRenderTargetPixels( idRT, 0, 0, w, h, idBuf );

		// restore
		renderer.setRenderTarget( prevRT );
		renderer.setClearColor( prevClear, prevAlpha );
		scene.background = prevBg;
		for ( const [ n, mat ] of restore ) n.material = mat;
		idRT.dispose();
		for ( const m of idMats.values() ) m.dispose();

		// --- 3) coverage stats ---
		const idAt = ( x, y ) => ( x < 0 || y < 0 || x >= w || y >= h ) ? 0 : Math.round( idBuf[ ( y * w + x ) * 4 ] );
		const chanMax = ( off, x, y ) => {

			let m = 0;
			for ( let dy = - radius; dy <= radius; dy ++ ) for ( let dx = - radius; dx <= radius; dx ++ ) {

				const xx = x + dx, yy = y + dy;
				if ( xx >= 0 && yy >= 0 && xx < w && yy < h ) m = Math.max( m, maskBuf[ ( yy * w + xx ) * 4 + off ] );

			}
			return m / 255;

		};

		const stats = {};
		for ( const b of buckets ) stats[ b ] = { area: 0, boundary: 0, inkedAny: 0, inkedDepth: 0, interior: 0, interiorInked: 0 };

		for ( let y = 0; y < h; y ++ ) for ( let x = 0; x < w; x ++ ) {

			const id = idAt( x, y );
			if ( id === 0 ) continue;
			const b = buckets[ id - 1 ];
			if ( ! b ) continue;
			const st = stats[ b ];
			st.area ++;

			const isBoundary = idAt( x + 1, y ) !== id || idAt( x - 1, y ) !== id || idAt( x, y + 1 ) !== id || idAt( x, y - 1 ) !== id;
			if ( isBoundary ) {

				st.boundary ++;
				if ( chanMax( 0, x, y ) >= edgeThresh ) st.inkedAny ++;
				if ( chanMax( 1, x, y ) >= edgeThresh ) st.inkedDepth ++;

			} else {

				st.interior ++;
				if ( maskBuf[ ( y * w + x ) * 4 ] / 255 >= edgeThresh ) st.interiorInked ++;

			}

		}

		const parts = {};
		for ( const b of buckets ) {

			const s = stats[ b ];
			if ( s.area === 0 ) continue;
			parts[ b ] = {
				areaPx: s.area,
				boundaryPx: s.boundary,
				silhouetteCovPct: + ( 100 * s.inkedAny / Math.max( 1, s.boundary ) ).toFixed( 1 ),
				depthCovPct: + ( 100 * s.inkedDepth / Math.max( 1, s.boundary ) ).toFixed( 1 ),
				interiorInkPct: + ( 100 * s.interiorInked / Math.max( 1, s.interior ) ).toFixed( 1 ),
			};

		}

		return { w, h, edgeThresh, radius, parts };

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
