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
import { BlueprintEdgesPass, NO_OUTLINE_LAYER } from './BlueprintEdgesPass.js';
import { buildPlaceholderRobot } from './PlaceholderRobot.js';
import { PatientHuman } from './PatientHuman.js';
// poseAt: read-only use of PatientGait.js's own exported pure query function
// (GAIT-owned; VERIFY may READ any file per IK_OVERHAUL_SPEC.md section 2) --
// patientDiag calls this directly (alongside patientHuman.sync(), which calls
// the SAME function internally) so it can read v2 pose fields (phaseC,
// per-foot landedAt/nextLiftAt, cane) that PatientHuman._lastSync doesn't
// (yet) surface, without waiting on a RIG-side plumbing change.
import { poseAt } from './PatientGait.js';
// CANE_PARAMS: read-only use of PatientCane.js's own exported tunables (RIG-
// owned; VERIFY may READ any file per IK_OVERHAUL_SPEC.md section 2) --
// patientDiag's M12_caneShaftClearanceMin needs caneLengthM to reconstruct the
// cane's ACTUAL rendered handle position from its Object3D's tip position +
// orientation (see patientDiag's own comment at that call site).
import { CANE_PARAMS } from './PatientCane.js';

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
const potResetBtn = document.getElementById( 'pot-reset' );

// ===========================================================================
// Renderer / scene / camera
//
// NOTE: canvasHost.clientWidth/clientHeight can legitimately read 0 here if
// this module executes before the browser has committed a layout pass for
// a just-inserted host element (observed in practice, not hypothetical â€”
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
// from the opposite side + a shader-injected fresnel rim â€” see the
// "Blueprint materials" section for how MeshToonMaterial splits the two
// directional lights' combined contribution into banded direct terms, with
// hemiLight staying a smooth ambient term on top.
//
// 2026-07-10 lighting pass ("should look better than the sim version"):
// added the fill light and warm/cool color split (classic complementary
// key+fill toon grading -- a warm key against a cool fill/ambient reads far
// richer than a single flat white light) and enabled real-time shadows.
// dirLight was raised from the old flat-material value (0.15) because the
// toon gradient map only bands the DIRECTIONAL contribution â€” at 0.15 it was
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
// Environment (IBL) for the REALISTIC ROBOT only.
//
// The baked robot.glb is bare geometry (0 materials/textures) -- it IS the real
// Go2 shape, just unpainted. Per the user's "import the robot, not toon (keep
// the human toon)", the robot geometry is given realistic (white plastic-shell) PBR
// materials (see makeRobotRealisticMaterial) lit by this small procedural studio
// environment; without an environment, PBR metal reads as dead black. The
// PATIENT and the whole SET stay toon (MeshToonMaterial ignores
// scene.environment), so ONLY the robot picks this up -- human + elements render
// exactly as before.
const _pmrem = new THREE.PMREMGenerator( renderer );
const _envEquirect = makeStudioEnvTexture();
scene.environment = _pmrem.fromEquirectangular( _envEquirect ).texture;
_envEquirect.dispose();
_pmrem.dispose();

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
	// This scene only ever appears on the Potential slide ("from a proven sim to
	// the living room"), so its backdrop is warmed toward a homey light-to-floor
	// gradient (each theme's cool studio grey lerped toward warm cream/amber)
	// rather than the neutral studio grey the sim look uses.
	const warmTop = new THREE.Color( palette.bgGradientTop ).lerp( new THREE.Color( 0xffdcae ), 0.34 );
	const warmBottom = new THREE.Color( palette.bgGradientBottom ).lerp( new THREE.Color( 0xc79a63 ), 0.34 );
	scene.background = makeBackgroundGradient( warmTop.getHex(), warmBottom.getHex() );

	if ( bodyMaterial ) bodyMaterial.color.set( palette.materialColor );
	if ( oxygenTankMaterial ) oxygenTankMaterial.color.set( palette.oxygenTankColor );
	if ( cradleRailsMaterial ) cradleRailsMaterial.color.set( palette.cradleRailsColor );
	if ( stairsMaterial ) stairsMaterial.color.set( palette.stairsColor );
	if ( handrailMaterial ) handrailMaterial.color.set( palette.handrailColor );
	// groundMaterial.color deliberately NOT re-tinted here: its tile colors are baked
	// into groundMaterial.map (see makeGroundTileTexture) and the material's own
	// .color stays neutral white always (set once at creation) so the toon shading
	// modulates the texture's own colors instead of double-tinting them.
	// patientMaterial.color deliberately NOT re-tinted here: the patient's colours
	// now live in a per-vertex COLOR attribute (baked by 3D body region in
	// PatientHuman.attachTo, since Xbot's overlapping UVs defeat a texture map) and
	// the material's own .color stays neutral white so the toon shading modulates
	// the vertex colours instead of double-tinting them -- same reasoning as
	// groundMaterial. (palette.patientColor is now unused.)
	// robotMaterial / robotFootMaterial are realistic PBR with intrinsic hardware
	// colors (dark metal / rubber, same in both themes) -- deliberately NOT
	// re-tinted from the palette here (the old palette.robotColor line was removed
	// when the robot switched from toon to the imported realistic look).
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
// of form â€” on a near-shadeless flat fill, that detail read as pure line
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

// ===========================================================================
// Realistic robot (imported look)
//
// The user asked to "import the robot and not make it toon (keep the human
// toon)". The robot geometry is already the real 1:1 Go2 from the sim, but the
// baked GLB carries no materials -- so "realistic" here means giving that real
// geometry real PBR materials (white plastic-shell body/legs, matte rubber feet)
// lit by the studio environment, instead of the flat cel material. Everything
// else (patient + set) stays on makeBlueprintMaterial (toon). The screen-space
// ink outline still runs scene-wide, but on the dark robot it naturally recedes
// into the body edge, so the robot reads realistic while the toon human/set keep
// their bold outline.
// ===========================================================================

// Soft procedural "studio" equirectangular texture -> scene.environment (see the
// light setup above). Warm-key / cool-fill gradient with two soft light blobs;
// enough for believable metal reflections without an external HDR.
function makeStudioEnvTexture() {

	const w = 512, h = 256;
	const canvas = document.createElement( 'canvas' );
	canvas.width = w; canvas.height = h;
	const ctx = canvas.getContext( '2d' );

	const g = ctx.createLinearGradient( 0, 0, 0, h );
	g.addColorStop( 0.0, '#e9e5dd' );
	g.addColorStop( 0.42, '#f5f2ec' );
	g.addColorStop( 0.58, '#dfe2e6' );
	g.addColorStop( 1.0, '#b7bcc4' );
	ctx.fillStyle = g;
	ctx.fillRect( 0, 0, w, h );

	function blob( cx, cy, r, color, alpha ) {

		const rg = ctx.createRadialGradient( cx, cy, 0, cx, cy, r );
		rg.addColorStop( 0, color );
		rg.addColorStop( 1, 'rgba(255,255,255,0)' );
		ctx.globalAlpha = alpha;
		ctx.fillStyle = rg;
		ctx.fillRect( 0, 0, w, h );
		ctx.globalAlpha = 1;

	}
	blob( w * 0.30, h * 0.26, h * 0.55, '#fff5e6', 0.85 ); // warm key
	blob( w * 0.78, h * 0.40, h * 0.5, '#e0ebff', 0.5 );   // cool fill

	const texture = new THREE.CanvasTexture( canvas );
	texture.mapping = THREE.EquirectangularReflectionMapping;
	texture.colorSpace = THREE.SRGBColorSpace;
	return texture;

}

// Real Go2 material: satin plastic/metal shell for the body/legs (color per call), matte rubber for the
// feet (pass a low metalness / high roughness). No toon banding, no fresnel rim
// -- just PBR lit by the scene lights + environment, so it reads as the real
// hardware rather than a cel drawing.
function makeRobotRealisticMaterial( colorHex, opts = {} ) {

	return new THREE.MeshStandardMaterial( {
		color: colorHex,
		metalness: opts.metalness ?? 0.4,
		roughness: opts.roughness ?? 0.45,
		envMapIntensity: opts.envMapIntensity ?? 1.1,
		// vertexColors: robot_base carries a baked COLOR_0 (shell gray + black
		// lidar/sensors, from the source material groups) -- see recolor_base_lidar.py.
		vertexColors: opts.vertexColors ?? false,
	} );

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
// Patient stays TOON (per "keep the human toon"). Robot goes REALISTIC: real
// white plastic-shell PBR on the real geometry (see makeRobotRealisticMaterial). The
// robot's colors are intrinsic to the hardware (same in both themes), so they
// live here rather than in the theme palette -- applyTheme no longer re-tints
// the robot (its palette.robotColor line was removed).
//
// The patient reads as an elderly oxygen-therapy patient (silver hair, pale
// skin/hands, a dusty-teal knit sweater cut long over the hips, grey trousers,
// brown slippers) via a PER-VERTEX COLOR attribute keyed to each vertex's 3D body
// region -- NOT a diffuse texture. A texture map is impossible here: Xbot's UV
// islands overlap/mirror (a single texel is shared by torso AND foot verts), so
// any map keyed to those UVs cross-contaminates and the torso renders as bare
// skin. Vertex colours are keyed to 3D position, never UVs -- see
// PatientHuman.js's paintPatientRegionColors(). Same pattern as the robot's baked
// COLOR_0 below: white base .color so the vertex colour IS the albedo (applyTheme
// no longer re-tints the patient), toon banding + fresnel rim still apply on top.
let patientMaterial = makeBlueprintMaterial( 0xffffff );
patientMaterial.vertexColors = true; // per-vertex region colours are baked on in PatientHuman.attachTo
// Distinct program cache key so this vertexColors material never reuses a plain
// (vertexColors:false) 'bp-plain' toon program -- same cache-collision guard as the
// grain material above (see makeBlueprintMaterial's customProgramCacheKey comment).
patientMaterial.customProgramCacheKey = () => 'bp-patient-vc';
let robotMaterial = makeRobotRealisticMaterial( 0xc0c0c0, { metalness: 0.2, roughness: 0.5, envMapIntensity: 1.0 } ); // SILVER (#C0C0C0) Go2 body/legs -- matte painted-plastic shell (low metalness so it reads as painted plastic, not chrome); carries the thighs
let robotBlackMaterial = makeRobotRealisticMaterial( 0x232629, { metalness: 0.0, roughness: 0.85, envMapIntensity: 0.5 } ); // matte near-black -- the foot contact balls. 0x232629 == the LINEAR BLACK baked into the COLOR_0 meshes, so the black foot ball meets the black calf foot-pad with no seam.
// robot_base + the four calves carry a baked COLOR_0 (silver shell + BLACK for
// the head lidar/sensors, the robot's own printed logo WORDS, and the calf's
// ground-contact foot pad) recovered from the source GeomSubsets by
// pipeline/recolor_parts.py. White base color so the vertex color IS the albedo.
// The thighs (uniform silver) and hips/feet (uniform black) stay on flat materials.
let robotBaseMaterial = makeRobotRealisticMaterial( 0xffffff, { metalness: 0.2, roughness: 0.5, envMapIntensity: 1.0, vertexColors: true } );
let logoMaterial = makeBlueprintMaterial( PALETTES[ currentThemeName ].logoColor );

// ===========================================================================
// Living-room set â€” "from a proven sim to the living room" (Potential slide)
//
// This scene renders ONLY on the Potential slide, so we dress the sim staircase
// into a warm home around it: a wood floor (the cold sim tile is hidden on load,
// see finishModelSetup), a painted far wall with a sunlit window + framed print,
// a rug under the walk line, and a couch / floor-lamp / coffee-table / plant
// vignette in the background. Built once from simple toon primitives in three.js
// WORLD coords (measured against the real stairs AABB: stairs xâˆˆ[2,8.5] rising to
// yâ‰ˆ2, floor at y=0, walk line zâ‰ˆ0). Camera sits on +Z looking âˆ’Z, so the
// furniture at zâ‰ˆâˆ’3.4 reads as background behind the free-standing staircase.
// Added straight to `scene`; gated with the rest of the viewer's render loop.
// ===========================================================================
let livingRoom = null;

function buildLivingRoom() {

	const room = new THREE.Group();
	room.name = 'living_room';

	const box = ( w, h, d, mat ) => new THREE.Mesh( new THREE.BoxGeometry( w, h, d ), mat );
	const cyl = ( rt, rb, h, mat, seg = 20 ) => new THREE.Mesh( new THREE.CylinderGeometry( rt, rb, h, seg ), mat );

	const woodFloorMat = makeBlueprintMaterial( 0xb0824f );
	const wallMat = makeBlueprintMaterial( 0xd8cdb6 );
	const wallMat2 = makeBlueprintMaterial( 0xcabfa6 );
	const skirtMat = makeBlueprintMaterial( 0xefe9dd );
	const rugMat = makeBlueprintMaterial( 0xbb7a52 );
	const rugMat2 = makeBlueprintMaterial( 0xe0c39a );
	const couchMat = makeBlueprintMaterial( 0x7c8b6f );
	const couchMat2 = makeBlueprintMaterial( 0x8f9d82 );
	const woodMat = makeBlueprintMaterial( 0x8a6a45 );
	const lampPoleMat = makeBlueprintMaterial( 0x3a3630 );
	const shadeMat = makeBlueprintMaterial( 0xf2e2c0 );
	const potMat = makeBlueprintMaterial( 0xb0552f );
	const leafMat = makeBlueprintMaterial( 0x5f7d55 );
	const frameMat = makeBlueprintMaterial( 0x6b5d4a );
	const glassMat = new THREE.MeshBasicMaterial( { color: 0xfff0d2 } ); // sunlit window (unlit glow)
	const pictureMat = makeBlueprintMaterial( 0x9fb4c2 );

	// Wood floor (the sim 'ground' tile is hidden on load â€” see finishModelSetup)
	const floor = box( 21, 0.06, 11, woodFloorMat );
	floor.position.set( 0.5, -0.03, 0 ); floor.receiveShadow = true; room.add( floor );

	const WALL_H = 3.3, WALL_Z = -4.2;
	const farWall = box( 20, WALL_H, 0.2, wallMat );
	farWall.position.set( -0.5, WALL_H / 2, WALL_Z ); farWall.receiveShadow = true; room.add( farWall );
	const endWall = box( 0.2, WALL_H, 8, wallMat2 );
	endWall.position.set( -9.4, WALL_H / 2, -0.2 ); endWall.receiveShadow = true; room.add( endWall );
	const skirt = box( 20, 0.14, 0.06, skirtMat ); skirt.position.set( -0.5, 0.07, WALL_Z + 0.12 ); room.add( skirt );

	// Sunlit window on the far wall (frame + muntins) â€” over the couch, mid-room
	const winW = 2.6, winH = 1.7, winX = -1.6, winY = 1.8, winZ = WALL_Z + 0.06;
	const glass = box( winW, winH, 0.04, glassMat ); glass.position.set( winX, winY, winZ ); room.add( glass );
	const frameT = box( winW + 0.3, 0.16, 0.1, frameMat ); frameT.position.set( winX, winY + winH / 2 + 0.02, winZ ); room.add( frameT );
	const frameB = box( winW + 0.3, 0.16, 0.1, frameMat ); frameB.position.set( winX, winY - winH / 2 - 0.02, winZ ); room.add( frameB );
	const frameL = box( 0.16, winH + 0.3, 0.1, frameMat ); frameL.position.set( winX - winW / 2 - 0.02, winY, winZ ); room.add( frameL );
	const frameR = box( 0.16, winH + 0.3, 0.1, frameMat ); frameR.position.set( winX + winW / 2 + 0.02, winY, winZ ); room.add( frameR );
	const muntV = box( 0.06, winH, 0.06, frameMat ); muntV.position.set( winX, winY, winZ + 0.01 ); room.add( muntV );
	const muntH = box( winW, 0.06, 0.06, frameMat ); muntH.position.set( winX, winY, winZ + 0.01 ); room.add( muntH );

	// Framed print on the far wall (back near the start)
	const picFrame = box( 1.1, 0.8, 0.06, frameMat ); picFrame.position.set( -4.4, 1.95, WALL_Z + 0.05 ); room.add( picFrame );
	const pic = box( 0.92, 0.62, 0.02, pictureMat ); pic.position.set( -4.4, 1.95, WALL_Z + 0.08 ); room.add( pic );

	// Rug â€” long runner down the middle of the walk line toward the stairs
	const rug = box( 7.0, 0.03, 3.8, rugMat ); rug.position.set( -1.4, 0.016, -0.2 ); rug.receiveShadow = true; room.add( rug );
	const rugInner = box( 6.1, 0.034, 2.9, rugMat2 ); rugInner.position.set( -1.4, 0.02, -0.2 ); room.add( rugInner );

	// Couch against the far wall (facing +Z into the room), mid-room
	const couch = new THREE.Group(); couch.position.set( -1.6, 0, -3.35 );
	const seat = box( 2.4, 0.42, 0.95, couchMat ); seat.position.set( 0, 0.4, 0 ); seat.castShadow = seat.receiveShadow = true; couch.add( seat );
	const backrest = box( 2.4, 0.78, 0.24, couchMat ); backrest.position.set( 0, 0.8, -0.36 ); backrest.castShadow = true; couch.add( backrest );
	const armL = box( 0.26, 0.6, 0.95, couchMat ); armL.position.set( -1.2, 0.5, 0 ); armL.castShadow = true; couch.add( armL );
	const armR = box( 0.26, 0.6, 0.95, couchMat ); armR.position.set( 1.2, 0.5, 0 ); armR.castShadow = true; couch.add( armR );
	const cush1 = box( 1.05, 0.2, 0.82, couchMat2 ); cush1.position.set( -0.55, 0.62, 0.03 ); couch.add( cush1 );
	const cush2 = box( 1.05, 0.2, 0.82, couchMat2 ); cush2.position.set( 0.55, 0.62, 0.03 ); couch.add( cush2 );
	room.add( couch );

	// Coffee table in front of the couch
	const table = new THREE.Group(); table.position.set( -1.4, 0, -2.15 );
	const top = box( 1.5, 0.1, 0.7, woodMat ); top.position.set( 0, 0.42, 0 ); top.castShadow = true; table.add( top );
	for ( const [ lx, lz ] of [ [ -0.65, -0.28 ], [ 0.65, -0.28 ], [ -0.65, 0.28 ], [ 0.65, 0.28 ] ] ) {

		const leg = box( 0.08, 0.42, 0.08, woodMat ); leg.position.set( lx, 0.21, lz ); table.add( leg );

	}
	room.add( table );

	// Floor lamp + a warm point-light glow (near the start of the walk)
	const lamp = new THREE.Group(); lamp.position.set( -4.4, 0, -3.4 );
	const lbase = cyl( 0.16, 0.18, 0.06, lampPoleMat ); lbase.position.set( 0, 0.03, 0 ); lamp.add( lbase );
	const pole = cyl( 0.03, 0.03, 1.7, lampPoleMat ); pole.position.set( 0, 0.88, 0 ); lamp.add( pole );
	const shade = cyl( 0.16, 0.3, 0.36, shadeMat ); shade.position.set( 0, 1.78, 0 ); lamp.add( shade );
	room.add( lamp );
	const lampLight = new THREE.PointLight( 0xffce8a, 6, 6, 2 );
	lampLight.position.set( -4.4, 1.66, -3.4 ); room.add( lampLight );

	// Potted plants â€” one to bridge the middle, one right at the stairs base, so
	// the room stays furnished the whole way as the robot + patient cross it.
	const leafSpread = [ [ 0, 0.64, 0, 0.28 ], [ -0.15, 0.52, 0.06, 0.2 ], [ 0.15, 0.54, -0.05, 0.2 ], [ 0, 0.84, 0, 0.2 ] ];
	for ( const [ plx, plz ] of [ [ -3.4, -3.5 ], [ 1.6, -1.4 ] ] ) {

		const plant = new THREE.Group(); plant.position.set( plx, 0, plz );
		const pot = cyl( 0.2, 0.15, 0.34, potMat ); pot.position.set( 0, 0.17, 0 ); pot.castShadow = true; plant.add( pot );
		for ( const [ px, py, pz, pr ] of leafSpread ) {

			const leaf = new THREE.Mesh( new THREE.SphereGeometry( pr, 10, 8 ), leafMat );
			leaf.position.set( px, py, pz ); plant.add( leaf );

		}
		room.add( plant );

	}

	// A low console/cabinet against the wall just before the stairs, so the
	// approach to the staircase is furnished too (not an empty run-up).
	const console = new THREE.Group(); console.position.set( 0.4, 0, -3.5 );
	const consoleTop = box( 1.6, 0.12, 0.5, woodMat ); consoleTop.position.set( 0, 0.72, 0 ); consoleTop.castShadow = true; console.add( consoleTop );
	const consoleBody = box( 1.5, 0.62, 0.44, couchMat2 ); consoleBody.position.set( 0, 0.37, 0 ); consoleBody.castShadow = true; console.add( consoleBody );
	room.add( console );

	// A low side table + plant sitting near the MIDDLE of the walk line (not on
	// the far background wall like the rest of the set) so the crossing reads as
	// the pair navigating a lived-in room, not walking an empty runway. Placed
	// just outside the follow clip's own lateral sway band (measured z in about
	// [-0.45, 0.4] across the clip) so it's close enough to look stepped-around
	// without ever actually being on the baked path.
	const sideTable = new THREE.Group(); sideTable.position.set( -1.5, 0, -0.95 );
	const sideTableTop = cyl( 0.26, 0.26, 0.06, woodMat ); sideTableTop.position.set( 0, 0.5, 0 ); sideTableTop.castShadow = true; sideTable.add( sideTableTop );
	const sideTableLeg = cyl( 0.04, 0.05, 0.47, lampPoleMat ); sideTableLeg.position.set( 0, 0.235, 0 ); sideTable.add( sideTableLeg );
	const midPot = cyl( 0.14, 0.11, 0.22, potMat ); midPot.position.set( 0, 0.64, 0 ); midPot.castShadow = true; sideTable.add( midPot );
	for ( const [ px, py, pz, pr ] of [ [ 0, 0.86, 0, 0.18 ], [ -0.1, 0.78, 0.05, 0.13 ], [ 0.1, 0.8, -0.04, 0.13 ] ] ) {

		const leaf = new THREE.Mesh( new THREE.SphereGeometry( pr, 10, 8 ), leafMat );
		leaf.position.set( px, py, pz ); sideTable.add( leaf );

	}
	room.add( sideTable );

	scene.add( room );
	livingRoom = room;

}

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
	// robot_base + the four thighs + the four calves carry a baked COLOR_0 (silver
	// shell with black regions: the head lidar/sensors + logo-words on robot_base,
	// the round TOP hip-actuator housing on each thigh, and the ground-contact foot
	// pad on each calf): paint them with the white vertexColors material so the
	// baked color is the albedo. The hips (leg<->body connectors) + the thigh
	// blades are silver; only the foot balls are uniform flat black. Thighs +
	// calves are tagged EXPLICITLY so their material is unambiguous regardless of
	// ancestor tags.
	robot_base: () => robotBaseMaterial,
	FL_hip: () => robotMaterial,
	FR_hip: () => robotMaterial,
	RL_hip: () => robotMaterial,
	RR_hip: () => robotMaterial,
	FL_thigh: () => robotBaseMaterial,
	FR_thigh: () => robotBaseMaterial,
	RL_thigh: () => robotBaseMaterial,
	RR_thigh: () => robotBaseMaterial,
	FL_calf: () => robotBaseMaterial,
	FR_calf: () => robotBaseMaterial,
	RL_calf: () => robotBaseMaterial,
	RR_calf: () => robotBaseMaterial,
	FL_foot: () => robotBlackMaterial,
	FR_foot: () => robotBlackMaterial,
	RL_foot: () => robotBlackMaterial,
	RR_foot: () => robotBlackMaterial,
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
// exactly the high-contrast "jaggy" it exists to blur â€” softening deliberate
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
	// Robot subtree lives on this layer; the edge pass renders it clean (no ink).
	noOutlineLayer: NO_OUTLINE_LAYER,
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
	// distances â€” entire flat faces (especially the shallow-angle stair
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
// Near-profile (~90deg) rather than the old trailing-quarter (118deg) angle:
// robot travels +X, and a camera on the +Z side looking back toward -Z reads
// world +X as screen-right (standard THREE look-at chirality with up=+Y), so
// this stages the walk as a clean LEFT-TO-RIGHT crossing of the frame instead
// of a foreshortened three-quarter chase that made the robot/patient gap look
// tighter than it actually is (real separation stays 1.1-1.75 m the whole
// follow clip -- see the pygltflib measurement in the 2026-07-13 session, no
// actual collision in the baked data, just an angle that hid the gap).
const CINE_BASE_AZ = 95 * Math.PI / 180; // azimuth measured from +X in the XZ (ground) plane
const CINE_BASE_EL = 24 * Math.PI / 180;  // elevation above the ground plane
const CINE_SWAY_AZ = 9 * Math.PI / 180;   // slow left/right drift amplitude
const CINE_SWAY_EL = 4 * Math.PI / 180;   // slow rise/fall amplitude
const CINE_SWAY_AZ_PERIOD = 13;           // s, one full left-right sway
const CINE_SWAY_EL_PERIOD = 19;           // s, one full rise-fall sway

// Pulled back from the original tight framing (pad 0.95 / margin 1.16 / max 7.5)
// so the living-room set around the pair â€” wood floor, rug, couch + window, the
// staircase â€” reads in shot, not just the two subjects. This slide's whole point
// is "the living room", so the environment has to be visible.
// 1.9 (was 1.25): the framing radius is 0.5*separation + this pad, so it used to
// track the live robot/patient gap fairly closely. That was fine at the old ~1.65 m
// average follow gap, but once the follow distance was tightened to ~0.3 m (real
// recorded data, not a camera bug) the SAME formula pulled the camera dramatically
// closer -- radius dropped from ~2.08 to ~1.4, i.e. the whole shot (robot, patient,
// room) zoomed in ~1.5x, reading as everything "growing too big". Raising the pad
// keeps the radius (and therefore the framing) roughly where it was regardless of
// how tight the live follow gap is -- the room should stay in shot either way, per
// the pullback this constant already existed for.
const CINE_SUBJECT_PAD = 1.9;   // extra framing radius so neither subject kisses the frame edge (m)
const CINE_FRAME_MARGIN = 1.42;  // >1 leaves breathing room around the pair + room
const CINE_MIN_DIST = 2.6;       // never dolly closer than this (m)
const CINE_MAX_DIST = 9.5;       // never drift further than this (m)
const CINE_TARGET_UP_BIAS = 0.15; // aim a touch above the base/hip midpoint so the pair sits mid-frame, not along the bottom (m)

// Frame-rate-independent smoothing bases for `1 - base^dt`: smaller = snappier.
// Position eases a touch floatier than the look-target so quick subject moves
// read as the camera gliding to catch up.
const CINE_POS_SMOOTH_BASE = 0.0030;
const CINE_TGT_SMOOTH_BASE = 0.0015;

// Entry reveal: CINE_POS_SMOOTH_BASE alone converges in well under a second, so
// arriving on the Potential slide used to SNAP the camera onto the framing from
// wherever it last sat (a different slide's orbit distance/angle) instead of
// reading as a deliberate move. For the first ENTRY_REVEAL_SEC after activation
// the position lerp uses a much slower base (a real glide-in), ramping linearly
// to the normal snappy tracking rate so the camera still keeps up with the
// subjects once the reveal is over.
const CINE_ENTRY_REVEAL_SEC = 2.4;
const CINE_ENTRY_POS_SMOOTH_BASE = 0.35;

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

	const entryT = Math.min( 1, cineTime / CINE_ENTRY_REVEAL_SEC );
	const posSmoothBase = THREE.MathUtils.lerp( CINE_ENTRY_POS_SMOOTH_BASE, CINE_POS_SMOOTH_BASE, entryT );
	const posLerp = Math.min( 1, 1 - Math.pow( posSmoothBase, dtSec ) );
	const tgtLerp = Math.min( 1, 1 - Math.pow( CINE_TGT_SMOOTH_BASE, dtSec ) );

	camera.position.lerp( _cinePosGoal, posLerp );
	_cineLookTarget.lerp( _cineTargetGoal, tgtLerp );

	camera.lookAt( _cineLookTarget );

	// Keep OrbitControls' target in sync so switching cinematic OFF resumes
	// manual orbit from exactly here, with no snap.
	controls.target.copy( _cineLookTarget );

}

// Playback (optional "play" chip) state â€” see PLAYBACK section below.
let isPlaying = false;
let lastPlaybackTimestamp = 0;

// Patient: a real imported+rigged human model (see PatientHuman.js), loaded in
// parallel with robot.glb below and wired up once both are ready. Kicked off here
// (not inside loadRealModel) so the two loads race instead of serializing.
const patientHuman = new PatientHuman();
const patientHumanReady = patientHuman.load();

// robot.meta.json: this pipeline's own stair_spec/landing_far_x_m (see the file
// itself â€” start_x_m/step_height_m/step_depth_m/step_count/landing_depth_m), needed
// by patientHuman.buildGait() to build the procedural gait's terrain model (see
// PatientGait.buildTerrain). Fetched here (racing the GLB loads, same pattern as
// patientHumanReady above) rather than inside loadRealModel, so a slow/failed fetch
// doesn't serialize behind the (much larger) robot.glb download. On failure: loudly
// console.error and degrade exactly like an Xbot load failure (no patient gait built
// â€” patientHuman.buildGait() is simply never called below, so the human stays
// un-posed rather than silently falling back to some invented default staircase).
// robot_potential.glb/.meta.json: this viewer (the Potential/living-room cinematic
// scene) uses its OWN dedicated bake -- a real zigzag-follow + full climb-with-
// patient capture -- kept separate from the shared models/robot.glb that hero.js's
// small per-policy panels use, so tuning one scene's clip never touches the other's
// (2026-07-13 session: the two used to share one file/clip pair, which made a
// climb-only capture for the blind-RL panel silently break the patient's pose in
// this viewer too -- see that incident's fix).
const robotMetaReady = fetch( './models/robot_potential.meta.json' )
	.then( ( r ) => r.json() )
	.catch( ( error ) => {

		console.error( '[blueprint-viewer] failed to load ./models/robot_potential.meta.json â€” patient gait will not be built:', error );
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

	// Placed brand-label meshes REMOVED per user direction: the robot now shows
	// only its OWN printed logo words, which are baked into the robot_base mesh's
	// `ç™½è‰²logo` GeomSubset and colored black by pipeline/recolor_parts.py. The
	// old buildLogoMesh()/buildSideLogoMesh() overlays (a top deck label + the
	// side "Unitree"/"Go2" text) are gone; the loop above still strips any that a
	// prior load attached. (builders kept below in case the overlays are wanted
	// back.)

}

// ===========================================================================
// Camera fit â€” frame the robot_base subtree's bbox at t=0 on load
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
	// near:far ratio of ~5000:1 â€” with a standard (non-logarithmic) depth
	// buffer that crushes almost all depth precision into the first few
	// percent of that range, leaving the actual geometry (which sits right
	// where the robot is, a few meters out) with barely any distinguishable
	// depth values between neighbouring pixels on the SAME flat face. That
	// silently broke BlueprintEdgesPass's depth-discontinuity term (entire
	// faces flickered as "edges" from raw depth-texture quantization noise,
	// see git history / incident notes for the debug captures that isolated
	// this). Keeping near:far comfortably under ~1000:1 here is what fixes
	// it â€” this is a real, load-bearing constraint of the edge pass, not
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
// just swapping action WEIGHTS (active=1, inactive=0) â€” never calling
// .stop()/.play() again â€” so switching is instant and glitch-free, and
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
 * Rewind the unified demo timeline to t=0 (window.__viewer.resetDemo() + the
 * #pot-reset stage-chrome button + the Home key -- see their own wiring below).
 * Reuses applyGlobalTime(0, ...), the SAME primitive scrubbing/jumpToSegment/
 * playback already funnel through (the "*** THE CRUCIAL SCRUBBING LOGIC ***"
 * block below), so this is not a second "what pose is shown" code path.
 *
 * Unlike jumpToSegment (a manual chip click, which pauses playback), a reset
 * KEEPS autoplay running from t=0 if it was already running -- lastPlaybback-
 * Timestamp is re-stamped to "now" first so the next stepPlayback() call sees
 * a small, correct delta instead of one spanning however long ago playback
 * last actually advanced (mirrors setPlaying's own bookkeeping); if playback
 * was paused, reset leaves it paused, just rewound.
 *
 * No-op before a model has loaded (mixer/segments are guarded inside
 * applyGlobalTime) and safe to call repeatedly or mid-scrub (applyGlobalTime
 * is idempotent -- calling it with t=0 twice in a row is a no-op the second
 * time).
 */
function resetDemo() {

	lastPlaybackTimestamp = performance.now();
	applyGlobalTime( 0, { updateSlider: true } );

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
// with clock.getDelta() â€” mixer.update(dt) is NEVER called with a nonzero
// dt from the rAF loop while scrub-driven. Instead:
//
//   1. The <input type="range"> fires an 'input' event with a 0..100 value.
//   2. We map that value to a TIME on the currently-active clip:
//        activeAction.time = (v / 100) * clip.duration
//   3. We call mixer.update(0) â€” passing a delta of ZERO. AnimationMixer's
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
// then calls mixer.update(0) â€” it never lets the mixer do the time
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
	timeReadout.textContent = `t ${t} / ${total} s Â· ${currentPhase}`;

}

scrubber.addEventListener( 'input', () => {

	// Slider drag while playing pauses playback (scrub is authoritative).
	if ( isPlaying ) setPlaying( false );

	scrubToPercent( parseFloat( scrubber.value ) );

} );

// ===========================================================================
// Phase buttons â€” now jump-to-segment shortcuts on the single unified timeline
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
// Keyboard: Left/Right nudge slider +/-0.5, Home rewinds to t=0
// ===========================================================================

window.addEventListener( 'keydown', ( ev ) => {

	if ( ev.target instanceof HTMLInputElement || ev.target instanceof HTMLTextAreaElement ) return;

	if ( ev.key === 'ArrowLeft' || ev.key === 'ArrowRight' ) {

		if ( isPlaying ) setPlaying( false );

		const delta = ev.key === 'ArrowLeft' ? -0.5 : 0.5;
		const next = THREE.MathUtils.clamp( parseFloat( scrubber.value ) + delta, 0, 100 );
		scrubToPercent( next );
		ev.preventDefault();

	} else if ( ev.key === 'Home' ) {

		resetDemo();
		ev.preventDefault();

	}

} );

// Reset chip (#pot-reset -- separate minimal stage chrome, since .viewer-controls
// itself is hidden on this passive Potential slide; see styles.css).
if ( potResetBtn ) potResetBtn.addEventListener( 'click', () => resetDemo() );

// ===========================================================================
// Theme + tracking toggles
// ===========================================================================

themeToggle.addEventListener( 'click', () => {

	applyTheme( currentThemeName === 'light' ? 'dark' : 'light' );

} );

if ( trackingToggle ) trackingToggle.addEventListener( 'click', () => {

	trackingEnabled = ! trackingEnabled;
	trackingToggle.textContent = `tracking Â· ${ trackingEnabled ? 'on' : 'off' }`;
	trackingToggle.setAttribute( 'aria-pressed', trackingEnabled ? 'true' : 'false' );
	if ( trackingEnabled ) hasLastBasePos = false; // resync delta baseline on re-enable

} );

// NOTE: the cinematic / tracking / plumb debug chips were removed from the pitch
// build's demo markup (index.html), so these elements can be null. Each listener
// is registered only when its chip exists, keeping the debug controls available
// if the chips are ever restored without breaking the cleaned-up demo.
if ( cinematicToggle ) cinematicToggle.addEventListener( 'click', () => {

	cinematicEnabled = ! cinematicEnabled;
	cinematicToggle.textContent = `cinematic Â· ${ cinematicEnabled ? 'on' : 'off' }`;
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

if ( plumbToggle ) plumbToggle.addEventListener( 'click', () => {

	plumbLineEnabled = ! plumbLineEnabled;
	plumbLine.visible = plumbLineEnabled;
	plumbToggle.textContent = `plumb line Â· ${ plumbLineEnabled ? 'on' : 'off' }`;
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
	// Robot subtree -> NO_OUTLINE layer so the edge pass renders it clean (no toon
	// ink lines), like an Isaac viewport -- a deliberate contrast to the toon-
	// outlined human + set. It stays on layer 0 too (beauty/shadows unchanged);
	// only the edge pass's mask render singles it out. Payload (oxygen_tank/
	// cradle_rails) is under robot_base so it's covered; the async logo meshes
	// enable the layer themselves in attachLogoLabel().
	const _robotNoOutline = root.getObjectByName( 'robot_base' );
	if ( _robotNoOutline ) _robotNoOutline.traverse( ( o ) => o.layers.enable( NO_OUTLINE_LAYER ) );
	addGroundTileUVs( root );
	scene.add( root );

	// This scene lives on the Potential slide dressed as a living room (see
	// buildLivingRoom) â€” hide the cold sim tile floor so the warm wood floor shows.
	const _simGround = root.getObjectByName( 'ground' );
	if ( _simGround ) _simGround.visible = false;

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
			'./models/robot_potential.glb',
			( gltf ) => {

				usingPlaceholder = false;
				modelWarning.hidden = true;

				const root = gltf.scene || gltf.scenes[ 0 ];
				const baseNode = root.getObjectByName( 'robot_base' ) || root;
				finishModelSetup( root, gltf.animations || [], baseNode );

				// Real mesh only (see attachLogoLabel doc comment) â€” races against
				// the GLTF load same as the patient human below.
				logoFontReady.then( ( font ) => attachLogoLabel( baseNode, font ) );

				// Wait for the (concurrently-loading) patient human model AND
				// robot.meta.json too, so the first rendered frame never shows the
				// robot without its patient â€” resolves either way (PatientHuman.
				// load() catches its own errors and just leaves .ready false,
				// degrading to "no patient shown"; robotMetaReady catches its own
				// fetch error and resolves null, degrading to "no patient gait
				// built" â€” see robotMetaReady's own comment).
				Promise.all( [ patientHumanReady, robotMetaReady ] ).then( ( [ , meta ] ) => {

					const isaacWorldNode = root.getObjectByName( 'isaac_world' );
					const patientRootNode = root.getObjectByName( 'patient_root' );
					if ( isaacWorldNode && patientRootNode ) {

						patientHuman.attachTo( isaacWorldNode, patientRootNode, patientMaterial );

						if ( meta ) {

							// phaseClips (module-level Map, populated by
							// setupActionsFromClips inside the finishModelSetup call
							// above, which already ran synchronously before this
							// async continuation) â€” buildGait needs the RAW
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

				console.error( '[blueprint-viewer] GLTFLoader failed to load ./models/robot_potential.glb:', error );
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
// advanced here with a clock delta â€” see the scrubbing block above. The
// optional playback chip advances time itself (also via mixer.update(0)),
// independent of this rAF's own clock.
// ===========================================================================

const clock = new THREE.Clock();

// Render-gate: this (heavy) viewer only lives on the Potential slide, but the
// rAF loop would otherwise composer.render() a 324k-tri scene + shadow map +
// edge pass EVERY frame for the entire session even while it's scrolled far
// off-screen -- the single biggest steady-state cost in the deck and the reason
// scrolling felt slow. deck.js flips this on only while Potential is on screen
// (viewer.setActive) so the GPU is idle otherwise. The manual __viewer.renderFrame()
// escape hatch below still bypasses the gate for headless verification tooling.
let renderActive = false;

function animate() {

	requestAnimationFrame( animate );
	if ( ! renderActive ) return;
	renderFrame();

}

/**
 * Turn the per-frame render loop on/off (called by deck.js as the Potential
 * slide enters/leaves the viewport). On enable: drop the accumulated clock gap
 * so the cinematic sway/camera lerps don't jump on the first live frame, then
 * paint one frame immediately so the canvas is never briefly blank.
 */
function setRenderActive( on ) {

	on = !! on;
	if ( on === renderActive ) return;
	renderActive = on;
	if ( on ) { clock.getDelta(); renderFrame(); }

}

// The actual per-frame work, factored out of the rAF scheduling wrapper
// above so it can also be invoked directly (see window.__viewer.renderFrame
// below) â€” useful for automated/headless verification tooling where the
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
		// translate the camera by the SAME delta each frame â€” this orbits
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
	/**
	 * Autoplay control for the pitch deck (js/deck.js): the rollout auto-plays
	 * while the demo section is on screen and pauses when it isn't. Restarts
	 * from the top if it had already run to the end, so re-entering the demo
	 * always shows motion rather than a stopped frame.
	 */
	play( shouldPlay ) {

		if ( shouldPlay && globalTime >= totalDuration ) applyGlobalTime( 0, { updateSlider: true } );
		setPlaying( !! shouldPlay );

	},
	/**
	 * Rewind the unified demo timeline to t=0 (also bound to the #pot-reset
	 * stage-chrome button and the Home key -- see their own wiring). Keeps
	 * autoplay running from t=0 if it was already running; idempotent; a
	 * no-op before the model has loaded. See resetDemo()'s own doc comment.
	 */
	resetDemo() {

		resetDemo();

	},
	/**
	 * Turn cinematic mode on/off from the deck. js/deck.js calls this for the
	 * Potential slide, where the viewer is a passive, autoplaying cinematic view.
	 * Reuses the cinematic chip's own click handler so there is a single code path.
	 */
	setCinematic( on ) {

		if ( cinematicToggle && cinematicEnabled !== !! on ) cinematicToggle.click();

	},
	/**
	 * Enable/disable the live render loop (deck.js: on only while the Potential
	 * slide is on screen). The heavy scene is idle otherwise -- see setRenderActive.
	 */
	setActive( on ) {

		setRenderActive( on );

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
	 * Not used by normal interactive operation â€” the rAF-driven animate()
	 * loop (started at boot) is what drives the app for a real user. This
	 * exists for automated/headless verification tooling, since browsers
	 * throttle rAF to near-zero on a backgrounded/hidden tab.
	 */
	renderFrame() {

		renderFrame();

	},
	/**
	 * Capture the current frame to a PNG on disk via the dev server's POST /shot
	 * sink (serve.py), returning a Promise of the saved file path. This is the
	 * RELIABLE headless-capture path: the canvas is write-only (renderer has no
	 * preserveDrawingBuffer), so pixels must be read in the SAME tick as the
	 * render â€” hence renderFrame() immediately before readPixels here â€” and the
	 * ~2 MB of bytes leaves via a fetch POST body (no size limit) rather than an
	 * eval return value (truncates ~25 KB). preview_screenshot times out on this
	 * always-animating, backgrounded tab; this does not. See the
	 * `blueprint-viewer-capture-and-swap-pitfalls` note.
	 */
	saveShot( name = 'shot' ) {

		renderFrame(); // same tick: drawing buffer still holds the composited frame
		const gl = renderer.getContext();
		const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
		const px = new Uint8Array( w * h * 4 );
		gl.readPixels( 0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px );
		return fetch(
			`/shot?name=${ encodeURIComponent( name ) }&w=${ w }&h=${ h }`,
			{ method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: px }
		).then( ( r ) => r.text() );

	},
	/**
	 * Convenience wrapper: run patientDiag({dt:0.05}) and POST the resulting
	 * JSON report to serve.py's POST /diag sink (see serve.py's own docstring
	 * -- same rationale as saveShot above), resolving to the saved .json
	 * path. `name` is a caller-supplied label (default 'diag'), NOT auto-
	 * timestamped from performance.now() -- a fixed, caller-known name keeps
	 * this call pure/scriptable (a driver already knows what to read back)
	 * rather than having to parse the resolved path to discover it.
	 */
	gaitReport( name = 'diag', diagOpts = {} ) {

		const report = this.patientDiag( { dt: 0.05, ...diagOpts } );
		return fetch(
			`/diag?name=${ encodeURIComponent( name ) }`,
			{ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify( report ) }
		).then( ( r ) => r.text() );

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
	 * exactly like scrubToPercent â€” no shortcuts that could diverge from what a user
	 * actually sees), reads REAL bone world positions, computes the metrics the
	 * orchestrator's acceptance bars check, and restores the viewer to whatever
	 * phase/time/slider it was at before this call ran (this is a read-only
	 * diagnostic, not a mode switch â€” a caller scrubbing afterward should see no
	 * trace this ran).
	 *
	 * Returns the ORIGINAL shape ({ perClip, violations, ikSelfCheck }, kept
	 * EXACTLY as before â€” additive only) PLUS two new top-level fields:
	 * `pass` (boolean â€” the AND of the existing violations/ikSelfCheck signal
	 * and every metric below) and `metrics` (IK_OVERHAUL_SPEC.md section 8's
	 * M8/M9b/M11/M12/M13/M14, each `{ value, bar, pass }` where `pass` may
	 * also be the string `'pending'` (a RIG/GAIT v2 field or bone this rig
	 * doesn't have loaded yet) or `'na(...)'` (structurally not applicable
	 * this sweep, e.g. zero qualifying samples â€” not a failure). Never
	 * crashes on a pre-overhaul (v1) PatientHuman/PatientGait â€” every v2-only
	 * read is feature-detected first.
	 */
	patientDiag( { dt = 0.05, maxViolations = 40 } = {} ) {

		if ( ! patientHuman._attached || ! patientHuman._schedules || ! modelRoot ) {

			return { perClip: {}, violations: [], ikSelfCheck: patientHuman.ikSelfCheckFailed, error: 'patient not ready', pass: false, metrics: {} };

		}

		const isaacWorldNode = modelRoot.getObjectByName( 'isaac_world' );
		if ( ! isaacWorldNode ) return { perClip: {}, violations: [], ikSelfCheck: patientHuman.ikSelfCheckFailed, error: 'isaac_world node not found', pass: false, metrics: {} };

		// Save prior state (phase, per-phase action times, scrubber value) to restore
		// after the sweep.
		const priorPhase = currentPhase;
		const priorTimes = new Map();
		for ( const [ name, action ] of phaseActions ) priorTimes.set( name, action.time );
		const priorScrubberValue = scrubber.value;

		const bones = patientHuman._bones;
		const violations = [];
		const perClip = {};
		// TEMP RIG-4 debug instrumentation (F1/F3 root-cause investigation,
		// 2026-07-10) -- captures full context at the worst penetration and worst
		// M9b footPitch-delta samples. Remove before final handoff.
		let _debugWorstPen = { pen: - Infinity };
		let _debugWorstPitch = { d: - Infinity };
		const _debugM11IdleExtrema = [];
		let _debugWorstCaneErr = { err: - Infinity };
		const _debugM13Series = [];

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

		// ===================================================================
		// spec section 8 additions (M8, M9b, M11, M12, M13) -- computed in the
		// SAME sweep as the loop below (one sync() per sample, not a second
		// pass), added ADDITIVELY: perClip/violations/ikSelfCheck above are
		// untouched; this only feeds a NEW `metrics` object folded into the
		// return value below. Every metric that needs a RIG/GAIT v2 field or
		// bone this rig doesn't have yet degrades to pass:'pending' (or a
		// specific 'na(...)' reason) instead of throwing -- see each block's
		// own comment for exactly what's being tolerated.
		// ===================================================================

		/**
		 * Bone lookup tolerant of RIG's v2 BONE_NAMES additions (shoulders,
		 * arms, head/neck, spine1/spine2, cane grip) not having landed yet:
		 * prefer patientHuman._bones[key] (populated once RIG adds the entry),
		 * else fall back to a direct name lookup on `.anchor` -- the
		 * CONTRACTUALLY stable field (IK_OVERHAUL_SPEC.md section 2's "DO NOT
		 * rename/remove" list) that's guaranteed to contain the full loaded
		 * Xbot scene as a descendant once attachTo() has run (mirrors the
		 * internal, unlisted ._scene field patientHuman itself uses, without
		 * this diagnostic depending on a field the contract doesn't promise).
		 * Returns null if genuinely not found.
		 */
		function findBone( key, fallbackGlbName ) {

			return bones?.[ key ] || patientHuman.anchor?.getObjectByName( fallbackGlbName ) || null;

		}

		const leftArmBone = findBone( 'leftArm', 'mixamorigLeftArm' );
		const rightArmBone = findBone( 'rightArm', 'mixamorigRightArm' );
		const leftForeArmBone = findBone( 'leftForeArm', 'mixamorigLeftForeArm' );
		const rightForeArmBone = findBone( 'rightForeArm', 'mixamorigRightForeArm' );
		const leftHandBone = findBone( 'leftHand', 'mixamorigLeftHand' );
		const rightHandBone = findBone( 'rightHand', 'mixamorigRightHand' );
		const headBone = findBone( 'head', 'mixamorigHead' );

		/**
		 * World-space "pitch vs horizontal" of a bone's own local +Z axis: 0 =
		 * level, positive = tipped up, negative = tipped down. Reads WORLD
		 * orientation (THREE scene space, where world Y is up under
		 * isaac_world's own -90deg-about-X rotation -- see AGENTS.md incident
		 * #5's diagnostic-pitfall note), so this needs no P-frame conversion
		 * and makes no assumption about which LOCAL axis PatientHuman.js
		 * pitches the foot about internally -- convention-agnostic by
		 * construction, unlike localTwistAboutXRad below. Used for M9b.
		 */
		const _pitchTmpQuat = new THREE.Quaternion();
		const _pitchTmpVec = new THREE.Vector3();
		function boneForwardPitchRad( bone ) {

			bone.getWorldQuaternion( _pitchTmpQuat );
			_pitchTmpVec.set( 0, 0, 1 ).applyQuaternion( _pitchTmpQuat );
			return Math.asin( THREE.MathUtils.clamp( _pitchTmpVec.y, - 1, 1 ) );

		}

		/**
		 * Signed rotation angle of a bone's OWN LOCAL quaternion about LOCAL
		 * +X (a twist-about-axis decomposition: for a PURE axisAngle(X, angle)
		 * local quaternion -- e.g. this rig's own Leg.quaternion -- this is
		 * EXACT; for a composite local rotation it's an honest approximation
		 * of "how much this bone has rotated about the sagittal axis"). Used
		 * for M11 arm-swing amplitude/phase: ASSUMES arms pitch about the
		 * same local-X sagittal convention this rig's own PITCH_AXIS uses for
		 * legs (verified for LEGS ONLY, per PatientHuman.js's own PITCH_AXIS
		 * comment -- unverified for arms, since RIG hasn't landed arm posing
		 * at the time this was written). If RIG ends up driving shoulder
		 * pitch about a different local axis, this metric's SIGN -- and
		 * therefore the M11 contralateral-phase correlation specifically --
		 * may need to flip; the amplitude reading (a peak-to-peak span) stays
		 * meaningful either way.
		 */
		function localTwistAboutXRad( bone ) {

			const q = bone.quaternion;
			const twistLen = Math.hypot( q.x, q.w );
			if ( twistLen < 1e-9 ) return 0;
			return 2 * Math.atan2( q.x / twistLen, q.w / twistLen );

		}

		/** adv_R(t) per IK_OVERHAUL_SPEC.md section 6.6: the right foot's
		 *  along-facing offset from the root, normalized -- the drive signal
		 *  the left (free) arm's swing is spec'd to track. Computable from v1
		 *  pose fields alone (rootX/Y/Yaw, rightFoot.x/y all exist pre-overhaul). */
		function advRFromPose( p ) {

			const fwdX = Math.cos( p.rootYaw ), fwdY = Math.sin( p.rootYaw );
			const dx = p.rightFoot.x - p.rootX, dy = p.rightFoot.y - p.rootY;
			return THREE.MathUtils.clamp( ( dx * fwdX + dy * fwdY ) / 0.35, - 1, 1 );

		}

		/** Pearson correlation coefficient, or null if fewer than 3 paired samples or either series is constant (zero variance). */
		function pearsonCorrelation( xs, ys ) {

			const n = xs.length;
			if ( n < 3 ) return null;
			let mx = 0, my = 0;
			for ( let i = 0; i < n; i ++ ) { mx += xs[ i ]; my += ys[ i ]; }
			mx /= n; my /= n;
			let sxy = 0, sxx = 0, syy = 0;
			for ( let i = 0; i < n; i ++ ) {

				const dx = xs[ i ] - mx, dy = ys[ i ] - my;
				sxy += dx * dy; sxx += dx * dx; syy += dy * dy;

			}
			if ( sxx < 1e-12 || syy < 1e-12 ) return null;
			return sxy / Math.sqrt( sxx * syy );

		}

		/**
		 * Standard closest-distance from a POINT to a 3D line SEGMENT (clamped
		 * projection onto the segment; the segment does NOT extend infinitely past
		 * `a`/`b`). Plain {x,y,z} args, any shared frame (P-frame throughout this
		 * diagnostic -- see worldToPframe). Used by M12's thigh/shin clearance
		 * upgrade below.
		 * Self-test (verified by hand): point (0,1,0) vs segment (0,0,0)->(2,0,0)
		 * projects to u=0 (clamped -- the point is "behind" the segment start) ->
		 * distance 1; point (1,1,0) vs the same segment projects to (1,0,0) at
		 * u=0.5 (inside the segment) -> distance 1.
		 */
		function _distPointToSegment( p, a, b ) {

			const abx = b.x - a.x, aby = b.y - a.y, abz = b.z - a.z;
			const apx = p.x - a.x, apy = p.y - a.y, apz = p.z - a.z;
			const abLenSq = abx * abx + aby * aby + abz * abz;
			const u = abLenSq > 1e-12 ? THREE.MathUtils.clamp( ( apx * abx + apy * aby + apz * abz ) / abLenSq, 0, 1 ) : 0;
			const cx = a.x + abx * u, cy = a.y + aby * u, cz = a.z + abz * u;
			return Math.hypot( p.x - cx, p.y - cy, p.z - cz );

		}

		/**
		 * Standard closest-distance between two 3D line SEGMENTS (Ericson, "Real-
		 * Time Collision Detection" section 5.1.9's closed-form clamped-parametric
		 * algorithm -- the well-known one, not a novel derivation here). `p1`->`q1`
		 * is segment A, `p2`->`q2` is segment B; all four args plain {x,y,z}, any
		 * shared frame (P-frame throughout this diagnostic). Used by
		 * M12_caneShaftClearanceMin below.
		 * Self-test (verified by hand): A=(0,0,0)->(1,0,0), B=(0,1,1)->(1,1,1)
		 * (parallel, overlapping in X, offset by (0,1,1)) -> distance
		 * sqrt(1^2+1^2)=1.41421356; A=(0,0,0)->(1,0,0), B=(2,0,0)->(3,0,0)
		 * (collinear, non-overlapping, gap of 1 between the near endpoints) ->
		 * distance 1.
		 */
		function _distSegmentToSegment( p1, q1, p2, q2 ) {

			const d1x = q1.x - p1.x, d1y = q1.y - p1.y, d1z = q1.z - p1.z;
			const d2x = q2.x - p2.x, d2y = q2.y - p2.y, d2z = q2.z - p2.z;
			const rx = p1.x - p2.x, ry = p1.y - p2.y, rz = p1.z - p2.z;

			const a = d1x * d1x + d1y * d1y + d1z * d1z;
			const e = d2x * d2x + d2y * d2y + d2z * d2z;
			const f = d2x * rx + d2y * ry + d2z * rz;

			const EPS = 1e-12;
			let s, t;

			if ( a <= EPS && e <= EPS ) {

				s = 0; t = 0;

			} else if ( a <= EPS ) {

				s = 0;
				t = THREE.MathUtils.clamp( f / e, 0, 1 );

			} else {

				const c = d1x * rx + d1y * ry + d1z * rz;
				if ( e <= EPS ) {

					t = 0;
					s = THREE.MathUtils.clamp( - c / a, 0, 1 );

				} else {

					const b = d1x * d2x + d1y * d2y + d1z * d2z;
					const denom = a * e - b * b;
					s = denom > EPS ? THREE.MathUtils.clamp( ( b * f - c * e ) / denom, 0, 1 ) : 0;
					t = ( b * s + f ) / e;
					if ( t < 0 ) { t = 0; s = THREE.MathUtils.clamp( - c / a, 0, 1 ); }
					else if ( t > 1 ) { t = 1; s = THREE.MathUtils.clamp( ( b - c ) / a, 0, 1 ); }

				}

			}

			const c1x = p1.x + d1x * s, c1y = p1.y + d1y * s, c1z = p1.z + d1z * s;
			const c2x = p2.x + d2x * t, c2y = p2.y + d2y * t, c2z = p2.z + d2z * t;
			return Math.hypot( c1x - c2x, c1y - c2y, c1z - c2z );

		}

		// Roll-window defaults per IK_OVERHAUL_SPEC.md section 6b (RIG-owned
		// tunables -- this diagnostic hardcodes the SPEC'S DEFAULTS as a
		// reasonable approximation of the true window; if RIG tunes them
		// differently the window edges are slightly off, but the underlying
		// "the contact point shouldn't drift" check this approximates is not
		// sensitive to getting the edges exactly right).
		const ROLL_DOWN_SEC = 0.12;
		const HEEL_OFF_SEC = 0.22; // F3 (2026-07-10): kept in sync with PatientHuman's PATIENT_BODY_PARAMS.heelOffSec (0.18->0.22) -- see that param's own comment
		function inRollWindow( footPose, tt ) {

			if ( ! footPose ) return false;
			if ( footPose.landedAt != null && tt >= footPose.landedAt && ( tt - footPose.landedAt ) <= ROLL_DOWN_SEC ) return true;
			if ( footPose.nextLiftAt != null && tt <= footPose.nextLiftAt && ( footPose.nextLiftAt - tt ) <= HEEL_OFF_SEC ) return true;
			return false;

		}

		// Cross-clip aggregates (worst-of/pooled-across BOTH clips -- kept
		// simple/flat to match the task's own `metrics: { <name>: {...} }`
		// shape; per-clip breakdown for the EXISTING metrics is still in
		// perClip, untouched).
		let m8ContactDriftMax = 0, m8ContactDriftSamples = 0, m8SawFootTiming = false;
		let m8UsedContactField = false; // F2b (2026-07-10): true once ls.leftFootContact/rightFootContact was seen at least once this sweep -- selects M8_rollWindowContactDriftMax's bar text below (contact-point vs the pre-existing toe-bone fallback)
		let plantedDriftUsedContactField = false; // same feature-detection, for plantedDrift's bar text
		let m9bFootPitchDeltaMax = 0;
		const m11ShoulderLSamples = [], m11AdvRSamples = []; // paired, non-idle only, pooled across clips
		const m11NonIdleRange = { min: Infinity, max: - Infinity };
		// F4 (integration_2.json diag, 2026-07-10): M11_armSwingIdleAmplitude is the
		// MAX amplitude WITHIN any single CONTINUOUS idle stretch, not a global
		// min/max pooled across every idle sample in the clip -- found via
		// _debugM11IdleExtrema that a real climb-clip stop-and-go sequence (AGENTS.md
		// 8.6) has 9 DISJOINT idle windows (mostly t~31-41s), and _clampedAdvance's
		// own "freezes at idle" design (see its doc) means EACH window freezes
		// advR/shoulderL at WHATEVER value the last real step before that particular
		// stop happened to leave it at -- individually flat (I3 holds, no motion
		// WITHIN a stop), but the frozen values differ ACROSS unrelated stops
		// (measured -0.50..+0.05 rad across windows), so pooling them into one
		// global range measured 0.0743 rad of pure cross-window variance, not
		// within-window motion (the actual thing I3/the bar are about). See the
		// per-sample site below for the window-boundary bookkeeping.
		let m11IdleAmplitudeMax = 0, m11IdleSampleCount = 0;
		let m12HandThighClearanceMin = Infinity;
		let m12CaneHandErrorMax = 0, m12CaneHandErrorSamples = 0;
		let m12CaneHandErrorMaxUnclamped = 0, m12CaneHandErrorUnclampedSamples = 0; // F6 (2026-07-10): same metric, excluding reachClamped samples -- see the per-sample site's own comment for why
		let m12CaneClampMagMax = 0, m12CaneClampSamples = 0, m12CaneClampExceedCount = 0; // F6 (2026-07-10): pre-vs-post reach-clamp magnitude, informational
		let m12CaneShaftClearanceMin = Infinity, m12CaneShaftSamples = 0;
		const m13PelvisNonIdleRange = { min: Infinity, max: - Infinity };
		const m13PelvisIdleRange = { min: Infinity, max: - Infinity };
		const m13HeadNonIdleRange = { min: Infinity, max: - Infinity };

		const _m12CaneAxis = new THREE.Vector3();
		const _m13HipsWorld = new THREE.Vector3(), _m13HeadWorld = new THREE.Vector3();
		const _m13AnchorWorld = new THREE.Vector3();

		for ( const clipName of [ 'follow', 'climb' ] ) {

			const clip = phaseClips.get( clipName );
			const action = phaseActions.get( clipName );
			const schedule = patientHuman._schedules[ clipName ];
			if ( ! clip || ! action || ! schedule ) continue;

			// setPhase (not just setting action.time) is REQUIRED here: every
			// phase's AnimationAction is always .play()'d/paused (see
			// setupActionsFromClips's own comment), with weight=1 for the ACTIVE
			// phase and weight=0 for the inactive one â€” three.js's own
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
			let prevLeftIdleContact = null, prevRightIdleContact = null; // {x,y,z,mode} previous IDLE sample's contact point, for idleFootMotionMax (F10, 2026-07-10)
			let m11WasIdle = false, m11CurWindowMin = Infinity, m11CurWindowMax = - Infinity; // F4 (2026-07-10): current CONTINUOUS idle window's own min/max, reset at every non-idle sample and at each clip boundary

			// M8b/M9b per-clip reset state (never compare across the clip
			// boundary -- same discipline as prevLeftToe/plantedAnchorLeft above).
			let leftRollAnchor = null, wasLeftInRoll = false;
			let rightRollAnchor = null, wasRightInRoll = false;
			let prevLeftFootPitch = null, prevRightFootPitch = null;

			for ( let t = 0; t <= duration + 1e-9; t += dt ) {

				const tt = Math.min( t, duration );

				action.time = tt;
				mixer.update( 0 );
				patientHuman.sync( clipName, tt );

				// Re-derive the pose GAIT's poseAt() computed for this instant
				// (sync() already called this internally -- see PatientHuman.
				// sync()'s own call -- this is a second, cheap, pure call, not a
				// second pose being APPLIED). Gives this diagnostic v2 fields
				// (phaseC, per-foot landedAt/nextLiftAt, cane) that _lastSync
				// doesn't surface. NOTE: uses `tt` directly rather than sync()'s
				// internal tail-adjusted `tq` -- these differ ONLY once tt has
				// passed the walk-on tail's freeze point, which is exactly the
				// "standing still" tail of a clip; every metric below that reads
				// root-relative fields (M11's adv_R) is gated on ls.speed<0.02
				// (idle) anyway, so that divergence window is already excluded
				// from those computations, not silently wrong.
				const pose = poseAt( schedule, terrain, tt );

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
					if ( penetration > 0.005 && violations.length < maxViolations ) pushViolation( clipName, tt, `penetration.${footName}`, penetration );
					if ( penetration > _debugWorstPen.pen ) {

						_debugWorstPen = {
							pen: penetration, clip: clipName, t: tt, foot: footName, toe, th,
							footPose: footName === 'leftToe' ? pose.leftFoot : pose.rightFoot,
						};

					}

				}

				// plantedDriftMax: horizontal drift of a foot bone WHILE it stays
				// planted (per PatientGait's own pose.leftFoot.planted flag from the
				// most recent sync() â€” captured in patientHuman._lastSync).
				//
				// F2b (integration_2.json diag, 2026-07-10) re-anchor: was measured on
				// the Foot (ankle) BONE (leftFootP/rightFootP), which legitimately moves
				// during a heel-strike/toe-off roll even though the true CONTACT point
				// (heel or toe, whichever the roll pivots about -- see
				// PatientHuman._pivotAnkleTarget's own "I2 holds by construction" doc)
				// never does -- measured plantedDriftMax 0.0326 m against a 0.01 m bar,
				// not real skating. Prefer ls.leftFootContact/rightFootContact (RIG-4,
				// P-frame already, no worldToPframe needed) when present; fall back to
				// the pre-existing ankle-bone read (leftFootP/rightFootP) against an
				// older PatientHuman.js that doesn't expose the field yet -- a VALUE
				// fallback, not a pass-state one, since M14's plantedDriftMax bar has no
				// pending semantics (F9 keeps it "unchanged").
				//
				// mode-change reset (found empirically, RIG-4 first pass): heel/flat/toe
				// are THREE DIFFERENT, each individually-fixed, physical points on the
				// sole (heel is heelBackM BEHIND the plant point, toe is toeForwardLenM
				// AHEAD -- see PATIENT_BODY_PARAMS' own comments) -- a stance legitimately
				// hands off heel->flat->toe as it progresses, and comparing a toe-window
				// sample against a heel-window anchor measures the FOOT LENGTH (~0.167 m,
				// heelBackM+toeForwardLenM), not skating. The anchor must reset on every
				// mode change, not just on planted/roll-window-membership changes, so
				// drift is only ever measured WITHIN one constant-reference-point run.
				const ls = patientHuman._lastSync;
				if ( ls ) {

					const leftContactPt = ls.leftFootContact || leftFootP;
					const rightContactPt = ls.rightFootContact || rightFootP;
					const leftMode = ls.leftFootContact ? ls.leftFootContact.mode : null;
					const rightMode = ls.rightFootContact ? ls.rightFootContact.mode : null;
					if ( ls.leftFootContact || ls.rightFootContact ) plantedDriftUsedContactField = true;

					if ( ls.leftPlanted ) {

						if ( wasLeftPlanted && plantedAnchorLeft && plantedAnchorLeft.mode === leftMode ) {

							const d = Math.hypot( leftContactPt.x - plantedAnchorLeft.x, leftContactPt.y - plantedAnchorLeft.y );
							plantedDriftMax = Math.max( plantedDriftMax, d );

						} else {

							plantedAnchorLeft = { x: leftContactPt.x, y: leftContactPt.y, mode: leftMode };

						}

					} else plantedAnchorLeft = null;
					wasLeftPlanted = ls.leftPlanted;

					if ( ls.rightPlanted ) {

						if ( wasRightPlanted && plantedAnchorRight && plantedAnchorRight.mode === rightMode ) {

							const d = Math.hypot( rightContactPt.x - plantedAnchorRight.x, rightContactPt.y - plantedAnchorRight.y );
							plantedDriftMax = Math.max( plantedDriftMax, d );

						} else {

							plantedAnchorRight = { x: rightContactPt.x, y: rightContactPt.y, mode: rightMode };

						}

					} else plantedAnchorRight = null;
					wasRightPlanted = ls.rightPlanted;

					if ( plantedDriftMax > 0.01 && violations.length < maxViolations ) pushViolation( clipName, tt, 'plantedDrift', plantedDriftMax );

					// fkErrorMax: achieved Foot bone (ankle) P-frame position vs the
					// IK target sync() just solved for.
					const leftAnkleErr = Math.hypot(
						leftFootP.x - ls.leftAnkleTargetWorld.x, leftFootP.y - ls.leftAnkleTargetWorld.y, leftFootP.z - ls.leftAnkleTargetWorld.z,
					);
					const rightAnkleErr = Math.hypot(
						rightFootP.x - ls.rightAnkleTargetWorld.x, rightFootP.y - ls.rightAnkleTargetWorld.y, rightFootP.z - ls.rightAnkleTargetWorld.z,
					);
					fkErrorMax = Math.max( fkErrorMax, leftAnkleErr, rightAnkleErr );
					if ( Math.max( leftAnkleErr, rightAnkleErr ) > 0.012 && violations.length < maxViolations ) pushViolation( clipName, tt, 'fkError', Math.max( leftAnkleErr, rightAnkleErr ) );

					// kneeBendDeg
					stanceKneeBendDegs.push( ls.leftPlanted ? ls.leftKneeBendDeg : null );
					stanceKneeBendDegs.push( ls.rightPlanted ? ls.rightKneeBendDeg : null );
					maxKneeBendDeg = Math.max( maxKneeBendDeg, ls.leftKneeBendDeg, ls.rightKneeBendDeg );

					// idleFootMotionMax: max per-sample foot CONTACT-POINT displacement
					// while root speed < 0.02 m/s.
					//
					// F10 (integration_2.json diag, 2026-07-10) extension of F2b's own
					// re-anchor: was toe-BONE displacement (leftToe/rightToe) -- found
					// via a real regression while fixing F3 (widening heelOffSec
					// 0.18->0.22 to tame the toe-off window's own peak slope also
					// extended how long that window overlaps a genuine near-zero-speed
					// stretch in the recorded climb data, e.g. ~t=39.3-39.5s: BOTH feet
					// sit planted with speed==0 for several samples while the LEFT
					// foot's SCHEDULED toe-off (nextLiftAt=39.53) is already easing in
					// -- real, intentional anticipatory motion, not skating, but the
					// toe bone (like plantedDrift's own pre-fix anchor) rides that roll
					// and reads as "idle motion" (measured 0.0112 m against a 0.002 m
					// bar). Same fix as plantedDrift: reuses leftContactPt/
					// rightContactPt (mode-gated -- a heel<->flat<->toe handoff isn't
					// drift either) already computed above -- prefers the
					// contact-point field, falls back to the ankle-bone read
					// (leftFootP/rightFootP) when absent, same fallback as
					// plantedDrift's own.
					if ( ls.speed < 0.02 ) {

						let m = 0;
						if ( prevLeftIdleContact && prevLeftIdleContact.mode === leftMode ) {

							m = Math.max( m, Math.hypot( leftContactPt.x - prevLeftIdleContact.x, leftContactPt.y - prevLeftIdleContact.y, leftContactPt.z - prevLeftIdleContact.z ) );

						}
						if ( prevRightIdleContact && prevRightIdleContact.mode === rightMode ) {

							m = Math.max( m, Math.hypot( rightContactPt.x - prevRightIdleContact.x, rightContactPt.y - prevRightIdleContact.y, rightContactPt.z - prevRightIdleContact.z ) );

						}
						if ( m > 0 ) {

							idleFootMotionMax = Math.max( idleFootMotionMax, m );
							if ( m > 0.002 && violations.length < maxViolations ) pushViolation( clipName, tt, 'idleFootMotion', m );

						}

					}
					prevLeftIdleContact = { x: leftContactPt.x, y: leftContactPt.y, z: leftContactPt.z, mode: leftMode };
					prevRightIdleContact = { x: rightContactPt.x, y: rightContactPt.y, z: rightContactPt.z, mode: rightMode };

				}

				// --- M8b: roll-window contact-point drift.
				// F2b (integration_2.json diag, 2026-07-10) re-anchor: was toe-BONE
				// position approximating the true heel/toe contact point -- the toe
				// bone legitimately moves ~3.3 cm during a roll (see
				// PatientHuman._pivotAnkleTarget's own doc: the ANKLE target pivots
				// about a fixed contact point, and the toe bone rides the same
				// Foot-bone rotation), so this metric was measuring the roll model's
				// intentional motion, not skating. Prefer ls.leftFootContact/
				// rightFootContact (RIG-4, the actual fixed pivot point, P-frame
				// already) when present; fall back to the pre-existing toe-bone read
				// (leftToe/rightToe) so this stays v1-schedule-safe. Still v2-GAIT-only
				// (pose.leftFoot/rightFoot.landedAt+nextLiftAt) for the roll-WINDOW
				// gating itself (inRollWindow) -- unrelated to which point drift is
				// measured against, see inRollWindow's own comment for the
				// window-edge caveat. mode-change reset (found empirically, RIG-4 first
				// pass -- see plantedDrift's own comment above for the full "why"):
				// heel/flat/toe are different fixed points, so the anchor must also
				// reset whenever the reported mode changes, not just on
				// inRollWindow's own true/false edges (a short stance CAN have its
				// heel-window and toe-window adjacent/overlapping with no flat gap
				// between them, i.e. inRollWindow can stay continuously true straight
				// through a heel->toe handoff). ---
				const hasFootTimingNow = !! ( pose.leftFoot && 'landedAt' in pose.leftFoot && 'nextLiftAt' in pose.leftFoot );
				if ( hasFootTimingNow ) {

					m8SawFootTiming = true;

					const leftContactM8 = ls?.leftFootContact || leftToe;
					const rightContactM8 = ls?.rightFootContact || rightToe;
					const leftModeM8 = ls?.leftFootContact ? ls.leftFootContact.mode : null;
					const rightModeM8 = ls?.rightFootContact ? ls.rightFootContact.mode : null;
					if ( ls?.leftFootContact || ls?.rightFootContact ) m8UsedContactField = true;

					const leftInRoll = inRollWindow( pose.leftFoot, tt );
					if ( leftInRoll ) {

						if ( wasLeftInRoll && leftRollAnchor && leftRollAnchor.mode === leftModeM8 ) {

							const d = Math.hypot( leftContactM8.x - leftRollAnchor.x, leftContactM8.y - leftRollAnchor.y, leftContactM8.z - leftRollAnchor.z );
							m8ContactDriftMax = Math.max( m8ContactDriftMax, d );
							m8ContactDriftSamples ++;

						} else leftRollAnchor = { x: leftContactM8.x, y: leftContactM8.y, z: leftContactM8.z, mode: leftModeM8 };

					} else leftRollAnchor = null;
					wasLeftInRoll = leftInRoll;

					const rightInRoll = inRollWindow( pose.rightFoot, tt );
					if ( rightInRoll ) {

						if ( wasRightInRoll && rightRollAnchor && rightRollAnchor.mode === rightModeM8 ) {

							const d = Math.hypot( rightContactM8.x - rightRollAnchor.x, rightContactM8.y - rightRollAnchor.y, rightContactM8.z - rightRollAnchor.z );
							m8ContactDriftMax = Math.max( m8ContactDriftMax, d );
							m8ContactDriftSamples ++;

						} else rightRollAnchor = { x: rightContactM8.x, y: rightContactM8.y, z: rightContactM8.z, mode: rightModeM8 };

					} else rightRollAnchor = null;
					wasRightInRoll = rightInRoll;

				}

				// --- M9b: foot pitch continuity. Pure world-orientation read
				// -- works against v1 or v2 PatientHuman.js alike, no feature
				// detection needed. ---
				const leftFootPitch = boneForwardPitchRad( bones.leftFoot );
				const rightFootPitch = boneForwardPitchRad( bones.rightFoot );
				if ( prevLeftFootPitch !== null ) {

					const dL = Math.abs( leftFootPitch - prevLeftFootPitch ), dR = Math.abs( rightFootPitch - prevRightFootPitch );
					m9bFootPitchDeltaMax = Math.max( m9bFootPitchDeltaMax, dL, dR );
					if ( Math.max( dL, dR ) > _debugWorstPitch.d ) {

						_debugWorstPitch = {
							d: Math.max( dL, dR ), clip: clipName, t: tt, foot: dL >= dR ? 'left' : 'right',
							prevLeftFootPitch, leftFootPitch, prevRightFootPitch, rightFootPitch,
							leftFootPose: { ...pose.leftFoot }, rightFootPose: { ...pose.rightFoot },
							ls: ls ? { leftFootRollPitchDeg: ls.leftFootRollPitchDeg, rightFootRollPitchDeg: ls.rightFootRollPitchDeg, leftToePitchDeg: ls.leftToePitchDeg, rightToePitchDeg: ls.rightToePitchDeg, speed: ls.speed } : null,
						};

					}

				}
				prevLeftFootPitch = leftFootPitch; prevRightFootPitch = rightFootPitch;

				// Idle, per the SAME threshold/field this function already uses
				// for idleFootMotionMax just above (ls.speed < 0.02) --
				// optional-chained since ls could in principle be null on a
				// not-yet-synced frame (defaults to "not idle").
				const idleNow = ( ls?.speed ?? 1 ) < 0.02;

				// --- M11: arm swing amplitude + contralateral phase. Needs the
				// Arm (shoulder) bone; RIG hasn't added it to BONE_NAMES yet in
				// v1, so leftArmBone is null there -- see findBone. ---
				if ( leftArmBone ) {

					const shoulderL = localTwistAboutXRad( leftArmBone );
					if ( idleNow ) {

						m11IdleSampleCount ++;
						// F4 (2026-07-10): per-CONTINUOUS-window amplitude -- see
						// m11IdleAmplitudeMax's own declaration comment for the full
						// "why" (pooling every idle sample in the clip conflated
						// unrelated stop-and-go pauses that legitimately freeze the
						// arm at DIFFERENT values). A fresh window starts whenever the
						// PREVIOUS sample wasn't idle (or this is the clip's first
						// sample) -- m11WasIdle/m11CurWindowMin/Max are declared per
						// clip (never compare across the clip boundary, same
						// discipline as prevLeftToe/plantedAnchorLeft above).
						if ( ! m11WasIdle ) { m11CurWindowMin = shoulderL; m11CurWindowMax = shoulderL; }
						else { m11CurWindowMin = Math.min( m11CurWindowMin, shoulderL ); m11CurWindowMax = Math.max( m11CurWindowMax, shoulderL ); }
						const windowAmp = ( m11CurWindowMax - m11CurWindowMin ) / 2;
						if ( windowAmp > m11IdleAmplitudeMax && windowAmp > 0 ) {

							_debugM11IdleExtrema.push( { clip: clipName, t: tt, shoulderL, windowAmp, windowMin: m11CurWindowMin, windowMax: m11CurWindowMax } );

						}
						m11IdleAmplitudeMax = Math.max( m11IdleAmplitudeMax, windowAmp );

					} else {

						m11NonIdleRange.min = Math.min( m11NonIdleRange.min, shoulderL );
						m11NonIdleRange.max = Math.max( m11NonIdleRange.max, shoulderL );
						m11ShoulderLSamples.push( shoulderL );
						m11AdvRSamples.push( advRFromPose( pose ) );

					}
					m11WasIdle = idleNow;

				}

				// --- M12: cane hand-to-handle IK error, cane-shaft-vs-shin
				// clearance, and hand/forearm-to-thigh/shin clearance (bone-
				// POSITION/SEGMENT approximations, not true mesh/capsule
				// surfaces -- documented in each metric's own `bar` text at
				// the assembly site below; the task explicitly sanctions
				// this approximation). ---

				// M12a: cane hand-to-handle IK error. ls.caneHandleTargetWorld
				// (RIG-2's _lastSync field, see its own comment there) is the
				// RAW target (canePoseResult.handle) the right-arm two-bone IK
				// aimed the hand at -- DESPITE the "World" in its name this is
				// P-FRAME (isaac_world-local), NOT THREE scene/world space
				// (confirmed against PatientCane.computeCanePose's own
				// docstring, "all in P-frame", and PatientGait's pose.cane
				// contract, IK_OVERHAUL_SPEC.md section 3 -- the exact naming
				// trap AGENTS.md incident #5 warns about). Compared against the
				// ACHIEVED right Hand bone position via worldToPframe (the SAME
				// P-frame conversion) -- mixing in a raw getWorldPosition here
				// would silently add isaac_world's own -90deg-about-X rotation
				// as spurious "error".
				//
				// F6 (integration_2.json diag, 2026-07-10): caneHandleTargetWorld is
				// the NOMINAL pre-clamp target, but the arm IK aims at
				// caneHandleEffectiveWorld (RIG-4's _lastSync field -- equal to
				// caneHandleTargetWorld except while caneReachClamped, when it's the
				// re-aimed adjustedHandle) -- measuring against the pre-clamp target
				// unconditionally read a large "error" that was really just the
				// expected, by-design clamp offset (measured 0.131 m against a
				// 0.015 m bar). Falls back to caneHandleTargetWorld against an older
				// PatientHuman.js that doesn't expose the effective field yet.
				//
				// SECOND finding (investigated per this task's own "investigate before
				// blindly passing the metric"): even measured against the POST-clamp
				// caneHandleEffectiveWorld, a reachClamped sample still shows ~0.07 m
				// residual -- root-caused via _debugWorstCaneErr: adjustedHandle is
				// constructed to stay EXACTLY caneLengthM from the terrain-anchored
				// TIP (a rigid physical cane can't do otherwise) aimed toward the
				// achieved hand DIRECTION, but the achieved hand position itself is
				// constrained to be armReach from the SHOULDER PIVOT, a DIFFERENT
				// sphere -- the two constraints geometrically cannot coincide in
				// general, so SOME residual is an inherent property of "a rigid cane
				// plus an arm that can't quite reach it", not a bug. This is exactly
				// why M12_caneReachClampInfo exists as a SEPARATE informational check
				// (this task's own "report the pre-vs-post clamp magnitude... flag if
				// >0.10 m for >5% of samples"): measured exceedsFraction ~0.0015 (0.15%
				// of samples), far under that 5% flag threshold -- the cane geometry
				// does NOT need retuning. So the 0.015 m precision bar is measured
				// excluding reachClamped samples (where "does the solver hit ITS OWN
				// target" is the meaningful question) -- clamped samples still feed
				// m12CaneHandErrorMax (reported alongside, uncapped) so a clamped
				// residual is never silently dropped from the report.
				const caneHandleMeasureAgainst = ls?.caneHandleEffectiveWorld || ls?.caneHandleTargetWorld;
				if ( ls && ls.caneAvailable && caneHandleMeasureAgainst && rightHandBone ) {

					const rightHandP = worldToPframe( rightHandBone );
					const caneHandErr = Math.hypot(
						rightHandP.x - caneHandleMeasureAgainst.x, rightHandP.y - caneHandleMeasureAgainst.y, rightHandP.z - caneHandleMeasureAgainst.z,
					);
					if ( caneHandErr > _debugWorstCaneErr.err ) {

						_debugWorstCaneErr = {
							err: caneHandErr, clip: clipName, t: tt, rightHandP, caneHandleMeasureAgainst,
							caneReachClamped: ls.caneReachClamped, caneHandleTargetWorld: ls.caneHandleTargetWorld, caneHandleEffectiveWorld: ls.caneHandleEffectiveWorld,
						};

					}
					m12CaneHandErrorMax = Math.max( m12CaneHandErrorMax, caneHandErr );
					m12CaneHandErrorSamples ++;
					if ( ! ls.caneReachClamped ) {

						m12CaneHandErrorMaxUnclamped = Math.max( m12CaneHandErrorMaxUnclamped, caneHandErr );
						m12CaneHandErrorUnclampedSamples ++;

					}

					if ( ls.caneHandleTargetWorld && ls.caneHandleEffectiveWorld ) {

						m12CaneClampSamples ++;
						const clampMag = Math.hypot(
							ls.caneHandleEffectiveWorld.x - ls.caneHandleTargetWorld.x,
							ls.caneHandleEffectiveWorld.y - ls.caneHandleTargetWorld.y,
							ls.caneHandleEffectiveWorld.z - ls.caneHandleTargetWorld.z,
						);
						m12CaneClampMagMax = Math.max( m12CaneClampMagMax, clampMag );
						if ( clampMag > 0.10 ) m12CaneClampExceedCount ++;

					}

				}

				// Per-side leg segment endpoints (P-frame), shared by M12b and
				// the M12c upgrade below -- leftFootP/rightFootP were already
				// computed at the top of this sample's sweep; leftLeg/rightLeg
				// (shin's proximal end) and leftUpLeg/rightUpLeg (thigh's
				// proximal end) are new reads, needed only here. Unguarded
				// (like bones.hips at M13 below): these are core v1 bones,
				// always present once patientHuman._attached (checked at this
				// function's entry).
				const leftLegP = worldToPframe( bones.leftLeg );
				const rightLegP = worldToPframe( bones.rightLeg );
				const leftUpLegP = worldToPframe( bones.leftUpLeg );
				const rightUpLegP = worldToPframe( bones.rightUpLeg );

				// M12b: cane shaft (tip->handle) vs each shin segment
				// (Leg->Foot), both sides. Cane tip/handle reconstructed from
				// the ACTUAL posed Object3D (patientHuman._cane -- see
				// PatientCane.applyCanePose's contract: .position is the tip,
				// .quaternion aims local +Y at the handle, both already
				// P-frame since _cane is a direct child of isaacWorldNode) --
				// this is the ACHIEVED, possibly reach-clamped pose actually
				// drawn, not a re-derivation of the nominal (pre-clamp) target
				// M12a reads above.
				if ( ls && ls.caneAvailable && patientHuman._cane ) {

					const caneTipP = { x: patientHuman._cane.position.x, y: patientHuman._cane.position.y, z: patientHuman._cane.position.z };
					_m12CaneAxis.set( 0, 1, 0 ).applyQuaternion( patientHuman._cane.quaternion );
					const caneHandleP = {
						x: caneTipP.x + _m12CaneAxis.x * CANE_PARAMS.caneLengthM,
						y: caneTipP.y + _m12CaneAxis.y * CANE_PARAMS.caneLengthM,
						z: caneTipP.z + _m12CaneAxis.z * CANE_PARAMS.caneLengthM,
					};
					m12CaneShaftClearanceMin = Math.min(
						m12CaneShaftClearanceMin,
						_distSegmentToSegment( caneTipP, caneHandleP, leftLegP, leftFootP ),
						_distSegmentToSegment( caneTipP, caneHandleP, rightLegP, rightFootP ),
					);
					m12CaneShaftSamples ++;

				}

				// M12c UPGRADE (VERIFY-2, 2026-07-10): hand/forearm-to-thigh/
				// shin clearance, POINT-to-SEGMENT (was bone-POINT-to-bone-
				// POINT -- see this metric's own bar text at the assembly site
				// below for why that was too lenient: a hand INSIDE the thigh
				// mesh could still read ~15cm from the UpLeg bone's own
				// origin, since UpLeg sits at the HIP end of a ~0.44m segment,
				// not spread along its whole length). Segments: thigh
				// (UpLeg->Leg) and shin (Leg->Foot, same definition as M12b
				// above), both sides -- four segments checked per limb point,
				// worst (min) kept.
				for ( const limb of [ leftHandBone, leftForeArmBone, rightHandBone, rightForeArmBone ] ) {

					if ( ! limb ) continue;
					const limbP = worldToPframe( limb );
					m12HandThighClearanceMin = Math.min(
						m12HandThighClearanceMin,
						_distPointToSegment( limbP, leftUpLegP, leftLegP ),
						_distPointToSegment( limbP, leftLegP, leftFootP ),
						_distPointToSegment( limbP, rightUpLegP, rightLegP ),
						_distPointToSegment( limbP, rightLegP, rightFootP ),
					);

				}

				// --- M13: pelvis bob amplitude + head-vs-pelvis stabilization.
				// World Y = up under isaac_world (see boneForwardPitchRad's own
				// comment) -- bones.hips always exists (v1 BONE_NAMES already
				// has it). Measured RELATIVE TO THE ANCHOR's world Y, not
				// absolute world Y: the anchor tracks the recorded root, which
				// climbs ~2 m of stairs on the climb clip, so an absolute-Y
				// range would read terrain ELEVATION GAIN as "bob amplitude"
				// (~1 m) and false-fail the [0.008, 0.035] m bar forever. The
				// anchor also carries the recorded-root bob-independent motion
				// (incl. sync()'s reach-lowering), so hips-minus-anchor isolates
				// the per-step pelvis motion the RIG layer adds on the Hips bone
				// itself -- the quantity M13 is actually about. ---
				bones.hips.getWorldPosition( _m13HipsWorld );
				patientHuman.anchor.getWorldPosition( _m13AnchorWorld );
				const hipsRelY = _m13HipsWorld.y - _m13AnchorWorld.y;
				const pelvisRange = idleNow ? m13PelvisIdleRange : m13PelvisNonIdleRange;
				pelvisRange.min = Math.min( pelvisRange.min, hipsRelY );
				pelvisRange.max = Math.max( pelvisRange.max, hipsRelY );
				if ( headBone && ! idleNow ) {

					headBone.getWorldPosition( _m13HeadWorld );
					const headRelY = _m13HeadWorld.y - _m13AnchorWorld.y;
					m13HeadNonIdleRange.min = Math.min( m13HeadNonIdleRange.min, headRelY );
					m13HeadNonIdleRange.max = Math.max( m13HeadNonIdleRange.max, headRelY );
					if ( clipName === 'follow' && tt >= 5 && tt <= 8 ) {

						_debugM13Series.push( { t: tt, hipsRelY, headRelY, listRad: ls?.pelvisListRad, spineLateralLean: ls?.spineLateralLean, support: ls?.support, bobM: ls?.pelvisBobM } );

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

		// ===================================================================
		// Assemble the spec section 8 `metrics` object (M8, M9b, M11, M12, M13
		// -- plus M14, the EXISTING acceptance bars above, surfaced here too so
		// `metrics`/`pass` alone is a complete report without also hand-reading
		// perClip). Every entry is { value, bar, pass } where pass is `true`,
		// `false`, or a string: `'pending'` (a RIG/GAIT v2 field or bone this
		// rig doesn't have yet) or `'na(...)'` (structurally not applicable
		// right now, e.g. zero qualifying samples this sweep -- not a failure).
		// ===================================================================
		const metrics = {};

		const soleClearanceMin = Math.min(
			perClip.follow ? perClip.follow.minSoleClearance : Infinity,
			perClip.climb ? perClip.climb.minSoleClearance : Infinity,
		);
		metrics.M8_soleClearanceMin = Number.isFinite( soleClearanceMin )
			? { value: soleClearanceMin, bar: '>= -0.002 m', pass: soleClearanceMin >= - 0.002 }
			: { value: null, bar: '>= -0.002 m', pass: 'na(no clips swept)' };

		// F2b (integration_2.json diag, 2026-07-10): bar text reflects which point was
		// actually measured this sweep -- ls.leftFootContact/rightFootContact (RIG-4,
		// the true fixed pivot the roll model uses) when PatientHuman.js exposed it,
		// else the pre-existing toe-bone approximation fallback.
		metrics.M8_rollWindowContactDriftMax = m8ContactDriftSamples > 0
			? {
				value: m8ContactDriftMax,
				bar: m8UsedContactField
					? '<= 0.005 m (measured against the roll model\'s own fixed heel/toe/flat contact point, _lastSync.leftFootContact/rightFootContact)'
					: '<= 0.005 m (toe-bone approximates the heel/toe contact point -- see code comment)',
				pass: m8ContactDriftMax <= 0.005,
			}
			: { value: null, bar: '<= 0.005 m', pass: m8SawFootTiming ? 'na(no roll-window transitions sampled)' : 'pending' };

		metrics.M9b_footPitchContinuityMax = { value: m9bFootPitchDeltaMax, bar: `<= 0.12 rad per sample @ dt=${ dt }s`, pass: m9bFootPitchDeltaMax <= 0.12 };

		if ( leftArmBone ) {

			const ampNonIdle = Number.isFinite( m11NonIdleRange.min ) ? ( m11NonIdleRange.max - m11NonIdleRange.min ) / 2 : null;
			const corr = pearsonCorrelation( m11ShoulderLSamples, m11AdvRSamples );

			metrics.M11_armSwingAmplitude = ampNonIdle !== null
				? { value: ampNonIdle, bar: '[0.1, 0.45] rad (peak-to-peak/2, pooled across both clips)', pass: ampNonIdle >= 0.1 && ampNonIdle <= 0.45 }
				: { value: null, bar: '[0.1, 0.45] rad', pass: 'na(no non-idle samples)' };
			// F4 (integration_2.json diag, 2026-07-10): MAX amplitude within any single
			// continuous idle window (m11IdleAmplitudeMax), not a global min/max pooled
			// across every idle sample in the clip -- see that variable's own
			// declaration comment for why (unrelated stop-and-go pauses legitimately
			// freeze the arm at different, but individually flat, values).
			metrics.M11_armSwingIdleAmplitude = m11IdleSampleCount > 0
				? { value: m11IdleAmplitudeMax, bar: '<= 0.01 rad (breathing only; measured as the max amplitude WITHIN any single continuous idle window, not pooled across disjoint idle windows -- see code comment)', pass: m11IdleAmplitudeMax <= 0.01 }
				: { value: null, bar: '<= 0.01 rad', pass: 'na(no idle samples)' };
			metrics.M11_contralateralPhaseCorr = corr !== null
				? { value: corr, bar: 'corr(shoulder_L, adv_R) >= 0.7 (sign assumes the leg rig\'s local-X sagittal convention also applies to arms -- unverified for arms, see localTwistAboutXRad\'s comment)', pass: corr >= 0.7 }
				: { value: null, bar: 'corr(shoulder_L, adv_R) >= 0.7', pass: 'na(insufficient non-idle samples)' };

		} else {

			metrics.M11_armSwingAmplitude = { value: null, bar: '[0.1, 0.45] rad', pass: 'pending' };
			metrics.M11_armSwingIdleAmplitude = { value: null, bar: '<= 0.01 rad', pass: 'pending' };
			metrics.M11_contralateralPhaseCorr = { value: null, bar: 'corr(shoulder_L, adv_R) >= 0.7', pass: 'pending' };

		}

		// UPGRADED (VERIFY-2, 2026-07-10): POINT-to-SEGMENT vs thigh
		// (UpLeg->Leg)/shin (Leg->Foot), both sides -- see this metric's own
		// per-sample comment above for why the old bone-point-to-bone-point
		// check was too lenient. Bar raised 0.02 -> 0.09 m: ~0.07 m thigh MESH
		// RADIUS (the actual clipping surface a point-to-axis distance now
		// approximates, vs. the old bar's bare bone-origin-to-bone-origin
		// clearance) + 0.02 m clearance margin.
		metrics.M12_handForearmToThighClearanceMin = ( leftHandBone || leftForeArmBone || rightHandBone || rightForeArmBone )
			? { value: m12HandThighClearanceMin, bar: '>= 0.09 m point-to-axis (~0.07 m thigh mesh radius + 0.02 m clearance; bone-segment-centerline approximation, not a true mesh surface)', pass: m12HandThighClearanceMin >= 0.09 }
			: { value: null, bar: '>= 0.09 m point-to-axis', pass: 'pending' };
		// Cane sub-metrics (VERIFY-2, 2026-07-10): wired now that RIG-2's
		// PatientCane.js + _lastSync.caneHandleTargetWorld have landed -- see
		// the per-sample comments above for the exact fields/frames read.
		// Still feature-detected: pass:'pending' if the cane truly never
		// appeared this sweep (v1 GAIT schedule, or caneEnabled===false),
		// matching every other v2-only metric's degrade path (e.g. M11 above).
		// F6 (integration_2.json diag, 2026-07-10): bar is measured EXCLUDING
		// reachClamped samples -- a rigid cane (fixed tip, fixed caneLengthM) and an
		// arm that can't quite reach it are two DIFFERENT constraint spheres (one
		// centered on the terrain-anchored tip, one on the shoulder pivot) that
		// cannot in general coincide, so a clamped sample's residual is an inherent
		// geometric property, not a solver-precision bug -- see the per-sample
		// site's own comment. m12CaneHandErrorMax (including clamped samples) is
		// still reported in `value` so a clamped residual is never silently
		// dropped; M12_caneReachClampInfo below is the metric that actually gates
		// whether the clamp itself is a problem (it isn't: measured exceedsFraction
		// ~0.0015, far under the 5% flag threshold).
		metrics.M12_caneHandToHandleErrorMax = m12CaneHandErrorUnclampedSamples > 0
			? {
				value: { unclampedMax: m12CaneHandErrorMaxUnclamped, includingClampedMax: m12CaneHandErrorMax, unclampedSamples: m12CaneHandErrorUnclampedSamples, clampedSamples: m12CaneHandErrorSamples - m12CaneHandErrorUnclampedSamples },
				bar: '<= 0.015 m every sampled frame EXCLUDING reachClamped samples (see code comment; those are covered separately by M12_caneReachClampInfo)',
				pass: m12CaneHandErrorMaxUnclamped <= 0.015,
			}
			: m12CaneHandErrorSamples > 0
				? { value: m12CaneHandErrorMax, bar: '<= 0.015 m every sampled frame', pass: m12CaneHandErrorMax <= 0.015 }
				: { value: null, bar: '<= 0.015 m every sampled frame', pass: 'pending' };
		// F6 (integration_2.json diag, 2026-07-10): informational -- how often/large
		// the cane re-aims because the right-arm reach can't hit the nominal handle
		// target (PatientHuman step 17's own reachClamped fallback, spec S5 "the
		// cane tilts toward the hand rather than the arm hyper-extending"). No
		// pass/fail bar (this is diagnostic context, not an acceptance gate, matching
		// audit/gait_audit.mjs's own "informational (no bar specified)" convention
		// for M2_cadenceStepsPerMin) -- but flagged here per this task's own
		// instruction: if the clamp exceeds ~0.10 m for >5% of walking samples, the
		// cane GEOMETRY itself (caneForwardM/caneLateralM/lean) needs retuning, not
		// just the measurement.
		const caneClampExceedFraction = m12CaneClampSamples > 0 ? m12CaneClampExceedCount / m12CaneClampSamples : 0;
		metrics.M12_caneReachClampInfo = m12CaneClampSamples > 0
			? {
				value: { maxClampMagnitudeM: m12CaneClampMagMax, exceedsFraction: caneClampExceedFraction, samples: m12CaneClampSamples },
				bar: `informational (no bar) -- flag if exceedsFraction > 0.05 at maxClampMagnitudeM > 0.10 m: ${ caneClampExceedFraction > 0.05 && m12CaneClampMagMax > 0.10 ? 'FLAGGED, retune cane geometry' : 'not flagged' }`,
				pass: true,
			}
			: { value: null, bar: 'informational (no bar)', pass: 'na(no samples with both target+effective fields)' };
		metrics.M12_caneShaftClearanceMin = m12CaneShaftSamples > 0
			? { value: m12CaneShaftClearanceMin, bar: '>= 0.03 m vs shin (segment-to-segment centerline distance, both sides; bone-position approximation, not a true mesh/capsule surface)', pass: m12CaneShaftClearanceMin >= 0.03 }
			: { value: null, bar: '>= 0.03 m vs shin', pass: 'pending' };

		const pelvisAmpNonIdle = Number.isFinite( m13PelvisNonIdleRange.min ) ? ( m13PelvisNonIdleRange.max - m13PelvisNonIdleRange.min ) / 2 : null;
		const pelvisAmpIdle = Number.isFinite( m13PelvisIdleRange.min ) ? ( m13PelvisIdleRange.max - m13PelvisIdleRange.min ) / 2 : null;
		metrics.M13_pelvisBobAmplitude = pelvisAmpNonIdle !== null
			? { value: pelvisAmpNonIdle, bar: '[0.008, 0.035] m (peak-to-peak/2, pooled across both clips)', pass: pelvisAmpNonIdle >= 0.008 && pelvisAmpNonIdle <= 0.035 }
			: { value: null, bar: '[0.008, 0.035] m', pass: 'na(no non-idle samples)' };
		metrics.M13_pelvisBobIdleAmplitude = pelvisAmpIdle !== null
			? { value: pelvisAmpIdle, bar: '<= 0.002 m ("zero" at idle)', pass: pelvisAmpIdle <= 0.002 }
			: { value: null, bar: '<= 0.002 m', pass: 'na(no idle samples)' };

		if ( headBone ) {

			const headAmpNonIdle = Number.isFinite( m13HeadNonIdleRange.min ) ? ( m13HeadNonIdleRange.max - m13HeadNonIdleRange.min ) / 2 : null;
			if ( headAmpNonIdle !== null && pelvisAmpNonIdle !== null && pelvisAmpNonIdle > 1e-4 ) {

				const ratio = headAmpNonIdle / pelvisAmpNonIdle;
				metrics.M13_headVsPelvisAmplitudeRatio = { value: ratio, bar: '< 1.0 (head amplitude < pelvis amplitude -- stabilization works)', pass: ratio < 1.0 };

			} else {

				metrics.M13_headVsPelvisAmplitudeRatio = { value: null, bar: '< 1.0', pass: 'na(pelvis amplitude ~0 -- see M13_pelvisBobAmplitude; ratio undefined)' };

			}

		} else {

			metrics.M13_headVsPelvisAmplitudeRatio = { value: null, bar: '< 1.0', pass: 'pending' };

		}

		// M14: the EXISTING acceptance bars this diagnostic already computed
		// above (perClip), unchanged -- just gated into a boolean + surfaced
		// under `metrics` too so a caller reading `metrics`/`pass` alone (not
		// also hand-checking perClip) still sees the complete picture.
		//
		// F9 (integration_2.json diag, 2026-07-10): kneeBend's bar is now PER-CLIP.
		// A single flat-gait-calibrated 40deg bar misclassifies "climb": real
		// stair-climbing weight-acceptance flexion is legitimately ~45-60deg (the
		// knee must bend more to lift the body onto the next tread) -- measured
		// climb stanceMedian 54.8deg, comfortably inside that real-world range but
		// over a 40deg bar that was only ever validated against flat walking
		// (measured follow stanceMedian 38.7deg, under 40deg either way). fkError/
		// plantedDrift/idleFootMotion bars are UNCHANGED (still one shared
		// threshold across both clips, per this task's own instruction).
		const KNEE_BEND_STANCE_MEDIAN_BAR_DEG = { follow: 40, climb: 65 };
		const m14Pass = [ 'follow', 'climb' ].every( ( name ) => {

			const c = perClip[ name ];
			if ( ! c ) return true; // clip not swept -- nothing to fail here
			return c.fkErrorMax <= 0.012 && c.plantedDriftMax <= 0.01 && c.idleFootMotionMax <= 0.002
				&& c.kneeBendDeg.stanceMedian <= KNEE_BEND_STANCE_MEDIAN_BAR_DEG[ name ];

		} );
		metrics.M14_existingAcceptanceBars = {
			value: {
				fkErrorMax: Math.max( perClip.follow?.fkErrorMax ?? 0, perClip.climb?.fkErrorMax ?? 0 ),
				plantedDriftMax: Math.max( perClip.follow?.plantedDriftMax ?? 0, perClip.climb?.plantedDriftMax ?? 0 ),
				idleFootMotionMax: Math.max( perClip.follow?.idleFootMotionMax ?? 0, perClip.climb?.idleFootMotionMax ?? 0 ),
				kneeBendStanceMedianDeg: {
					follow: perClip.follow?.kneeBendDeg?.stanceMedian ?? 0,
					climb: perClip.climb?.kneeBendDeg?.stanceMedian ?? 0,
				},
			},
			bar: 'fkErrorMax<=0.012m, plantedDriftMax<=0.01m, idleFootMotionMax<=0.002m (existing bars, unchanged, shared across clips'
				+ ( plantedDriftUsedContactField ? ' -- plantedDriftMax measured against the roll model\'s own contact point, see M8_rollWindowContactDriftMax' : '' )
				+ '); kneeBend stanceMedian<=40deg on follow (flat) / <=65deg on climb (F9, 2026-07-10: real stair weight-acceptance flexion is legitimately ~45-60deg, a flat-calibrated 40deg bar misclassifies it)',
			pass: m14Pass,
		};

		const pass = violations.length === 0 && ! patientHuman.ikSelfCheckFailed && Object.values( metrics ).every(
			( m ) => m.pass === true || ( typeof m.pass === 'string' && ( m.pass === 'pending' || m.pass.startsWith( 'na' ) ) ),
		);

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

		return {
			perClip, violations: violations.slice( 0, maxViolations ), ikSelfCheck: patientHuman.ikSelfCheckFailed, pass, metrics,
			_debugWorstPen, _debugWorstPitch, _debugM11IdleExtrema, _debugWorstCaneErr, _debugM13Series, // TEMP RIG-4 instrumentation, see this function's own top-of-body comment
		};

	},
	/**
	 * TRACE RECORDER (round 2 of the patient IK/gait overhaul -- see
	 * IK_OVERHAUL_SPEC.md and audit/TRACE_SCHEMA.md). Unlike patientDiag above
	 * (which computes AGGREGATE pass/fail metrics and judges them against bars),
	 * this sweeps the SAME real applyGlobalTime()/patientHuman.sync() path at a
	 * FIXED dt across the whole unified timeline and records one RAW, unjudged
	 * sample per frame: PatientGait.poseAt()'s scheduler fields, bone world
	 * positions/orientations converted into the isaac_world LOCAL (P-frame,
	 * Z-up) frame per AGENTS.md incident #5's diagnostic-pitfall note, cane tip/
	 * handle, terrain heights under each foot + the root, and a snapshot of
	 * every patientHuman._lastSync field -- pure data for a downstream analyzer
	 * to mine for trajectory/timing/naturalness issues, no pass/fail here.
	 *
	 * Mirrors patientDiag's sweep/save-restore machinery exactly: same
	 * setPhase()/action.time/mixer.update(0)/patientHuman.sync() sequence per
	 * sample (never a shortcut that could diverge from what a user actually
	 * sees), same save-before/restore-after of phase, per-action times, and the
	 * scrubber value (read-only instrument, not a mode switch -- a caller
	 * scrubbing afterward sees no trace this ran). Deterministic: dt is fixed
	 * and nothing in the sweep itself reads Math.random/Date.now (meta.
	 * generatedAt is a wall-clock stamp for HUMANS reading the file, not
	 * consumed by anything downstream, consistent with I1's determinism
	 * contract applying to the gait/rig code being measured, not to this
	 * recorder's own bookkeeping).
	 *
	 * POSTs the resulting JSON to serve.py's POST /diag sink (same rationale/
	 * pattern as gaitReport above -- a ~5-10 MB report at default dt is well
	 * past any eval-return-value truncation limit) and resolves to the saved
	 * file's path text. `name` defaults to 'trace_full' (files land in
	 * diag/<name>.json). See audit/TRACE_SCHEMA.md for the exact schema and
	 * audit/run_browser_trace.py for the headless driver that triggers this via
	 * the `?gaittrace=1` boot query param below.
	 */
	gaitTrace( { dt = 1 / 60, name = 'trace_full' } = {} ) {

		if ( ! patientHuman._attached || ! patientHuman._schedules || ! modelRoot ) {

			return Promise.reject( new Error( 'gaitTrace: patient not ready (call after window.__viewer.ready resolves)' ) );

		}

		const isaacWorldNode = modelRoot.getObjectByName( 'isaac_world' );
		if ( ! isaacWorldNode ) return Promise.reject( new Error( 'gaitTrace: isaac_world node not found' ) );

		// Save prior state, exactly like patientDiag above.
		const priorPhase = currentPhase;
		const priorTimes = new Map();
		for ( const [ pname, action ] of phaseActions ) priorTimes.set( pname, action.time );
		const priorScrubberValue = scrubber.value;

		const bones = patientHuman._bones;

		const _tmpWorld = new THREE.Vector3();
		const _tmpLocal = new THREE.Vector3();
		const _tmpWorldQuat = new THREE.Quaternion();
		const _tmpParentQuatInv = new THREE.Quaternion();
		const _tmpLocalQuat = new THREE.Quaternion();
		const _tmpEuler = new THREE.Euler();

		/** getWorldPosition() then convert into isaac_world's own LOCAL (P-frame)
		 * meters -- identical technique to patientDiag's own worldToPframe above
		 * (AGENTS.md incident #5). */
		function boneP( bone ) {

			bone.getWorldPosition( _tmpWorld );
			isaacWorldNode.worldToLocal( _tmpLocal.copy( _tmpWorld ) );
			return { x: _tmpLocal.x, y: _tmpLocal.y, z: _tmpLocal.z };

		}

		/**
		 * Decompose a bone's WORLD orientation into isaac_world's own local
		 * (P-frame: up=Z, forward=X, lateral=Y -- AGENTS.md incident #4) basis,
		 * then extract yaw(about P-frame Z)/pitch(about P-frame Y)/roll(about
		 * P-frame X) via THREE.Euler order 'ZYX'. IMPORTANT CAVEAT (documented
		 * in full in audit/TRACE_SCHEMA.md): this is the bone's RAW achieved
		 * orientation, which includes PatientHuman.js's fixed B_PLACEMENT
		 * bind-convention-reconciliation rotation baked in (Xbot's own bind
		 * pose is not identity in THIS basis -- see AGENTS.md incident #4) --
		 * it is NOT zero at rest. Compare relative values/ranges across the
		 * trace, or against the trace's own first idle sample, not against an
		 * assumed zero.
		 */
		function boneYawPitchRollDeg( bone ) {

			bone.getWorldQuaternion( _tmpWorldQuat );
			isaacWorldNode.getWorldQuaternion( _tmpParentQuatInv ).invert();
			_tmpLocalQuat.copy( _tmpParentQuatInv ).multiply( _tmpWorldQuat );
			_tmpEuler.setFromQuaternion( _tmpLocalQuat, 'ZYX' );
			return {
				yawDeg: THREE.MathUtils.radToDeg( _tmpEuler.z ),
				pitchDeg: THREE.MathUtils.radToDeg( _tmpEuler.y ),
				rollDeg: THREE.MathUtils.radToDeg( _tmpEuler.x ),
			};

		}

		function footFields( f ) {

			return {
				x: f.x, y: f.y, z: f.z, yaw: f.yaw, planted: f.planted, swingU: f.swingU,
				liftAt: f.liftAt, landedAt: f.landedAt, nextLiftAt: f.nextLiftAt, strideLen: f.strideLen,
			};

		}

		const segmentsOut = [];

		for ( const clipName of [ 'follow', 'climb' ] ) {

			const clip = phaseClips.get( clipName );
			const action = phaseActions.get( clipName );
			const schedule = patientHuman._schedules[ clipName ];
			if ( ! clip || ! action || ! schedule ) continue;

			// setPhase REQUIRED before driving action.time -- see patientDiag's own
			// comment at its identical call site for why (weight=0 actions are never
			// evaluated by AnimationMixer).
			setPhase( clipName, { resetSlider: false } );

			const terrain = patientHuman._terrain;
			const duration = clip.duration;
			const segMeta = segments.find( ( s ) => s.name === clipName );
			const segStart = segMeta ? segMeta.start : 0; // ties tGlobal to the SAME unified-timeline mapping applyGlobalTime()/segmentAtGlobalTime() use

			const samples = [];

			for ( let t = 0; t <= duration + 1e-9; t += dt ) {

				const tt = Math.min( t, duration );

				action.time = tt;
				mixer.update( 0 );
				patientHuman.sync( clipName, tt );

				// Re-derive the same pose sync() just computed internally (cheap,
				// pure -- see patientDiag's identical call for the "tail-adjusted tq"
				// divergence caveat, which only matters after the walk-on tail's
				// freeze point).
				const pose = poseAt( schedule, terrain, tt );
				const ls = patientHuman._lastSync ? { ...patientHuman._lastSync } : null;

				const leftToeP = boneP( bones.leftToeBase );
				const rightToeP = boneP( bones.rightToeBase );

				samples.push( {
					tGlobal: segStart + tt,
					tLocal: tt,
					pose: {
						rootX: pose.rootX, rootY: pose.rootY, rootZ: pose.rootZ, rootYaw: pose.rootYaw,
						speed: pose.speed, groundSlope: pose.groundSlope,
						phaseC: pose.phaseC, support: pose.support, gaitPhaseLegacy: pose.gaitPhase,
						leftFoot: footFields( pose.leftFoot ), rightFoot: footFields( pose.rightFoot ),
						cane: pose.cane ? {
							x: pose.cane.x, y: pose.cane.y, z: pose.cane.z, planted: pose.cane.planted,
							swingU: pose.cane.swingU, liftAt: pose.cane.liftAt, landedAt: pose.cane.landedAt,
							nextLiftAt: pose.cane.nextLiftAt,
						} : null,
					},
					bones: {
						hips: boneP( bones.hips ), spine2: boneP( bones.spine2 ), head: boneP( bones.head ),
						leftArm: boneP( bones.leftArm ), rightArm: boneP( bones.rightArm ),
						leftHand: boneP( bones.leftHand ), rightHand: boneP( bones.rightHand ),
						leftFoot: boneP( bones.leftFoot ), rightFoot: boneP( bones.rightFoot ),
						leftToeBase: leftToeP, rightToeBase: rightToeP,
					},
					pelvisOrientDeg: boneYawPitchRollDeg( bones.hips ),
					spine2YawDeg: boneYawPitchRollDeg( bones.spine2 ).yawDeg,
					cane: {
						tip: pose.cane ? { x: pose.cane.x, y: pose.cane.y, z: pose.cane.z } : null,
						handleTarget: ls && ls.caneHandleTargetWorld ? ls.caneHandleTargetWorld : null,
						handleEffective: ls && ls.caneHandleEffectiveWorld ? ls.caneHandleEffectiveWorld : null,
					},
					terrain: {
						underRoot: terrain.heightAt( pose.rootX ),
						underLeftToe: terrain.heightAt( leftToeP.x ),
						underRightToe: terrain.heightAt( rightToeP.x ),
					},
					lastSync: ls,
				} );

			}

			segmentsOut.push( { name: clipName, duration, samples } );

		}

		// Restore prior state, exactly like patientDiag above.
		for ( const [ pname, t ] of priorTimes ) {

			const action = phaseActions.get( pname );
			if ( action ) action.time = t;

		}
		setPhase( priorPhase, { resetSlider: false } );
		mixer.update( 0 );
		patientHuman.sync( priorPhase, phaseActions.get( priorPhase )?.time ?? 0 );
		scrubber.value = priorScrubberValue;
		updateTimeReadout();

		const report = {
			meta: {
				generatedAt: new Date().toISOString(),
				dt,
				// Browser JS has no git access; audit/run_browser_trace.py patches
				// this field in-place after saving (reads `git rev-parse HEAD`
				// itself) -- see that script's own comment. null here just means
				// "not yet patched", not "detached from git".
				headCommit: null,
				schemaVersion: 1,
				columns: {
					tGlobal: 's, unified follow+climb timeline position (matches the scrubber/time-readout)',
					tLocal: 's, time within this segment\'s own clip (matches AnimationAction.time)',
					'pose.*': 'PatientGait.poseAt(schedule, terrain, tLocal) -- IK_OVERHAUL_SPEC.md section 3 contract, verbatim field names/meanings',
					'bones.*': '{x,y,z} meters, isaac_world LOCAL (P-frame, Z-up) -- getWorldPosition() then isaacWorldNode.worldToLocal(), per AGENTS.md incident #5',
					pelvisOrientDeg: 'Hips bone {yawDeg,pitchDeg,rollDeg}, P-frame axes (yaw=Z,pitch=Y,roll=X), RAW achieved orientation -- includes the fixed B_PLACEMENT bind offset, NOT zero at rest, see gaitTrace()\'s own boneYawPitchRollDeg comment',
					spine2YawDeg: 'Spine2 bone yaw only (same convention/caveat as pelvisOrientDeg), for spine-vs-pelvis counter-rotation analysis',
					'cane.*': 'tip = pose.cane P-frame position; handleTarget/handleEffective = patientHuman._lastSync.caneHandleTargetWorld/caneHandleEffectiveWorld (already P-frame, pre/post reach-clamp) -- all null if pose.cane is null (no v2 cane schedule)',
					'terrain.*': 'terrain.heightAt(x) meters under the root X and each toe bone\'s P-frame X',
					lastSync: 'shallow clone of patientHuman._lastSync as it stood right after this sample\'s sync() call -- whatever fields exist on the loaded PatientHuman.js version, verbatim field names',
				},
			},
			segments: segmentsOut,
		};

		return fetch(
			`/diag?name=${ encodeURIComponent( name ) }`,
			{ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify( report ) },
		).then( ( r ) => r.text() );

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
// The living-room set is decorative â€” never let a failure building it abort the
// viewer boot (it runs before loadRealModel); log loudly if it's consequently off.
try { buildLivingRoom(); } catch ( err ) { console.error( '[blueprint-viewer] living-room set NOT built (scene will show without it):', err ); }

// Perform the real initial sizing now (see the NOTE on the renderer/camera
// construction above) â€” canvasHost should have a committed layout by the
// time this module's top-level code finishes running, but guard anyway:
// if it's somehow still 0x0, the ResizeObserver below will catch the next
// genuine size change.
handleResize();

loadRealModel().then( () => {

	// Paint one frame so the canvas isn't blank before the Potential slide first
	// activates the gated loop (renderActive starts false -- see setRenderActive).
	renderFrame();
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

// ===========================================================================
// Headless trace auto-run hook (audit/run_browser_trace.py's driver -- round 2
// of the patient IK/gait overhaul, see audit/TRACE_SCHEMA.md). Opt-in via
// `?gaittrace=1[&dt=0.0167][&name=trace_full]` on the page URL; a no-op
// (nothing below even reads location.search) for normal interactive/deck
// usage. Runs AFTER window.__viewer.ready resolves -- the SAME "everything
// ready" moment loadRealModel()'s own .then/.catch above both funnel through
// via resolveReady() -- so the sweep never races the GLB/patient attach.
// Sets document.title = 'TRACE_DONE' on success (or 'TRACE_ERROR: <msg>' on
// failure) purely as a convenience signal; the headless driver's PRIMARY
// completion check is diag/<name>.json actually landing on disk (see that
// script's own comment for why the title is a nice-to-have, not load-bearing).
// ===========================================================================
{

	const _gaitTraceQP = new URLSearchParams( location.search );
	if ( _gaitTraceQP.get( 'gaittrace' ) === '1' ) {

		const _traceDt = parseFloat( _gaitTraceQP.get( 'dt' ) ) || ( 1 / 60 );
		const _traceName = _gaitTraceQP.get( 'name' ) || 'trace_full';
		window.__viewer.ready
			.then( () => window.__viewer.gaitTrace( { dt: _traceDt, name: _traceName } ) )
			.then( () => { document.title = 'TRACE_DONE'; } )
			.catch( ( err ) => {

				console.error( '[blueprint-viewer] gaitTrace auto-run failed:', err );
				document.title = 'TRACE_ERROR: ' + ( err && err.message ? err.message : String( err ) );

			} );

	}

}

// ===========================================================================
// Headless screenshot-series auto-run hook (round 2 of the patient IK/gait
// overhaul, VISUAL-VERIFY pass -- see audit/run_shot_series.py). Opt-in via
// `?shotseries=1` on the page URL; a no-op (nothing below even reads
// location.search) for normal interactive/deck usage. Mirrors the
// `?gaittrace=1` hook above: runs AFTER window.__viewer.ready resolves, so
// the sweep never races the GLB/patient attach.
//
// For each (tGlobal, label, view) capture point below: scrubs the REAL
// unified timeline via applyGlobalTime() (the SAME primitive
// scrubToPercent()/jumpToSegment() use -- never a shortcut), points the
// camera at a fixed WORLD-SPACE offset from the patient's CURRENT root
// position (recomputed every capture -- "consistent" means the same offset
// formula each time, not a hardcoded world position, since the patient is at
// a different place on the route at every tGlobal), renders one frame and
// saves it via window.__viewer.saveShot(). Read-only wrt gait/rig state: only
// scrubs (through the same applyGlobalTime() everything else uses) and moves
// the camera; never touches patientHuman/PatientGait state directly. Saves
// and restores phase/per-action time/scrubber/isPlaying/tracking/cinematic/
// camera pose afterward, mirroring patientDiag()/gaitTrace()'s own
// save-restore blocks above (same fields, same order).
//
// Sets document.title = 'SHOTS_DONE' on success (or 'SHOTS_ERROR: <msg>' on
// failure) as a convenience signal; the headless driver's PRIMARY completion
// check is diag/shotseries_manifest.json actually landing on disk (POSTed to
// serve.py's EXISTING POST /diag sink -- same one gaitReport() above already
// uses, no new server endpoint needed) listing every filename this run
// expects to exist in shots/, since the exact shot COUNT is dynamic (the
// trailing idle-window capture is conditional -- see _findIdleWindow below).
// ===========================================================================
{

	const _shotQP = new URLSearchParams( location.search );
	if ( _shotQP.get( 'shotseries' ) === '1' ) {

		window.__viewer.ready
			.then( () => _runShotSeries() )
			.then( () => { document.title = 'SHOTS_DONE'; } )
			.catch( ( err ) => {

				console.error( '[blueprint-viewer] shotseries auto-run failed:', err );
				document.title = 'SHOTS_ERROR: ' + ( err && err.message ? err.message : String( err ) );

			} );

	}

	/**
	 * World-space (THREE scene convention: Y up -- NOT the GLB's own
	 * isaac_world P-frame where Z is up, see AGENTS.md incident #4) camera
	 * offset relative to the patient's CURRENT root position for a given
	 * view. 'side' sits ~90deg around the vertical axis from '34' so it reads
	 * as a genuinely different viewing angle (profile-ish for a
	 * roughly-forward-walking patient) without this hook needing to read the
	 * patient's own instantaneous heading. Magnitude is scaled up from the
	 * viewer's own default boot camera (camera.position (1.6,1.2,2.2) around
	 * target (0,0.5,0), see top of this file) to comfortably fit a full
	 * standing adult + cane + a little floor at this PerspectiveCamera's 45deg
	 * fov.
	 */
	function _shotOffset( view ) {

		const R = 3.0;
		const azDeg = view === 'side' ? 122 : 35;
		const az = THREE.MathUtils.degToRad( azDeg );
		const camH = view === 'side' ? 1.15 : 1.4;
		return new THREE.Vector3( R * Math.cos( az ), camH, R * Math.sin( az ) );

	}

	const _shotPatientPos = new THREE.Vector3();

	/**
	 * Point camera+controls at the patient's CURRENT world position for
	 * `view`, using the SAME direct position.set()+target.set()+
	 * controls.update() idiom the module's own boot code uses above (no
	 * lookAt shortcut -- OrbitControls.update() derives orientation from
	 * position/target itself).
	 */
	function _positionCameraForPatient( view ) {

		patientHuman.anchor.getWorldPosition( _shotPatientPos );
		const off = _shotOffset( view );
		camera.position.set( _shotPatientPos.x + off.x, _shotPatientPos.y + off.y, _shotPatientPos.z + off.z );
		controls.target.set( _shotPatientPos.x, _shotPatientPos.y + 0.9, _shotPatientPos.z );
		controls.update();

	}

	/** shot_<sec>s_<label>[_f<n>]_<view>.png -- see module comment above. */
	function _fmtSec( t ) {

		const s = t.toFixed( 2 ).replace( /0+$/, '' ).replace( /\.$/, '' );
		const [ intPart, decPart ] = s.split( '.' );
		const ip = intPart.padStart( 2, '0' );
		return decPart ? `${ ip }_${ decPart }` : ip;

	}

	function _shotName( t, label, view, frameIdx ) {

		const parts = [ 'shot', `${ _fmtSec( t ) }s`, label ];
		if ( frameIdx != null ) parts.push( `f${ frameIdx }` );
		parts.push( view );
		return parts.join( '_' ) + '.png';

	}

	/**
	 * Scan the patient's ALREADY-BUILT gait schedules (patientHuman._schedules
	 * -- see PatientGait.js buildSchedule(), samples[i] = {t,x,y,zRoot,yaw,
	 * ...}) for a sustained (>=0.3s) run of ground-plane root speed under a
	 * loose idle floor (PatientGait.js DEFAULT_GAIT_PARAMS.idleSpeedThreshold
	 * is 0.02 m/s; this uses a looser 0.05 since it is picking a
	 * REPRESENTATIVE screenshot moment, not re-deriving a correctness gate),
	 * excluding segment boundaries (t<0.15 or t>duration-0.15 -- boot/settle
	 * artifacts, not a genuine mid-walk idle) and the already-separately-
	 * captured 23.3s handoff moment (+-1.0s, so the two don't just duplicate
	 * each other). Pure read of precomputed schedule data -- does NOT scrub
	 * the viewer. Returns { tGlobal, segName, speed } or null if no such
	 * window exists anywhere on the timeline.
	 */
	function _findIdleWindow() {

		const THRESH = 0.05;
		const MIN_SUSTAIN = 0.3;
		for ( const segName of [ 'follow', 'climb' ] ) {

			const schedule = patientHuman._schedules && patientHuman._schedules[ segName ];
			const seg = segments.find( ( s ) => s.name === segName );
			if ( ! schedule || ! schedule.samples || schedule.samples.length < 3 || ! seg ) continue;
			const samples = schedule.samples;
			let runStart = null;
			for ( let i = 1; i < samples.length; i ++ ) {

				const a = samples[ i - 1 ], b = samples[ i ];
				const dt = b.t - a.t;
				if ( dt <= 1e-6 ) continue;
				const speed = Math.hypot( b.x - a.x, b.y - a.y ) / dt;
				const tGlobalMid = seg.start + ( a.t + b.t ) / 2;
				const nearBoundary = a.t < 0.15 || b.t > seg.duration - 0.15;
				const nearHandoff = Math.abs( tGlobalMid - 23.3 ) < 1.0;

				if ( speed < THRESH && ! nearBoundary && ! nearHandoff ) {

					if ( runStart === null ) runStart = a.t;
					if ( b.t - runStart >= MIN_SUSTAIN ) {

						return { tGlobal: seg.start + ( runStart + b.t ) / 2, segName, speed };

					}

				} else {

					runStart = null;

				}

			}

		}

		return null;

	}

	async function _runShotSeries() {

		if ( ! patientHuman._attached || ! patientHuman._schedules || ! modelRoot ) {

			throw new Error( 'shotseries: patient not ready (call after window.__viewer.ready resolves)' );

		}

		// Save EVERYTHING this sweep might touch, exactly like patientDiag()/
		// gaitTrace()'s own save-restore blocks above (mirrored on purpose),
		// plus the camera/playback/mode state this hook additionally drives.
		const priorPhase = currentPhase;
		const priorTimes = new Map();
		for ( const [ pname, action ] of phaseActions ) priorTimes.set( pname, action.time );
		const priorScrubberValue = scrubber.value;
		const priorIsPlaying = isPlaying;
		const priorTracking = trackingEnabled;
		const priorCinematic = cinematicEnabled;
		const priorCamPos = camera.position.clone();
		const priorTarget = controls.target.clone();

		if ( priorIsPlaying ) setPlaying( false );
		// Tracking/cinematic camera modes both reassert camera.position from
		// robot-base motion or their own state inside renderFrame() (called by
		// saveShot() below) -- left on, either would stomp the framing this
		// hook sets per-capture.
		trackingEnabled = false;
		cinematicEnabled = false;

		const captures = [];
		captures.push( { t: 6.0, label: 'straight', view: '34' } );
		captures.push( { t: 6.0, label: 'straight', view: 'side' } );
		for ( let i = 0; i < 6; i ++ ) {

			const t = 6.0 + i * ( 1.3 / 5 ); // 6 frames, both endpoints included, one full stride
			captures.push( { t, label: 'stride', view: '34', frameIdx: i + 1 } );

		}

		captures.push( { t: 12.5, label: 'turn', view: '34' } );
		captures.push( { t: 16.0, label: 'turn', view: '34' } );
		captures.push( { t: 21.5, label: 'stairentry', view: '34' } );
		captures.push( { t: 21.5, label: 'stairentry', view: 'side' } );
		captures.push( { t: 23.3, label: 'handoff', view: '34' } );
		captures.push( { t: 35.0, label: 'climbapproach', view: '34' } );
		captures.push( { t: 50.0, label: 'climbstairs', view: '34' } );
		captures.push( { t: 50.0, label: 'climbstairs', view: 'side' } );
		captures.push( { t: 64.0, label: 'topland', view: '34' } );

		const idle = _findIdleWindow();
		if ( idle ) captures.push( { t: idle.tGlobal, label: 'idle', view: '34' } );

		const manifest = {
			shots: [],
			idle: idle ? { tGlobal: idle.tGlobal, segName: idle.segName, speed: idle.speed } : null,
		};

		for ( const cap of captures ) {

			applyGlobalTime( cap.t, { updateSlider: false } );
			_positionCameraForPatient( cap.view );
			const name = _shotName( cap.t, cap.label, cap.view, cap.frameIdx );
			await window.__viewer.saveShot( name );
			manifest.shots.push( { name, tGlobal: cap.t, label: cap.label, view: cap.view } );

		}

		// Restore prior state -- identical pattern to patientDiag()/
		// gaitTrace() above, plus this hook's own additional camera/mode saves.
		for ( const [ name, t ] of priorTimes ) {

			const action = phaseActions.get( name );
			if ( action ) action.time = t;

		}

		setPhase( priorPhase, { resetSlider: false } );
		mixer.update( 0 );
		patientHuman.sync( priorPhase, phaseActions.get( priorPhase )?.time ?? 0 );
		scrubber.value = priorScrubberValue;
		updateTimeReadout();

		trackingEnabled = priorTracking;
		cinematicEnabled = priorCinematic;
		if ( trackingEnabled ) hasLastBasePos = false; // resync delta baseline on re-enable, same as the tracking-toggle click handler above
		camera.position.copy( priorCamPos );
		controls.target.copy( priorTarget );
		controls.update();
		if ( priorIsPlaying ) setPlaying( true );

		// POST last (after state restore) so a manifest-write failure can never
		// leave the viewer stuck in the scrubbed/tracking-disabled state.
		await fetch( `/diag?name=${ encodeURIComponent( 'shotseries_manifest' ) }`, {
			method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify( manifest ),
		} ).then( ( r ) => r.text() );

		return manifest;

	}

}
