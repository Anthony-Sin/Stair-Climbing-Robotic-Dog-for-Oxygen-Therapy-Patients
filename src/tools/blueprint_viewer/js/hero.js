// hero.js
//
// ACT 1 of the page. LEFT: an animated 3D stage — the robot changes behaviour
// per chapter (spins in place / walks in place / climbs the real staircase
// with a right-side follow-cam). Leader lines point from the robot to key
// parts, on both sides of the stage. RIGHT: a chaptered explainer panel
// (paragraph + a richer policy SVG diagram + an optional mp4 clip). Arrow
// buttons step chapters; zoom buttons dolly the camera (the wheel is left to
// the page so scrolling is never hijacked).
//
// Separate three.js scene from the interactive viewer (js/main.js). Reuses
// ./models/robot.glb as an ASSET, keeping the FULL baked hierarchy
// (isaac_world > robot_base + stairs + …) so it can drive the real
// `follow`/`climb` clips and show the real stairs. Shares palette.js + the
// BlueprintEdgesPass ink outlines, read-only.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import { PALETTES } from './palette.js';
import { BlueprintEdgesPass } from './BlueprintEdgesPass.js';

// ===========================================================================
// Chapters. `motion` selects the stage behaviour; `points` are the leader
// callouts, split between the stage's left/right gutters (mostly base-mounted
// parts so they stay steady). `clip` may be null (no mp4 for that chapter);
// `clip.src` is a real Isaac Sim rollout cut from log/.../scene_view.mp4.
// `fx` names the stage overlay effect (camera-scan / feel), null for none.
// ===========================================================================

const CHAPTERS = [
	{
		id: 'architecture',
		title: 'System architecture',
		motion: 'spin',
		fx: null,
		body: 'A Unitree Go2 carries a patient’s oxygen concentrator on a shock-isolated cradle. The entire control stack — perception plus the learned policy — runs inside one Docker container on a Jetson Orin: the same container whether Isaac Sim or the real robot feeds it. Only the sensor source is swapped at the container boundary, so a policy proven in simulation is expected to hold on hardware.',
		clip: null, // no clip for this chapter
		points: [
			{ node: 'oxygen_tank', side: 'right', label: 'O₂ concentrator', sub: 'patient payload' },
			{ node: 'cradle_rails', side: 'right', label: 'payload cradle', sub: 'shock-isolated' },
			{ node: 'robot_base', side: 'left', label: 'onboard compute', sub: 'Jetson Orin' },
			{ node: 'FR_hip', side: 'left', label: '12× joint actuators', sub: 'three per leg' },
		],
	},
	{
		id: 'walking',
		title: 'Walking policy',
		motion: 'walk',
		fx: 'scan',
		body: 'On flat ground the robot follows the patient by sight: a YOLO-World detector locates the person in every camera frame, and a learned trot gait steers to keep pace while holding the oxygen payload level — rejecting the disturbances of a shifting load and an uneven floor at each step.',
		clip: { badge: 'clip 01', cap: 'Isaac Sim rollout — flat-ground follow gait.', src: './assets/clips/walk.mp4' },
		points: [
			{ node: 'robot_base', side: 'left', label: 'gait controller', sub: 'trot clock' },
			{ node: 'FL_hip', side: 'left', label: 'hip abduction', sub: 'lateral balance' },
			{ node: 'cradle_rails', side: 'right', label: 'payload held level', sub: 'load balancing' },
			{ node: 'FR_calf', side: 'right', label: 'calf drive', sub: 'ground clearance' },
		],
	},
	{
		id: 'blind-rl',
		title: 'Blind RL policy',
		motion: 'climb',
		fx: 'feel',
		body: 'The staircase is climbed on feel alone. Cameras can’t see the steps underfoot, so a reinforcement-learning policy leans entirely on proprioception and foot contact — sensing each riser as a paw lands — to place its feet and drive the payload upward, step after step, while keeping the concentrator upright on the incline.',
		clip: { badge: 'clip 02', cap: 'Isaac Sim rollout — blind stair traversal.', src: './assets/clips/stairs.mp4' },
		points: [
			{ node: 'robot_base', side: 'left', label: 'IMU · body attitude', sub: 'stays upright' },
			{ node: 'FR_hip', side: 'left', label: 'joint feedback', sub: 'proprioception' },
			{ node: 'oxygen_tank', side: 'right', label: 'payload upright', sub: 'on the incline' },
			{ node: 'cradle_rails', side: 'right', label: 'kept level', sub: 'active balancing' },
		],
	},
];

// Policy SVG schematics — richer than a bare 3-box flow, still clean line-art.
// Theme-aware: strokes/text use currentColor (#hero-diagram sets color: var(--ink)).
const DIAGRAMS = {
	architecture: `<svg viewBox="0 0 320 178" role="img" aria-label="Docker / simulation split architecture diagram">
		<rect class="dg-box" x="6" y="44" width="86" height="28" rx="5"/><text class="dg-t" x="49" y="62" text-anchor="middle">Isaac Sim</text>
		<rect class="dg-box" x="6" y="98" width="86" height="28" rx="5"/><text class="dg-t" x="49" y="116" text-anchor="middle">real Go2</text>
		<path class="dg-ln" d="M92 58 H102 V71 H116"/>
		<path class="dg-ln" d="M92 112 H102 V71 H116"/>
		<path class="dg-ah" d="M110 67l6 4-6 4"/>
		<text class="dg-c" x="174" y="40" text-anchor="middle">docker container</text>
		<rect class="dg-dock" x="108" y="46" width="130" height="82" rx="8"/>
		<rect class="dg-box" x="116" y="58" width="116" height="26" rx="5"/><text class="dg-t" x="174" y="75" text-anchor="middle">perception</text>
		<rect class="dg-box" x="116" y="94" width="116" height="26" rx="5"/><text class="dg-t" x="174" y="111" text-anchor="middle">policy π · frozen</text>
		<path class="dg-ln" d="M174 84 V94"/><path class="dg-ah" d="M170 90l4 4 4-4"/>
		<rect class="dg-box" x="252" y="92" width="64" height="30" rx="5"/><text class="dg-t" x="284" y="111" text-anchor="middle">12× joints</text>
		<path class="dg-ln" d="M232 107 H250"/><path class="dg-ah" d="M244 103l6 4-6 4"/>
		<text class="dg-c" x="160" y="150" text-anchor="middle">identical container · sim ⇄ real</text>
		<text class="dg-c" x="160" y="165" text-anchor="middle">only the sensor source is swapped</text>
	</svg>`,
	walking: `<svg viewBox="0 0 320 178" role="img" aria-label="Walking control loop diagram">
		<rect class="dg-box" x="6" y="26" width="76" height="38" rx="5"/><text class="dg-t" x="44" y="49" text-anchor="middle">state est.</text>
		<rect class="dg-box" x="122" y="26" width="80" height="38" rx="5"/><text class="dg-t" x="162" y="44" text-anchor="middle">policy π</text><text class="dg-c" x="162" y="57" text-anchor="middle">MLP</text>
		<rect class="dg-box" x="242" y="26" width="72" height="38" rx="5"/><text class="dg-t" x="278" y="44" text-anchor="middle">PD joint</text><text class="dg-t" x="278" y="56" text-anchor="middle">targets</text>
		<path class="dg-ln" d="M82 45 H120"/><path class="dg-ah" d="M114 41l6 4-6 4"/>
		<path class="dg-ln" d="M202 45 H240"/><path class="dg-ah" d="M236 41l6 4-6 4"/>
		<rect class="dg-box" x="118" y="104" width="88" height="36" rx="5"/><text class="dg-t" x="162" y="120" text-anchor="middle">Go2 · 12 DoF</text><text class="dg-c" x="162" y="132" text-anchor="middle">rigid-body plant</text>
		<path class="dg-ln" d="M278 64 V122 H208"/><path class="dg-ah" d="M214 118l-6 4 6 4"/>
		<path class="dg-ln" d="M118 122 H44 V64"/><path class="dg-ah" d="M40 70l4-6 4 6"/>
		<text class="dg-c" x="235" y="96" text-anchor="middle">torque</text>
		<text class="dg-c" x="66" y="96" text-anchor="middle">imu · contacts</text>
	</svg>`,
	'blind-rl': `<svg viewBox="0 0 320 178" role="img" aria-label="Blind RL closed-loop policy diagram">
		<rect class="dg-box" x="16" y="14" width="108" height="22" rx="5"/><text class="dg-t" x="70" y="30" text-anchor="middle">proprioception ×N</text>
		<rect class="dg-box" x="16" y="42" width="108" height="22" rx="5"/><text class="dg-t" x="70" y="58" text-anchor="middle">foot contact</text>
		<rect class="dg-box" x="16" y="70" width="108" height="22" rx="5"/><text class="dg-t" x="70" y="86" text-anchor="middle">velocity command</text>
		<rect class="dg-box" x="140" y="34" width="64" height="44" rx="6"/><text class="dg-t" x="172" y="53" text-anchor="middle">policy π</text><text class="dg-c" x="172" y="67" text-anchor="middle">MLP</text>
		<rect class="dg-box" x="224" y="42" width="88" height="30" rx="5"/><text class="dg-t" x="268" y="61" text-anchor="middle">joint targets</text>
		<path class="dg-ln" d="M124 25 H132 V50 H140"/><path class="dg-ln" d="M124 53 H140"/><path class="dg-ln" d="M124 81 H132 V60 H140"/>
		<path class="dg-ah" d="M134 50l6 4-6 4"/>
		<path class="dg-ln" d="M204 56 H222"/><path class="dg-ah" d="M216 52l6 4-6 4"/>
		<rect class="dg-box" x="120" y="112" width="96" height="30" rx="5"/><text class="dg-t" x="168" y="131" text-anchor="middle">Go2 on stairs</text>
		<path class="dg-ln" d="M268 72 V127 H216"/><path class="dg-ah" d="M222 123l-6 4 6 4"/>
		<path class="dg-ln" d="M120 127 H8 V25 H16"/><path class="dg-ah" d="M10 21l6 4-6 4"/>
		<text class="dg-c" x="64" y="108" text-anchor="middle">feels each riser</text>
		<text class="dg-c" x="164" y="164" text-anchor="middle">closed proprioceptive loop · climbs by feel</text>
	</svg>`,
};

const SVG_NS = 'http://www.w3.org/2000/svg';
const NARROW_PX = 820;

const stage = document.getElementById( 'hero-stage' );
if ( stage ) boot( stage );

function boot( host ) {

	// -------------------------------------------------------------------
	// Theme + defensive colors
	// -------------------------------------------------------------------
	let themeName = document.documentElement.getAttribute( 'data-theme' ) || 'light';
	if ( themeName !== 'light' && themeName !== 'dark' ) themeName = 'light';
	let palette = PALETTES[ themeName ];

	function heroColors( p ) {

		const pick = ( ...v ) => v.find( ( x ) => x !== undefined && x !== null );
		return {
			bg: pick( p.sceneBackground, 0xd6d2ca ),
			ink: pick( p.inkColorGl, 0x2f2c28 ),
			robot: pick( p.robotColor, p.materialColor, 0xe7e3d9 ),
			tank: pick( p.oxygenTankColor, p.materialColor, 0xf7f6f2 ),
			cradle: pick( p.cradleRailsColor, p.materialColor, 0x333333 ),
			stairs: pick( p.stairsColor, p.materialColor, 0x9c6b3a ),
			rail: pick( p.handrailColor, p.materialColor, 0x332e29 ),
		};

	}
	let colors = heroColors( palette );

	// -------------------------------------------------------------------
	// Renderer / scene / camera / controls
	// -------------------------------------------------------------------
	const renderer = new THREE.WebGLRenderer( { antialias: true, alpha: false } );
	renderer.setPixelRatio( Math.min( window.devicePixelRatio || 1, 2 ) );
	renderer.setSize( 1, 1 );
	renderer.shadowMap.enabled = true;
	renderer.shadowMap.type = THREE.PCFSoftShadowMap;
	host.insertBefore( renderer.domElement, host.firstChild );

	const scene = new THREE.Scene();
	scene.background = makeStudioBackdrop( colors.bg );

	const camera = new THREE.PerspectiveCamera( 36, 1, 0.08, 60 );
	camera.position.set( 1.2, 0.7, 1.6 );

	const controls = new OrbitControls( camera, renderer.domElement );
	controls.enableDamping = true;
	controls.dampingFactor = 0.09;
	controls.enablePan = false;
	controls.enableZoom = false;
	controls.autoRotate = false;
	controls.autoRotateSpeed = 0.9;
	controls.minPolarAngle = 0.5;
	controls.maxPolarAngle = Math.PI / 2 - 0.03;
	controls.target.set( 0, 0.35, 0 );

	// -------------------------------------------------------------------
	// Lights + shadow
	// -------------------------------------------------------------------
	scene.add( new THREE.HemisphereLight( 0xffffff, 0xd8d4cc, 1.15 ) );
	const keyLight = new THREE.DirectionalLight( 0xffffff, 1.75 );
	keyLight.position.set( 2.4, 4.2, 2.6 );
	keyLight.castShadow = true;
	keyLight.shadow.mapSize.set( 2048, 2048 );
	keyLight.shadow.camera.near = 0.5;
	keyLight.shadow.camera.far = 22;
	keyLight.shadow.camera.left = -3.5;
	keyLight.shadow.camera.right = 3.5;
	keyLight.shadow.camera.top = 3.5;
	keyLight.shadow.camera.bottom = -3.5;
	keyLight.shadow.bias = -0.0006;
	keyLight.shadow.radius = 5;
	scene.add( keyLight );
	const fillLight = new THREE.DirectionalLight( 0xffffff, 0.35 );
	fillLight.position.set( -3, 1.6, -1.8 );
	scene.add( fillLight );

	// -------------------------------------------------------------------
	// Toon materials
	// -------------------------------------------------------------------
	const CEL_GRADIENT_MAP = makeToonGradientMap( [ 0.4, 0.72, 1.0 ] );
	const RIM_COLOR = new THREE.Color( 0xffffff );

	function makeToonGradientMap( levels ) {

		const data = new Uint8Array( levels.length );
		for ( let i = 0; i < levels.length; i ++ ) data[ i ] = Math.round( THREE.MathUtils.clamp( levels[ i ], 0, 1 ) * 255 );
		const t = new THREE.DataTexture( data, levels.length, 1, THREE.RedFormat );
		t.minFilter = THREE.NearestFilter; t.magFilter = THREE.NearestFilter; t.generateMipmaps = false; t.needsUpdate = true;
		return t;

	}

	function makeToonMaterial( colorHex ) {

		const m = new THREE.MeshToonMaterial( { color: colorHex, gradientMap: CEL_GRADIENT_MAP } );
		m.onBeforeCompile = ( shader ) => {

			shader.uniforms.uRimColor = { value: RIM_COLOR };
			shader.uniforms.uRimPower = { value: 2.4 };
			shader.uniforms.uRimIntensity = { value: 0.4 };
			shader.fragmentShader = shader.fragmentShader
				.replace( '#define TOON', '#define TOON\nuniform vec3 uRimColor;\nuniform float uRimPower;\nuniform float uRimIntensity;' )
				.replace(
					'vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;',
					'vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;\n' +
					'\tfloat rimFresnel = pow( 1.0 - max( dot( normalize( vNormal ), normalize( vViewPosition ) ), 0.0 ), uRimPower );\n' +
					'\toutgoingLight += rimFresnel * uRimIntensity * uRimColor;',
				);

		};
		return m;

	}

	const robotMaterial = makeToonMaterial( colors.robot );
	const oxygenTankMaterial = makeToonMaterial( colors.tank );
	const cradleRailsMaterial = makeToonMaterial( colors.cradle );
	const stairsMaterial = makeToonMaterial( colors.stairs );
	const handrailMaterial = makeToonMaterial( colors.rail );
	const PLINTH_COLOR = 0x9a9284;
	const plinthMaterial = makeToonMaterial( PLINTH_COLOR );

	const TINT = {
		oxygen_tank: () => oxygenTankMaterial,
		cradle_rails: () => cradleRailsMaterial,
		stairs: () => stairsMaterial,
		handrails: () => handrailMaterial, // separate node in the newer rebake; absent (merged) in older glbs
	};

	function tintFor( mesh ) {

		let p = mesh;
		while ( p ) { const fn = TINT[ p.name ]; if ( fn ) return fn(); p = p.parent; }
		return robotMaterial;

	}

	// -------------------------------------------------------------------
	// Post-processing
	// -------------------------------------------------------------------
	const composer = new EffectComposer( renderer );
	composer.addPass( new RenderPass( scene, camera ) );
	const edgesPass = new BlueprintEdgesPass( scene, camera, {
		inkColor: colors.ink, normalThreshold: 0.55, depthThreshold: 0.03, thickness: 1.2,
	} );
	composer.addPass( edgesPass );
	composer.addPass( new OutputPass() );

	// -------------------------------------------------------------------
	// Overlay DOM refs
	// -------------------------------------------------------------------
	const heroInner = document.getElementById( 'hero-inner' );
	const leadersSvg = document.getElementById( 'hero-leaders' );
	const calloutsEl = document.getElementById( 'hero-callouts' );
	const chapterIndexEl = document.getElementById( 'hero-chapter-index' );
	const chapterTitleEl = document.getElementById( 'hero-chapter-title' );
	const bodyEl = document.getElementById( 'hero-body' );
	const diagramEl = document.getElementById( 'hero-diagram' );
	const clipFigureEl = document.querySelector( '.hero-clip' );
	const clipBadgeEl = document.getElementById( 'hero-clip-badge' );
	const clipCapEl = document.getElementById( 'hero-clip-cap' );
	const dotsEl = document.getElementById( 'hero-dots' );
	const clipVideo = document.getElementById( 'hero-clip-video' );
	const clipFrame = document.getElementById( 'hero-clip-frame' );
	const clipPlayBtn = document.getElementById( 'hero-clip-play' );
	const clipDurEl = document.getElementById( 'hero-clip-dur' );
	const fxEl = document.getElementById( 'hero-fx' );

	// -------------------------------------------------------------------
	// Inline mp4 clip (real Isaac Sim rollout, cut per chapter) — muted +
	// looped so it plays inline; the corner button toggles play/pause.
	// -------------------------------------------------------------------
	function setClip( src ) {

		if ( ! clipVideo || ! src ) return;
		const abs = new URL( src, location.href ).href;
		if ( clipVideo.src !== abs ) clipVideo.src = src;
		try { clipVideo.currentTime = 0; } catch ( e ) { /* not seekable yet */ }
		const p = clipVideo.play();
		if ( p && p.catch ) p.catch( () => {} ); // autoplay may be blocked; button still works
		updateClipButton();

	}

	function stopClip() { if ( clipVideo ) clipVideo.pause(); }

	function updateClipButton() {

		if ( clipFrame && clipVideo ) clipFrame.classList.toggle( 'playing', ! clipVideo.paused && ! clipVideo.ended );

	}

	if ( clipVideo && clipPlayBtn ) {

		clipPlayBtn.addEventListener( 'click', () => {

			if ( clipVideo.paused ) { const p = clipVideo.play(); if ( p && p.catch ) p.catch( () => {} ); }
			else clipVideo.pause();
			updateClipButton();

		} );
		clipVideo.addEventListener( 'play', updateClipButton );
		clipVideo.addEventListener( 'pause', updateClipButton );
		clipVideo.addEventListener( 'loadedmetadata', () => {

			if ( clipDurEl && isFinite( clipVideo.duration ) ) clipDurEl.textContent = Math.round( clipVideo.duration ) + 's';

		} );

	}

	// -------------------------------------------------------------------
	// Per-policy stage FX overlay: built once, toggled via a class on
	// #hero-fx (set in applyMotion), and positioned each frame (updateFx).
	//   walk  -> a scan cone projected in FRONT of the robot (+ YOLO HUD)
	//   climb -> orange proprioceptive "feeling" waves at EACH foot
	// -------------------------------------------------------------------
	let currentFx = null;
	let fxConfEl = null, fxFeet = [];
	let fxConeSvg = null, fxConeFill = null, fxConeSweep = null, fxConeGrad = null;
	let fxConf = 0.94, fxConfTarget = 0.94, fxConfAccum = 0, fxConeT = 0;
	const _fxBox = new THREE.Box3();
	const FX_FEET = [ 'FL_foot', 'FR_foot', 'RL_foot', 'RR_foot' ];
	const _cA = new THREE.Vector3(), _cB = new THREE.Vector3(), _cC = new THREE.Vector3(), _cD = new THREE.Vector3();
	const _cFwd = new THREE.Vector3(), _cApex = new THREE.Vector3(), _cFar = new THREE.Vector3();

	buildFx();

	function buildFx() {

		if ( ! fxEl ) return;
		fxEl.innerHTML =
			// walk: a scan cone projected in front of the robot + YOLO HUD
			'<svg class="fx-cone" aria-hidden="true">' +
				'<defs><linearGradient id="fxConeGrad" gradientUnits="userSpaceOnUse">' +
					'<stop class="fx-cone-s0" offset="0"/><stop class="fx-cone-s1" offset="1"/>' +
				'</linearGradient></defs>' +
				'<polygon class="fx-cone-fill"/><line class="fx-cone-sweep"/>' +
			'</svg>' +
			'<div class="fx-hud fx-hud-yolo"><span class="fx-hud-dot"></span>YOLO-World<span class="fx-hud-conf">person 0.00</span></div>' +
			// climb: orange proprioceptive "feeling" waves at each foot + sense HUD
			'<div class="fx-feet">' + FX_FEET.map( ( id ) => `<div class="fx-foot" data-leg="${ id }"><span></span><span></span><span></span></div>` ).join( '' ) + '</div>' +
			'<div class="fx-hud fx-hud-sense"><span class="fx-hud-dot"></span>proprioception · contact sensing</div>';
		fxConeSvg = fxEl.querySelector( '.fx-cone' );
		fxConeFill = fxEl.querySelector( '.fx-cone-fill' );
		fxConeSweep = fxEl.querySelector( '.fx-cone-sweep' );
		fxConeGrad = fxEl.querySelector( '#fxConeGrad' );
		fxConfEl = fxEl.querySelector( '.fx-hud-conf' );
		fxFeet = [ ...fxEl.querySelectorAll( '.fx-foot' ) ].map( ( el ) => ( { el, node: null, id: el.dataset.leg } ) );

	}

	function updateFx( dt ) {

		if ( ! fxEl || ! currentFx ) return;
		const sr = host.getBoundingClientRect();
		const W = sr.width, H = sr.height;

		if ( currentFx === 'scan' ) {

			updateCone( dt, W, H );

			// Wander the "confidence" so the YOLO HUD reads live (not a real score).
			fxConfAccum += dt;
			if ( fxConfAccum > 0.55 ) { fxConfAccum = 0; fxConfTarget = 0.9 + Math.random() * 0.09; }
			fxConf += ( fxConfTarget - fxConf ) * Math.min( 1, dt * 4 );
			if ( fxConfEl ) fxConfEl.textContent = 'person ' + fxConf.toFixed( 2 );

		} else if ( currentFx === 'feel' ) {

			// Orange "feeling" waves ripple out from EACH foot as it senses a riser.
			for ( const foot of fxFeet ) {

				if ( ! foot.node ) foot.node = scene.getObjectByName( foot.id );
				if ( ! foot.node ) { foot.el.classList.add( 'fx-off' ); continue; }
				foot.node.getWorldPosition( _tmpVec );
				const p = _tmpVec.project( camera );
				if ( p.z > 1 || Math.abs( p.x ) > 1 || Math.abs( p.y ) > 1 ) { foot.el.classList.add( 'fx-off' ); continue; }
				foot.el.classList.remove( 'fx-off' );
				foot.el.style.left = ( ( p.x * 0.5 + 0.5 ) * W ) + 'px';
				foot.el.style.top = ( ( - p.y * 0.5 + 0.5 ) * H ) + 'px';

			}

		}

	}

	// Scan cone: apex at the front of the robot, fanning forward along its
	// facing (front-feet minus rear-feet). Rebuilds the polygon, the gradient
	// axis and a sweeping bar (apex -> far) every frame.
	function updateCone( dt, W, H ) {

		if ( ! fxConeSvg ) return;

		for ( const f of fxFeet ) if ( ! f.node ) f.node = scene.getObjectByName( f.id );
		if ( ! robotBase || fxFeet.some( ( f ) => ! f.node ) ) { fxConeSvg.classList.add( 'fx-off' ); return; }

		fxFeet[ 0 ].node.getWorldPosition( _cA ); // FL
		fxFeet[ 1 ].node.getWorldPosition( _cB ); // FR
		fxFeet[ 2 ].node.getWorldPosition( _cC ); // RL
		fxFeet[ 3 ].node.getWorldPosition( _cD ); // RR
		_cFwd.copy( _cA ).add( _cB ).sub( _cC ).sub( _cD ).multiplyScalar( 0.5 ).setY( 0 ); // front mid - rear mid
		if ( _cFwd.lengthSq() < 1e-6 ) { fxConeSvg.classList.add( 'fx-off' ); return; }
		_cFwd.normalize();

		robotBase.getWorldPosition( _cA ); // reuse as base
		_cApex.copy( _cA ).addScaledVector( _cFwd, 0.3 ); _cApex.y += 0.07; // at the front sensor/camera (nose), not mid-body
		_cFar.copy( _cApex ).addScaledVector( _cFwd, 0.8 );

		_cApex.project( camera );
		_cFar.project( camera );
		if ( _cApex.z > 1 || _cFar.z > 1 ) { fxConeSvg.classList.add( 'fx-off' ); return; }

		const ax = ( _cApex.x * 0.5 + 0.5 ) * W, ay = ( - _cApex.y * 0.5 + 0.5 ) * H;
		const fxp = ( _cFar.x * 0.5 + 0.5 ) * W, fyp = ( - _cFar.y * 0.5 + 0.5 ) * H;
		let dx = fxp - ax, dy = fyp - ay;
		const L = Math.hypot( dx, dy );
		if ( L < 2 ) { fxConeSvg.classList.add( 'fx-off' ); return; }
		dx /= L; dy /= L;
		const nx = - dy, ny = dx, halfW = L * 0.42; // perpendicular + cone half-width
		const c1x = fxp + nx * halfW, c1y = fyp + ny * halfW;
		const c2x = fxp - nx * halfW, c2y = fyp - ny * halfW;

		fxConeSvg.classList.remove( 'fx-off' );
		// Fade with view angle: when the view direction aligns with the cone's
		// forward axis (looking head-on / from behind), a flat cone reads badly,
		// so fade it out; full opacity when the view is side-on.
		const align = Math.abs( _cFwd.dot( camera.getWorldDirection( _cB ) ) );
		fxConeSvg.style.opacity = THREE.MathUtils.clamp( ( 1 - align ) / 0.35, 0, 1 ).toFixed( 3 );
		fxConeSvg.setAttribute( 'viewBox', `0 0 ${ W } ${ H }` );
		fxConeFill.setAttribute( 'points', `${ ax },${ ay } ${ c1x },${ c1y } ${ c2x },${ c2y }` );
		fxConeGrad.setAttribute( 'x1', ax ); fxConeGrad.setAttribute( 'y1', ay );
		fxConeGrad.setAttribute( 'x2', fxp ); fxConeGrad.setAttribute( 'y2', fyp );

		fxConeT += dt;
		const s = 0.2 + 0.78 * ( 0.5 + 0.5 * Math.sin( fxConeT * 2.4 ) ); // sweep phase
		const spx = ax + dx * L * s, spy = ay + dy * L * s, sw = halfW * s;
		fxConeSweep.setAttribute( 'x1', spx + nx * sw ); fxConeSweep.setAttribute( 'y1', spy + ny * sw );
		fxConeSweep.setAttribute( 'x2', spx - nx * sw ); fxConeSweep.setAttribute( 'y2', spy - ny * sw );

	}

	// -------------------------------------------------------------------
	// Scene / animation state
	// -------------------------------------------------------------------
	let robotReady = false, chapterIdx = 0, entries = [];
	let robotBase = null, stairsNode = null, handrailsNode = null, plinth = null, mixer = null;
	let followAction = null, climbAction = null;
	const baseP0 = new THREE.Vector3();
	const baseQ0 = new THREE.Quaternion();
	let currentMotion = 'spin';

	const WALK_TIMESCALE = 0.6; // slow the flat-ground gait down (per feedback)

	let robotTarget = new THREE.Vector3( 0, 0.35, 0 );
	let robotDist = 2.2;
	let camTransition = null;
	// Climb follow-cam (right-side, tracks the robot up the stairs).
	let climbFollow = false;
	const climbDir = new THREE.Vector3( 0.22, 0.4, 1 ).normalize();
	let climbDist = 3;
	const _clock = new THREE.Clock();
	const _tmpVec = new THREE.Vector3(), _tmpTarget = new THREE.Vector3(), _tmpDesired = new THREE.Vector3(), _curBase = new THREE.Vector3();

	function fitDist( maxDim, factor ) {

		return ( maxDim * factor ) / Math.tan( ( camera.fov * Math.PI ) / 360 );

	}

	// -------------------------------------------------------------------
	// Load robot.glb (full baked hierarchy)
	// -------------------------------------------------------------------
	new GLTFLoader().load(
		'./models/robot.glb',
		( gltf ) => {

			const root = gltf.scene || gltf.scenes[ 0 ];
			scene.add( root );

			robotBase = root.getObjectByName( 'robot_base' );
			stairsNode = root.getObjectByName( 'stairs' );
			handrailsNode = root.getObjectByName( 'handrails' ); // null in older (merged) glbs
			const groundNode = root.getObjectByName( 'ground' );
			const patientRoot = root.getObjectByName( 'patient_root' );
			const patientAnchor = root.getObjectByName( 'patient_human_anchor' );
			if ( ! robotBase ) { console.error( '[hero] robot_base not found — hero skipped' ); return; }

			root.traverse( ( n ) => {

				if ( ! n.isMesh ) return;
				n.material = tintFor( n );
				n.castShadow = true;
				n.receiveShadow = ( n.name === 'stairs' );

			} );

			if ( groundNode ) groundNode.visible = false;
			if ( patientRoot ) patientRoot.visible = false;
			if ( patientAnchor ) patientAnchor.visible = false;

			const clips = gltf.animations || [];
			const followClip = clips.find( ( c ) => c.name === 'follow' ) || clips[ 0 ];
			const climbClip = clips.find( ( c ) => c.name === 'climb' ) || clips[ 1 ] || clips[ 0 ];
			mixer = new THREE.AnimationMixer( root );
			followAction = mixer.clipAction( followClip );
			climbAction = mixer.clipAction( climbClip );
			for ( const a of [ followAction, climbAction ] ) { a.setLoop( THREE.LoopRepeat ); a.play(); a.paused = true; a.weight = 0; }

			followAction.time = 0; followAction.weight = 1; mixer.update( 0 );
			baseP0.copy( robotBase.position );
			baseQ0.copy( robotBase.quaternion );

			const rbox = new THREE.Box3().setFromObject( robotBase );
			const rSize = rbox.getSize( new THREE.Vector3() );
			robotTarget = rbox.getCenter( new THREE.Vector3() );
			const robotMaxDim = Math.max( rSize.x, rSize.y, rSize.z ) || 1;
			robotDist = fitDist( robotMaxDim, 1.25 );
			climbDist = robotDist * 1.5; // frames the robot large enough that the leader arrows read on the stairs, without the old too-tight crop

			const feetY = rbox.min.y;
			const footprint = Math.max( rSize.x, rSize.z ) * 0.62 + 0.12;
			plinth = new THREE.Mesh(
				new THREE.CylinderGeometry( footprint, footprint * 1.03, 0.12, 72 ),
				plinthMaterial,
			);
			plinth.position.set( robotTarget.x, feetY - 0.06, robotTarget.z );
			plinth.receiveShadow = true;
			scene.add( plinth );

			robotReady = true;
			buildDots();
			goToChapter( 0, { immediate: true } );

		},
		undefined,
		( error ) => console.error( '[hero] failed to load ./models/robot.glb:', error ),
	);

	// -------------------------------------------------------------------
	// Chapter dots + nav
	// -------------------------------------------------------------------
	function buildDots() {

		dotsEl.innerHTML = '';
		CHAPTERS.forEach( ( ch, i ) => {

			const d = document.createElement( 'button' );
			d.type = 'button'; d.className = 'hero-dot'; d.setAttribute( 'role', 'tab' ); d.setAttribute( 'aria-label', ch.title );
			d.addEventListener( 'click', () => goToChapter( i ) );
			dotsEl.appendChild( d );

		} );

	}

	function syncDots() {

		[ ...dotsEl.children ].forEach( ( d, i ) => d.classList.toggle( 'active', i === chapterIdx ) );

	}

	document.getElementById( 'hero-prev' ).addEventListener( 'click', () => goToChapter( chapterIdx - 1 ) );
	document.getElementById( 'hero-next' ).addEventListener( 'click', () => goToChapter( chapterIdx + 1 ) );

	// -------------------------------------------------------------------
	// Leader callouts (left + right gutters) — build / draw
	// -------------------------------------------------------------------
	function clearEntries() {

		calloutsEl.innerHTML = '';
		while ( leadersSvg.firstChild ) leadersSvg.removeChild( leadersSvg.firstChild );
		entries = [];

	}

	function linspace( a, b, n ) {

		if ( n <= 1 ) return [ ( a + b ) / 2 ];
		const out = [];
		for ( let i = 0; i < n; i ++ ) out.push( a + ( i * ( b - a ) ) / ( n - 1 ) );
		return out;

	}

	function buildEntries( chapter ) {

		const resolved = chapter.points
			.map( ( pt ) => ( { pt, node: scene.getObjectByName( pt.node ) } ) )
			.filter( ( e ) => e.node );
		const left = resolved.filter( ( e ) => e.pt.side !== 'right' );
		const right = resolved.filter( ( e ) => e.pt.side === 'right' );
		// Bottom bound kept well clear of the bottom-third #hero-body caption
		// (which can grow to several lines on the longer chapters) so leader
		// callouts never sit underneath it.
		const leftTops = linspace( 20, 62, left.length );
		const rightTops = linspace( 20, 62, right.length );
		let li = 0, ri = 0;

		for ( const { pt, node } of resolved ) {

			const isRight = pt.side === 'right';
			const card = document.createElement( 'div' );
			card.className = `hero-callout ${ isRight ? 'side-right' : 'side-left' }`;
			card.style.top = `${ isRight ? rightTops[ ri ++ ] : leftTops[ li ++ ] }%`;
			card.innerHTML = `<span class="hero-callout-label">${ pt.label }</span>` +
				( pt.sub ? `<span class="hero-callout-sub">${ pt.sub }</span>` : '' );
			calloutsEl.appendChild( card );

			const line = document.createElementNS( SVG_NS, 'polyline' );
			line.setAttribute( 'class', 'hero-leader' ); line.setAttribute( 'pathLength', '1' );
			leadersSvg.appendChild( line );

			const dot = document.createElementNS( SVG_NS, 'circle' );
			dot.setAttribute( 'class', 'hero-leader-dot' ); dot.setAttribute( 'r', '3' );
			leadersSvg.appendChild( dot );

			const anchor = document.createElementNS( SVG_NS, 'circle' );
			anchor.setAttribute( 'class', 'hero-leader-anchor' ); anchor.setAttribute( 'r', '2.4' );
			leadersSvg.appendChild( anchor );

			entries.push( { pt, node, card, line, dot, anchor, isRight } );

		}

	}

	function goToChapter( idx, { immediate = false } = {} ) {

		if ( ! robotReady ) return;
		const n = CHAPTERS.length;
		chapterIdx = ( ( idx % n ) + n ) % n;
		const chapter = CHAPTERS[ chapterIdx ];
		currentMotion = chapter.motion;

		const applyContent = () => {

			chapterIndexEl.textContent = `${ String( chapterIdx + 1 ).padStart( 2, '0' ) } / ${ String( n ).padStart( 2, '0' ) }`;
			chapterTitleEl.textContent = chapter.title;
			bodyEl.textContent = chapter.body;
			diagramEl.innerHTML = DIAGRAMS[ chapter.id ] || '';
			if ( chapter.clip ) {

				clipFigureEl.style.display = '';
				clipBadgeEl.textContent = chapter.clip.badge;
				clipCapEl.textContent = chapter.clip.cap;
				setClip( chapter.clip.src );

			} else {

				clipFigureEl.style.display = 'none';
				stopClip();

			}
			syncDots();
			applyMotion( chapter );

		};

		const build = () => {

			clearEntries();
			applyContent();
			buildEntries( chapter );
			updateLeaders();
			requestAnimationFrame( () => entries.forEach( ( e ) => e.line.classList.add( 'drawn' ) ) );
			heroInner.classList.remove( 'switching' );

		};

		if ( immediate ) build();
		else { heroInner.classList.add( 'switching' ); setTimeout( build, 230 ); }

	}

	// -------------------------------------------------------------------
	// Motion: clip selection + plinth/stairs visibility + camera vantage.
	// -------------------------------------------------------------------
	function applyMotion( chapter ) {

		const motion = chapter.motion;
		climbFollow = ( motion === 'climb' );

		currentFx = chapter.fx || null;
		if ( fxEl ) {

			fxEl.classList.toggle( 'fx-mode-scan', currentFx === 'scan' );
			fxEl.classList.toggle( 'fx-mode-feel', currentFx === 'feel' );

		}

		if ( motion === 'climb' ) {

			climbAction.weight = 1; climbAction.paused = false; climbAction.time = 0; climbAction.timeScale = 1;
			followAction.weight = 0; followAction.paused = true;

		} else {

			followAction.weight = 1; followAction.paused = ( motion === 'spin' ); followAction.time = 0;
			followAction.timeScale = ( motion === 'walk' ) ? WALK_TIMESCALE : 1;
			climbAction.weight = 0; climbAction.paused = true;
			mixer.update( 0 );
			pinBase();

		}

		if ( plinth ) plinth.visible = ( motion !== 'climb' );
		if ( stairsNode ) stairsNode.visible = ( motion === 'climb' );
		if ( handrailsNode ) handrailsNode.visible = ( motion === 'climb' );

		if ( motion === 'climb' ) {

			// Follow-cam handles framing each frame; nothing to transition here.
			camTransition = null;
			controls.autoRotate = false;
			return;

		}

		let dir, dist, autoRotate = false;
		if ( motion === 'spin' ) {

			dir = new THREE.Vector3( 0.62, 0.42, 1 ).normalize();
			dist = robotDist; autoRotate = true;

		} else { // walk — right-side 3/4 so the gait reads

			dir = new THREE.Vector3( 0.32, 0.32, 1 ).normalize();
			dist = robotDist * 1.35; // pulled back so the leader callouts aren't cramped (was 1.05, too tight)

		}

		const target = robotTarget.clone();
		const pos = target.clone().addScaledVector( dir, dist );
		camTransition = { p0: camera.position.clone(), t0: controls.target.clone(), p1: pos, t1: target, t: 0, dur: 0.7, autoRotate };
		controls.autoRotate = false;

	}

	// Walk in place with a robotic body bob: pin the FORWARD/lateral drift and
	// facing to the t=0 pose, but KEEP the clip's vertical (z) motion so the
	// body bobs up and down as it steps. Spin is fully static.
	function pinBase() {

		if ( currentMotion === 'spin' ) {

			robotBase.position.copy( baseP0 ); robotBase.quaternion.copy( baseQ0 );

		} else if ( currentMotion === 'walk' ) {

			robotBase.position.x = baseP0.x; robotBase.position.y = baseP0.y; // keep z (bob)
			robotBase.quaternion.copy( baseQ0 );

		}

	}

	// -------------------------------------------------------------------
	// Leader lines (per frame)
	// -------------------------------------------------------------------
	function updateLeaders() {

		if ( entries.length === 0 ) return;
		const sr = host.getBoundingClientRect();
		const W = sr.width, H = sr.height;
		const narrow = window.innerWidth <= NARROW_PX;
		leadersSvg.setAttribute( 'viewBox', `0 0 ${ W } ${ H }` );

		for ( const e of entries ) {

			if ( narrow ) { e.line.style.display = 'none'; e.dot.style.display = 'none'; e.anchor.style.display = 'none'; continue; }

			e.node.getWorldPosition( _tmpVec );
			const p = _tmpVec.project( camera );
			const hidden = p.z > 1 || p.z < -1 || p.x < -1 || p.x > 1 || p.y < -1 || p.y > 1;
			if ( hidden ) { e.line.style.display = 'none'; e.dot.style.display = 'none'; e.anchor.style.display = 'none'; continue; }

			e.line.style.display = ''; e.dot.style.display = ''; e.anchor.style.display = '';
			const sx = ( p.x * 0.5 + 0.5 ) * W;
			const sy = ( - p.y * 0.5 + 0.5 ) * H;

			const cr = e.card.getBoundingClientRect();
			const ax = ( e.isRight ? cr.left : cr.right ) - sr.left;
			const ay = cr.top + cr.height / 2 - sr.top;
			const elbowX = ax + ( e.isRight ? 22 : -22 );

			e.line.setAttribute( 'points', `${ sx },${ sy } ${ elbowX },${ ay } ${ ax },${ ay }` );
			e.dot.setAttribute( 'cx', String( sx ) ); e.dot.setAttribute( 'cy', String( sy ) );
			e.anchor.setAttribute( 'cx', String( ax ) ); e.anchor.setAttribute( 'cy', String( ay ) );

		}

	}

	// -------------------------------------------------------------------
	// Zoom controls
	// -------------------------------------------------------------------
	function setDist( d ) {

		const dir = camera.position.clone().sub( controls.target );
		const clamped = THREE.MathUtils.clamp( d, robotDist * 0.55, robotDist * 2.4 );
		camera.position.copy( controls.target ).addScaledVector( dir.normalize(), clamped );
		controls.update();

	}
	function zoomBy( factor ) {

		if ( climbFollow ) { climbDist = THREE.MathUtils.clamp( climbDist * factor, robotDist * 0.7, robotDist * 3 ); return; }
		camTransition = null;
		setDist( camera.position.distanceTo( controls.target ) * factor );

	}
	document.getElementById( 'hero-zoom-in' ).addEventListener( 'click', () => zoomBy( 0.82 ) );
	document.getElementById( 'hero-zoom-out' ).addEventListener( 'click', () => zoomBy( 1.22 ) );
	document.getElementById( 'hero-zoom-reset' ).addEventListener( 'click', () => { climbDist = robotDist * 1.5; applyMotion( CHAPTERS[ chapterIdx ] ); } );

	// -------------------------------------------------------------------
	// Theme sync
	// -------------------------------------------------------------------
	new MutationObserver( () => {

		let name = document.documentElement.getAttribute( 'data-theme' ) || 'light';
		if ( name !== 'light' && name !== 'dark' ) name = 'light';
		if ( name === themeName ) return;
		themeName = name; palette = PALETTES[ name ]; colors = heroColors( palette );
		scene.background = makeStudioBackdrop( colors.bg );
		edgesPass.setInkColor( colors.ink );
		robotMaterial.color.set( colors.robot );
		oxygenTankMaterial.color.set( colors.tank );
		cradleRailsMaterial.color.set( colors.cradle );
		stairsMaterial.color.set( colors.stairs );
		handrailMaterial.color.set( colors.rail );

	} ).observe( document.documentElement, { attributes: true, attributeFilter: [ 'data-theme' ] } );

	// -------------------------------------------------------------------
	// Resize
	// -------------------------------------------------------------------
	function handleResize() {

		const w = host.clientWidth, h = host.clientHeight;
		if ( w === 0 || h === 0 ) return;
		const pr = Math.min( window.devicePixelRatio || 1, 2 );
		renderer.setPixelRatio( pr ); renderer.setSize( w, h );
		composer.setPixelRatio( pr ); composer.setSize( w, h );
		camera.aspect = w / h; camera.updateProjectionMatrix();

	}
	new ResizeObserver( handleResize ).observe( host );
	window.addEventListener( 'resize', handleResize );
	handleResize();

	// -------------------------------------------------------------------
	// Render loop
	// -------------------------------------------------------------------
	let visible = true;
	new IntersectionObserver( ( e ) => { visible = e[ 0 ].isIntersecting; }, { threshold: 0.02 } )
		.observe( document.getElementById( 'hero-section' ) );

	const easeInOut = ( k ) => ( k < 0.5 ? 2 * k * k : 1 - Math.pow( -2 * k + 2, 2 ) / 2 );

	function animate() {

		requestAnimationFrame( animate );
		if ( ! visible ) return;
		const dt = Math.min( _clock.getDelta(), 0.05 );

		if ( robotReady && mixer ) {

			mixer.update( dt );
			if ( currentMotion !== 'climb' ) pinBase();

		}

		if ( camTransition ) {

			camTransition.t += dt / camTransition.dur;
			const k = easeInOut( Math.min( 1, camTransition.t ) );
			camera.position.lerpVectors( camTransition.p0, camTransition.p1, k );
			_tmpVec.lerpVectors( camTransition.t0, camTransition.t1, k );
			controls.target.copy( _tmpVec ); camera.lookAt( _tmpVec );
			if ( camTransition.t >= 1 ) { controls.autoRotate = camTransition.autoRotate; camTransition = null; }

		} else if ( climbFollow && robotBase ) {

			// Right-side follow-cam: keep a fixed offset from the climbing robot.
			robotBase.getWorldPosition( _curBase );
			_tmpTarget.copy( _curBase ); _tmpTarget.y += 0.12;
			_tmpDesired.copy( _tmpTarget ).addScaledVector( climbDir, climbDist );
			camera.position.lerp( _tmpDesired, 0.1 );
			controls.target.lerp( _tmpTarget, 0.1 );
			camera.lookAt( controls.target );

		} else {

			controls.update();

		}

		updateLeaders();
		updateFx( dt );
		composer.render();

	}
	animate();

	window.__hero = {
		scene, camera, renderer, composer, controls,
		materials: { robotMaterial, oxygenTankMaterial, cradleRailsMaterial, stairsMaterial, plinthMaterial },
		goToChapter, get chapterIdx() { return chapterIdx; }, get motion() { return currentMotion; },
		get baseZ() { return robotBase ? robotBase.getWorldPosition( new THREE.Vector3() ) : null; },
		get robotTarget() { return robotTarget; }, get robotDist() { return robotDist; },
		_robotAABB() { _fxBox.setFromObject( robotBase ); return { min: _fxBox.min.toArray(), max: _fxBox.max.toArray() }; },
		_tick( dt ) {

			if ( ! mixer ) return;
			mixer.update( dt );
			if ( currentMotion !== 'climb' ) pinBase();

		},
		_fx( dt ) { updateFx( dt || 0.016 ); }, // manual FX pump (rAF is paused when tab is hidden)
		_actions() {

			return {
				followW: followAction && followAction.weight, followPaused: followAction && followAction.paused, followTS: followAction && followAction.timeScale, followT: followAction && +followAction.time.toFixed( 3 ),
				climbW: climbAction && climbAction.weight, climbPaused: climbAction && climbAction.paused, climbT: climbAction && +climbAction.time.toFixed( 3 ),
				climbFollow, climbDist: +climbDist.toFixed( 2 ),
			};

		},
	};

}

// ===========================================================================
// Studio backdrop: soft radial gradient (lighter centre -> darker edge).
// ===========================================================================

function makeStudioBackdrop( bgInt ) {

	const size = 512;
	const canvas = document.createElement( 'canvas' );
	canvas.width = canvas.height = size;
	const ctx = canvas.getContext( '2d' );
	const base = new THREE.Color( bgInt );
	const center = base.clone().offsetHSL( 0, 0, 0.05 );
	const edge = base.clone().offsetHSL( 0, 0, -0.07 );
	const grad = ctx.createRadialGradient( size * 0.5, size * 0.42, size * 0.06, size * 0.5, size * 0.5, size * 0.72 );
	grad.addColorStop( 0, '#' + center.getHexString() );
	grad.addColorStop( 1, '#' + edge.getHexString() );
	ctx.fillStyle = grad; ctx.fillRect( 0, 0, size, size );
	const tex = new THREE.CanvasTexture( canvas );
	tex.colorSpace = THREE.SRGBColorSpace;
	return tex;

}
