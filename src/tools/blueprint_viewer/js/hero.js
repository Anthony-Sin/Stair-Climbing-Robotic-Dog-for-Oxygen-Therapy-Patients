// hero.js
//
// The shared 3D stage for the "Solution & technology" scroll sections. It is
// PINNED (its host #hero-stage is sticky) on the left while the three step-
// sections (architecture / walking / blind-RL climb) scroll past on the right;
// js/deck.js calls showPolicy() as each step becomes active and setStageVisible()
// as the Solution block enters/leaves the viewport. The robot spins / walks in
// place / climbs the real staircase per policy, with leader-line callouts and a
// per-policy FX overlay (walk = camera-scan cone + YOLO HUD; climb = per-foot
// proprioceptive "feeling" waves).
//
// Separate three.js scene from the demo viewer (js/main.js). Reuses
// ./models/robot.glb as an ASSET (full baked hierarchy) so it can drive the
// real follow/climb clips and show the real stairs. Shares palette.js + the
// BlueprintEdgesPass ink outlines, read-only.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import { PALETTES } from './palette.js';
import { BlueprintEdgesPass, NO_OUTLINE_LAYER } from './BlueprintEdgesPass.js';
import { POLICIES } from './content.js';

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
	const renderer = new THREE.WebGLRenderer( { antialias: true, alpha: true } );
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
	// Realistic robot materials + studio IBL env
	// -------------------------------------------------------------------
	// The robot is REAL PBR (not toon) so it matches the interactive viewer -- a
	// deliberate contrast against the toon set below. Only the robot reads
	// scene.environment (the toon MeshToonMaterials ignore it).
	const _pmrem = new THREE.PMREMGenerator( renderer );
	scene.environment = _pmrem.fromEquirectangular( makeStudioEnvTexture() ).texture;
	_pmrem.dispose();

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

	function makeRobotRealisticMaterial( colorHex, opts = {} ) {

		return new THREE.MeshStandardMaterial( {
			color: colorHex,
			metalness: opts.metalness ?? 0.4,
			roughness: opts.roughness ?? 0.45,
			envMapIntensity: opts.envMapIntensity ?? 1.1,
			vertexColors: opts.vertexColors ?? false,
		} );

	}

	// -------------------------------------------------------------------
	// Toon materials  (the set: stairs, rails, tank, cradle, plinth)
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

	// Robot: realistic silver PBR shell (theme-independent, intrinsic hardware
	// colors) + per-part COLOR_0 via the white vertexColors material (see TINT).
	// robotMaterial = silver body/hips; robotBaseMaterial = base/thighs/calves
	// (baked black head/words/thigh-housing/knee/foot-pad); robotBlackMaterial = feet.
	const robotMaterial = makeRobotRealisticMaterial( 0xc0c0c0, { metalness: 0.2, roughness: 0.5, envMapIntensity: 1.0 } );
	const robotBlackMaterial = makeRobotRealisticMaterial( 0x232629, { metalness: 0.0, roughness: 0.85, envMapIntensity: 0.5 } );
	const robotBaseMaterial = makeRobotRealisticMaterial( 0xffffff, { metalness: 0.2, roughness: 0.5, envMapIntensity: 1.0, vertexColors: true } );
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
		handrails: () => handrailMaterial,
		// Robot per-part (matches the viewer): base + thighs + calves carry a baked
		// COLOR_0 (silver shell + black head/words/thigh-housing/knee/foot-pad) painted
		// via the white vertexColors material; hips silver; foot balls flat black.
		robot_base: () => robotBaseMaterial,
		FL_hip: () => robotMaterial, FR_hip: () => robotMaterial, RL_hip: () => robotMaterial, RR_hip: () => robotMaterial,
		FL_thigh: () => robotBaseMaterial, FR_thigh: () => robotBaseMaterial, RL_thigh: () => robotBaseMaterial, RR_thigh: () => robotBaseMaterial,
		FL_calf: () => robotBaseMaterial, FR_calf: () => robotBaseMaterial, RL_calf: () => robotBaseMaterial, RR_calf: () => robotBaseMaterial,
		FL_foot: () => robotBlackMaterial, FR_foot: () => robotBlackMaterial, RL_foot: () => robotBlackMaterial, RR_foot: () => robotBlackMaterial,
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
		noOutlineLayer: NO_OUTLINE_LAYER, // robot subtree -> no ink outline (Isaac-viewport look), like the viewer
	} );
	composer.addPass( edgesPass );
	composer.addPass( new OutputPass() );

	// -------------------------------------------------------------------
	// Overlay DOM refs
	// -------------------------------------------------------------------
	const leadersSvg = document.getElementById( 'hero-leaders' );
	const calloutsEl = document.getElementById( 'hero-callouts' );
	const fxEl = document.getElementById( 'hero-fx' );

	// -------------------------------------------------------------------
	// Per-policy stage FX overlay (built once, toggled via a class on #hero-fx
	// in applyMotion, positioned each frame in updateFx).
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
			'<svg class="fx-cone" aria-hidden="true">' +
				'<defs><linearGradient id="fxConeGrad" gradientUnits="userSpaceOnUse">' +
					'<stop class="fx-cone-s0" offset="0"/><stop class="fx-cone-s1" offset="1"/>' +
				'</linearGradient></defs>' +
				'<polygon class="fx-cone-fill"/><line class="fx-cone-sweep"/>' +
			'</svg>' +
			'<div class="fx-hud fx-hud-yolo"><span class="fx-hud-dot"></span>YOLO-World<span class="fx-hud-conf">person 0.00</span></div>' +
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

			fxConfAccum += dt;
			if ( fxConfAccum > 0.55 ) { fxConfAccum = 0; fxConfTarget = 0.9 + Math.random() * 0.09; }
			fxConf += ( fxConfTarget - fxConf ) * Math.min( 1, dt * 4 );
			if ( fxConfEl ) fxConfEl.textContent = 'person ' + fxConf.toFixed( 2 );

		} else if ( currentFx === 'feel' ) {

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

	function updateCone( dt, W, H ) {

		if ( ! fxConeSvg ) return;

		for ( const f of fxFeet ) if ( ! f.node ) f.node = scene.getObjectByName( f.id );
		if ( ! robotBase || fxFeet.some( ( f ) => ! f.node ) ) { fxConeSvg.classList.add( 'fx-off' ); return; }

		fxFeet[ 0 ].node.getWorldPosition( _cA );
		fxFeet[ 1 ].node.getWorldPosition( _cB );
		fxFeet[ 2 ].node.getWorldPosition( _cC );
		fxFeet[ 3 ].node.getWorldPosition( _cD );
		_cFwd.copy( _cA ).add( _cB ).sub( _cC ).sub( _cD ).multiplyScalar( 0.5 ).setY( 0 );
		if ( _cFwd.lengthSq() < 1e-6 ) { fxConeSvg.classList.add( 'fx-off' ); return; }
		_cFwd.normalize();

		robotBase.getWorldPosition( _cA );
		// Apex sits at the front camera. Lowered (was +0.07 above the base) so the
		// scan cone reads as coming FROM the head camera, not floating above it.
		_cApex.copy( _cA ).addScaledVector( _cFwd, 0.3 ); _cApex.y -= 0.04;
		_cFar.copy( _cApex ).addScaledVector( _cFwd, 0.8 ); _cFar.y -= 0.03;

		_cApex.project( camera );
		_cFar.project( camera );
		if ( _cApex.z > 1 || _cFar.z > 1 ) { fxConeSvg.classList.add( 'fx-off' ); return; }

		const ax = ( _cApex.x * 0.5 + 0.5 ) * W, ay = ( - _cApex.y * 0.5 + 0.5 ) * H;
		const fxp = ( _cFar.x * 0.5 + 0.5 ) * W, fyp = ( - _cFar.y * 0.5 + 0.5 ) * H;
		let dx = fxp - ax, dy = fyp - ay;
		const L = Math.hypot( dx, dy );
		if ( L < 2 ) { fxConeSvg.classList.add( 'fx-off' ); return; }
		dx /= L; dy /= L;
		const nx = - dy, ny = dx, halfW = L * 0.42;
		const c1x = fxp + nx * halfW, c1y = fyp + ny * halfW;
		const c2x = fxp - nx * halfW, c2y = fyp - ny * halfW;

		fxConeSvg.classList.remove( 'fx-off' );
		const align = Math.abs( _cFwd.dot( camera.getWorldDirection( _cB ) ) );
		fxConeSvg.style.opacity = THREE.MathUtils.clamp( ( 1 - align ) / 0.35, 0, 1 ).toFixed( 3 );
		fxConeSvg.setAttribute( 'viewBox', `0 0 ${ W } ${ H }` );
		fxConeFill.setAttribute( 'points', `${ ax },${ ay } ${ c1x },${ c1y } ${ c2x },${ c2y }` );
		fxConeGrad.setAttribute( 'x1', ax ); fxConeGrad.setAttribute( 'y1', ay );
		fxConeGrad.setAttribute( 'x2', fxp ); fxConeGrad.setAttribute( 'y2', fyp );

		fxConeT += dt;
		const s = 0.2 + 0.78 * ( 0.5 + 0.5 * Math.sin( fxConeT * 2.4 ) );
		const spx = ax + dx * L * s, spy = ay + dy * L * s, sw = halfW * s;
		fxConeSweep.setAttribute( 'x1', spx + nx * sw ); fxConeSweep.setAttribute( 'y1', spy + ny * sw );
		fxConeSweep.setAttribute( 'x2', spx - nx * sw ); fxConeSweep.setAttribute( 'y2', spy - ny * sw );

	}

	// -------------------------------------------------------------------
	// Scene / animation state
	// -------------------------------------------------------------------
	let robotReady = false, entries = [];
	let stageVisible = false;      // driven by deck.js via setStageVisible()
	let currentPolicyId = 'architecture';
	let robotBase = null, stairsNode = null, handrailsNode = null, plinth = null, mixer = null;
	let followAction = null, climbAction = null;
	const baseP0 = new THREE.Vector3();
	const baseQ0 = new THREE.Quaternion();
	let currentMotion = 'spin';

	const WALK_TIMESCALE = 0.6;

	let robotTarget = new THREE.Vector3( 0, 0.35, 0 );
	let robotDist = 2.2;
	let camTransition = null;
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
			handrailsNode = root.getObjectByName( 'handrails' );
			const groundNode = root.getObjectByName( 'ground' );
			const patientRoot = root.getObjectByName( 'patient_root' );
			const patientAnchor = root.getObjectByName( 'patient_human_anchor' );
			if ( ! robotBase ) { console.error( '[hero] robot_base not found — hero stage skipped' ); return; }

			root.traverse( ( n ) => {

				if ( ! n.isMesh ) return;
				n.material = tintFor( n );
				n.castShadow = true;
				n.receiveShadow = ( n.name === 'stairs' );

			} );

			// Robot subtree -> NO_OUTLINE layer: the realistic robot renders clean (no
			// ink outlines), like the viewer. The toon set keeps its outlines.
			robotBase.traverse( ( o ) => o.layers.enable( NO_OUTLINE_LAYER ) );

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
			climbDist = robotDist * 1.5;

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
			// Prime the currently-requested policy so the stage is ready the moment
			// it scrolls into view (deck.js re-calls showPolicy as steps activate).
			showPolicy( currentPolicyId );

		},
		undefined,
		( error ) => console.error( '[hero] failed to load ./models/robot.glb:', error ),
	);

	// -------------------------------------------------------------------
	// Leader callouts — build / draw
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

	function buildEntries( points ) {

		if ( ! points ) return;
		const resolved = points
			.map( ( pt ) => ( { pt, node: scene.getObjectByName( pt.node ) } ) )
			.filter( ( e ) => e.node );
		const left = resolved.filter( ( e ) => e.pt.side !== 'right' );
		const right = resolved.filter( ( e ) => e.pt.side === 'right' );
		const leftTops = linspace( 18, 78, left.length );
		const rightTops = linspace( 18, 78, right.length );
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

	// -------------------------------------------------------------------
	// PUBLIC: switch policy (called by deck.js as each Solution step activates)
	// -------------------------------------------------------------------
	function showPolicy( name ) {

		const policy = POLICIES[ name ];
		if ( ! policy ) return;
		currentPolicyId = name;
		currentMotion = policy.motion;

		clearEntries();
		if ( ! robotReady ) return;

		applyMotion( policy );
		buildEntries( policy.points );
		updateLeaders();
		requestAnimationFrame( () => entries.forEach( ( e ) => e.line.classList.add( 'drawn' ) ) );

	}

	// -------------------------------------------------------------------
	// Motion: clip selection + plinth/stairs visibility + camera vantage.
	// -------------------------------------------------------------------
	function applyMotion( policy ) {

		const motion = policy.motion;
		currentMotion = motion;
		climbFollow = ( motion === 'climb' );

		currentFx = policy.fx || null;
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

			camTransition = null;
			controls.autoRotate = false;
			return;

		}

		let dir, dist, autoRotate = false;
		if ( motion === 'spin' ) {

			dir = new THREE.Vector3( 0.62, 0.42, 1 ).normalize();
			dist = robotDist; autoRotate = true;

		} else {

			dir = new THREE.Vector3( 0.32, 0.32, 1 ).normalize();
			dist = robotDist * 1.35;

		}

		const target = robotTarget.clone();
		const pos = target.clone().addScaledVector( dir, dist );
		camTransition = { p0: camera.position.clone(), t0: controls.target.clone(), p1: pos, t1: target, t: 0, dur: 0.7, autoRotate };
		controls.autoRotate = false;

	}

	function pinBase() {

		if ( currentMotion === 'spin' ) {

			robotBase.position.copy( baseP0 ); robotBase.quaternion.copy( baseQ0 );

		} else if ( currentMotion === 'walk' ) {

			robotBase.position.x = baseP0.x; robotBase.position.y = baseP0.y;
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
	document.getElementById( 'hero-zoom-reset' ).addEventListener( 'click', () => { climbDist = robotDist * 1.5; if ( robotReady ) applyMotion( POLICIES[ currentPolicyId ] ); } );

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
		// robot is realistic PBR with intrinsic (theme-independent) colors -- no retint
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
		updateLeaders();

	}
	new ResizeObserver( handleResize ).observe( host );
	window.addEventListener( 'resize', handleResize );
	handleResize();

	// -------------------------------------------------------------------
	// Render loop — only runs while the Solution stage is on screen.
	// -------------------------------------------------------------------
	const easeInOut = ( k ) => ( k < 0.5 ? 2 * k * k : 1 - Math.pow( -2 * k + 2, 2 ) / 2 );

	function animate() {

		requestAnimationFrame( animate );
		if ( ! stageVisible ) return;
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

	function setStageVisible( v ) {

		v = !! v;
		if ( v && ! stageVisible ) _clock.getDelta(); // drop the accumulated gap so motion doesn't jump
		stageVisible = v;

	}

	window.__hero = {
		scene, camera, renderer, composer, controls,
		materials: { robotMaterial, oxygenTankMaterial, cradleRailsMaterial, stairsMaterial, plinthMaterial },
		showPolicy, setStageVisible,
		get policy() { return currentPolicyId; }, get motion() { return currentMotion; },
		get stageVisible() { return stageVisible; },
		get robotTarget() { return robotTarget; }, get robotDist() { return robotDist; },
		_robotAABB() { _fxBox.setFromObject( robotBase ); return { min: _fxBox.min.toArray(), max: _fxBox.max.toArray() }; },
		_tick( dt ) { if ( mixer ) { mixer.update( dt ); if ( currentMotion !== 'climb' ) pinBase(); } },
		_fx( dt ) { updateFx( dt || 0.016 ); },
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
