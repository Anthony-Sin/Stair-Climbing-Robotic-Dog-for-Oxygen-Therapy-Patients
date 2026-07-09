// topple.js
//
// The dramatic, HAND-AUTHORED synthetic topple for the "Engineering challenges"
// slide (#s-challenges) — the "it kept falling" story the slide talks through.
//
// It stages the REAL climb: the baked `climb` clip drives robot_base up the real
// staircase with its real churning leg motion, and then — partway up — the top-
// heavy O₂ payload wins and the whole assembly rears back over its hind feet and
// topples down the steps. The climb clip keeps playing through the rear-back (legs
// still churning, "trying to climb") and only goes limp on impact, so the failure
// reads as "lost the fight with the weight mid-climb", not a canned tip-over.
//
// Mechanics: during the climb phase the mixer owns robot_base (recorded root
// motion up the stairs). At the fall trigger we capture that world pose + a pivot
// at the hind feet, then each frame OVERRIDE robot_base with a rigid rotation
// about that pivot (see applyTopple), while still advancing the clip so the leg
// joints — children of robot_base — keep animating.
//
// Render-gated: an IntersectionObserver on the slide means the loop only draws
// while the slide is on screen (and restarts the climb-then-fall on arrival).

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

import { PALETTES } from './palette.js';
import { BlueprintEdgesPass, NO_OUTLINE_LAYER } from './BlueprintEdgesPass.js';

const host = document.getElementById( 'topple-stage' );
const section = document.getElementById( 's-challenges' );
if ( host && section ) boot( host, section );

function boot( host, section ) {

	let themeName = document.documentElement.getAttribute( 'data-theme' ) || 'light';
	if ( themeName !== 'light' && themeName !== 'dark' ) themeName = 'light';
	let palette = PALETTES[ themeName ];

	// -------------------------------------------------------------------
	// Renderer / scene / camera
	// -------------------------------------------------------------------
	const renderer = new THREE.WebGLRenderer( { antialias: true, alpha: true } );
	renderer.setPixelRatio( Math.min( window.devicePixelRatio || 1, 2 ) );
	renderer.setSize( 1, 1 );
	renderer.shadowMap.enabled = true;
	renderer.shadowMap.type = THREE.PCFSoftShadowMap;
	renderer.domElement.style.transition = 'opacity 0.35s ease';
	host.insertBefore( renderer.domElement, host.firstChild );

	const scene = new THREE.Scene();
	scene.background = null; // alpha canvas -> the DOM sheet shows through

	const camera = new THREE.PerspectiveCamera( 40, 1, 0.05, 60 );

	// -------------------------------------------------------------------
	// Lights (warm key + cool fill, one shadow) + studio IBL for the robot PBR
	// -------------------------------------------------------------------
	scene.add( new THREE.HemisphereLight( 0xffffff, 0xd8d4cc, 1.25 ) );
	const keyLight = new THREE.DirectionalLight( 0xfff2df, 1.7 );
	keyLight.position.set( 2.6, 4.4, 2.2 );
	keyLight.castShadow = true;
	keyLight.shadow.mapSize.set( 1024, 1024 );
	keyLight.shadow.camera.near = 0.5;
	keyLight.shadow.camera.far = 20;
	keyLight.shadow.camera.left = -3;
	keyLight.shadow.camera.right = 3;
	keyLight.shadow.camera.top = 3;
	keyLight.shadow.camera.bottom = -3;
	keyLight.shadow.bias = -0.0008;
	keyLight.shadow.radius = 4;
	scene.add( keyLight );
	scene.add( keyLight.target );
	const fillLight = new THREE.DirectionalLight( 0xb9d3ff, 0.4 );
	fillLight.position.set( -3, 1.8, -2 );
	scene.add( fillLight );

	const _pmrem = new THREE.PMREMGenerator( renderer );
	scene.environment = _pmrem.fromEquirectangular( makeStudioEnvTexture() ).texture;
	_pmrem.dispose();

	function makeStudioEnvTexture() {

		const w = 256, h = 128;
		const canvas = document.createElement( 'canvas' );
		canvas.width = w; canvas.height = h;
		const ctx = canvas.getContext( '2d' );
		const g = ctx.createLinearGradient( 0, 0, 0, h );
		g.addColorStop( 0, '#efeae1' ); g.addColorStop( 0.55, '#e0e3e8' ); g.addColorStop( 1, '#bcc1c9' );
		ctx.fillStyle = g; ctx.fillRect( 0, 0, w, h );
		const rg = ctx.createRadialGradient( w * 0.3, h * 0.28, 0, w * 0.3, h * 0.28, h * 0.7 );
		rg.addColorStop( 0, 'rgba(255,245,230,0.9)' ); rg.addColorStop( 1, 'rgba(255,245,230,0)' );
		ctx.fillStyle = rg; ctx.fillRect( 0, 0, w, h );
		const tex = new THREE.CanvasTexture( canvas );
		tex.mapping = THREE.EquirectangularReflectionMapping;
		tex.colorSpace = THREE.SRGBColorSpace;
		return tex;

	}

	// -------------------------------------------------------------------
	// Materials (mirror hero.js: realistic robot PBR + toon payload/stairs)
	// -------------------------------------------------------------------
	const CEL = makeToonGradientMap( [ 0.4, 0.72, 1.0 ] );
	const RIM = new THREE.Color( 0xffffff );

	function makeToonGradientMap( levels ) {

		const data = new Uint8Array( levels.length );
		for ( let i = 0; i < levels.length; i ++ ) data[ i ] = Math.round( THREE.MathUtils.clamp( levels[ i ], 0, 1 ) * 255 );
		const t = new THREE.DataTexture( data, levels.length, 1, THREE.RedFormat );
		t.minFilter = THREE.NearestFilter; t.magFilter = THREE.NearestFilter; t.generateMipmaps = false; t.needsUpdate = true;
		return t;

	}

	function makeToon( colorHex ) {

		const m = new THREE.MeshToonMaterial( { color: colorHex, gradientMap: CEL } );
		m.onBeforeCompile = ( shader ) => {

			shader.uniforms.uRimColor = { value: RIM };
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

	function makeRobotMat( colorHex, opts = {} ) {

		return new THREE.MeshStandardMaterial( {
			color: colorHex,
			metalness: opts.metalness ?? 0.4,
			roughness: opts.roughness ?? 0.45,
			envMapIntensity: opts.envMapIntensity ?? 1.0,
			vertexColors: opts.vertexColors ?? false,
		} );

	}

	const robotMaterial = makeRobotMat( 0xc0c0c0, { metalness: 0.2, roughness: 0.5 } );
	const robotBlackMaterial = makeRobotMat( 0x232629, { metalness: 0.0, roughness: 0.85, envMapIntensity: 0.5 } );
	const robotBaseMaterial = makeRobotMat( 0xffffff, { metalness: 0.2, roughness: 0.5, vertexColors: true } );
	const oxygenTankMaterial = makeToon( 0x8fc4cf ); // clean medical teal-cyan (matches palette oxygenTankColor)
	const cradleRailsMaterial = makeToon( 0x333333 );
	const stairsMaterial = makeToon( 0xc07d3c );
	const handrailMaterial = makeToon( 0x332e29 );

	const TINT = {
		oxygen_tank: () => oxygenTankMaterial,
		cradle_rails: () => cradleRailsMaterial,
		stairs: () => stairsMaterial,
		handrails: () => handrailMaterial,
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
	// Post: RenderPass -> ink edges (toon set only) -> Output
	// -------------------------------------------------------------------
	const composer = new EffectComposer( renderer );
	composer.addPass( new RenderPass( scene, camera ) );
	const edgesPass = new BlueprintEdgesPass( scene, camera, {
		inkColor: palette.inkColorGl, normalThreshold: 0.6, depthThreshold: 0.03, thickness: 1.3,
		noOutlineLayer: NO_OUTLINE_LAYER,
	} );
	composer.addPass( edgesPass );
	composer.addPass( new OutputPass() );

	// -------------------------------------------------------------------
	// Timeline (seconds). climbEnd = the mixer-driven climb; after it the payload
	// wins and the assembly rears back over its hind feet and topples.
	// -------------------------------------------------------------------
	const T = { climbEnd: 2.4, rearEnd: 3.0, fallEnd: 3.7, bounceEnd: 4.0, settleEnd: 4.9, holdEnd: 5.7, end: 6.4 };
	const REAR_ANGLE = THREE.MathUtils.degToRad( 34 );
	const DOWN_ANGLE = THREE.MathUtils.degToRad( 110 ); // past vertical -> onto its back, down the steps
	const ROLL_MAX = THREE.MathUtils.degToRad( 12 );    // slight sideways spill toward camera
	const CLIMB_SPEED = 1.7;
	const WORLD_Z = new THREE.Vector3( 0, 0, 1 ); // lateral axis: +Z rotation rears the nose (+X) up
	const WORLD_X = new THREE.Vector3( 1, 0, 0 );

	let robotBase = null, mixer = null, climbAction = null;
	let ready = false, visible = false;
	let toppleT = 0, impactFired = false, captured = false;

	const fallStartPos = new THREE.Vector3();
	const fallStartQuat = new THREE.Quaternion();
	const pivot = new THREE.Vector3();
	const parentMatrixInv = new THREE.Matrix4();
	const parentQuatInv = new THREE.Quaternion();

	const _dQ = new THREE.Quaternion();
	const _rollQ = new THREE.Quaternion();
	const _offset = new THREE.Vector3();
	const _newPos = new THREE.Vector3();
	const _newQuat = new THREE.Quaternion();
	const _localPos = new THREE.Vector3();
	const _localQuat = new THREE.Quaternion();
	const _proj = new THREE.Vector3();
	const _robotPos = new THREE.Vector3();
	const _shake = new THREE.Vector3();

	const camOffset = new THREE.Vector3();
	const camTarget = new THREE.Vector3();
	const _camGoal = new THREE.Vector3();
	let upBias = 0.25;

	const clock = new THREE.Clock();

	// Overlay chrome refs
	const hudEl = document.getElementById( 'topple-hud' );
	const replayBtn = document.getElementById( 'topple-replay' );
	const dustEl = document.createElement( 'div' );
	dustEl.id = 'topple-dust'; dustEl.setAttribute( 'aria-hidden', 'true' );
	host.appendChild( dustEl );

	// -------------------------------------------------------------------
	// Load the robot + REAL staircase (keep stairs/handrails; hide patient/ground)
	// -------------------------------------------------------------------
	new GLTFLoader().load(
		'./models/robot.glb',
		( gltf ) => {

			const root = gltf.scene || gltf.scenes[ 0 ];
			scene.add( root );

			robotBase = root.getObjectByName( 'robot_base' );
			if ( ! robotBase ) { console.error( '[topple] robot_base not found — topple stage skipped' ); return; }

			for ( const name of [ 'ground', 'patient_root', 'patient_human_anchor' ] ) {

				const n = root.getObjectByName( name );
				if ( n ) n.visible = false;

			}

			root.traverse( ( n ) => {

				if ( ! n.isMesh ) return;
				n.material = tintFor( n );
				n.castShadow = true;
				n.receiveShadow = ( n.name === 'stairs' || n.name === 'handrails' );

			} );

			robotBase.traverse( ( o ) => o.layers.enable( NO_OUTLINE_LAYER ) );

			// Real climb clip drives root motion up the stairs + churning legs.
			const clips = gltf.animations || [];
			const climbClip = clips.find( ( c ) => c.name === 'climb' ) || clips[ 1 ] || clips[ 0 ];
			if ( ! climbClip ) { console.error( '[topple] no climb clip — topple stage skipped' ); return; }
			mixer = new THREE.AnimationMixer( root );
			climbAction = mixer.clipAction( climbClip );
			climbAction.setLoop( THREE.LoopRepeat );
			climbAction.play();
			climbAction.timeScale = CLIMB_SPEED;
			climbAction.time = 0;
			mixer.update( 0 );

			scene.updateMatrixWorld( true );
			parentMatrixInv.copy( robotBase.parent.matrixWorld ).invert();
			robotBase.parent.getWorldQuaternion( parentQuatInv ); parentQuatInv.invert();

			// Frame: sit to the near/side of the robot at the climb start, looking up
			// the stairs at it, so both the dog and the steps ahead are in shot.
			const box = new THREE.Box3().setFromObject( robotBase );
			const size = box.getSize( new THREE.Vector3() );
			const maxDim = Math.max( size.x, size.y, size.z ) || 1;
			robotBase.getWorldPosition( _robotPos );
			upBias = size.y * 0.35;

			const dist = ( maxDim * 1.5 ) / Math.tan( ( camera.fov * Math.PI ) / 360 );
			camOffset.copy( new THREE.Vector3( 0.58, 0.30, 1.0 ).normalize() ).multiplyScalar( dist );
			camTarget.copy( _robotPos ); camTarget.y += upBias;
			camera.position.copy( camTarget ).add( camOffset );
			camera.lookAt( camTarget );
			camera.near = Math.max( 0.02, dist * 0.02 );
			camera.far = dist * 8;
			camera.updateProjectionMatrix();

			ready = true;
			startTopple();

		},
		undefined,
		( error ) => console.error( '[topple] failed to load ./models/robot.glb:', error ),
	);

	// -------------------------------------------------------------------
	// Topple math (angle is the rear-back rotation past the climb pose)
	// -------------------------------------------------------------------
	const easeInCubic = ( k ) => k * k * k;
	const D2R = THREE.MathUtils.degToRad;

	function toppleAngle( t ) {

		if ( t < T.climbEnd ) return 0;
		if ( t < T.rearEnd ) { const k = ( t - T.climbEnd ) / ( T.rearEnd - T.climbEnd ); return easeInCubic( k ) * REAR_ANGLE; }
		if ( t < T.fallEnd ) { const k = ( t - T.rearEnd ) / ( T.fallEnd - T.rearEnd ); return REAR_ANGLE + easeInCubic( k ) * ( DOWN_ANGLE - REAR_ANGLE ); }
		if ( t < T.bounceEnd ) { const k = ( t - T.fallEnd ) / ( T.bounceEnd - T.fallEnd ); return DOWN_ANGLE + Math.sin( k * Math.PI ) * D2R( 5 ); }
		if ( t < T.settleEnd ) { const k = ( t - T.bounceEnd ) / ( T.settleEnd - T.bounceEnd ); return DOWN_ANGLE - D2R( 2 ) + Math.sin( k * Math.PI * 3 ) * D2R( 2 ) * ( 1 - k ); }
		return DOWN_ANGLE - D2R( 2 );

	}

	function toppleRoll( t ) {

		return THREE.MathUtils.clamp( ( t - T.climbEnd ) / ( T.fallEnd - T.climbEnd ), 0, 1 ) * ROLL_MAX;

	}

	function applyTopple( t ) {

		_dQ.setFromAxisAngle( WORLD_Z, toppleAngle( t ) );
		_rollQ.setFromAxisAngle( WORLD_X, toppleRoll( t ) );
		_dQ.multiply( _rollQ );

		_offset.copy( fallStartPos ).sub( pivot ).applyQuaternion( _dQ );
		_newPos.copy( pivot ).add( _offset );
		_newQuat.copy( _dQ ).multiply( fallStartQuat );

		_localPos.copy( _newPos ).applyMatrix4( parentMatrixInv );
		_localQuat.copy( parentQuatInv ).multiply( _newQuat );

		robotBase.position.copy( _localPos );
		robotBase.quaternion.copy( _localQuat );

	}

	// Capture the climb-end world pose + hind-foot pivot the first fall frame.
	function captureFallStart() {

		robotBase.getWorldPosition( fallStartPos );
		robotBase.getWorldQuaternion( fallStartQuat );
		const box = new THREE.Box3().setFromObject( robotBase );
		const center = box.getCenter( new THREE.Vector3() );
		// Pivot: hind feet (rear = min travel-X, at foot level), so it rears back
		// over its hind legs the way a top-heavy load pulls it.
		pivot.set( box.min.x + ( center.x - box.min.x ) * 0.28, box.min.y, center.z );
		captured = true;

	}

	// -------------------------------------------------------------------
	// Impact FX: dust ring at the pivot's screen point + HUD blip + shake
	// -------------------------------------------------------------------
	function fireImpact() {

		const r = host.getBoundingClientRect();
		_proj.copy( pivot ).project( camera );
		dustEl.style.left = ( ( _proj.x * 0.5 + 0.5 ) * r.width ) + 'px';
		dustEl.style.top = ( ( - _proj.y * 0.5 + 0.5 ) * r.height ) + 'px';
		dustEl.classList.remove( 'burst' );
		void dustEl.offsetWidth;
		dustEl.classList.add( 'burst' );

	}

	function updateHud( t ) {

		if ( ! hudEl ) return;
		hudEl.classList.toggle( 'on', t >= T.fallEnd && t < T.fallEnd + 1.7 );

	}

	function startTopple() {

		toppleT = 0;
		impactFired = false;
		captured = false;
		if ( hudEl ) hudEl.classList.remove( 'on' );
		if ( ready ) {

			climbAction.time = 0;
			climbAction.timeScale = CLIMB_SPEED;
			mixer.update( 0 );
			robotBase.getWorldPosition( _robotPos );
			camTarget.copy( _robotPos ); camTarget.y += upBias;

		}

	}

	if ( replayBtn ) replayBtn.addEventListener( 'click', startTopple );

	// -------------------------------------------------------------------
	// Theme sync
	// -------------------------------------------------------------------
	new MutationObserver( () => {

		let name = document.documentElement.getAttribute( 'data-theme' ) || 'light';
		if ( name !== 'light' && name !== 'dark' ) name = 'light';
		if ( name === themeName ) return;
		themeName = name; palette = PALETTES[ name ];
		edgesPass.setInkColor( palette.inkColorGl );

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
	// Render loop — gated on slide visibility (restart the climb+fall on arrival)
	// -------------------------------------------------------------------
	new IntersectionObserver( ( entries ) => {

		const vis = entries[ 0 ].isIntersecting && entries[ 0 ].intersectionRatio >= 0.4;
		if ( vis && ! visible ) { clock.getDelta(); startTopple(); }
		visible = vis;

	}, { threshold: [ 0.4 ] } ).observe( section );

	function animate() {

		requestAnimationFrame( animate );
		if ( ! visible || ! ready ) return;

		const dt = Math.min( clock.getDelta(), 0.05 );
		toppleT += dt;
		if ( toppleT >= T.end ) { toppleT = 0; impactFired = false; captured = false; climbAction.time = 0; }

		const climbing = toppleT < T.climbEnd;

		// Legs churn full-speed while climbing + rearing; go limp on impact.
		climbAction.timeScale = ( toppleT < T.fallEnd ) ? CLIMB_SPEED : 0;
		mixer.update( dt );

		if ( climbing ) {

			// Clip owns robot_base (recorded climb up the stairs). Follow it up.
			robotBase.getWorldPosition( _robotPos );
			_camGoal.copy( _robotPos ); _camGoal.y += upBias;
			const k = 1 - Math.pow( 0.0025, dt );
			camTarget.lerp( _camGoal, Math.min( 1, k ) );

		} else {

			// Payload wins: rear back over the hind feet and topple. mixer.update above
			// still animated the leg joints; here we override the body transform.
			if ( ! captured ) captureFallStart();
			applyTopple( toppleT );

			// Impact events fire once as it lands.
			if ( ! impactFired && toppleT >= T.fallEnd ) { impactFired = true; fireImpact(); }

		}

		updateHud( toppleT );

		// Camera: follow target (locked once the fall starts) + a short impact shake.
		camera.position.copy( camTarget ).add( camOffset );
		const sinceImpact = toppleT - T.fallEnd;
		if ( sinceImpact >= 0 && sinceImpact < 0.3 ) {

			const amp = 0.035 * ( 1 - sinceImpact / 0.3 );
			_shake.set( ( Math.random() - 0.5 ) * amp, ( Math.random() - 0.5 ) * amp, 0 );
			camera.position.add( _shake );

		}
		camera.lookAt( camTarget );
		keyLight.target.position.copy( camTarget );

		// Soft dissolve at the loop wrap.
		let opacity = 1;
		if ( toppleT > T.holdEnd ) opacity = 1 - ( toppleT - T.holdEnd ) / ( T.end - T.holdEnd );
		else if ( toppleT < 0.4 ) opacity = toppleT / 0.4;
		renderer.domElement.style.opacity = opacity.toFixed( 3 );

		composer.render();

	}
	animate();

	window.__topple = { scene, camera, renderer, startTopple, get t() { return toppleT; }, get visible() { return visible; } };

}
