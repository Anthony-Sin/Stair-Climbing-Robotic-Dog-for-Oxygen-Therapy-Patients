// BlueprintEdgesPass.js
//
// Custom EffectComposer Pass that draws crisp dark "technical pen" outlines
// over the beauty render, recalculated every frame from the CURRENT camera
// and geometry (so outlines stay correct as the robot/scrubber moves, and
// as OrbitControls / the follow-cam reorients).
//
// Approach (4 sub-passes, each a cheap full-screen quad):
//   1. Render the scene with scene.overrideMaterial = MeshNormalMaterial
//      into an internal WebGLRenderTarget that also has a DepthTexture
//      attached (one pass gives us both view-space normals AND depth).
//   2. MASK: a full-screen shader samples that normal buffer + depth buffer
//      at the four neighbours of each texel (Roberts-cross / mini-Sobel) and
//      flags an edge where either normals discontinue sharply OR depth
//      discontinues by more than a distance-scaled threshold (so edges stay
//      ~constant pixel width regardless of how far the surface is from the
//      camera). Written to a small mask texture (0..1 in .r).
//   3. DILATE: a 5-tap cross max-filter over the mask.
//   4. COMPOSITE: a 5-tap cross min-filter over the DILATED mask (completing
//      a morphological "closing" — dilate then erode) mixed with the beauty
//      color using uInkColor.
//
// Steps 3+4 (closing) exist because on coarsely-tessellated relief (e.g. the
// real Isaac Go2 mesh's embossed logo lettering), the raw per-pixel mask
// from step 2 comes out as disconnected dash/dot fragments along a letter's
// stroke — the surface normal genuinely doesn't change enough between two
// nearby facets for step 2 to fire there, so no threshold tuning on step 2
// closes those gaps (verified live: identical gap pattern from
// uNormalThreshold 0.55 down to 0.15). A morphological closing bridges gaps
// up to ~2*uCloseRadius px WITHOUT fattening already-continuous lines
// elsewhere, unlike raising uThickness (which fattens the whole drawing
// uniformly and still didn't fully connect the letters even at 3x the
// normal value). Setting uCloseRadius to 0 makes both filters a no-op
// (every tap samples the same center texel), so this is strictly additive
// and can be disabled per-instance.
//
// This is intentionally a from-scratch shader (not three's SobelOperator
// example) so normals AND depth are combined in one pass with tunable,
// resolution-aware thickness.

import {
	Color,
	DepthTexture,
	FloatType,
	HalfFloatType,
	LinearFilter,
	MeshBasicMaterial,
	MeshNormalMaterial,
	NearestFilter,
	NoBlending,
	RGBAFormat,
	ShaderMaterial,
	UniformsUtils,
	Vector2,
	WebGLRenderTarget,
} from 'three';
import { Pass, FullScreenQuad } from 'three/addons/postprocessing/Pass.js';

// three.js layer (0-31) reserved for meshes that must NOT receive the ink
// outline. main.js puts the whole robot subtree on this layer; the edge pass's
// mask render (below) singles it out and suppresses ink there, so the robot
// renders clean like an Isaac viewport while the toon human + set keep their
// outline. Objects stay on layer 0 too, so beauty/shadows are unaffected.
export const NO_OUTLINE_LAYER = 11;

const BlueprintMaskShader = {
	name: 'BlueprintMaskShader',

	uniforms: {
		tNormal: { value: null },
		tDepth: { value: null },
		uResolution: { value: new Vector2( 1, 1 ) },
		uNormalThreshold: { value: 0.4 },
		// See the depthThreshold constructor-option comment in main.js: a
		// too-tight value here makes whole flat faces flicker as false
		// edges from ordinary depth-texture quantization noise. 0.025 is
		// the verified-clean default; callers can still override via the
		// constructor's `options.depthThreshold`.
		uDepthThreshold: { value: 0.025 },
		uThickness: { value: 1.2 }, // pixels
		uCameraNear: { value: 0.1 },
		uCameraFar: { value: 100 },
		// Distance (view-space, world units) over which INTERIOR (normal-
		// discontinuity) edges fade out. The dense panel-seam/rivet crease
		// lines on the 324k-tri robot are separated and legible up close, but
		// once the mesh is far enough that many of them land within a few
		// pixels they pile into a solid black smudge ("all the little lines
		// combined make one big black line"). Fading them by distance leaves
		// distant geometry with only its clean silhouette (depth) edges while
		// keeping full crease detail up close. See the fragment tail.
		uInteriorFadeNear: { value: 2.0 }, // full interior detail nearer than this
		uInteriorFadeFar: { value: 6.5 },  // interior creases fully gone past this
		// Global multiplier on the INTERIOR (normal-discontinuity) crease lines,
		// independent of the distance fade above. 1 = normal (interior creases inked
		// per the fade band); 0 = interior creases fully suppressed at ALL distances,
		// leaving ONLY the clean silhouette (depth) outline. The cinematic view sets
		// this to 0 for a clean "one solid black outline, no busy interior lines" look
		// (see main.js's cinematic toggle); the default orbit view keeps it at 1.
		uInteriorStrength: { value: 1.0 },
		// White where the robot (NO_OUTLINE_LAYER) is on screen, black elsewhere.
		// The fragment clears ALL ink there (dilated a few px) so the robot has no
		// toon outline; the human/set are untouched (mask is black over them).
		tRobotMask: { value: null },
	},

	vertexShader: /* glsl */ `
		varying vec2 vUv;
		void main() {
			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
		}
	`,

	fragmentShader: /* glsl */ `
		uniform sampler2D tNormal;
		uniform sampler2D tDepth;
		uniform vec2 uResolution;
		uniform float uNormalThreshold;
		uniform float uDepthThreshold;
		uniform float uThickness;
		uniform float uCameraNear;
		uniform float uCameraFar;
		uniform float uInteriorFadeNear;
		uniform float uInteriorFadeFar;
		uniform float uInteriorStrength;
			uniform sampler2D tRobotMask;

		varying vec2 vUv;

		// Perspective depth -> linear view-space distance (0..1 over near..far).
		float linearizeDepth( float z ) {
			float ndc = z * 2.0 - 1.0;
			return ( 2.0 * uCameraNear * uCameraFar ) /
				( uCameraFar + uCameraNear - ndc * ( uCameraFar - uCameraNear ) );
		}

		void main() {
			vec2 texel = ( uThickness / uResolution );

			// Roberts-cross sample offsets (diagonal 2x2), scaled by uThickness.
			vec2 uv0 = vUv + texel * vec2( -0.5, -0.5 );
			vec2 uv1 = vUv + texel * vec2(  0.5,  0.5 );
			vec2 uv2 = vUv + texel * vec2(  0.5, -0.5 );
			vec2 uv3 = vUv + texel * vec2( -0.5,  0.5 );

			// --- Normal discontinuity (Roberts-cross over view-space normals) ---
			vec3 n0 = normalize( texture2D( tNormal, uv0 ).rgb * 2.0 - 1.0 );
			vec3 n1 = normalize( texture2D( tNormal, uv1 ).rgb * 2.0 - 1.0 );
			vec3 n2 = normalize( texture2D( tNormal, uv2 ).rgb * 2.0 - 1.0 );
			vec3 n3 = normalize( texture2D( tNormal, uv3 ).rgb * 2.0 - 1.0 );

			float normalEdge = length( n0 - n1 ) + length( n2 - n3 );
			// The +0.35 this used to be made borderline creases (magnitude just
			// above uNormalThreshold) blend in as a faint, semi-transparent grey
			// smear instead of committing to ink or nothing — visible as faint
			// "dashed" lines along shallow seams. +0.10 still gives ~1px of AA
			// at the boundary but snaps everything else to fully-inked or
			// fully-clear (the closing filter downstream handles genuine gaps).
			normalEdge = smoothstep( uNormalThreshold, uNormalThreshold + 0.10, normalEdge );

			// --- Depth discontinuity (linearized, distance-scaled) ---
			float d0 = linearizeDepth( texture2D( tDepth, uv0 ).r );
			float d1 = linearizeDepth( texture2D( tDepth, uv1 ).r );
			float d2 = linearizeDepth( texture2D( tDepth, uv2 ).r );
			float d3 = linearizeDepth( texture2D( tDepth, uv3 ).r );

			// Scale the threshold by distance from camera so a fixed real-world
			// gap (e.g. a leg silhouette) produces a comparable edge response
			// whether it's close to or far from the camera.
			float refDepth = max( d0, 0.0001 );
			float depthEdge = ( abs( d0 - d1 ) + abs( d2 - d3 ) ) / refDepth;
			// Tightened from *4.0 alongside the normalEdge change above (same
			// "faint/dashed instead of bold-or-absent" complaint), but NOT all
			// the way down to a hard step: verified live that anything tighter
			// than roughly *3.0 reintroduces the flat-face flicker this ratio
			// was originally loosened to fix (a whole shallow-angle panel lit
			// up as a false edge at *1.1). *3.0 is the tightest value that
			// stayed clean on a flat panel test region while still snapping
			// real edges to solid ink.
			depthEdge = smoothstep( uDepthThreshold, uDepthThreshold * 3.0, depthEdge );

			// --- Distance level-of-detail (the fix for "little lines merge into
			// one big black blob far away") ---
			// refDepth is this fragment's view-space distance in world units. The
			// two edge types are treated differently on purpose, matching "keep a
			// clean black outline, drop the dense clustered lines":
			//   * depthEdge = the SILHOUETTE / occlusion outline (a clean isolated
			//     boundary). Kept at FULL ink strength at every distance, so the
			//     robot and patient always read with a clear solid-black outline.
			//   * normalEdge = the dense INTERIOR surface creases (panel seams,
			//     rivets) that pile into a black smudge far away. These fade to
			//     nothing across uInteriorFadeNear..uInteriorFadeFar, leaving the
			//     toon shading (the "gray") inside the outline. Up close every
			//     crease is still drawn.
			float distFade = smoothstep( uInteriorFadeNear, uInteriorFadeFar, refDepth );
			normalEdge *= ( 1.0 - distFade );
			// Global interior-crease strength (0 in cinematic -> only the
			// silhouette/depth outline survives; see uInteriorStrength above).
			normalEdge *= uInteriorStrength;

			// Grazing-angle suppression (the main "respawning outline" fix): fade
			// interior normal-edges on faces nearly EDGE-ON to the camera (view-space
			// normal z ~ 0). Such faces (the payload box's side, the sides of the
			// cylindrical leg links) spray a dense, unstable cluster of crease lines
			// that pop in/out and change count as the body rotates a hair during the
			// walk -- exactly the flicker complaint. Their real boundary is already
			// drawn by the depth silhouette edge, so dropping their interior creases
			// removes the shimmer with no loss of outline. n0 is the view-space normal
			// of the first Roberts-cross tap. Applies in every view (a pure quality
			// win); silhouette/depth edges are untouched.
			float facing = abs( n0.z );
			normalEdge *= smoothstep( 0.20, 0.45, facing );

			float edge = clamp( max( normalEdge, depthEdge ), 0.0, 1.0 );

			// --- Robot outline suppression (clean "Isaac viewport" robot) ---
			// The robot subtree is drawn white into tRobotMask; clear ALL ink where
			// the robot is, DILATED outward a few px so the silhouette line just
			// OUTSIDE the boundary is cleared too (else a thin outline hugs it). The
			// robot still occupies the depth buffer, so edges of objects BEHIND it
			// stay occluded and never bleed across it. Human/set: mask is black -> no-op.
			vec2 rmOff = 4.0 / uResolution;
			float rm = texture2D( tRobotMask, vUv ).r;
			rm = max( rm, texture2D( tRobotMask, vUv + vec2(  rmOff.x, 0.0 ) ).r );
			rm = max( rm, texture2D( tRobotMask, vUv + vec2( -rmOff.x, 0.0 ) ).r );
			rm = max( rm, texture2D( tRobotMask, vUv + vec2( 0.0,  rmOff.y ) ).r );
			rm = max( rm, texture2D( tRobotMask, vUv + vec2( 0.0, -rmOff.y ) ).r );
			rm = max( rm, texture2D( tRobotMask, vUv + rmOff ).r );
			rm = max( rm, texture2D( tRobotMask, vUv - rmOff ).r );
			rm = max( rm, texture2D( tRobotMask, vUv + vec2(  rmOff.x, -rmOff.y ) ).r );
			rm = max( rm, texture2D( tRobotMask, vUv + vec2( -rmOff.x,  rmOff.y ) ).r );
			edge *= ( 1.0 - step( 0.5, rm ) );

			// .r = combined edge (the ONLY channel the downstream dilate/erode +
			// composite consume). .g/.b expose the depth (silhouette) and normal
			// (interior, post uInteriorStrength) components SEPARATELY so the
			// edge-coverage probe (main.js __viewer.edgeCoverage) can tell whether a
			// given part boundary is being inked by a silhouette edge vs an interior
			// crease. Purely diagnostic -- .g/.b never affect the rendered image.
			gl_FragColor = vec4( edge, depthEdge, normalEdge, 1.0 );
		}
	`,
};

// 5-tap cross max-filter: step 3 of the closing (dilate). Cheap — a single
// texture (not tNormal/tDepth) at 5 taps instead of the mask shader's 8.
const BlueprintDilateShader = {
	name: 'BlueprintDilateShader',

	uniforms: {
		tMask: { value: null },
		uResolution: { value: new Vector2( 1, 1 ) },
		uCloseRadius: { value: 0 }, // pixels; 0 makes this a no-op passthrough
	},

	vertexShader: /* glsl */ `
		varying vec2 vUv;
		void main() {
			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
		}
	`,

	fragmentShader: /* glsl */ `
		uniform sampler2D tMask;
		uniform vec2 uResolution;
		uniform float uCloseRadius;
		varying vec2 vUv;

		void main() {
			vec2 texel = uCloseRadius / uResolution;
			float m = texture2D( tMask, vUv ).r;
			m = max( m, texture2D( tMask, vUv + texel * vec2(  1.0,  0.0 ) ).r );
			m = max( m, texture2D( tMask, vUv + texel * vec2( -1.0,  0.0 ) ).r );
			m = max( m, texture2D( tMask, vUv + texel * vec2(  0.0,  1.0 ) ).r );
			m = max( m, texture2D( tMask, vUv + texel * vec2(  0.0, -1.0 ) ).r );
			gl_FragColor = vec4( m, m, m, 1.0 );
		}
	`,
};

// Step 4: 5-tap cross min-filter over the DILATED mask (the "erode" half of
// the closing — same radius as the dilate, so a proper closing rather than
// a net dilation) composited with the beauty render.
const BlueprintEdgesShader = {
	name: 'BlueprintEdgesShader',

	uniforms: {
		tDiffuse: { value: null },
		tDilated: { value: null },
		uResolution: { value: new Vector2( 1, 1 ) },
		uInkColor: { value: new Color( 0x2f2c28 ) },
		uCloseRadius: { value: 0 },
		uOpacity: { value: 1.0 },
	},

	vertexShader: /* glsl */ `
		varying vec2 vUv;
		void main() {
			vUv = uv;
			gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
		}
	`,

	fragmentShader: /* glsl */ `
		uniform sampler2D tDiffuse;
		uniform sampler2D tDilated;
		uniform vec2 uResolution;
		uniform vec3 uInkColor;
		uniform float uCloseRadius;
		uniform float uOpacity;

		varying vec2 vUv;

		void main() {
			vec2 texel = uCloseRadius / uResolution;
			float edge = texture2D( tDilated, vUv ).r;
			edge = min( edge, texture2D( tDilated, vUv + texel * vec2(  1.0,  0.0 ) ).r );
			edge = min( edge, texture2D( tDilated, vUv + texel * vec2( -1.0,  0.0 ) ).r );
			edge = min( edge, texture2D( tDilated, vUv + texel * vec2(  0.0,  1.0 ) ).r );
			edge = min( edge, texture2D( tDilated, vUv + texel * vec2(  0.0, -1.0 ) ).r );

			vec4 beauty = texture2D( tDiffuse, vUv );
			vec3 outColor = mix( beauty.rgb, uInkColor, edge * uOpacity );

			gl_FragColor = vec4( outColor, beauty.a );
		}
	`,
};

export class BlueprintEdgesPass extends Pass {

	constructor( scene, camera, options = {} ) {

		super();

		this.scene = scene;
		this.camera = camera;

		this.needsSwap = true;
		this.clear = false;

		// --- Internal normal+depth render target ---
		this._normalMaterial = new MeshNormalMaterial();
		// MeshNormalMaterial by default skips morph/skin normal handling
		// unless the source material had it; three enables this
		// automatically per-object when using overrideMaterial + skinning,
		// so nothing else to configure here.

		const depthTexture = new DepthTexture();
		depthTexture.type = FloatType;
		depthTexture.minFilter = NearestFilter;
		depthTexture.magFilter = NearestFilter;

		// samples: 4 -> MSAA on the normal+depth geometry buffer the edge detector
		// samples. The beauty pass is already MSAA'd (renderer antialias:true), but
		// the EDGE mask was computed from an aliased normal/depth buffer, so thin
		// MOVING features (swinging legs, feet) had jagged, shimmering, "respawning"
		// outlines during the walk. Multisampling the geometry buffer (WebGL2
		// resolves both color and the attached depth texture on read) gives the
		// Roberts-cross smooth normals/depth to difference, killing most of that
		// per-frame edge jitter. Only this buffer needs it; the mask/dilate/composite
		// targets stay single-sampled.
		this._normalTarget = new WebGLRenderTarget( 1, 1, {
			minFilter: LinearFilter,
			magFilter: LinearFilter,
			format: RGBAFormat,
			type: HalfFloatType,
			depthTexture,
			samples: 4,
		} );
		this._normalTarget.texture.name = 'BlueprintEdgesPass.normal';

		// --- Mask target (raw, possibly-fragmented edge mask) ---
		this._maskTarget = new WebGLRenderTarget( 1, 1, {
			minFilter: NearestFilter,
			magFilter: NearestFilter,
			format: RGBAFormat,
		} );
		this._maskTarget.texture.name = 'BlueprintEdgesPass.mask';

		this._maskMaterial = new ShaderMaterial( {
			name: BlueprintMaskShader.name,
			uniforms: UniformsUtils.clone( BlueprintMaskShader.uniforms ),
			vertexShader: BlueprintMaskShader.vertexShader,
			fragmentShader: BlueprintMaskShader.fragmentShader,
			blending: NoBlending,
			depthTest: false,
			depthWrite: false,
		} );
		this._maskMaterial.uniforms.tNormal.value = this._normalTarget.texture;
		this._maskMaterial.uniforms.tDepth.value = this._normalTarget.depthTexture;
		this._maskQuad = new FullScreenQuad( this._maskMaterial );

		// --- Robot no-outline mask (see NO_OUTLINE_LAYER) ---
		// A white-on-black render of ONLY the robot subtree, used by the mask shader
		// to clear ink there. The robot still renders into the normal/depth buffer in
		// Pass 1, so this removes ONLY its own outline -- occlusion and everyone
		// else's outlines are unchanged.
		this._noOutlineLayer = options.noOutlineLayer ?? null;
		this._robotMaskTarget = new WebGLRenderTarget( 1, 1, {
			minFilter: NearestFilter,
			magFilter: NearestFilter,
			format: RGBAFormat,
		} );
		this._robotMaskTarget.texture.name = 'BlueprintEdgesPass.robotMask';
		this._robotWhiteMaterial = new MeshBasicMaterial( { color: 0xffffff, toneMapped: false, fog: false } );
		this._maskMaterial.uniforms.tRobotMask.value = this._robotMaskTarget.texture;

		// --- Dilate target + material (closing step 1 of 2) ---
		this._dilateTarget = new WebGLRenderTarget( 1, 1, {
			minFilter: NearestFilter,
			magFilter: NearestFilter,
			format: RGBAFormat,
		} );
		this._dilateTarget.texture.name = 'BlueprintEdgesPass.dilate';

		this._dilateMaterial = new ShaderMaterial( {
			name: BlueprintDilateShader.name,
			uniforms: UniformsUtils.clone( BlueprintDilateShader.uniforms ),
			vertexShader: BlueprintDilateShader.vertexShader,
			fragmentShader: BlueprintDilateShader.fragmentShader,
			blending: NoBlending,
			depthTest: false,
			depthWrite: false,
		} );
		this._dilateMaterial.uniforms.tMask.value = this._maskTarget.texture;
		this._dilateQuad = new FullScreenQuad( this._dilateMaterial );

		// --- Full-screen composite quad (closing step 2 of 2 + ink mix) ---
		this._material = new ShaderMaterial( {
			name: BlueprintEdgesShader.name,
			uniforms: UniformsUtils.clone( BlueprintEdgesShader.uniforms ),
			vertexShader: BlueprintEdgesShader.vertexShader,
			fragmentShader: BlueprintEdgesShader.fragmentShader,
			blending: NoBlending,
			depthTest: false,
			depthWrite: false,
		} );
		this._material.uniforms.tDilated.value = this._dilateTarget.texture;

		if ( options.inkColor !== undefined ) this.setInkColor( options.inkColor );
		if ( options.normalThreshold !== undefined ) this._maskMaterial.uniforms.uNormalThreshold.value = options.normalThreshold;
		if ( options.depthThreshold !== undefined ) this._maskMaterial.uniforms.uDepthThreshold.value = options.depthThreshold;
		if ( options.thickness !== undefined ) this._maskMaterial.uniforms.uThickness.value = options.thickness;
		if ( options.interiorFadeNear !== undefined ) this._maskMaterial.uniforms.uInteriorFadeNear.value = options.interiorFadeNear;
		if ( options.interiorFadeFar !== undefined ) this._maskMaterial.uniforms.uInteriorFadeFar.value = options.interiorFadeFar;
		if ( options.interiorStrength !== undefined ) this._maskMaterial.uniforms.uInteriorStrength.value = options.interiorStrength;
		this.setCloseRadius( options.closeRadius !== undefined ? options.closeRadius : 1.5 );

		this._fsQuad = new FullScreenQuad( this._material );

	}

	/**
	 * Combined view of every tunable uniform across the mask/dilate/composite
	 * sub-materials, keyed the same as before the 4-pass split (so external
	 * live-tuning code, e.g. `edgesPass.uniforms.uNormalThreshold.value = x`,
	 * keeps working unchanged).
	 */
	get uniforms() {

		return {
			...this._maskMaterial.uniforms,
			...this._material.uniforms,
		};

	}

	setInkColor( colorLike ) {

		this._material.uniforms.uInkColor.value.set( colorLike );

	}

	/**
	 * Interior-crease strength: 1 = normal (interior normal-discontinuity lines
	 * inked per the distance-fade band), 0 = interior creases fully suppressed at
	 * every distance, leaving ONLY the clean silhouette (depth) outline. The
	 * cinematic view sets this to 0 for a clean single-outline render.
	 */
	setInteriorStrength( strength ) {

		this._maskMaterial.uniforms.uInteriorStrength.value = strength;

	}

	/**
	 * Normal-discontinuity (interior crease) threshold: HIGHER = only the
	 * strongest structural creases ink (fewer lines); LOWER = more surface detail.
	 * Cinematic raises this so the robot keeps a few strong "robotic" structure
	 * lines (leg-body joins, the camera mount, major panel seams) without the busy
	 * rivet/seam clutter -- see main.js's cinematic edge style.
	 */
	setNormalThreshold( threshold ) {

		this._maskMaterial.uniforms.uNormalThreshold.value = threshold;

	}

	/** Interior-crease distance-fade band (view-space metres): creases fade from full at `near` to gone at `far`. Pushed far out in cinematic so the (now sparse) structural lines survive at the pulled-back framing distance. */
	setInteriorFade( near, far ) {

		this._maskMaterial.uniforms.uInteriorFadeNear.value = near;
		this._maskMaterial.uniforms.uInteriorFadeFar.value = far;

	}

	/** Morphological closing radius in pixels (0 disables — pure passthrough). */
	setCloseRadius( radiusPx ) {

		this._dilateMaterial.uniforms.uCloseRadius.value = radiusPx;
		this._material.uniforms.uCloseRadius.value = radiusPx;

	}

	setSize( width, height ) {

		const w = Math.max( 1, width );
		const h = Math.max( 1, height );

		this._normalTarget.setSize( w, h );
		this._maskTarget.setSize( w, h );
		this._dilateTarget.setSize( w, h );
		if ( this._robotMaskTarget ) this._robotMaskTarget.setSize( w, h );

		this._maskMaterial.uniforms.uResolution.value.set( w, h );
		this._dilateMaterial.uniforms.uResolution.value.set( w, h );
		this._material.uniforms.uResolution.value.set( w, h );

	}

	render( renderer, writeBuffer, readBuffer /*, deltaTime, maskActive */ ) {

		// Keep near/far in sync in case the camera changed since construction.
		this._maskMaterial.uniforms.uCameraNear.value = this.camera.near;
		this._maskMaterial.uniforms.uCameraFar.value = this.camera.far;

		// --- Pass 1: render scene normals + depth into our own RT ---
		const previousOverrideMaterial = this.scene.overrideMaterial;
		const previousBackground = this.scene.background;
		const previousRenderTarget = renderer.getRenderTarget();
		const previousClearColor = new Color();
		renderer.getClearColor( previousClearColor );
		const previousClearAlpha = renderer.getClearAlpha();

		this.scene.overrideMaterial = this._normalMaterial;
		// Neutral background for the normal pass so empty pixels don't read
		// as a fake "surface" (they'll just show max depth / flat normal,
		// which the edge shader treats as no discontinuity against sky).
		this.scene.background = null;

		renderer.setRenderTarget( this._normalTarget );
		renderer.setClearColor( 0x7777ff, 1 ); // "flat" encoded normal (0,0,1)
		renderer.clear( true, true, false );
		renderer.render( this.scene, this.camera );

		this.scene.overrideMaterial = previousOverrideMaterial;
		this.scene.background = previousBackground;
		renderer.setClearColor( previousClearColor, previousClearAlpha );

		// --- Pass 1.5: robot-only white mask (outline suppression) ---
		// Render JUST the NO_OUTLINE_LAYER (robot subtree) as flat white so the mask
		// shader can clear ink there. The main camera's layer mask is switched to that
		// layer for this internal render only, then restored -- beauty + shadows (run
		// in the prior RenderPass) are untouched. Shadow auto-update is paused so this
		// extra render doesn't needlessly re-render the shadow map.
		if ( this._noOutlineLayer !== null ) {

			const prevShadowAuto = renderer.shadowMap.autoUpdate;
			renderer.shadowMap.autoUpdate = false;
			const savedCamMask = this.camera.layers.mask;

			this.scene.overrideMaterial = this._robotWhiteMaterial;
			this.scene.background = null;
			this.camera.layers.set( this._noOutlineLayer );

			renderer.setRenderTarget( this._robotMaskTarget );
			renderer.setClearColor( 0x000000, 1 );
			renderer.clear( true, true, false );
			renderer.render( this.scene, this.camera );

			this.camera.layers.mask = savedCamMask;
			this.scene.overrideMaterial = previousOverrideMaterial;
			this.scene.background = previousBackground;
			renderer.setClearColor( previousClearColor, previousClearAlpha );
			renderer.shadowMap.autoUpdate = prevShadowAuto;

		}

		// --- Pass 2: raw (possibly-fragmented) edge mask ---
		renderer.setRenderTarget( this._maskTarget );
		this._maskQuad.render( renderer );

		// --- Pass 3: dilate (closing, step 1/2) ---
		renderer.setRenderTarget( this._dilateTarget );
		this._dilateQuad.render( renderer );

		// --- Pass 4: erode (closing, step 2/2) + composite with beauty ---
		this._material.uniforms.tDiffuse.value = readBuffer.texture;

		if ( this.renderToScreen ) {

			renderer.setRenderTarget( null );
			this._fsQuad.render( renderer );

		} else {

			renderer.setRenderTarget( writeBuffer );
			if ( this.clear ) renderer.clear( renderer.autoClearColor, renderer.autoClearDepth, renderer.autoClearStencil );
			this._fsQuad.render( renderer );

		}

		renderer.setRenderTarget( previousRenderTarget );

	}

	dispose() {

		this._normalTarget.dispose();
		this._normalMaterial.dispose();
		this._maskTarget.dispose();
		this._maskMaterial.dispose();
		this._maskQuad.dispose();
		this._dilateTarget.dispose();
		this._dilateMaterial.dispose();
		this._dilateQuad.dispose();
		if ( this._robotMaskTarget ) this._robotMaskTarget.dispose();
		if ( this._robotWhiteMaterial ) this._robotWhiteMaterial.dispose();
		this._material.dispose();
		this._fsQuad.dispose();

	}

}
