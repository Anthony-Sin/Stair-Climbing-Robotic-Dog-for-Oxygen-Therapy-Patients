// BlueprintEdgesPass.js
//
// Custom EffectComposer Pass that draws crisp dark "technical pen" outlines
// over the beauty render, recalculated every frame from the CURRENT camera
// and geometry (so outlines stay correct as the robot/scrubber moves, and
// as OrbitControls / the follow-cam reorients).
//
// Approach:
//   1. Render the scene with scene.overrideMaterial = MeshNormalMaterial
//      into an internal WebGLRenderTarget that also has a DepthTexture
//      attached (one pass gives us both view-space normals AND depth).
//   2. A full-screen shader samples that normal buffer + depth buffer at
//      the four neighbours of each texel (Roberts-cross / mini-Sobel) and
//      flags an edge where either normals discontinue sharply OR depth
//      discontinues by more than a distance-scaled threshold (so edges
//      stay ~constant pixel width regardless of how far the surface is
//      from the camera).
//   3. The edge mask is mixed with the beauty color using uInkColor.
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

const BlueprintEdgesShader = {
	name: 'BlueprintEdgesShader',

	uniforms: {
		tDiffuse: { value: null },
		tNormal: { value: null },
		tDepth: { value: null },
		uResolution: { value: new Vector2( 1, 1 ) },
		uInkColor: { value: new Color( 0x2f2c28 ) },
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
		uniform sampler2D tNormal;
		uniform sampler2D tDepth;
		uniform vec2 uResolution;
		uniform vec3 uInkColor;
		uniform float uNormalThreshold;
		uniform float uDepthThreshold;
		uniform float uThickness;
		uniform float uCameraNear;
		uniform float uCameraFar;
		uniform float uOpacity;

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
			normalEdge = smoothstep( uNormalThreshold, uNormalThreshold + 0.35, normalEdge );

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
			depthEdge = smoothstep( uDepthThreshold, uDepthThreshold * 4.0, depthEdge );

			float edge = clamp( max( normalEdge, depthEdge ), 0.0, 1.0 );

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

		this._normalTarget = new WebGLRenderTarget( 1, 1, {
			minFilter: LinearFilter,
			magFilter: LinearFilter,
			format: RGBAFormat,
			type: HalfFloatType,
			depthTexture,
		} );
		this._normalTarget.texture.name = 'BlueprintEdgesPass.normal';

		// --- Full-screen composite quad ---
		this._material = new ShaderMaterial( {
			name: BlueprintEdgesShader.name,
			uniforms: UniformsUtils.clone( BlueprintEdgesShader.uniforms ),
			vertexShader: BlueprintEdgesShader.vertexShader,
			fragmentShader: BlueprintEdgesShader.fragmentShader,
			blending: NoBlending,
			depthTest: false,
			depthWrite: false,
		} );
		this._material.uniforms.tNormal.value = this._normalTarget.texture;
		this._material.uniforms.tDepth.value = this._normalTarget.depthTexture;

		if ( options.inkColor !== undefined ) this.setInkColor( options.inkColor );
		if ( options.normalThreshold !== undefined ) this._material.uniforms.uNormalThreshold.value = options.normalThreshold;
		if ( options.depthThreshold !== undefined ) this._material.uniforms.uDepthThreshold.value = options.depthThreshold;
		if ( options.thickness !== undefined ) this._material.uniforms.uThickness.value = options.thickness;

		this._fsQuad = new FullScreenQuad( this._material );

	}

	get uniforms() {

		return this._material.uniforms;

	}

	setInkColor( colorLike ) {

		this._material.uniforms.uInkColor.value.set( colorLike );

	}

	setSize( width, height ) {

		this._normalTarget.setSize( Math.max( 1, width ), Math.max( 1, height ) );
		this._material.uniforms.uResolution.value.set( Math.max( 1, width ), Math.max( 1, height ) );

	}

	render( renderer, writeBuffer, readBuffer /*, deltaTime, maskActive */ ) {

		// Keep near/far in sync in case the camera changed since construction.
		this._material.uniforms.uCameraNear.value = this.camera.near;
		this._material.uniforms.uCameraFar.value = this.camera.far;

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

		// --- Pass 2: composite beauty (readBuffer) + normal/depth edges ---
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
		this._material.dispose();
		this._fsQuad.dispose();

	}

}
