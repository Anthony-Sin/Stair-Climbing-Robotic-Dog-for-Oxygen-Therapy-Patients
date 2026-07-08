// PartLabels.js
//
// SVG leader-line callouts, drawn over the WebGL canvas each frame:
// a 1px ink polyline with a 45-degree elbow from a labelled 3D node's
// projected screen position out to a text anchor, mono 13px ink text.
//
// Text anchors stack in two fixed columns (right column = upper area,
// left column = lower area, matching the reference layout) rather than
// tracking the node laterally, so labels stay legible and non-overlapping
// even while the robot/camera moves. A label is hidden outright if its
// node is missing from the current model (skip silently) or if the node
// projects off-screen / behind the camera.

import { Vector3 } from 'three';

const DEFAULT_CONFIG = [
	{ node: 'oxygen_tank', text: 'o2 concentrator', side: 'right' },
	{ node: 'cradle_rails', text: 'cradle rail', side: 'right' },
	{ node: 'head', text: 'depth camera', side: 'right' },
	{ node: 'FL_thigh', text: 'thigh actuator', side: 'left' },
	{ node: 'FR_calf', text: 'calf link', side: 'left' },
	{ node: 'FR_foot', text: 'foot pad', side: 'left' },
	{ node: 'stairs', text: 'staircase · 0.13 m rise', side: 'left' },
	{ node: 'patient_root', text: 'patient', side: 'right' },
];

const SVG_NS = 'http://www.w3.org/2000/svg';

export class PartLabels {

	/**
	 * @param {HTMLElement} container element to mount the SVG overlay into (position: relative/absolute parent)
	 * @param {THREE.Camera} camera
	 * @param {Array} config optional override of the default label config
	 */
	constructor( container, camera, config = DEFAULT_CONFIG ) {

		this.camera = camera;
		this.config = config;
		this.inkColor = '#2f2c28';

		this.svg = document.createElementNS( SVG_NS, 'svg' );
		this.svg.setAttribute( 'class', 'part-labels-svg' );
		this.svg.style.position = 'absolute';
		this.svg.style.inset = '0';
		this.svg.style.width = '100%';
		this.svg.style.height = '100%';
		this.svg.style.pointerEvents = 'none';
		this.svg.style.zIndex = '5';
		container.appendChild( this.svg );

		this._entries = [];
		this._tmpVec = new Vector3();

		this._nodesResolved = false;

	}

	setInkColor( cssColor ) {

		this.inkColor = cssColor;
		for ( const entry of this._entries ) {

			entry.line.setAttribute( 'stroke', cssColor );
			entry.text.setAttribute( 'fill', cssColor );

		}

	}

	/**
	 * Resolve config node names against the loaded scene root. Call once
	 * after a model (real or placeholder) is loaded/rebuilt. Missing nodes
	 * are silently skipped (per spec) rather than erroring.
	 */
	setSceneRoot( root ) {

		// Clear any previous entries.
		for ( const entry of this._entries ) {

			this.svg.removeChild( entry.line );
			this.svg.removeChild( entry.text );

		}

		this._entries = [];

		const rightSideConfigs = this.config.filter( ( c ) => c.side === 'right' );
		const leftSideConfigs = this.config.filter( ( c ) => c.side === 'left' );

		let rightIndex = 0;
		let leftIndex = 0;

		for ( const cfg of this.config ) {

			const node = root.getObjectByName( cfg.node );
			if ( ! node ) continue; // skip silently: node absent from this model

			const line = document.createElementNS( SVG_NS, 'polyline' );
			line.setAttribute( 'fill', 'none' );
			line.setAttribute( 'stroke', this.inkColor );
			line.setAttribute( 'stroke-width', '1' );
			this.svg.appendChild( line );

			const text = document.createElementNS( SVG_NS, 'text' );
			text.setAttribute( 'fill', this.inkColor );
			text.setAttribute( 'font-family', "'Cascadia Mono','JetBrains Mono',Consolas,monospace" );
			text.setAttribute( 'font-size', '13' );
			text.textContent = cfg.text;
			this.svg.appendChild( text );

			// Column slot index within its side (0 = topmost/first).
			const columnIndex = cfg.side === 'right' ? rightIndex ++ : leftIndex ++;

			this._entries.push( { cfg, node, line, text, columnIndex } );

		}

		this._nodesResolved = true;

	}

	/**
	 * Update leader lines + text anchor positions from the current camera
	 * and node world transforms. Call once per rendered frame.
	 * @param {number} viewportWidth
	 * @param {number} viewportHeight
	 */
	update( viewportWidth, viewportHeight ) {

		if ( ! this._nodesResolved || this._entries.length === 0 ) return;

		this.svg.setAttribute( 'viewBox', `0 0 ${viewportWidth} ${viewportHeight}` );

		// Fixed column anchor geometry, matching the reference's stacked
		// leader layout: right column sits in the upper-right, left column
		// in the lower-left, each entry offset vertically by its slot index.
		const rightColumnX = viewportWidth - 24;
		const rightColumnTopY = viewportHeight * 0.16;
		const leftColumnX = 24;
		const leftColumnTopY = viewportHeight * 0.52;
		const rowPitch = 26;
		const elbowInset = 34; // horizontal run of the 45-degree elbow segment

		for ( const entry of this._entries ) {

			const node = entry.node;

			node.getWorldPosition( this._tmpVec );
			const projected = this._tmpVec.clone().project( this.camera );

			const behindCamera = projected.z > 1 || projected.z < -1;
			const offScreen =
				projected.x < -1 || projected.x > 1 || projected.y < -1 || projected.y > 1;

			if ( behindCamera || offScreen ) {

				entry.line.style.display = 'none';
				entry.text.style.display = 'none';
				continue;

			}

			entry.line.style.display = '';
			entry.text.style.display = '';

			const screenX = ( projected.x * 0.5 + 0.5 ) * viewportWidth;
			const screenY = ( - projected.y * 0.5 + 0.5 ) * viewportHeight;

			const isRight = entry.cfg.side === 'right';
			const anchorX = isRight ? rightColumnX : leftColumnX;
			const anchorY = ( isRight ? rightColumnTopY : leftColumnTopY ) + entry.columnIndex * rowPitch;

			// 45-degree elbow: from the node, travel diagonally toward the
			// column, then run horizontally to the text anchor.
			const elbowDir = isRight ? 1 : -1;
			const elbowX = anchorX - elbowDir * elbowInset;
			const elbowY = anchorY;

			const points = `${screenX},${screenY} ${elbowX},${elbowY} ${anchorX},${anchorY}`;
			entry.line.setAttribute( 'points', points );

			const textX = isRight ? anchorX - elbowInset + 6 : anchorX + elbowInset - 6;
			entry.text.setAttribute( 'x', String( textX ) );
			entry.text.setAttribute( 'y', String( anchorY + 4 ) );
			entry.text.setAttribute( 'text-anchor', isRight ? 'end' : 'start' );

		}

	}

	dispose() {

		this.svg.remove();

	}

}
