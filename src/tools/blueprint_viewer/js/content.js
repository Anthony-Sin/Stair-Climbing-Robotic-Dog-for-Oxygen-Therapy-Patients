// content.js
//
// Single source of truth for the pitch deck's copy + data visuals, shared by
// the 3D stage engine (hero.js, which needs the Solution policies' motion/fx/
// leader points) and the scroll orchestrator (deck.js, which injects every
// section's HTML and builds the nav). All data visuals are native SVG in the
// paper/ink palette (currentColor + accents teal #0e9b8e, oak #c07d3c,
// terracotta #b0552f, red #e5484d) — deliberately NOT the dark matplotlib PNGs,
// which clash with the deck's vibe. Numbers are the real training/sweep values
// (follow_sweep_20260705_203945).

// ---------------------------------------------------------------------------
// Ordered sections — drives the nav rail + IntersectionObserver in deck.js.
// `kind`: 'content' (static HTML card) | 'stage' (Solution, shared 3D stage) |
// 'demo' (the interactive ACT-2 viewer, owned by main.js).
// ---------------------------------------------------------------------------
export const SECTIONS = [
	{ id: 's-problem',   label: 'Problem',      group: 'Problem',                kind: 'content' },
	{ id: 's-arch',      label: 'Architecture', group: 'Solution & technology',  kind: 'stage', policy: 'architecture' },
	{ id: 's-walk',      label: 'Walking',      group: 'Solution & technology',  kind: 'stage', policy: 'walking' },
	{ id: 's-climb',     label: 'Climbing',     group: 'Solution & technology',  kind: 'stage', policy: 'blind-rl' },
	{ id: 's-demo',      label: 'Live demo',    group: 'Live demo',              kind: 'demo' },
	{ id: 's-training',  label: 'Training',     group: 'Progress & results',     kind: 'content' },
	{ id: 's-sweep',     label: 'Height sweep', group: 'Progress & results',     kind: 'content' },
	{ id: 's-challenges',label: 'Challenges',   group: 'Engineering challenges',  kind: 'content' },
	{ id: 's-potential', label: 'Potential',    group: 'Real-world potential',   kind: 'content' },
	// 's-team' (the "Anthony Sinchi" credits slide) is hidden from the deck --
	// its CONTENT/copy entry below is left intact, just not in SECTIONS, so
	// deck.js's SECTIONS-driven nav/scroll never renders or links to it.
];

// ---------------------------------------------------------------------------
// Solution & technology policies (the 3 shared-stage sections). `motion`/`fx`
// drive the 3D stage; `points` are the leader callouts; `diagram`/`clip`/`body`
// fill the section's side panel.
// ---------------------------------------------------------------------------
export const POLICIES = {
	architecture: {
		id: 'architecture', motion: 'spin', fx: null,
		title: 'System architecture',
		body: 'A <b>Unitree Go2</b> carries the patient’s oxygen concentrator on a <b>shock-isolated cradle</b>. The whole control stack — <b>perception plus the learned policy</b> — runs inside <b>one Docker container</b> on a <b>Jetson Orin</b>: the same container whether Isaac Sim or the real robot feeds it. <b>Only the sensor source is swapped</b>, so a policy proven in simulation is expected to hold on hardware.',
		clip: null,
		points: [
			{ node: 'oxygen_tank', side: 'left', label: 'O₂ concentrator', sub: 'patient payload' },
			{ node: 'cradle_rails', side: 'left', label: 'payload cradle', sub: 'shock-isolated' },
			{ node: 'robot_base', side: 'left', label: 'onboard compute', sub: 'Jetson Orin' },
			{ node: 'FR_hip', side: 'left', label: '12× actuators', sub: 'three per leg' },
		],
	},
	walking: {
		id: 'walking', motion: 'walk', fx: 'scan',
		title: 'Walking policy — it follows you',
		body: 'On flat ground the robot <b>follows the patient by sight</b>: a <b>YOLO-World detector</b> locates the person in <b>every camera frame</b>, and a <b>learned trot gait</b> steers to keep pace while <b>holding the oxygen payload level</b> — rejecting the disturbances of a shifting load and an uneven floor at each step.',
		clip: { src: './assets/clips/walk.mp4', cap: 'Isaac Sim rollout — flat-ground follow gait.' },
		points: [
			{ node: 'robot_base', side: 'left', label: 'gait controller', sub: 'trot clock' },
			{ node: 'FL_hip', side: 'left', label: 'hip abduction', sub: 'lateral balance' },
			{ node: 'cradle_rails', side: 'left', label: 'payload held level', sub: 'load balancing' },
			{ node: 'FR_calf', side: 'left', label: 'calf drive', sub: 'ground clearance' },
		],
	},
	'blind-rl': {
		id: 'blind-rl', motion: 'climb', fx: 'feel',
		title: 'Blind-RL policy — it climbs by feel',
		body: 'The staircase is <b>climbed on feel alone</b>. Cameras <b>can’t see the steps underfoot</b>, so a reinforcement-learning policy leans entirely on <b>proprioception and foot contact</b> — <b>sensing each riser as a paw lands</b> — to place its feet and drive the payload upward, step after step, while <b>keeping the concentrator upright</b> on the incline.',
		clip: { src: './assets/clips/climb_0p130_trimmed.mp4', cap: 'Isaac Sim rollout — 0.15 m riser climb.' },
		points: [
			{ node: 'robot_base', side: 'left', label: 'IMU · attitude', sub: 'stays upright' },
			{ node: 'FR_hip', side: 'left', label: 'joint feedback', sub: 'proprioception' },
			{ node: 'oxygen_tank', side: 'left', label: 'payload upright', sub: 'on the incline' },
			{ node: 'FL_calf', side: 'left', label: 'foot contact', sub: 'feels each riser' },
		],
	},
};

// Policy SVG schematics (theme-aware via currentColor).
export const DIAGRAMS = {
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
		<rect class="dg-box" x="24" y="12" width="56" height="28" rx="5"/><text class="dg-t" x="52" y="30" text-anchor="middle">camera</text>
		<rect class="dg-box" x="108" y="12" width="84" height="28" rx="5"/><text class="dg-t" x="150" y="30" text-anchor="middle">YOLO-World</text>
		<rect class="dg-box" x="108" y="58" width="84" height="28" rx="5"/><text class="dg-t" x="150" y="76" text-anchor="middle">state est.</text>
		<rect class="dg-box" x="232" y="26" width="64" height="42" rx="6"/><text class="dg-t" x="264" y="45" text-anchor="middle">policy π</text><text class="dg-c" x="264" y="59" text-anchor="middle">MLP</text>
		<rect class="dg-box" x="232" y="94" width="64" height="28" rx="5"/><text class="dg-t" x="264" y="112" text-anchor="middle">PD control</text>
		<rect class="dg-box" x="90" y="140" width="112" height="28" rx="6"/><text class="dg-t" x="146" y="156" text-anchor="middle">Go2 · 12 DoF</text><text class="dg-c" x="146" y="167" text-anchor="middle">rigid-body plant</text>
		<path class="dg-ln" d="M80 26 H108"/><path class="dg-ah" d="M102 22 l6 4 -6 4"/>
		<text class="dg-c" x="94" y="22" text-anchor="middle">RGB</text>
		<path class="dg-ln" d="M192 26 H212 V38 H232"/><path class="dg-ah" d="M226 34 l6 4 -6 4"/>
		<text class="dg-c" x="212" y="22" text-anchor="middle">target</text>
		<path class="dg-ln" d="M192 72 H212 V54 H232"/><path class="dg-ah" d="M226 50 l6 4 -6 4"/>
		<path class="dg-ln" d="M264 68 V94"/><path class="dg-ah" d="M260 88 l4 6 4 -6"/>
		<path class="dg-ln" d="M264 122 V154 H202"/><path class="dg-ah" d="M208 150 l-6 4 6 4"/>
		<text class="dg-c" x="240" y="148" text-anchor="middle">torque</text>
		<path class="dg-ln" d="M90 154 H12 V26 H24"/><path class="dg-ah" d="M18 22 l6 4 -6 4"/>
		<path class="dg-ln" d="M12 72 H108"/><path class="dg-ah" d="M102 68 l6 4 -6 4"/>
		<text class="dg-c" x="60" y="68" text-anchor="middle">IMU · joints</text>
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

// ---------------------------------------------------------------------------
// Data-chart + illustration builders (native SVG).
// ---------------------------------------------------------------------------

// Shared axis grid: faint horizontal gridlines + a denser set of numeric tick
// labels on both axes, so the training charts read as real plotted data (more
// numbers), not just a headline value. yTicks/xTicks are [{ v/label, y/x }].
function chartGrid( yTicks, xTicks ) {
	const grid = yTicks.map( ( t ) => `<line class="ch-grid" x1="46" y1="${ t.y }" x2="464" y2="${ t.y }"/>` ).join( '' );
	const yl = yTicks.map( ( t ) => `<text class="ch-tick" x="42" y="${ t.y + 3.2 }" text-anchor="end">${ t.label }</text>` ).join( '' );
	const xl = xTicks.map( ( t ) => `<text class="ch-tick" x="${ t.x }" y="282" text-anchor="middle">${ t.label }</text>` ).join( '' );
	return grid + yl + xl;
}

// iteration x-axis is shared across the 6000-iter training charts.
const ITER_XTICKS = [
	{ x: 46, label: '0' }, { x: 150.5, label: '1.5k' }, { x: 255, label: '3k' },
	{ x: 359.5, label: '4.5k' }, { x: 464, label: '6k' },
];

function chartLearningCurve() {
	return `<div class="hero-chart"><p class="hero-chart-title">Learning curve · mean episode reward</p>
	<svg viewBox="0 0 480 300" role="img" aria-label="Reward rises to 95.7 over 6000 iterations">
		<rect class="ch-frame" x="46" y="30" width="418" height="236"/>
		<line class="ch-base" x1="46" y1="246.3" x2="464" y2="246.3"/>
		<path class="ch-area" fill="#0e9b8e" d="M46,197.2 L73.9,256.2 L115.7,216.8 L150.5,167.7 L185.3,124.4 L220.2,89 L255,69.3 L289.8,61.5 L359.5,58.9 L464,58.1 L464,266 L46,266 Z"/>
		<polyline class="ch-line" stroke="#0e9b8e" points="46,197.2 73.9,256.2 115.7,216.8 150.5,167.7 185.3,124.4 220.2,89 255,69.3 289.8,61.5 359.5,58.9 464,58.1"/>
		<circle class="ch-dot" fill="#0e9b8e" cx="464" cy="58.1" r="4"/>
		<text class="ch-val" x="458" y="52" text-anchor="end">95.7</text>
		${ chartGrid( [ { y: 51, label: '100' }, { y: 100.75, label: '75' }, { y: 150.5, label: '50' }, { y: 200.25, label: '25' }, { y: 250, label: '0' } ], ITER_XTICKS ) }
		<text class="ch-axlbl" x="255" y="296" text-anchor="middle">training iterations</text>
	</svg></div>`;
}

function chartCurriculum() {
	return `<div class="hero-chart"><p class="hero-chart-title">Curriculum · stair riser height reached</p>
	<svg viewBox="0 0 480 300" role="img" aria-label="Curriculum riser height settles at 138 mm">
		<rect class="ch-frame" x="46" y="30" width="418" height="236"/>
		<line class="ch-target" x1="46" y1="113.3" x2="464" y2="113.3"/>
		<text class="ch-note" x="50" y="108">0.15 m · real target</text>
		<path class="ch-area" fill="#c07d3c" d="M46,182.7 L87.8,235.4 L115.7,224.3 L185.3,193.8 L255,168.8 L324.7,149.4 L394.3,135.5 L464,130 L464,266 L46,266 Z"/>
		<polyline class="ch-line" stroke="#c07d3c" points="46,182.7 87.8,235.4 115.7,224.3 185.3,193.8 255,168.8 324.7,149.4 394.3,135.5 464,130"/>
		<circle class="ch-dot" fill="#c07d3c" cx="464" cy="130" r="4"/>
		<text class="ch-val" x="458" y="124" text-anchor="end">138 mm</text>
		${ chartGrid( [ { y: 47, label: '200' }, { y: 95.75, label: '165' }, { y: 144.5, label: '130' }, { y: 193.25, label: '95' }, { y: 242, label: '60' } ], ITER_XTICKS ) }
		<text class="ch-axlbl" x="255" y="296" text-anchor="middle">training iterations · riser mm</text>
	</svg></div>`;
}

// "It learns to stop falling": share of training episodes that end in a topple,
// collapsing from ~92% early to ~3% at the plateau (ties to the "≈100% end
// upright" headline stat). Same smooth-polyline convention as the learning curve.
function chartFallRate() {
	return `<div class="hero-chart"><p class="hero-chart-title">Falls · episodes ending in a topple (%)</p>
	<svg viewBox="0 0 480 300" role="img" aria-label="Fall rate drops from 92% to about 3% over 6000 iterations">
		<rect class="ch-frame" x="46" y="30" width="418" height="236"/>
		<line class="ch-base" x1="46" y1="246.3" x2="464" y2="246.3"/>
		<path class="ch-area" fill="#e5484d" d="M46,67 L87.8,94.8 L129.6,140.5 L171.4,174.4 L213.2,202.2 L275.9,226.1 L338.6,238.1 L401.3,242 L464,244 L464,266 L46,266 Z"/>
		<polyline class="ch-line" stroke="#e5484d" points="46,67 87.8,94.8 129.6,140.5 171.4,174.4 213.2,202.2 275.9,226.1 338.6,238.1 401.3,242 464,244"/>
		<circle class="ch-dot" fill="#e5484d" cx="464" cy="244" r="4"/>
		<text class="ch-val" x="458" y="237" text-anchor="end">3%</text>
		${ chartGrid( [ { y: 51, label: '100' }, { y: 100.75, label: '75' }, { y: 150.5, label: '50' }, { y: 200.25, label: '25' }, { y: 250, label: '0' } ], ITER_XTICKS ) }
		<text class="ch-axlbl" x="255" y="296" text-anchor="middle">training iterations</text>
	</svg></div>`;
}

// "It survives longer": mean seconds an episode runs before it terminates —
// early policies fall in ~2 s, the trained one carries the payload the full
// climb (~19 s) before the episode ends. A distinct read from reward: how FAR
// it gets, not how well it's scored.
function chartSurvival() {
	return `<div class="hero-chart"><p class="hero-chart-title">Survival · mean episode length (s)</p>
	<svg viewBox="0 0 480 300" role="img" aria-label="Mean episode length rises from about 2 s to 19 s over 6000 iterations">
		<rect class="ch-frame" x="46" y="30" width="418" height="236"/>
		<line class="ch-base" x1="46" y1="246.3" x2="464" y2="246.3"/>
		<path class="ch-area" fill="#3a7ce5" d="M46,232.1 L87.8,220.2 L129.6,190.3 L171.4,155.5 L234.1,120.7 L296.8,90.8 L359.5,70.9 L422.2,61 L464,57 L464,266 L46,266 Z"/>
		<polyline class="ch-line" stroke="#3a7ce5" points="46,232.1 87.8,220.2 129.6,190.3 171.4,155.5 234.1,120.7 296.8,90.8 359.5,70.9 422.2,61 464,57"/>
		<circle class="ch-dot" fill="#3a7ce5" cx="464" cy="57" r="4"/>
		<text class="ch-val" x="458" y="51" text-anchor="end">≈19 s</text>
		${ chartGrid( [ { y: 51, label: '20' }, { y: 100.75, label: '15' }, { y: 150.5, label: '10' }, { y: 200.25, label: '5' }, { y: 250, label: '0' } ], ITER_XTICKS ) }
		<text class="ch-axlbl" x="255" y="296" text-anchor="middle">training iterations</text>
	</svg></div>`;
}

function chartSweep() {
	const bars = [
		{ x: 52.33, h: 103.2, y: 144.8, cls: 'ch-bar-reached', v: '28.4°', m: '0.12',  n: '4.7″' },
		{ x: 123.0, h: 86.5,  y: 161.5, cls: 'ch-bar-reached', v: '23.8°', m: '0.13',  n: '5.1″' },
		{ x: 193.67, h: 99.2,  y: 148.8, cls: 'ch-bar-reached', v: '27.3°', m: '0.14',  n: '5.5″' },
		{ x: 264.33, h: 111.2, y: 136.8, cls: 'ch-bar-reached', v: '30.6°', m: '0.15',  n: '6″ ADA' },
		{ x: 335.0, h: 115.2, y: 132.8, cls: 'ch-bar-partial', v: '31.7°', m: '0.175', n: '7″' },
		{ x: 405.67, h: 21.8,  y: 226.2, cls: 'ch-bar-none',    v: '6.0°',  m: '0.198', n: '7.75″' },
	];
	const rects = bars.map( ( b ) => `<rect class="${ b.cls }" x="${ b.x }" y="${ b.y }" width="46" height="${ b.h }" rx="3"/>` ).join( '' );
	const vals = bars.map( ( b ) => `<text class="ch-val" x="${ b.x + 23 }" y="${ b.y - 6 }" text-anchor="middle">${ b.v }</text>` ).join( '' );
	const labs = bars.map( ( b ) =>
		`<text class="ch-tick" x="${ b.x + 23 }" y="262" text-anchor="middle">${ b.m }</text>` +
		`<text class="ch-tick" x="${ b.x + 23 }" y="274" text-anchor="middle">${ b.n }</text>`,
	).join( '' );
	return `<div class="hero-chart"><p class="hero-chart-title">Stair-height sweep · peak body tilt (deg)</p>
	<svg viewBox="0 0 480 300" role="img" aria-label="Peak body tilt at each riser height">
		<line class="ch-target" x1="40" y1="30" x2="464" y2="30"/>
		<text class="ch-note" x="462" y="24" text-anchor="end">fall line · 60°</text>
		<line class="ch-axis" x1="40" y1="248" x2="464" y2="248"/>
		<text class="ch-tick" x="36" y="33" text-anchor="end">60°</text>
		<text class="ch-tick" x="36" y="251" text-anchor="end">0°</text>
		${ rects }${ vals }${ labs }
	</svg></div>`;
}

function artProblem() {
	return `<div class="hero-art"><svg viewBox="0 0 440 340" role="img" aria-label="An elderly person with a cane and an oxygen tank at the foot of a staircase">
		<line x1="20" y1="302" x2="420" y2="302" stroke="currentColor" stroke-width="1.4" opacity="0.5"/>
		<path d="M250 302 V272 H292 V242 H334 V212 H376 V182 H418 V302 Z" fill="#c07d3c" fill-opacity="0.9" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
		<path d="M250 272 H292 M292 242 H334 M334 212 H376 M376 182 H418" fill="none" stroke="currentColor" stroke-width="1" opacity="0.5"/>
		<line x1="256" y1="250" x2="424" y2="130" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
		<line x1="262" y1="272" x2="262" y2="256" stroke="currentColor" stroke-width="1.6"/>
		<line x1="418" y1="182" x2="418" y2="150" stroke="currentColor" stroke-width="1.6"/>
		<path d="M300 252 L360 208" fill="none" stroke="currentColor" stroke-width="1.4" stroke-dasharray="4 4" opacity="0.55"/>
		<path d="M360 208 l-11 1 4 -10" fill="none" stroke="currentColor" stroke-width="1.4" opacity="0.55"/>
		<path d="M403 150 l9 16 -18 0 Z" fill="none" stroke="#e5484d" stroke-width="1.8" stroke-linejoin="round"/>
		<line x1="403" y1="156" x2="403" y2="161" stroke="#e5484d" stroke-width="1.8" stroke-linecap="round"/>
		<circle cx="403" cy="164.5" r="0.9" fill="#e5484d"/>
		<circle cx="150" cy="188" r="14" fill="none" stroke="currentColor" stroke-width="2.4"/>
		<path d="M143 204 Q168 210 162 250 Q160 258 150 258 Q140 258 142 246 Q145 224 138 208 Z" fill="#b0552f" fill-opacity="0.9" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
		<path d="M148 256 L140 300 M156 254 L168 300" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>
		<path d="M158 224 L182 250" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/>
		<path d="M186 246 q6 -2 6 4 L194 300" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
		<rect x="86" y="236" width="30" height="64" rx="10" fill="#f4f2ec" stroke="currentColor" stroke-width="2"/>
		<rect x="95" y="224" width="12" height="16" rx="3" fill="#333" stroke="currentColor" stroke-width="1.4"/>
		<line x1="101" y1="224" x2="101" y2="216" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
		<path d="M101 236 Q120 200 146 206" fill="none" stroke="currentColor" stroke-width="1.6" opacity="0.7"/>
		<path d="M107 230 Q130 200 144 190" fill="none" stroke="#3a7ce5" stroke-width="1.8" stroke-linecap="round"/>
	</svg></div>`;
}

function artRoadmap() {
	return `<div class="hero-art"><svg viewBox="0 0 480 300" role="img" aria-label="Deployment roadmap from proven simulation to a home pilot">
		<line x1="60" y1="96" x2="180" y2="96" stroke="currentColor" stroke-width="2" opacity="0.75"/>
		<line x1="180" y1="96" x2="300" y2="96" stroke="currentColor" stroke-width="2" opacity="0.75"/>
		<line x1="300" y1="96" x2="420" y2="96" stroke="currentColor" stroke-width="2" stroke-dasharray="5 4" opacity="0.55"/>
		<circle cx="60"  cy="96" r="11" fill="#0e9b8e" stroke="var(--sheet)" stroke-width="2"/>
		<circle cx="180" cy="96" r="11" fill="#0e9b8e" stroke="var(--sheet)" stroke-width="2"/>
		<circle cx="300" cy="96" r="11" fill="none" stroke="currentColor" stroke-width="2.4"/>
		<circle cx="420" cy="96" r="11" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="4 3"/>
		<g text-anchor="middle">
			<text class="ch-val" x="60"  y="70">Isaac Sim</text><text class="ch-note" x="60"  y="122">full climb ✓</text>
			<text class="ch-val" x="180" y="70">One stack</text><text class="ch-note" x="180" y="122">sim ⇄ robot</text>
			<text class="ch-val" x="300" y="70">Jetson Orin</text><text class="ch-note" x="300" y="122">on the Go2</text>
			<text class="ch-val" x="420" y="70">Home pilot</text><text class="ch-note" x="420" y="122">supervised</text>
		</g>
		<text class="ch-axlbl" x="60" y="188">who benefits</text>
		<g>
			<rect x="60"  y="200" width="108" height="34" rx="8" fill="none" stroke="currentColor" stroke-width="1.3" opacity="0.6"/>
			<text class="ch-note" x="114" y="221" text-anchor="middle" font-size="11">patients</text>
			<rect x="186" y="200" width="108" height="34" rx="8" fill="none" stroke="currentColor" stroke-width="1.3" opacity="0.6"/>
			<text class="ch-note" x="240" y="221" text-anchor="middle" font-size="11">caregivers</text>
			<rect x="312" y="200" width="120" height="34" rx="8" fill="none" stroke="currentColor" stroke-width="1.3" opacity="0.6"/>
			<text class="ch-note" x="372" y="221" text-anchor="middle" font-size="11">clinics &amp; rehab</text>
		</g>
	</svg></div>`;
}

function teamCard() {
	return `<div class="hero-teamcard">
		<div class="tc-avatar"><img src="./assets/team/anthony.png" alt="Anthony Sinchi" /></div>
		<div class="tc-name">Anthony Sinchi</div>
		<div class="tc-role">Solo build</div>
		<div class="tc-rule"></div>
		<div class="tc-cred">Reinforcement learning · Perception · Isaac Sim · Sim ⇄ real deploy</div>
		<div class="tc-tags">
			<span>RL &amp; controls</span><span>perception</span><span>Isaac Sim</span><span>Jetson deploy</span>
		</div>
	</div>`;
}

// Opening / title visual: two Isaac Sim renders leaning open like a box's
// flaps, with the REAL hardware shot resting flat as the box beneath them —
// so the problem slide shows the thing already exists, not just a render.
function problemPhotos() {
	return `<div class="hero-photos">
		<figure class="hp-back"><img src="./assets/renders/stairs_climb.png" alt="The robot dog climbing a full staircase behind an oxygen-therapy patient" /></figure>
		<figure class="hp-front"><img src="./assets/renders/patient_carry.png" alt="Close-up: the Go2 carrying the patient's oxygen concentrator up the stairs" /></figure>
		<figure class="hp-real">
			<img src="./assets/renders/real_go2.png" alt="The real Unitree Go2 fitted with the oxygen-concentrator payload, on the floor of a lab" />
			<figcaption>the real Go2 + O₂ payload</figcaption>
		</figure>
	</div>
	<figcaption class="hero-photos-cap">Isaac Sim renders — and the real hardware carrying the oxygen</figcaption>`;
}

// "Higher stairs slow it down": real per-height climb time (s) at the heights it
// completed. Rises 35.5 -> 51.2 s as the riser grows 0.12 -> 0.15 m (sweep data).
function chartClimbTime() {
	const bars = [
		{ x: 60,  m: '0.12', n: '4.7″', t: 35.5 },
		{ x: 160, m: '0.13', n: '5.1″', t: 35.9 },
		{ x: 260, m: '0.14', n: '5.5″', t: 43.4 },
		{ x: 360, m: '0.15', n: '6″ ADA', t: 51.2 },
	];
	const maxT = 56, y0 = 250, top = 30, w = 60;
	const px = ( t ) => y0 - ( t / maxT ) * ( y0 - top );
	const rects = bars.map( ( b ) => `<rect class="ch-bar-slow" x="${ b.x }" y="${ px( b.t ) }" width="${ w }" height="${ y0 - px( b.t ) }" rx="3"/>` ).join( '' );
	const vals = bars.map( ( b ) => `<text class="ch-val" x="${ b.x + w / 2 }" y="${ px( b.t ) - 6 }" text-anchor="middle">${ b.t.toFixed( 1 ) }s</text>` ).join( '' );
	const labs = bars.map( ( b ) =>
		`<text class="ch-tick" x="${ b.x + w / 2 }" y="266" text-anchor="middle">${ b.m }</text>` +
		`<text class="ch-tick" x="${ b.x + w / 2 }" y="278" text-anchor="middle">${ b.n }</text>`,
	).join( '' );
	return `<div class="hero-chart"><p class="hero-chart-title">Higher stairs, slower climb · time to the top (s)</p>
	<svg viewBox="0 0 480 300" role="img" aria-label="Climb time rises from 35.5 s to 51.2 s as the riser grows">
		<line class="ch-axis" x1="40" y1="250" x2="464" y2="250"/>
		<polyline class="ch-trend" points="90,${ px( 35.5 ) } 190,${ px( 35.9 ) } 290,${ px( 43.4 ) } 390,${ px( 51.2 ) }"/>
		${ rects }${ vals }${ labs }
	</svg></div>`;
}

const VISUALS = {
	problem: problemPhotos,
	roadmap: artRoadmap,
	team: teamCard,
	training: () => `<div class="hero-charts cols-2">${ chartLearningCurve() }${ chartCurriculum() }${ chartSurvival() }${ chartFallRate() }</div>`,
	sweep: () => `<div class="sweep-visual">
		<div class="hero-charts cols-2">${ chartSweep() }${ chartClimbTime() }</div>
		<figure class="sweep-fall">
			<div class="sweep-fall-frame"><video muted loop playsinline preload="none" data-src="./assets/clips/fall.mp4"></video></div>
			<figcaption>Past code-legal risers the gait rears up against the step and can’t get over it — real Isaac Sim, 0.26 m riser.</figcaption>
		</figure>
	</div>`,
};

export function buildVisualHTML( key ) {
	return ( VISUALS[ key ] || ( () => '' ) )();
}

export function buildCopyHTML( copy ) {
	if ( ! copy ) return '';
	let html = '';
	if ( copy.lead ) html += `<p class="hero-copy-lead">${ copy.lead }</p>`;
	if ( copy.body ) html += `<p class="hero-copy-body">${ copy.body }</p>`;
	// `faults`: a "what broke → the fix" list (Challenges slide). Each row pairs a
	// failure with the change that fixed it, for a talk-through of the engineering.
	if ( copy.faults && copy.faults.length ) {
		html += '<ul class="hero-faults">' + copy.faults.map( ( f ) =>
			`<li class="hero-fault"><span class="hf-p">${ f.p }</span><span class="hf-f">${ f.f }</span></li>`,
		).join( '' ) + '</ul>';
	}
	if ( copy.stats && copy.stats.length ) {
		html += '<div class="hero-stats">' + copy.stats.map( ( s ) =>
			`<div class="hero-stat"><span class="hero-stat-v">${ s.v }</span><span class="hero-stat-l">${ s.l }</span></div>`,
		).join( '' ) + '</div>';
	}
	if ( copy.tags && copy.tags.length ) {
		html += '<div class="hero-tags">' + copy.tags.map( ( t ) => `<span class="hero-tag">${ t }</span>` ).join( '' ) + '</div>';
	}
	return html;
}

// ---------------------------------------------------------------------------
// Content sections (everything except the 3 Solution stages + the demo).
// `visual` is a VISUALS key rendered into the left column; `copy` fills the
// right column.
// ---------------------------------------------------------------------------
export const CONTENT = {
	's-problem': {
		group: 'Problem', title: 'TASH — Stair-Climbing Robotic Oxygen Carrier',
		visual: 'problem',
		copy: {
			lead: 'Oxygen-therapy patients are <b>tethered to their supply</b> — wherever they go, the concentrator goes too.',
			body: 'Millions of people with <b>COPD</b>, <b>pulmonary fibrosis</b> and other chronic lung disease are prescribed <b>long-term oxygen</b> for everyday life at home. The equipment is <b>heavy and awkward</b>, and a staircase turns an ordinary trip into a <b>two-hands-full balancing act</b>. Many simply <b>stop using the stairs</b> — and lose a whole floor of their own home. Our idea is simple: let a <b>robot dog carry the oxygen</b> and climb alongside them, so they don’t have to.',
			stats: [
				{ v: '~1.5 M', l: 'Americans on home oxygen (est.)' },
				{ v: '2–3 kg', l: 'portable unit to carry' },
				{ v: 'Leading', l: 'injury cause, age 65+' },
			],
			tags: [ 'independence', 'fall risk', 'daily burden' ],
		},
	},
	's-training': {
		group: 'Progress & results', title: 'We trained it to climb — 6,000 iterations',
		visual: 'training',
		copy: {
			lead: 'A <b>blind reinforcement-learning policy</b> learned to drive the payload upstairs <b>on feel alone</b>.',
			body: 'Across <b>6,000 training iterations</b> the mean episode reward climbed to <b>95.7</b> and held. A <b>difficulty curriculum</b> pushed the stairs steeper over time, settling at a <b>138 mm riser</b> — right at the real-world target height. Along the way the policy learned to <b>stop falling</b> — topples collapsed from <b>92% of episodes to about 3%</b> — and to <b>survive longer</b>, carrying the payload the full climb instead of tipping in the first couple of seconds. The gait it found is <b>cautious, not graceful</b> — it noses down and leans into each riser rather than stepping cleanly — but inside that trained envelope it <b>ends upright almost every time</b>, keeping the payload level.',
			stats: [
				{ v: '95.7', l: 'mean reward at plateau' },
				{ v: '138 mm', l: 'riser the curriculum settled at' },
				{ v: '≈100%', l: 'episodes end upright, not a fall' },
			],
			tags: [ 'blind RL', 'Go2 + 2.22 kg O₂', 'Isaac Sim' ],
		},
	},
	's-sweep': {
		group: 'Progress & results', title: 'How high it climbs — and where it stops',
		visual: 'sweep',
		copy: {
			lead: 'We swept real staircase heights from <b>4.7″ up past the code maximum</b> — carrying the oxygen tank the whole way, and pushing until it fails.',
			body: 'Up to the <b>ADA-legal 6-inch riser</b> it climbs the <b>full 14-step staircase</b> and follows the patient to the top — but not for free: <b>every extra inch of riser slows it down</b>, from <b>35 s at 4.7″</b> to <b>51 s at 6″</b>. Push past code-legal stairs and it <b>runs out of reach</b> — a <b>7-inch riser</b> gets it barely a third of the way up — and steeper still it <b>rears up against the step and can’t get over it</b> at all. That limit sits right about where a person would want a <b>stair-lift</b> too.',
			stats: [
				{ v: '6″ ADA', l: 'highest riser it fully tops out' },
				{ v: '+44%', l: 'slower climb from 4.7″ to 6″' },
				{ v: '7″ +', l: 'too steep — stalls out, can’t climb' },
			],
			tags: [ 'ADA 6″ ✓', 'higher = slower', 'past code-max = stalls out' ],
		},
	},
	's-challenges': {
		// Visual is the hand-authored three.js topple (js/topple.js mounts into
		// #topple-stage) — no `visual` key, so deck.js leaves the bespoke host alone.
		group: 'Engineering challenges', title: 'Everything that broke — and the fix',
		copy: {
			lead: 'Getting a <b>blind</b> dog to climb stairs — carrying a payload, beside a person — <b>broke in every way you’d expect</b>. Each failure sent us back to the sim with a fix.',
			faults: [
				{ p: 'The first policy was <b>reckless</b> — it lunged at the steps and crashed.', f: '<b>Retrained from a stronger checkpoint</b> until the gait commits to each step <b>cautiously and stays upright</b>.' },
				{ p: 'It <b>couldn’t see</b> the stairs underfoot.', f: 'A <b>blind RL policy</b> that climbs by feel — <b>proprioception + foot contact</b>, no camera needed on the steps.' },
				{ p: 'It mistook the <b>patient</b> for a staircase.', f: 'Mask the person out of the depth view and require a <b>real stair detection</b> before ever committing to climb mode.' },
				{ p: 'It <b>crowded the patient</b> on the incline.', f: '<b>Gap-aware pacing</b> so the dog trails the person up the stairs instead of closing in on them.' },
			],
			tags: [ 'sim-in-the-loop', 'fail, fix, repeat', 'safety first' ],
		},
	},
	's-potential': {
		group: 'Real-world potential', title: 'From a proven sim to the living room',
		visual: 'roadmap',
		copy: {
			lead: 'The <b>same control container</b> runs in simulation and on the robot — <b>only the sensor source changes</b>.',
			body: 'That is the whole deployment bet: a policy <b>proven in Isaac Sim</b> is expected to hold on an <b>NVIDIA Jetson Orin</b> carried by a real Go2, because it runs the <b>identical Docker stack</b>. The payoff is <b>a floor of the house given back</b> — <b>patients keep their stairs</b>, caregivers get a hand, and clinics and rehab facilities get an assistant that never tires. Next up: <b>hardware bring-up</b>, gait polish, and a <b>supervised home pilot</b>.',
			stats: [
				{ v: 'Sim ✓', l: 'full climb, proven' },
				{ v: '1 stack', l: 'identical sim ⇄ robot' },
				{ v: 'Jetson', l: 'Orin edge compute' },
			],
			tags: [ 'home pilot', 'care facilities', 'aging in place' ],
		},
	},
	's-team': {
		group: 'Team', title: 'Built by one',
		visual: 'team',
		copy: {
			lead: 'One person — <b>the whole stack</b>.',
			body: 'Anthony designed and built everything you just saw: the <b>reinforcement-learning climb and walking policies</b>, the <b>YOLO-based perception</b> and person-following, the <b>Isaac Sim world and payload physics</b>, the <b>sim ⇄ real Docker deploy stack</b> for the Jetson Orin robot — and <b>this interactive presentation</b> itself.',
			tags: [ 'RL & controls', 'perception', 'Isaac Sim', 'sim ⇄ real deploy' ],
		},
	},
};
