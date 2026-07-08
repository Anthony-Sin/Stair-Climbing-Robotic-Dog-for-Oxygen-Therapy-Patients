# Vendored third-party assets

## Xbot.glb

Source: `three.js` official examples repo, `examples/models/gltf/Xbot.glb`
(https://github.com/mrdoob/three.js), fetched 2026-07-07 from
`raw.githubusercontent.com/mrdoob/three.js/master/examples/models/gltf/Xbot.glb`.

This is Adobe Mixamo's default "X Bot" rigged mannequin (67-joint skeleton,
untextured), bundled by three.js for its animation-blending example
(`webgl_animation_skinning_blending`) and used across countless three.js
tutorials/demos under that repo's MIT license. No textures (flat PBR
`baseColorFactor` materials only), which is why it was picked over the other
stock three.js character (`Soldier.glb`, a textured sci-fi "Vanguard" model) —
it tints cleanly with this viewer's flat toon material instead of clashing
with a pre-authored diffuse texture.

Used by the blueprint viewer as the patient mannequin's mesh + skeleton +
"walk" AnimationClip (arm swing / spine sway / hip-bob only — the leg bones
are overridden every frame by this pipeline's own procedurally-computed
angles; see `js/main.js`'s patient-retargeting code and
`pipeline/anim_bake.py`'s `patient_pose` output for why: a canned mocap walk
cycle has no idea where OUR stairs' risers are, so blindly playing it through
a climb would clip through/float above the treads).
