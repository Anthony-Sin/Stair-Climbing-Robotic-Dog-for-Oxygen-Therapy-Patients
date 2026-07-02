"""Fine-tuning scaffold for the Extreme-Parkour Go2 depth-vision policy.

This package fine-tunes the perceptive *depth encoder* (`RecurrentDepthBackbone`,
shipped as ``sim/models/locomotion/parkour/vision_weight.pt``) for the
oxygen-therapy environment, on a cloud GPU (RunPod RTX 6000 Ada).

It deliberately reuses the live model code in ``sim/isaac/`` as the single source
of truth (see ``sim_model_source.py``) rather than copying architecture/weights, and writes
checkpoints in the exact ``depth_encoder_state_dict`` format the runtime already
loads -- so a fine-tuned encoder is drop-in for the robot via
``--parkour-vision-weight``.

Entry points:
  * ``python fine_tuning/preflight.py``           -- environment/credential report
  * ``python fine_tuning/train.py --smoke-test``  -- end-to-end smoke test (no data needed)
  * ``python fine_tuning/train.py --episodes DIR`` -- real fine-tune (once the sim emits data)

The real sim->training data emitter is built separately; this package defines the
data contract it must satisfy in ``data/contract.py``.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
