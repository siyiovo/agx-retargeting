---
name: agx-adapt-output
description: Add or correct a dexterous-hand model or output adapter in agx-retargeting, including URDF assets, landmark YAML, palm alignment, independent-joint mapping and actuator commands. Use for hand geometry, kinematics or output SDK integration; operate hardware only within the requested scope.
---

# Adapt a Dexterous Hand

Integrate the requested hand through URDF assets and YAML configuration. Extend shared interfaces only where the existing model cannot represent the hand.

## Locate the integration points

From the target repository root, read `agx_retargeting/types.py`, `agx_retargeting/adapter.py`, `agx_retargeting/core/hand_model.py` and the closest `agx_retargeting/config/*_params.yaml`. Landmark names and `HandInput` live in `types.py`; `QMethodInitializer` lives in `core/retargeter.py`; task evaluation, numerical kernels and `VelocityProjection` live in `core/residual.py`; optimizer backends live in `opt/`. Treat current source as authoritative. The core currently assumes five fingers and 21 human landmarks; identify necessary core changes for other hand structures.

Establish the requested hand sides, model provenance and landmark origins from available files. Ask only when missing information prevents a correct mapping, while completing independent work. Model-only integration does not require a hardware SDK or actuator calibration.

## Model contract

- Place URDFs in `agx_retargeting/description/urdf/<model>/` and geometry in `agx_retargeting/description/meshes/<model>/`. Relative `model.urdf_path` values resolve under `description/urdf/` beside the configuration directory. Preserve licenses and resolvable relative references. `HandModel` currently supports `revolute` and `prismatic` joints with limits; mimic joints must directly reference independent joints. Explain unsupported structures rather than removing physical coupling to bypass a limitation.
- Create or update `agx_retargeting/config/<model>_params.yaml`. Order `links.fingers` as thumb, index, middle, ring and pinky. Templates accept `{side}`, `{side_short}` and `{finger}`, with thumb-specific role overrides. Required landmarks are the wrist, five TIPs, thumb IP and four non-thumb DIPs. Additional anatomical correspondences participate in bone-end initialization.
- Keep `links.palm` fixed to the URDF root. Automatic palm-basis estimation uses the neutral-pose wrist and four non-thumb MCPs. If MCP frames are unavailable, specify four suitable `palm_landmarks` in index, middle, ring and pinky order. Check geometric origins, not just link names; palm references and DIP/IP-to-TIP directions must be nondegenerate.
- Omit `basis_rotation` for automatic estimation; an explicit value replaces that basis. `input_rotation` adds calibration in the robot root frame. Both use extrinsic `xyz` Euler angles in degrees and support separate hand sides. `scale` is a positive length multiplier. Device coordinate conversion remains in the input adapter.
- Configure explicit bone endpoint correspondences for q-method initialization. Online geometry uses five wrist-to-TIP vectors, four thumb-to-TIP vectors and five DIP/IP-to-TIP directions. Direction residuals compare spherical logarithms in fixed tangent planes defined by the model's joint-range midpoint; `bone_sigma` uses radians. Verify the model's directional range stays within each reference's regular chart. `HandModel.evaluate()` returns base-frame positions and position Jacobians; `evaluate_poses()` provides full SE(3) poses with LOCAL body Jacobians when needed by external consumers. Match the structural `OutputModel` protocol and keep independent-joint coordinates throughout.
- Check that local targets use the observed palm basis and remain fixed during each solve. Wrist correction has direct translation and an SO(3) rotation vector; its SE(3) prior and world position-error change task share the online implementation used by offline evaluation.

## Application and output integration

Load custom models with `python -m agx_retargeting.inputs.webXR --config agx_retargeting/config/<model>_params.yaml`. Import model and retargeting classes from `agx_retargeting.core`, optimizer interfaces from `agx_retargeting.opt`, shared data from `agx_retargeting.types`, and adapter protocols from `agx_retargeting.adapter`. Resolve bundled profiles through `agx_retargeting.config.CONFIG_DIR`. Register a built-in alias in `agx_retargeting/inputs/webXR/app.py` and `session.py` when requested. Package-data patterns in `pyproject.toml` collect configuration and model assets under `agx_retargeting`; verify an installed wheel can resolve the new assets and retains model license notices. Change solver settings or dependencies only when the integration requires it.

Implement a pure conversion class satisfying `OutputAdapter[Command].adapt(q)`, following `Revo2Adapter` in `agx_retargeting/outputs/revo2.py`. Map independent coordinates by `hand.joint_names`; revolute values use radians and prismatic values use metres. Keep transmission, deadlines and feedback in the output driver wrapper. The application owns connection lifetime. `Retargeter` exposes the solved joint configuration as `q_output`; the adapter converts that command. Joint motion priors belong to the optimizer. Inspect the actual SDK for channel order, direction, units and calibration endpoints. Establish physical calibration with measurements; URDF limits provide geometric bounds.

## Completion criteria

Load `HandConfig` and `HandModel` for the requested sides. Verify configured frames, mesh references, independent-joint order, effective limits, palm basis, FK, TIP poses and bone endpoint references. Use available real recordings to check finite bounded results and analytic Jacobians against finite differences. Verify pure command mapping at measured or documented endpoints and validate transport with the real SDK. Hardware calibration requires device measurements. Preserve recording provenance and put temporary results in ignored `.work/`.

Deliver the changed assets and configuration, a minimal run command, landmark/joint mappings and verification results. Distinguish model loading, working retargeting and calibrated hardware operation. Update only affected README usage and parameters; keep each prose paragraph on one source line.
