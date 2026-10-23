---
name: agx-adapt-input
description: Add or correct an adapter from camera, glove, tracking SDK or recorded data to agx-retargeting HandInput. Use for landmark semantics, units, coordinates, timestamps and tracking recovery; exclude robot hand kinematics and actuator control.
---

# Adapt an Input Device

Convert device data or recordings into `HandInput` through a device adapter. Implement conversion, acquisition or replay, lifecycle handling, and verification required by the requested integration.

## Locate the integration points

From the target repository root, read `agx_retargeting/types.py`, `agx_retargeting/adapter.py`, `agx_retargeting/core/retargeter.py` and `agx_retargeting/inputs/webXR/adapter.py`. Consult `agx_retargeting/core/hand_model.py` for coordinate alignment and `agx_retargeting/inputs/webXR/server.py` when live-stream lifecycle handling is relevant. Establish semantics from the supplied SDK, documentation and samples. `InputAdapter[Source].adapt(source, side=..., timestamp=...)` converts one device sample; acquisition exposes `InputModel.read()` and `close()`. `HandInput` holds immutable batched arrays for computation. These interfaces use structural Python protocols.

For controller poses, joint angles or incomplete skeletons, determine whether documented bone geometry and calibration can recover the required 3D points. Complete the known interface work and ask for missing skeleton or calibration data where necessary. Do not present invented points, zero padding or undocumented bone lengths as valid tracking.

## Conversion contract

- A valid frame contains finite positions with shape `(21, 3)`, ordered by `KEYPOINT_NAMES`: wrist; thumb CMC/MCP/IP/TIP; then MCP/PIP/DIP/TIP for index, middle, ring and pinky. Map anatomical meanings explicitly rather than truncating an SDK array or assuming identical names imply identical origins.
- Express all points in one common right-handed frame, in metres. Accumulate local transforms before conversion and account for axis conversions already performed by the SDK. `HandInput.from_positions()` converts `m`, `cm` and `mm`; convert other units in the adapter.
- Preserve available device orientations as `(21, 4)` finite, nonzero xyzw quaternions representing local joint axes in that same source frame. `HandInput` normalizes them. Position-only devices set `quaternions=None`; the online tasks derive palm orientation and terminal directions from measured positions. Optional `(21, 3)` velocities use metres per second in the source frame.
- Set `side` to the actual `left` or `right` hand. `frame_id` is metadata and triggers no TF lookup or coordinate transform. The core aligns human and robot palm frames, so device conversion must not depend on a particular robot's fixed alignment rotation.

## Time and tracking state

Valid timestamps are finite, nonnegative seconds and strictly increase for each hand. Use the source clock for frame intervals and host monotonic receipt time for live-data expiry; never subtract timestamps from different clocks. Drop duplicate frames before solving. On a new session or clock rollback, emit tracking loss or call `Retargeter.reset()` before accepting the new time sequence.

For reported tracking loss or missing required landmarks, return `HandInput(None, side, timestamp, tracked=False, frame_id=...)`. The caller passes that state to the retargeter to clear history. Reject malformed numeric positions and orientations immediately. A stream with no new sample retains its previous tracking state until receipt-age expiry. Live streams favor the latest frame; offline replay advances using recorded timestamps.

## Modules and dependencies

Place a conversion class in `agx_retargeting/inputs/<device>/adapter.py` that satisfies `InputAdapter`; follow `WebXRAdapter.adapt()` for an existing implementation. Import shared data from `agx_retargeting.types`, adapter protocols from `agx_retargeting.adapter`, and retargeting classes from `agx_retargeting.core`. Resolve bundled YAML files through `agx_retargeting.config.CONFIG_DIR`. Keep acquisition or replay in its own module and release connections, threads and services. Keep device-specific formats in the input layer and extend the core when the interface requires it. Add device SDKs to the relevant optional dependency group and import them directly in the device modules. Subpackages are discovered under `agx_retargeting`; check CLI entry points in `pyproject.toml`. The WebXR application runs with `python -m agx_retargeting.inputs.webXR`.

## Completion criteria

Use available real recorded samples to verify landmark semantics, scale, hand side, quaternion conventions and axis directions, and preserve their provenance. Cover the missing-landmark, nonfinite-data, duplicate-time, restart and tracking-recovery cases relevant to the change; verify expiry and latest-frame behavior for live streams. With an existing hand model, confirm valid frames solve in `hand.joint_names` order, loss returns `None`, and recovery performs q-method bone-end initialization with fresh time and motion history. Use the real numerical and device libraries in verification. Put temporary results in ignored `.work/`.

Deliver conversion code, required entry points and dependencies, a minimal run command, and landmark, coordinate, unit and clock conventions. State sample or device limitations. Update only affected README installation and usage instructions; keep each prose paragraph on one source line.
