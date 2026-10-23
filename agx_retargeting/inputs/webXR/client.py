def patched_client(source: str) -> str:
    def replace(before: str, after: str) -> None:
        nonlocal source
        if after in source:
            return
        if source.count(before) != 1:
            raise ValueError(
                "Unexpected xr_media client; install the pinned webxr extra"
            )
        source = source.replace(before, after, 1)

    guard = (
        "function onFrame(_time, frame) {\n  if (!session || !frame || !space) return;"
    )
    if guard not in source:
        before = "function onFrame(_time, frame) {\n  if (!session || !frame) return;"
        if before not in source:
            before = "function onFrame(_time, frame) {\n  if (!session) return;"
        replace(before, guard)
    replace(
        '    renderer.xr.setReferenceSpaceType("local-floor");\n    await renderer.xr.setSession(session);',
        '    renderer.xr.setReferenceSpaceType("local-floor");\n'
        "    renderer.setAnimationLoop(onFrame);\n    await renderer.xr.setSession(session);",
    )
    old_loop = "  renderer.setAnimationLoop(onFrame);\n}\n\nfunction setLocked"
    if old_loop in source:
        source = source.replace(old_loop, "}\n\nfunction setLocked", 1)
    if source.count("renderer.setAnimationLoop(onFrame);") != 1:
        raise ValueError("Unexpected XR animation callback registration count")
    replace(
        '    reportError("initialization", error);\n    await session.end()',
        '    reportError("initialization", error);\n'
        "    renderer?.setAnimationLoop(null);\n    await session.end()",
    )
    # Hand retargeting requires joints; optional features may be silently denied.
    replace(
        'requiredFeatures:["local-floor"], optionalFeatures:["dom-overlay","hand-tracking"]',
        'requiredFeatures:["local-floor","hand-tracking"], optionalFeatures:["dom-overlay"]',
    )
    replace(
        'reportPhase("session-created", { mode, overlay:!!session.domOverlayState });',
        'reportPhase("session-created", { mode, overlay:!!session.domOverlayState, '
        "features:Array.from(session.enabledFeatures || []) });",
    )
    replace(
        "function renderFrame(_time, frame) {",
        'let lastInputStatus = "";\nfunction renderFrame(_time, frame) {',
    )
    replace(
        "  const zUpHmd = yupToZup(hmd);",
        "  const inputStatus = {\n"
        "    visibility:session.visibilityState,\n"
        "    sources:Array.from(session.inputSources, source => ({\n"
        "      side:source.handedness, hand:!!source.hand,\n"
        "    })),\n"
        "    left:{ tracked:hands.left.tracked, joints:Object.keys(hands.left.joints).length },\n"
        "    right:{ tracked:hands.right.tracked, joints:Object.keys(hands.right.joints).length },\n"
        "  };\n"
        "  const inputKey = JSON.stringify(inputStatus);\n"
        "  if (inputKey !== lastInputStatus) {\n"
        "    lastInputStatus = inputKey;\n"
        '    reportPhase("input-status", inputStatus);\n'
        "    ui.ready.textContent = `XR active · L ${inputStatus.left.joints}/25 · R ${inputStatus.right.joints}/25`;\n"
        "  }\n"
        "  const zUpHmd = yupToZup(hmd);",
    )
    return source
