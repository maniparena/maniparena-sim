"""Enable the OpenCV lens-distortion model on the QUANTA_X1 camera prims.

The robot USD authors the calibrated intrinsics (``omni:lensdistortion:opencv*:*``) but not the
``omni:lensdistortion:model`` token, and Isaac Lab skips ``spawn.func`` for camera prims that
already exist, so without this the cameras render as a plain perspective camera (default 12 mm
focal length, ~110 deg horizontal FOV) instead of the calibrated lens used for data collection.
"""

from __future__ import annotations

# camera prim name -> (model token, schema API)
_LENS_MODELS = {
    "Head_Front_Color_Camera": ("opencvPinhole", "OmniLensDistortionOpenCvPinholeAPI"),
    "Left_Gripper_Camera": ("opencvFisheye", "OmniLensDistortionOpenCvFisheyeAPI"),
    "Right_Gripper_Camera": ("opencvFisheye", "OmniLensDistortionOpenCvFisheyeAPI"),
}


def enable_usd_lens_models(stage=None) -> int:
    """Set the lens-distortion model token on every matching camera prim; returns the count."""
    import omni.usd
    from pxr import Sdf, Tf

    stage = stage or omni.usd.get_context().get_stage()
    count = 0
    for prim in stage.Traverse():
        spec = _LENS_MODELS.get(prim.GetName())
        if spec is None or prim.GetTypeName() != "Camera":
            continue
        model, api = spec
        try:
            prim.ApplyAPI(api)
        except (Tf.ErrorException, RuntimeError):
            pass  # schema not registered: the raw attributes below are enough
        attr = prim.GetAttribute("omni:lensdistortion:model")
        if not attr:
            attr = prim.CreateAttribute("omni:lensdistortion:model", Sdf.ValueTypeNames.Token)
        attr.Set(model)
        # A leftover ftheta projection would override the OpenCV model.
        proj = prim.GetAttribute("cameraProjectionType")
        if proj and proj.Get() == "fisheyePolynomial":
            proj.Clear()
        count += 1
    return count
