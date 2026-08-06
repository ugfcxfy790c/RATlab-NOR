"""
Shared geometry helpers -- no SLEAP/OpenCV dependency, so both the
(GUI) object picker and the (headless) scoring module can use it.
"""


def hitbox_half_width_px(cfg):
    """Half-width, in pixels, of the square hitbox drawn around each
    object: the real-world object footprint (OBJECT_SIZE_CM) plus a
    padding margin (OBJECT_PADDING_CM) on every side, converted to pixels
    via PX_PER_CM."""
    return (cfg.OBJECT_SIZE_CM / 2.0 + cfg.OBJECT_PADDING_CM) * cfg.PX_PER_CM


def object_half_width_px(cfg):
    """Half-width, in pixels, of the object's actual physical footprint
    (unpadded) -- the target region for the head-orientation test, as
    opposed to hitbox_half_width_px()'s padded proximity region.

    Grown by OBJECT_FOOTPRINT_GROW_PX as a small rotation/off-axis
    tolerance, independent of OBJECT_SIZE_CM/OBJECT_PADDING_CM so it
    doesn't also grow the padded hitbox or CLIMBING_TORSO_DISTANCE_PX.
    """
    grow = getattr(cfg, "OBJECT_FOOTPRINT_GROW_PX", 0.0)
    return max(0.0, (cfg.OBJECT_SIZE_CM / 2.0) * cfg.PX_PER_CM + grow)
