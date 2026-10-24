"""Stream A: environment adapters (LIBERO, LIBERO-Pro, desk sort, bin sort; Panda and UR5e).

Heavy simulator imports are lazy so ``strip_privileged`` stays cheap to import.
"""
from motormem.env.privileged import strip_privileged

__all__ = ["ArmKinematics", "CAMERA_MAP", "RobotEnv", "make_env", "strip_privileged"]


def make_env(spec, **kwargs):
    """Build (not reset) the ``RobotEnv`` for an ``EpisodeSpec``."""
    from motormem.env.libero import make_env as _make_env
    return _make_env(spec, **kwargs)


def __getattr__(name):
    if name in ("ArmKinematics", "RobotEnv", "CAMERA_MAP"):
        from motormem.env import robot_env
        return getattr(robot_env, name)
    raise AttributeError(name)
