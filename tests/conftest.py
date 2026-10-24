import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "sim: needs the MuJoCo simulator (slower; EGL rendering)")


@pytest.fixture(scope="session")
def mujoco_gl_egl():
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
