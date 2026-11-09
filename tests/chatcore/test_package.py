import importlib

import pytest

import chatcore


def test_package_exposes_version():
    assert chatcore.__version__ == "0.0.1"


def test_package_is_installed_from_src():
    assert chatcore.__file__ is not None and "src/chatcore" in chatcore.__file__.replace("\\", "/")


def test_unknown_submodule_is_not_importable():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("chatcore.does_not_exist")
