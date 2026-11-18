import pytest

from src.core.registry import BACKBONES, STEMS, Registry, RegistryError


def test_builds_from_name_only():
    b = BACKBONES.build("set_transformer")
    assert b.d_model == 256


def test_build_from_mapping_with_kwargs():
    b = BACKBONES.build({"name": "set_transformer", "d_model": 64, "n_layers": 1})
    assert b.d_model == 64


def test_unknown_name_suggests_alternatives():
    with pytest.raises(RegistryError) as e:
        BACKBONES.build({"name": "set_transfomer"})
    assert "set_transformer" in str(e.value)


def test_underscore_keys_are_annotations_not_kwargs():
    b = BACKBONES.build({"name": "set_transformer", "d_model": 64, "_note": "hi"})
    assert b.d_model == 64


def test_bad_kwarg_reports_signature():
    with pytest.raises(RegistryError) as e:
        BACKBONES.build({"name": "set_transformer", "nonexistent": 1})
    assert "signature" in str(e.value)


def test_double_registration_refused():
    r = Registry("thing")

    @r.register("a")
    class A:
        pass

    with pytest.raises(RegistryError):
        @r.register("a")
        class B:
            pass


def test_base_class_is_enforced():
    with pytest.raises(RegistryError):
        @STEMS.register("not_a_stem")
        class NotAStem:
            pass
