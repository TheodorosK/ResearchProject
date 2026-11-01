import pytest

from lumen import config, embedder


def test_every_model_choice_is_complete():
    for key, spec in config.MODELS.items():
        for field in ("label", "description", "image", "text"):
            assert spec[field], (key, field)


def test_large_siglip_is_registered_with_fastembed():
    pytest.importorskip("fastembed")
    from fastembed.image import siglip_embedding as image_mod
    from fastembed.text import siglip_embedding as text_mod

    embedder.register_models()
    embedder.register_models()  # harmless to repeat
    spec = config.MODELS["siglip2-large"]
    image = [m for m in image_mod.supported_siglip_models if m.model == spec["image"]]
    text = [m for m in text_mod.supported_siglip_models if m.model == spec["text"]]
    assert len(image) == len(text) == 1
    assert image[0].model_file == spec["vision_file"] and text[0].model_file == spec["text_file"]
    assert image[0].dim == text[0].dim == spec["dim"]
    assert image[0].sources.hf == spec["hf"]


def test_no_int8_weights_for_siglip_choices():
    # int8 exports measurably hurt (Large) or break (So400m) SigLIP 2 retrieval.
    for spec in config.MODELS.values():
        assert "int8" not in spec.get("vision_file", "") + spec.get("text_file", "")
