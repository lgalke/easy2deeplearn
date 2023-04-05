from language import load_input_language


def test_language():
    lang1 = load_input_language("B1")

    assert isinstance(lang1.get_word_by_id(2), str)

    scene = lang1.get_scene_by_id(2)
    assert isinstance(scene, tuple)
    shape, angle = scene
    print(shape)
    print(angle)
    assert isinstance(shape, int)
    assert isinstance(angle, int)
