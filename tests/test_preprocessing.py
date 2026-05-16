import numpy as np

from preprocessing import crop_face


def test_crop_face_clamps_boxes():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    crop = crop_face(frame, (-10, -10, 50, 50), output_size=112)
    assert crop is not None
    assert crop.image.size == (112, 112)
