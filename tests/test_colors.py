import unittest

import numpy as np

from gatevision.colors import clothing_colors, dominant_color, vehicle_color


def patch(bgr, h=120, w=160):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bgr
    return img


class ColorTests(unittest.TestCase):
    def test_basic_names(self):
        cases = {"white": (245, 245, 245), "black": (15, 15, 15), "red": (30, 30, 200),
                 "blue": (200, 80, 20), "green": (40, 160, 40), "yellow": (30, 220, 230),
                 "silver": (170, 172, 175), "gray": (100, 100, 100)}
        for name, bgr in cases.items():
            self.assertEqual(dominant_color(patch(bgr)), name, name)

    def test_vehicle_ignores_edges(self):
        img = patch((245, 245, 245))
        img[:, :10] = (20, 20, 20)      # dark tyre/edge strip
        img[:25, :] = (20, 20, 20)      # windscreen band
        self.assertEqual(vehicle_color(img), "white")

    def test_clothing_top_and_bottom(self):
        img = np.zeros((300, 100, 3), np.uint8)
        img[:150] = (30, 30, 200)       # red shirt (upper half)
        img[150:] = (150, 60, 20)       # blue trousers
        self.assertEqual(clothing_colors(img), ("red", "blue"))


if __name__ == "__main__":
    unittest.main()
