import unittest

import numpy as np

from drr_pipeline import ctrate_geometry as G

ROW = {"XYSpacing": "[0.611328125, 0.611328125]", "ZSpacing": "1", "RescaleSlope": "1", "RescaleIntercept": "-8192"}


class GeometryTests(unittest.TestCase):
    def test_spacing(self):
        self.assertEqual(G.spacing_from_metadata(ROW), (0.611328125, 0.611328125, 1.0))
        with self.assertRaises(ValueError):
            G.spacing_from_metadata({**ROW, "ZSpacing": "0"})

    def test_hu(self):
        raw = np.array([0, 7192, 8192, 17038])
        hu = G.raw_to_hu(raw, ROW)
        self.assertEqual(hu.dtype, np.int16)
        self.assertEqual(hu.tolist(), [-1024, -1000, 0, 3071])

    def test_affine(self):
        a = G.oriented_affine((0.6, 0.7, 1.0))
        self.assertEqual(np.diag(a).tolist(), [-0.6, -0.7, 1.0, 1.0])

    def test_labelmap(self):
        m = np.zeros((3, 4, 4, 4), np.uint8)
        m[1, 0, 0, 0] = 2
        m[2, 3, 3, 3] = 1
        lab = G.sign_labelmap(m, [1], (4, 4, 4))
        self.assertEqual(int(lab.sum()), 1)
        self.assertEqual(int(G.sign_labelmap(m, [], (4, 4, 4)).sum()), 0)
        with self.assertRaises(ValueError):
            G.sign_labelmap(m, [1], (5, 5, 5))


if __name__ == "__main__":
    unittest.main()
