import unittest

from gatevision.plates import PlateRead, correct_pk_plate, drop_year, is_year_only, pretty, vote


class PlateTests(unittest.TestCase):
    def test_corrects_lookalikes_by_position(self):
        self.assertEqual(correct_pk_plate("A5Y3549"), ("ASY3549", True))
        self.assertEqual(correct_pk_plate("ARG8O8"), ("ARG808", True))
        self.assertEqual(correct_pk_plate("0H121"), ("OH121", True))
        self.assertEqual(correct_pk_plate("DH-121"), ("DH121", True))
        self.assertEqual(correct_pk_plate("LEA091234"), ("LEA091234", True))

    def test_leaves_unknown_formats_alone(self):
        text, ok = correct_pk_plate("12345")
        self.assertEqual(text, "12345")
        self.assertFalse(ok)

    def test_pretty(self):
        self.assertEqual(pretty("ASY3549"), "ASY-3549")
        self.assertEqual(pretty("??"), "??")

    def test_vote_prefers_consistent_valid_reads(self):
        reads = [PlateRead("ASY3549", "x", 0.9, True), PlateRead("ASY3549", "x", 0.8, True),
                 PlateRead("A5Y3S49", "x", 0.95, False), PlateRead("ASY3649", "x", 0.5, True)]
        text, conf, best = vote(reads)
        self.assertEqual(text, "ASY3549")
        self.assertGreater(conf, 0.5)
        self.assertEqual(vote([])[0], None)


if __name__ == "__main__":
    unittest.main()


class ExtraOcrTests(unittest.TestCase):
    def test_crop_variants_and_extra_reads_vote_out_a_single_bad_read(self):
        import types
        import numpy as np
        from gatevision.plates import PlateReader, crop_variants, vote
        names = [n for n, _ in crop_variants(np.zeros((60, 150, 3), np.uint8))]
        self.assertNotIn("top", names)      # never cut the number line off
        self.assertIn("rot12", names)

        class Ocr:
            def __init__(self):
                self.n = 0

            def predict(self, img):
                self.n += 1
                txt = "ISK8004" if self.n == 1 else "TSK8104"
                return types.SimpleNamespace(text=txt, confidence=0.8)

        class Alpr:
            ocr = Ocr()

            def predict(self, img):
                bb = types.SimpleNamespace(x1=10, y1=10, x2=140, y2=70)
                det = types.SimpleNamespace(bounding_box=bb, confidence=0.9)
                return [types.SimpleNamespace(detection=det, ocr=types.SimpleNamespace(text="ISK8004", confidence=0.59))]

        reader = PlateReader.__new__(PlateReader)
        reader.alpr = Alpr()
        reads = reader.read(np.zeros((100, 200, 3), np.uint8))
        text, share, _ = vote(reads)
        self.assertEqual(text, "TSK8104")


class RegistrationYearTests(unittest.TestCase):
    def test_year_is_dropped_from_letters_year_number(self):
        self.assertEqual(drop_year("MNC17515"), "MNC515")
        self.assertEqual(drop_year("LEA175989"), "LEA5989")
        self.assertEqual(drop_year("LEF123503"), "LEF3503")
        self.assertEqual(drop_year("ASY3549"), "ASY3549")     # plain plates untouched
        self.assertEqual(drop_year("LEF1981"), "LEF1981")     # 4 digits: cannot tell, keep
        self.assertEqual(drop_year("ABC995000"), "ABC995000") # 99 is not a year

    def test_letters_plus_year_only_is_recognised_as_partial(self):
        self.assertTrue(is_year_only("MNC17"))
        self.assertFalse(is_year_only("MNC515"))
        self.assertFalse(is_year_only("DH121"))               # 3 digits = a real number

    def test_vote_prefers_the_full_read_over_letters_plus_year(self):
        reads = [PlateRead("MNC17", "", 0.9, True)] * 6 + [PlateRead("MNC515", "", 0.7, True)] * 3
        text, _share, _best = vote(reads)
        self.assertEqual(text, "MNC515")

    def test_partial_is_kept_when_nothing_better_exists(self):
        text, _s, _b = vote([PlateRead("MNC17", "", 0.8, True)])
        self.assertEqual(text, "MNC17")
