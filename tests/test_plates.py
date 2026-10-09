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

    def test_vote_merges_spellings_that_differ_by_a_character(self):
        # a clear bike plate, each frame/variant misreading a different character:
        # no single spelling wins much weight, but together they agree on the plate
        reads = [PlateRead(t, t, c, True) for t, c in [
            ("AWS5573", 0.93), ("AWS5578", 0.88), ("ARS5573", 0.88), ("AWS5673", 0.82),
            ("AWG5573", 0.86), ("AWS5573", 0.89), ("ANS5573", 0.70), ("JGL4569", 0.60)]]
        text, share, best = vote(reads)
        self.assertEqual(text, "AWS5573")
        self.assertGreater(share, 0.8)
        self.assertGreater(best.conf, 0.8)


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


def _two_line_plate(top="MNF 17", bottom="811"):
    import cv2
    import numpy as np
    img = np.full((140, 250, 3), 235, np.uint8)
    img[:, :45] = (80, 160, 60)                                 # green emblem strip
    cv2.putText(img, top.split()[0], (60, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 4)
    cv2.putText(img, top.split()[1], (185, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)
    cv2.putText(img, bottom, (70, 126), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (20, 20, 20), 4)
    cv2.rectangle(img, (2, 2), (247, 137), (30, 30, 30), 3)
    return img


class TwoLinePlateTests(unittest.TestCase):
    def test_two_line_plate_is_split_and_one_line_plate_is_not(self):
        import cv2
        import numpy as np
        from gatevision.plates import split_rows
        top, bottom = split_rows(_two_line_plate())
        self.assertLess(top.shape[0], 100)
        self.assertLess(bottom.shape[0], 100)
        one = np.full((90, 260, 3), 235, np.uint8)
        cv2.putText(one, "TSK 8104", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 4)
        self.assertIsNone(split_rows(one))
        cap = np.full((100, 260, 3), 235, np.uint8)             # one line + small caption underneath
        cv2.putText(cap, "TSK 8104", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 4)
        cv2.putText(cap, "BAJWA", (80, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2)
        self.assertIsNone(split_rows(cap))

    def test_compose_takes_letters_from_top_and_number_from_bottom(self):
        from gatevision.plates import compose_two_line
        self.assertEqual(compose_two_line("MNF17", "811"), "MNF811")
        self.assertEqual(compose_two_line("MNE·16", "254"), "MNE254")
        self.assertEqual(compose_two_line("LED14", "9"), "LED9")
        self.assertEqual(compose_two_line("MNF1", "811"), "MNF811")     # year only half read
        self.assertEqual(compose_two_line("MNFI7", "811"), "MNF811")    # year digit read as a letter
        self.assertEqual(compose_two_line("LEA", "5989"), "LEA5989")    # no year on the plate
        self.assertIsNone(compose_two_line("17", "811"))          # no letters on top
        self.assertIsNone(compose_two_line("MNF17", "ABC"))       # no digits below

    def test_reader_reads_the_rows_separately_so_the_year_never_mixes_into_the_number(self):
        import types
        import numpy as np
        from gatevision.plates import PlateReader, vote
        plate = _two_line_plate()
        frame = np.full((300, 400, 3), 90, np.uint8)
        frame[40:40 + plate.shape[0], 60:60 + plate.shape[1]] = plate

        class Ocr:
            n = 0

            def predict(self, img):
                Ocr.n += 1                                       # rows come top, bottom, top, bottom ...
                text = "MNF17" if Ocr.n % 2 == 1 else "811"
                return types.SimpleNamespace(text=text, confidence=0.9)

        class Alpr:
            ocr = Ocr()

            def predict(self, img):
                bb = types.SimpleNamespace(x1=60, y1=40, x2=60 + plate.shape[1], y2=40 + plate.shape[0])
                det = types.SimpleNamespace(bounding_box=bb, confidence=0.9)
                return [types.SimpleNamespace(detection=det, ocr=types.SimpleNamespace(text="MNF1811", confidence=0.5))]

        reader = PlateReader.__new__(PlateReader)
        reader.alpr = Alpr()
        reads = reader.read(frame)
        self.assertEqual({r.text for r in reads}, {"MNF811"})    # the mixed-up "MNF1811" never appears
        self.assertEqual(vote(reads)[0], "MNF811")
