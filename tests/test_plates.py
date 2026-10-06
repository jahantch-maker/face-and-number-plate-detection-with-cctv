import unittest

from gatevision.plates import PlateRead, correct_pk_plate, pretty, vote


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
