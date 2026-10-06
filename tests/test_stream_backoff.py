import unittest

from gatevision import stream


class BackoffTests(unittest.TestCase):
    def setUp(self):
        stream._fail_times.clear()
        stream._pause_until = 0.0
        stream._connected.clear()

    def test_pauses_all_cameras_after_three_quick_failures(self):
        self.assertEqual(stream.register_failure(1000.0), 15.0)
        self.assertEqual(stream.register_failure(1001.0), 15.0)
        self.assertEqual(stream.register_failure(1002.0), 600.0)      # 3rd within 2 min -> long pause
        self.assertEqual(stream.register_failure(1700.0), 15.0)       # pause over, counter restarted

    def test_pause_applies_to_every_camera(self):
        for t in (1000.0, 1001.0, 1002.0):
            stream.register_failure(t)
        self.assertAlmostEqual(stream.pause_remaining(1003.0), 599.0)   # other cameras must hold off too
        self.assertEqual(stream.pause_remaining(1700.0), 0.0)
        self.assertGreaterEqual(stream.register_failure(1100.0), 500.0)  # a failure during the pause keeps waiting

    def test_switched_off_camera_never_triggers_pause_while_another_streams(self):
        stream.register_success("cam_ok")
        for t in (1000.0, 1001.0, 1002.0, 1003.0):
            self.assertEqual(stream.register_failure(t, name="cam_off"), 30.0)
        self.assertEqual(stream.pause_remaining(1004.0), 0.0)
        stream.register_disconnect("cam_ok")                          # now nothing streams -> protection is back
        stream.register_failure(1010.0, name="cam_off")
        stream.register_failure(1011.0, name="cam_off")
        self.assertGreaterEqual(stream.register_failure(1012.0, name="cam_off"), 500.0)

    def test_old_failures_expire(self):
        stream.register_failure(1000.0)
        stream.register_failure(1001.0)
        self.assertEqual(stream.register_failure(1200.0), 15.0)       # first two are >2 min old

    def test_success_resets(self):
        stream.register_failure(1000.0)
        stream.register_failure(1001.0)
        stream.register_success()
        self.assertEqual(stream.register_failure(1002.0), 15.0)


if __name__ == "__main__":
    unittest.main()
