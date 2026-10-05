from types import SimpleNamespace
import unittest

from deployment.robotlab_g1.video import ensure_offscreen_framebuffer


class VideoRecorderTest(unittest.TestCase):
    def test_offscreen_framebuffer_grows_to_requested_video_size(self):
        model = SimpleNamespace(
            vis=SimpleNamespace(global_=SimpleNamespace(offwidth=640, offheight=480))
        )

        size = ensure_offscreen_framebuffer(model, width=1280, height=720)

        self.assertEqual(size, (1280, 720))
        self.assertEqual(model.vis.global_.offwidth, 1280)
        self.assertEqual(model.vis.global_.offheight, 720)

    def test_offscreen_framebuffer_never_shrinks_existing_buffer(self):
        model = SimpleNamespace(
            vis=SimpleNamespace(global_=SimpleNamespace(offwidth=1920, offheight=1080))
        )

        size = ensure_offscreen_framebuffer(model, width=1280, height=720)

        self.assertEqual(size, (1920, 1080))


if __name__ == "__main__":
    unittest.main()
