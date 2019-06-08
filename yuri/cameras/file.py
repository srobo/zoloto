from cv2 import VideoCapture, imread

from .base import BaseCamera


class ImageFileCamera(BaseCamera):
    def __init__(self, image_path, **kwargs):
        self.image_path = image_path
        super().__init__(**kwargs)

    def capture_frame(self):
        return imread(self.image_path)


class VideoFileCamera(BaseCamera):
    def __init__(self, video_path: str, **kwargs):
        super().__init__(**kwargs)
        self.video_capture = self.get_video_capture(video_path)

    def get_video_capture(self, video_path):
        return VideoCapture(video_path)

    def capture_frame(self):
        _, frame = self.video_capture.read()
        return frame

    def close(self):
        super().close()
        self.video_capture.release()
