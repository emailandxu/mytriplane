import ffmpeg
from pathlib import Path

def save_video(video, path: str, res:int, fps=30):
    """video is a sequence of numpy image in shape, (t, w, h, 3)"""
    Path(path).parent.mkdir(exist_ok=True)
    video_buffer = video.tobytes()
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', r=fps, s='{}x{}'.format(res, res))
        .vflip()
        .output(path, pix_fmt='yuv420p', r=60, loglevel="quiet")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    process.communicate(input=video_buffer)
