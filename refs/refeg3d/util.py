import ffmpeg
from pathlib import Path

def save_video(video, path: str, width:int, height:int, fps=30):
    """video is a sequence of numpy image in shape, (t, w, h, 3)"""
    # print("save_video", width, height)
    Path(path).parent.mkdir(exist_ok=True)
    video = video.transpose((0, 2, 1, 3))
    video_buffer = video.tobytes()
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', r=fps, s='{}x{}'.format(width, height))
        .output(path, pix_fmt='yuv420p', r=fps, loglevel="quiet")
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    process.communicate(input=video_buffer)
