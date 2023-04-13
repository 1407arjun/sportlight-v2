from multiprocessing.pool import ThreadPool
from moviepy.editor import VideoFileClip
from threading import Lock
import whisper

# Directory constants
SAMPLES_DIR = "samples/"
ETC_DIR = SAMPLES_DIR + "etc/"
AUDIO_DIR = SAMPLES_DIR + "audio/"
OUT_DIR = SAMPLES_DIR + "out/"

# File constants
VIDEO_NAME = "Untitled.mp4"
AUDIO_NAME = "Untitled.wav"

model = whisper.load_model("base")


class SegmentList:
    def __init__(self):
        # Initialize lists
        self.text = list()
        self.start = list()
        self.end = list()

        # Initialize lock
        self.lock = Lock()

    def append(self, text, start, end):
        # Only thread with lock can append
        with self.lock:
            self.text.append(text)
            self.start.append(start)
            self.end.append(end)


def extract_text(segments: SegmentList, vid: VideoFileClip, start, end):
    global model
    # Create subclip
    clip = vid.subclip(start, end)
    # Save the subclip
    path = f"{AUDIO_DIR}{str(int(start + end))}.mp3"
    clip.audio.write_audiofile(path)

    # Transcribe audio
    output = {}
    try:
        output = model.transcribe(path)
        return output
    except:
        output = {"text": "", "segments": []}

    # Add segments to the thread safe segment list
    for i in output["segments"]:
        segments.append(i["text"], i["start"], i["end"])


if __name__ == "__main__":
    # Get the video duration
    vid = VideoFileClip(SAMPLES_DIR + VIDEO_NAME)
    duration = vid.duration

    segments = SegmentList()
    # Create the thread pool for extraction
    with ThreadPool() as pool:
        # No. of processes
        np = pool._processes
        # Chunk size
        chunk = duration / np
        # Excess time for overlaps
        excess = 5
        # Create partitions
        args = [[chunk * (i) - excess, chunk * (i + 1) + excess] for i in range(np)]
        args[0][0] += excess
        args[-1][1] -= excess
        print(args)

        pool.starmap(extract_text, [tuple([segments, vid]) + tuple(i) for i in args])
