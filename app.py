from multiprocessing.pool import ThreadPool
from threading import Lock

from moviepy.editor import VideoFileClip

import whisper

import nltk
from nltk.corpus import wordnet

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Directory constants
SAMPLES_DIR = "samples/"
ETC_DIR = SAMPLES_DIR + "etc/"
AUDIO_DIR = SAMPLES_DIR + "audio/"
OUT_DIR = SAMPLES_DIR + "out/"

# File constants
VIDEO_NAME = "Untitled.mp4"
AUDIO_NAME = "Untitled.wav"

model = whisper.load_model("base")
nltk.download('wordnet')
nltk.download('omw-1.4')

positive = list()

# Words indicating highlights
add = ["Straight", "biggie", "Cover", "OnDrive", "Square", "Forward", "stadium", "Defence", "Sweep", "Reverse",
           "FrontFoot ", "LegGlance ", "BackFoot", "SquareCut", "Pull ", "Shot", "Hook", "Uppercut", "Cut", "Helicopter ", "SwitchHit",
           "Dilscoop", "class", "bounce", "Upper", "Uppish", "Scoop ", "Inside", "Out", "Shots", "Bouncer", "Outswinger", "Inswinger",
           "ReverseSwing", "played", "LegCutter", "OffCutter", "Yorker", "Slower", "Spin", "LegBreak ", "OffBreak", "Googly ",
           "Doosra", "Topspin ", "CarromBall", "Slider", "ArmBall", "Infield", "InnerRing", "Outfield", "Catching", "Wicketkeeper",
           "Slip", "Gully", "LegSlip", "LegGully", "Sillypoint", "Sillymidoff", "Shortleg", "Sillymidon", "InnerRing", "Point", "BackwardPoint",
           "MidOff", "Cover", "MidOn", "SquareLeg", "Backward ", "SquareLeg", "MidWicket", "FineLeg", "Outfield", "ThirdMan",
           "DeepPoint", "BackwardPoint", "ExtraCover", "LongOff", "FineLeg", "LongLeg", "LongOn", "Deep", "Cover", "played", "account"
           "cricket", "hard", "sides", "man", "finishes", "one", "crucial", "Captain", "shot", "six", "four", "boundary", "line", "drive",
           "celebrate", "placement", "beauty", "fifty", "century", "perfect", "magnifcient", "world", "cup", "batting", "fielding", "bowling",
           "catch", "caught", "out", "stumped", "one", "bowled", "night", "final", "room", "taken", "edged", "wicket", "review", "DRS", "cuts", "out", "short"]

for i in add:
    for synset in wordnet.synsets(i):
        for lemma in synset.lemmas():
            positive.append(lemma.name())

strings = ' '.join(positive)
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

def create_dataframe(matrix, tokens):
    doc_names = [f'doc_{i+1}' for i, _ in enumerate(matrix)]
    df = pd.DataFrame(data=matrix, index=doc_names, columns=tokens)
    return (df)

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

def calculate_similarity():


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
