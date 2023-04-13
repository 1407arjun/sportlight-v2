from multiprocessing.pool import ThreadPool
from multiprocessing import Lock

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

def extract_text(l: Lock, segments, vid: VideoFileClip, start, end):
    global model
    # Create subclip
    clip = vid.subclip(start, end)
    # Save the subclip
    path = f"{AUDIO_DIR}{str(int(start + end))}.mp3"
    clip.audio.write_audiofile(path)

    # Transcribe audio
    output = {"text": "", "segments": []}
    try:
        output = model.transcribe(path)
        return output
    finally:
        # Add segments to the thread safe segment list
        for i in output["segments"]:
            l.acquire()
            try:
                segments.append({"text": i["text"], "start": i["start"], "end": i["end"]})
            finally:
                l.release()


def calculate_similarity(l: Lock, result, segment):
    d = [segment["text"], strings]
    # Vectorize the strings
    Tfidf_vect = TfidfVectorizer()
    vector_matrix = Tfidf_vect.fit_transform(d)
    tokens = Tfidf_vect.get_feature_names_out()
    create_dataframe(vector_matrix.toarray(), tokens)

    # Calculate cosine similarity score
    cosine_similarity_matrix = cosine_similarity(vector_matrix)
    r = create_dataframe(
        cosine_similarity_matrix, ['Phrase', 'Strings'])
    score = r['Phrase'].values[1]

    # Accept as highlight if greater than threshold
    if (score[i] >= 0.00500000000000):
        l.acquire()
        try:
            result.append([segment["start"], segment["end"]])
        finally:
            l.release()


if __name__ == "__main__":
    # Get the video duration
    vid = VideoFileClip(SAMPLES_DIR + VIDEO_NAME)
    duration = vid.duration

    segments = list()
    # Create a lock to append to the segments extracted
    sl = Lock()
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

        pool.starmap(extract_text, [tuple([sl, segments, vid]) + tuple(i) for i in args])

    result = list()
    # Create a lock to append to the result
    rl = Lock()
    # Create the thread pool for similarity check
    with ThreadPool() as pool:
        # Chunk size calculated dynamically
        pool.starmap(calculate_similarity, [tuple([rl, result]) + tuple(i) for i in segments])