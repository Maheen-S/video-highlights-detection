# -*- coding: utf-8 -*-
"""rb_highlights_txt.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1enEqJmj1DK9IyTfScHJbBlQQxjC9tfHA
"""

!pip install ffmpeg-python

# pip install --upgrade decorator==4.4.2

import ffmpeg
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# extract audio from video
def extract_audio_from_video(video_path, audio_output_path):
    ffmpeg.input(video_path).output(audio_output_path).run()


# video_path = "/kaggle/input/sample-cricket-video-clips/Sample Cricket Video Clips/Sample Cricket Video Clips/DC vs KXIP IPL 2nd match highlights HD 2020 ipl2020.mp4"
video_path = "/kaggle/input/7-min-long-video-test/y2mate.com - 6  6  6  Shahid Afridi vs Chris Woakes  Pakistan vs England  2nd T20I 2015  PCB  MA2A_1080pFH.mp4"
output_path = "output_audio1.wav"
extract_audio_from_video(video_path , output_path)



"""# Denoise"""

!pip install pydub noisereduce

import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
import matplotlib.pyplot as plt

# Load your audio file
audio_path = "/kaggle/working/output_audio1.wav"
y, sr = librosa.load(audio_path, sr=16000)

# Plot the audio signal for visualization (optional)
plt.figure(figsize=(14, 5))
librosa.display.waveshow(y, sr=sr)
plt.title("Original Audio Signal")
plt.show()

# Apply noise reduction
reduced_noise = nr.reduce_noise(y=y, sr=sr)

# Save the denoised audio
output_path = "/kaggle/working/output_audio_denoised.wav"
sf.write(output_path, reduced_noise, sr)

# Load and visualize the denoised audio (optional)
denoised_audio, sr = librosa.load(output_path, sr=16000)
plt.figure(figsize=(14, 5))
librosa.display.waveshow(denoised_audio, sr=sr)
plt.title("Denoised Audio Signal")
plt.show()



"""# Whisper

## 30 second divisions for timestamps
"""

# import torch
# import librosa
# from transformers import pipeline

from pydub import AudioSegment
import torch
from transformers import pipeline
import numpy as np
import json

def split_audio(audio_path, chunk_length_ms=30000):
    audio = AudioSegment.from_wav(audio_path)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

def audiosegment_to_np(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        samples = samples.reshape((-1, 2))
    return samples.astype(np.float32) / np.iinfo(samples.dtype).max

def transcribe_chunk(chunk, start_time_s, pipe, sample_rate):
    chunk.export("/tmp/temp_chunk.wav", format="wav")
    chunk_np = audiosegment_to_np(chunk)
    transcription = pipe({"array": chunk_np, "sampling_rate": sample_rate}, return_timestamps=True)

    for item in transcription["chunks"]:
        item["timestamp"] = (
            item["timestamp"][0] + start_time_s,
            item["timestamp"][1] + start_time_s
        )
    return transcription["chunks"]

# Load the Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

# Split the audio into chunks
audio_path = "/kaggle/working/output_audio_denoised.wav"
chunks = split_audio(audio_path)

# Transcribe each chunk
all_transcriptions = []
chunk_length_s = 30
sample_rate = 16000

for i, chunk in enumerate(chunks):
    start_time_s = i * chunk_length_s
    transcriptions = transcribe_chunk(chunk, start_time_s, pipe, sample_rate)
    all_transcriptions.extend(transcriptions)

# Compile the final transcription
final_transcription = {
    "transcription": " ".join([t["text"] for t in all_transcriptions]),
    "timestamps": all_transcriptions
}

# Save the final transcription to a JSON file
with open("/kaggle/working/final_transcription.json", "w") as f:
    json.dump(final_transcription, f, indent=4)

final_transcription

"""# Spacy"""

import json
import spacy

nlp = spacy.load("en_core_web_sm")

# Load the JSON file
json_file_path = "/kaggle/working/final_transcription.json"
with open(json_file_path, "r") as file:
    transcription_with_timestamps = json.load(file)

# Function to extract events with spaCy
def extract_events_with_spacy(transcription_with_timestamps, keywords):
    events = []
    for segment in transcription_with_timestamps.get("timestamps", []):
        if isinstance(segment, dict):
            text = segment.get("text", "")
            timestamp = segment.get("timestamp", [])
            if text and isinstance(timestamp, (list, tuple)) and len(timestamp) == 2:
                start, end = timestamp
                doc = nlp(text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                for keyword in keywords:
                    if keyword in text.lower():
                        events.append({
                            "text": text,
                            "event": keyword,
                            "entities": entities,
                            "start": start,
                            "end": end
                        })
                        break
    return events

keywords = [
    'six', 'four', 'boundary', 'shot', 'drive', 'pull', 'hook', 'sweep', 'reverse sweep',
    'cover drive', 'cut shot', 'flick', 'straight drive', 'lofted shot', 'single', 'double', 'triple', 'run',
    'wicket', 'out', 'bowled', 'lbw', 'catch', 'stumped', 'run-out', 'appeal', 'no-ball', 'wide', 'yorker',
    'bouncer', 'swing', 'spin', 'googly', 'off-cutter', 'leg-cutter', 'full-toss', 'catch', 'dropped', 'fielded',
    'boundary save', 'direct hit', 'misfield', 'century', 'fifty', 'hat-trick', 'partnership', 'maiden over', 'over',
    'session', 'innings', 'powerplay','review', 'DRS', 'umpire', 'captain', 'run rate'
]

# Extract events with timestamps using spaCy
events_with_timestamps = extract_events_with_spacy(transcription_with_timestamps, keywords)

# Save the extracted events to a JSON file
output_json_path = "/kaggle/working/extracted_events_with_spacy.json"
with open(output_json_path, "w") as json_file:
    json.dump(events_with_timestamps, json_file, indent=4)

print(f"Extracted events with timestamps saved to {output_json_path}")

events_with_timestamps.sort(key=lambda x: x["start"])

events_with_timestamps

len(events_with_timestamps)



"""# Extract Audio Segments Based on Timestamps"""

from pydub import AudioSegment
import json

# Load the JSON file with timestamps
timestamps_json_path = "/kaggle/working/extracted_events_with_spacy.json"
with open(timestamps_json_path, "r") as file:
    timestamps = json.load(file)

# Print the timestamps for debugging
print("Original Timestamps:", timestamps)

timestamps.sort(key=lambda x: x["start"])

audio_path = "/kaggle/working/output_audio_denoised.wav"
audio = AudioSegment.from_wav(audio_path)

audio_duration_s = len(audio) / 1000
print(f"Audio duration: {audio_duration_s} sec")

# Function to extract and compile audio segments
def compile_audio_segments(audio, timestamps, output_file_path="/kaggle/working/compiled_audio.wav"):
    combined = AudioSegment.empty()
    for i, ts in enumerate(timestamps):
        start_ms = ts["start"] * 1000
        end_ms = ts["end"] * 1000

        # Check if the timestamps are within the audio duration
        if start_ms < 0 or end_ms > len(audio):
            print(f"Invalid segment {i+1} from {ts['start']}s to {ts['end']}s, skipping...")
            continue

        # Extract the audio segment and add it to the combined audio
        audio_segment = audio[start_ms:end_ms]
        combined += audio_segment
        print(f"Added segment {i+1} from {ts['start']}s to {ts['end']}s")

    # Export the combined audio segment
    combined.export(output_file_path, format="wav")
    print(f"Compiled audio saved as {output_file_path}")

# Compile audio segments based on sorted timestamps
compile_audio_segments(audio, timestamps)



"""# Extract Video Segments Based on Timestamps"""

import json
import cv2


json_file_path = "/kaggle/working/extracted_events_with_spacy.json"
video_file_path = "/kaggle/input/7-min-long-video-test/y2mate.com - 6  6  6  Shahid Afridi vs Chris Woakes  Pakistan vs England  2nd T20I 2015  PCB  MA2A_1080pFH.mp4"
compiled_video_path = "/kaggle/working/compiled_videoCV2.mp4"

# Load the JSON file with timestamps
with open(json_file_path, "r") as file:
    timestamps = json.load(file)

# Sort timestamps and ensure no repetitions
sorted_timestamps = sorted(timestamps, key=lambda x: x["start"])
unique_timestamps = []
for ts in sorted_timestamps:
    if not unique_timestamps or unique_timestamps[-1]["end"] != ts["start"]:
        unique_timestamps.append(ts)

# Load the video file
try:
    cap = cv2.VideoCapture(video_file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
except Exception as e:
    print(f"Error loading video file: {e}")

# Function to compile video segments
def compile_video_segments(video, timestamps, output_file_path, fps, width, height):
    try:
        out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for i, ts in enumerate(timestamps):
            try:
                start = ts["start"]
                end = ts["end"]
                if start >= end:
                    print(f"Invalid segment: start time {start} is not less than end time {end}")
                    continue

                video.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
                while video.get(cv2.CAP_PROP_POS_MSEC) < end * 1000:
                    ret, frame = video.read()
                    if not ret:
                        break
                    out.write(frame)
                print(f"Added segment {i+1} from {start}s to {end}s")
            except Exception as e:
                print(f"Error processing segment {i+1}: {e}")

        out.release()
        print(f"Compiled video saved as {output_file_path}")
    except Exception as e:
        print(f"Error during video writing: {e}")

# Compile video segments based on timestamps
compile_video_segments(cap, unique_timestamps, compiled_video_path, fps, width, height)



"""# Combine Timestamps
## Combine audio and video timestamps based on an 80% overlap
"""



import json

# Load the JSON files
with open('/kaggle/input/new-yolo-times/final_timestamps.json', 'r') as file:
    spacy_timestamps = json.load(file)

with open('/kaggle/input/video-highlights-7-min-json-latest/video_highlights-7-min.json', 'r') as file:
    yolo_timestamps = json.load(file)

    print(yolo_timestamps)

def overlap_percentage(start1, end1, start2, end2):
    """Calculate the overlap percentage between two time intervals."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_duration = max(0, overlap_end - overlap_start)

    duration1 = end1 - start1
    duration2 = end2 - start2

    overlap_ratio1 = overlap_duration / duration1
    overlap_ratio2 = overlap_duration / duration2

    return max(overlap_ratio1, overlap_ratio2)

def find_common_timestamps(timestamps1, timestamps2, overlap_threshold=0.9):
    common_timestamps = []
    for ts1 in timestamps1:
        for ts2 in timestamps2:
            start1, end1 = ts1["start"], ts1["end"]
            start2, end2 = ts2["start"], ts2["end"]
            if overlap_percentage(start1, end1, start2, end2) >= overlap_threshold:
                common_timestamps.append({
                    "start": max(start1, start2),
                    "end": min(end1, end2)
                })
    return common_timestamps

# Find common timestamps
common_timestamps = find_common_timestamps(spacy_timestamps, yolo_timestamps)

# Save the common timestamps to a new JSON file
output_path = '/kaggle/working/matching_timestamps.json'
with open(output_path, 'w') as file:
    json.dump(common_timestamps, file, indent=4)

print(f"Common timestamps saved to {output_path}")

common_timestamps



"""# Pick out the matching_timestamps segments from video"""

# !pip install moviepy

len(combined_highlights)

"""# Sets text as base [Spacy]"""

import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def align_video_to_text(text_events, video_highlights):
    combined_highlights = []

    for event in text_events:
        event_start = event['start']
        event_end = event['end']
        best_match = None
        best_score = float('inf')

        for video_highlight in video_highlights:
            video_start = video_highlight['start']
            video_end = video_highlight['end']

            # Calculate overlap duration
            overlap_start = max(event_start, video_start)
            overlap_end = min(event_end, video_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            # Ensure valid interval
            if overlap_start >= overlap_end:
                continue

            # Calculate proximity score
            proximity_score = abs(event_start - video_start) + abs(event_end - video_end)

            # Calculate total score (considering overlap and proximity)
            total_score = proximity_score - overlap_duration

            if total_score < best_score:
                best_score = total_score
                best_match = {
                    'start_time': event_start,
                    'end_time': event_end,
                    'confidence': video_highlight.get('confidence', 0.0)
                }

        if best_match and best_match not in combined_highlights:
            combined_highlights.append(best_match)

    return combined_highlights

# Load JSON files
text_events_file = '/kaggle/working/extracted_events_with_spacy.json'
video_highlights_file = '/kaggle/input/new-yolo-times/final_timestamps.json'

text_events = load_json(text_events_file)
video_highlights = load_json(video_highlights_file)

# Align video highlights to text events
combined_highlights = align_video_to_text(text_events, video_highlights)

# Save combined highlights to JSON file
combined_highlights_file = 'combined_highlights_optim_TEXT.json'
with open(combined_highlights_file, 'w') as f:
    json.dump(combined_highlights, f, indent=4)

print(f"Combined highlights saved to {combined_highlights_file}")

combined_highlights



"""# add +1.2 second before and after actual segment
## Either run above cell or this
"""

import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def align_video_to_text(text_events, video_highlights):
    combined_highlights = []
    buffer_time = 1.2  # 2-second buffer

    for event in text_events:
        event_start = max(event['start'] - buffer_time, 0)  # Ensure start time is not negative
        event_end = event['end'] + buffer_time
        best_match = None
        best_score = float('inf')

        for video_highlight in video_highlights:
            video_start = video_highlight['start']
            video_end = video_highlight['end']

            # Calculate overlap duration
            overlap_start = max(event_start, video_start)
            overlap_end = min(event_end, video_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            # Ensure valid interval
            if overlap_start >= overlap_end:
                continue

            # Calculate proximity score
            proximity_score = abs(event_start - video_start) + abs(event_end - video_end)

            # Calculate total score (considering overlap and proximity)
            total_score = proximity_score - overlap_duration

            if total_score < best_score:
                best_score = total_score
                best_match = {
                    'start_time': event_start,
                    'end_time': event_end,
                    'confidence': video_highlight.get('confidence', 0.0)
                }

        if best_match and best_match not in combined_highlights:
            combined_highlights.append(best_match)

    return combined_highlights

# Load JSON files
text_events_file = '/kaggle/working/extracted_events_with_spacy.json'
video_highlights_file = '/kaggle/input/new-yolo-times/final_timestamps.json'

text_events = load_json(text_events_file)
video_highlights = load_json(video_highlights_file)

# Align video highlights to text events
combined_highlights = align_video_to_text(text_events, video_highlights)

# Save combined highlights to JSON file
combined_highlights_file = 'combined_highlights_1_2sec.json'
with open(combined_highlights_file, 'w') as f:
    json.dump(combined_highlights, f, indent=4)

print(f"Combined highlights saved to {combined_highlights_file}")

combined_highlights



"""# Generate video"""

import json
import cv2

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def create_final_video(input_video_path, combined_highlights, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for highlight in combined_highlights:
        start_time = highlight['start_time']
        end_time = highlight['end_time']
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in range(start_frame, end_frame):
            success, frame = cap.read()
            if not success:
                break
            out.write(frame)

    cap.release()
    out.release()

# Load JSON files
# combined_highlights_file = 'combined_highlights_unique.json'
combined_highlights_file = '/kaggle/working/combined_highlights_1_2sec.json'
combined_highlights = load_json(combined_highlights_file)

# Paths
input_video_path = '/kaggle/input/7-min-long-video-test/y2mate.com - 6  6  6  Shahid Afridi vs Chris Woakes  Pakistan vs England  2nd T20I 2015  PCB  MA2A_1080pFH.mp4'
output_video_path = 'combined_highlights_1_2sec.mp4'

# Create final video
create_final_video(input_video_path, combined_highlights, output_video_path)

print(f"Final highlights video saved to {output_video_path}")

