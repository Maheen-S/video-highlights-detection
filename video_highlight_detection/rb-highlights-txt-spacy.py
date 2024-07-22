#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


get_ipython().system('pip install ffmpeg-python')


# In[ ]:





# In[3]:


# pip install --upgrade decorator==4.4.2


# In[4]:


import ffmpeg
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


# In[5]:


# extract audio from video 
def extract_audio_from_video(video_path, audio_output_path):
    ffmpeg.input(video_path).output(audio_output_path).run()
    

# video_path = "/kaggle/input/sample-cricket-video-clips/Sample Cricket Video Clips/Sample Cricket Video Clips/DC vs KXIP IPL 2nd match highlights HD 2020 ipl2020.mp4"
video_path = "/kaggle/input/7-min-long-video-test/y2mate.com - 6  6  6  Shahid Afridi vs Chris Woakes  Pakistan vs England  2nd T20I 2015  PCB  MA2A_1080pFH.mp4"
output_path = "output_audio1.wav"
extract_audio_from_video(video_path , output_path)


# In[ ]:





# # Denoise

# In[6]:


get_ipython().system('pip install pydub noisereduce')


# In[7]:


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


# In[ ]:





# In[8]:


# # Equalization 
# # denoised_audio = denoised_audio.low_pass_filter(3000).high_pass_filter(300)

# # Compression 
# denoised_audio = denoised_audio.compress_dynamic_range()

# enhanced_output_path = "/kaggle/working/enganced_audio.wav"
# denoised_audio.export(enhanced_output_path, format="wav")
# play(denoised_audio)


# In[ ]:





# In[ ]:





# # Whisper

# In[9]:


# # saves it to a txt file
# import torch
# import librosa
# from transformers import pipeline

# # Load the Whisper model
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

# # Load your audio file
# audio_path = "path/to/your/audio/file.wav"
# audio, sampling_rate = librosa.load(audio_path, sr=16000)

# # Convert audio to required format
# audio_input = {"array": audio, "sampling_rate": sampling_rate}

# # Get transcription
# result = pipe(audio_input, max_new_tokens=256)
# transcription = result["text"]

# # Save the transcription to a text file
# output_text_path = "path/to/your/output/transcription.txt"
# with open(output_text_path, "w") as file:
#     file.write(transcription)

# print(f"Transcription saved to {output_text_path}")


# ## orignal

# In[10]:


# import torch
# import librosa
# from transformers import pipeline

# # Load the Whisper model
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

# # Load your audio file
# audio_path = "/kaggle/working/output_audio1.wav"
# audio, sampling_rate = librosa.load(audio_path, sr=16000)

# # Convert audio to required format
# audio_input = {"array": audio, "sampling_rate": sampling_rate}

# # Get transcription
# transcription = pipe(audio_input, max_new_tokens=256)
# print("Transcription:", transcription["text"])


# In[ ]:





# In[11]:


# transcription['text']


# ## denoised

# In[12]:


# import torch
# import librosa
# from transformers import pipeline

# # Load the Whisper model
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

# # Load your audio file
# audio_path = "/kaggle/working/output_audio_denoised.wav"
# audio, sampling_rate = librosa.load(audio_path, sr=16000)

# # Convert audio to required format
# audio_input = {"array": audio, "sampling_rate": sampling_rate}

# # Get transcription
# transcription = pipe(audio_input, max_new_tokens=256)
# print("Transcription:", transcription["text"])


# In[ ]:





# 

# In[13]:


# import torch
# import librosa
# from transformers import pipeline
# import json

# # Load the Whisper model
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

# # Load your audio file
# audio_path = "/kaggle/working/output_audio_denoised.wav"
# audio, sampling_rate = librosa.load(audio_path, sr=16000)

# # Convert audio to required format
# audio_input = {"array": audio, "sampling_rate": sampling_rate}

# # Get transcription with timestamps
# result = pipe(audio_input, return_timestamps=True)  # Ensure return_timestamps is set to True

# # Print the result to inspect its structure
# print(json.dumps(result, indent=4))

# # Assuming the structure contains a 'chunks' key with the required information
# transcription_with_timestamps = result['chunks'] if 'chunks' in result else []

# # Save the transcription with timestamps to a JSON file
# output_json_path = "/kaggle/working/transcription_with_timestamps.json"
# with open(output_json_path, "w") as json_file:
#     json.dump(transcription_with_timestamps, json_file)

# print(f"Transcription with timestamps saved to {output_json_path}")


# # 30 second divisions-> whisper

# In[14]:


# !pip uninstall -y jax jaxlib transformers
# !pip install transformers==4.26.1
# !pip install jax==0.3.25
# !pip install jaxlib==0.3.25


# In[15]:


# !pip install -U jax


# In[16]:


# import torch
# import librosa
# from transformers import pipeline


# In[17]:


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


# In[18]:


final_transcription


# # Spacy

# In[19]:


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


# In[20]:


events_with_timestamps.sort(key=lambda x: x["start"])


# In[21]:


events_with_timestamps


# In[22]:


len(events_with_timestamps)


# In[ ]:





# # Extract timestamps only

# In[ ]:


# json_file_path = "/kaggle/working/extracted_events_with_spacy.json"
# with open(json_file_path, "r") as file:
#     transcription_with_timestamps = json.load(file)

# print(json.dumps(transcription_with_timestamps, indent=4))

# # Check the structure of the first few segments
# for i, segment in enumerate(transcription_with_timestamps[:5]):
#     print(f"Segment {i}: {segment}")

# # Extract timestamps from the JSON data
# timestamps = []
# for segment in transcription_with_timestamps:
#     if "timestamp" in segment and len(segment["timestamp"]) == 2:
#         timestamps.append({
#             "start": segment["timestamp"][0],
#             "end": segment["timestamp"][1]
#         })

# # Save the extracted timestamps to a new JSON file
# timestamps_output_path = "/kaggle/working/extracted_timestamps.json"
# with open(timestamps_output_path, "w") as json_file:
#     json.dump(timestamps, json_file, indent=4)

# print(f"Extracted timestamps saved to {timestamps_output_path}")


# In[ ]:





# # Extract Audio Segments Based on Timestamps

# In[ ]:


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


# In[ ]:





# # Extract Video Segments Based on Timestamps

# In[ ]:


# !pip install ffmpeg --upgrade
# !pip install moviepy --upgrade


# In[ ]:


# import json
# import cv2

# # Paths to input and output files
# json_file_path = "/kaggle/working/extracted_events_with_spacy.json"
# # video_file_path = "/kaggle/input/sample-cricket-video-clips/Sample Cricket Video Clips/Sample Cricket Video Clips/DC vs KXIP IPL 2nd match highlights HD 2020 ipl2020.mp4"
# video_file_path = "/kaggle/input/7-min-long-video-test/y2mate.com - 6  6  6  Shahid Afridi vs Chris Woakes  Pakistan vs England  2nd T20I 2015  PCB  MA2A_1080pFH.mp4"
# compiled_video_path = "/kaggle/working/compiled_videoCV2.mp4"

# # Load the JSON file with timestamps
# with open(json_file_path, "r") as file:
#     timestamps = json.load(file)

# # Sort timestamps and ensure no repetitions
# sorted_timestamps = sorted(timestamps, key=lambda x: x["start"])
# unique_timestamps = []
# for ts in sorted_timestamps:
#     if not unique_timestamps or unique_timestamps[-1]["end"] != ts["start"]:
#         unique_timestamps.append(ts)

# # Load the video file
# try:
#     cap = cv2.VideoCapture(video_file_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# except Exception as e:
#     print(f"Error loading video file: {e}")

# # Function to compile video segments
# def compile_video_segments(video, timestamps, output_file_path, fps, width, height):
#     clips = []
#     for i, ts in enumerate(timestamps):
#         try:
#             start = ts["start"]
#             end = ts["end"]
#             if start >= end:
#                 print(f"Invalid segment: start time {start} is not less than end time {end}")
#                 continue

#             video.set(cv2.CAP_PROP_POS_MSEC, start * 1000)
#             frames = []
#             while video.get(cv2.CAP_PROP_POS_MSEC) < end * 1000:
#                 ret, frame = video.read()
#                 if not ret:
#                     break
#                 frames.append(frame)

#             clips.extend(frames)
#             print(f"Added segment {i+1} from {start}s to {end}s")
#         except Exception as e:
#             print(f"Error processing segment {i+1}: {e}")

#     if not clips:
#         print("No valid video segments to compile.")
#         return

#     # Save the compiled video using OpenCV
#     try:
#         out = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
#         for frame in clips:
#             out.write(frame)
#         out.release()
#         print(f"Compiled video saved as {output_file_path}")
#     except Exception as e:
#         print(f"Error during video writing: {e}")

# # Compile video segments based on timestamps
# compile_video_segments(cap, unique_timestamps, compiled_video_path, fps, width, height)


# In[ ]:


# trying to optimise above code:
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


# # Combine Timestamps
# ## Combine audio and video timestamps based on an 80% overlap

# In[ ]:





# # TRYING WITH BB

# In[ ]:


model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# In[ ]:


import cv2
import json

def process_video(video_path, model, condition_func, fps=30):
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()

    if not success:
        print("Failed to open video")
        return []

    frame_count = 0
    timestamps = []
    frames_with_boxes = []
    
    while success:
        results = model(frame)
        boxes = results.xyxy[0]

        if condition_func(frame, boxes):
            timestamp = frame_count / fps
            timestamps.append({"start": timestamp, "end": timestamp + (1 / fps)})
            
            for box in boxes:
                x1, y1, x2, y2, conf, cls = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4], int(box[5])
                label = model.names[int(cls)]
                confidence = box[4]

                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label_text = f'{label} {confidence:.2f}'
                label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_ymin = max(y1, label_size[1] + 10)
                frame = cv2.rectangle(frame, (x1, label_ymin - label_size[1] - 10), 
                                      (x1 + label_size[0], label_ymin), (0, 255, 0), cv2.FILLED)
                frame = cv2.putText(frame, label_text, (x1, label_ymin - 7), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            frames_with_boxes.append((frame_count, frame))

        success, frame = vidcap.read()
        frame_count += 1

    vidcap.release()

    timestamps_output_path = "/kaggle/working/yolo_timestamps.json"
    with open(timestamps_output_path, "w") as json_file:
        json.dump(timestamps, json_file, indent=4)

    print(f"Processed {frame_count} frames.")
    print(f"Timestamps saved to {timestamps_output_path}")
    return frames_with_boxes

def condition_func(frame, boxes):
    return any(box[4] > 0.8 for box in boxes)

# Parameters
video_path = '/kaggle/input/sample-cricket-video-clips/Sample Cricket Video Clips/Sample Cricket Video Clips/DC vs KXIP IPL 2nd match highlights HD 2020 ipl2020.mp4'
frames_with_boxes = process_video(video_path, model, condition_func, fps=30)


# In[ ]:


# /kaggle/input/sample-cricket-video-clips/Sample Cricket Video Clips/Sample Cricket Video Clips/DC vs KXIP IPL 2nd match highlights HD 2020 ipl2020.mp4


# In[ ]:


import json

# Load audio and video timestamps
with open("/kaggle/working/extracted_timestamps.json", "r") as file:
    audio_timestamps = json.load(file)

with open("/kaggle/input/yolo-detected-timestamps/yolo_detected_timestamps.json", "r") as file:
    video_timestamps = json.load(file)

def overlap_percentage(ts1, ts2):
    start1, end1 = ts1["start"], ts1["end"]
    start2, end2 = ts2["start"], ts2["end"]
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_duration = max(0, overlap_end - overlap_start)
    ts1_duration = end1 - start1
    return (overlap_duration / ts1_duration) * 100

combined_timestamps = []
for audio_ts in audio_timestamps:
    for video_ts in video_timestamps:
        if overlap_percentage(audio_ts, video_ts) >= 80:
            combined_timestamps.append({
                "start": max(audio_ts["start"], video_ts["start"]),
                "end": min(audio_ts["end"], video_ts["end"])
            })

combined_timestamps_path = "/kaggle/working/combined_timestamps.json"
with open(combined_timestamps_path, "w") as json_file:
    json.dump(combined_timestamps, json_file, indent=4)

print(f"Combined timestamps saved to {combined_timestamps_path}")


# In[ ]:


get_ipython().system('pip install moviepy')


# In[ ]:


import json
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageSequenceClip

# Paths to input and output files
video_file_path = "/kaggle/input/sample-cricket-video-clips/Sample Cricket Video Clips/Sample Cricket Video Clips/DC vs KXIP IPL 2nd match highlights HD 2020 ipl2020.mp4"
compiled_video_path = "/kaggle/working/final_compiled_video.mp4"

# Load the JSON file with combined timestamps
with open("/kaggle/working/combined_timestamps.json", "r") as file:
    combined_timestamps = json.load(file)

# Function to compile video segments with bounding boxes
def compile_video_segments_with_boxes(video, frames_with_boxes, timestamps, output_file_path, fps=30):
    clips = []
    for ts in timestamps:
        start = ts["start"]
        end = ts["end"]
        video_segment = video.subclip(start, end)
        segment_frames = []

        for frame_count, frame in frames_with_boxes:
            frame_time = frame_count / fps
            if start <= frame_time <= end:
                segment_frames.append(frame)

        if segment_frames:
            segment_clip = ImageSequenceClip(segment_frames, fps=fps)
            clips.append(segment_clip)
        else:
            clips.append(video_segment)
    
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_file_path, codec="libx264", fps=fps)
    print(f"Compiled video saved as {output_file_path}")

# Load the video file
video = VideoFileClip(video_file_path)

# Ensure frames_with_boxes is populated from the previous YOLO processing
frames_with_boxes = process_video(video_file_path, model, condition_func, fps=30)

# Verify the FPS of the video
video_fps = video.fps
if video_fps is None:
    video_fps = fps  # Use default fps if not specified

# Compile video segments based on combined timestamps and frames with boxes
compile_video_segments_with_boxes(video, frames_with_boxes, combined_timestamps, compiled_video_path, fps=video_fps)


# In[ ]:


# Adjusted Script with Enhanced YOLO Processing:
# import cv2
# import json
# from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageSequenceClip

# # Paths to input and output files
# video_file_path = "/kaggle/input/sample-cricket-video-clips/Sample Cricket Video Clips/Sample Cricket Video Clips/DC vs KXIP IPL 2nd match highlights HD 2020 ipl2020.mp4"
# compiled_video_path = "/kaggle/working/final_compiled_video.mp4"

# # Load the JSON file with combined timestamps
# with open("/kaggle/working/combined_timestamps.json", "r") as file:
#     combined_timestamps = json.load(file)

# # Function to process video and collect frames with bounding boxes
# def process_video(video_path, model, condition_func, fps=30):
#     vidcap = cv2.VideoCapture(video_path)
#     success, frame = vidcap.read()

#     if not success:
#         print("Failed to open video")
#         return []

#     frame_count = 0
#     frames_with_boxes = []
    
#     while success:
#         results = model(frame)
#         boxes = results.xyxy[0]

#         if condition_func(frame, boxes):
#             for box in boxes:
#                 x1, y1, x2, y2, conf, cls = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4], int(box[5])
#                 label = model.names[int(cls)]
#                 confidence = box[4]

#                 frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 label_text = f'{label} {confidence:.2f}'
#                 label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
#                 label_ymin = max(y1, label_size[1] + 10)
#                 frame = cv2.rectangle(frame, (x1, label_ymin - label_size[1] - 10), 
#                                       (x1 + label_size[0], label_ymin), (0, 255, 0), cv2.FILLED)
#                 frame = cv2.putText(frame, label_text, (x1, label_ymin - 7), 
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
#             frames_with_boxes.append((frame_count, frame))

#         success, frame = vidcap.read()
#         frame_count += 1

#     vidcap.release()
#     print(f"Processed {frame_count} frames.")
#     return frames_with_boxes

# def condition_func(frame, boxes, threshold=0.5):
#     return any(box[4] > threshold for box in boxes)

# # Function to compile video segments with bounding boxes
# def compile_video_segments_with_boxes(video, frames_with_boxes, timestamps, output_file_path, fps=30):
#     clips = []
#     for ts in timestamps:
#         start = ts["start"]
#         end = ts["end"]
#         video_segment = video.subclip(start, end)
#         segment_frames = []

#         for frame_count, frame in frames_with_boxes:
#             frame_time = frame_count / fps
#             if start <= frame_time <= end:
#                 segment_frames.append(frame)

#         if segment_frames:
#             segment_clip = ImageSequenceClip(segment_frames, fps=fps)
#             clips.append(segment_clip)
#         else:
#             clips.append(video_segment)
    
#     final_clip = concatenate_videoclips(clips)
#     final_clip.write_videofile(output_file_path, codec="libx264", fps=fps)
#     print(f"Compiled video saved as {output_file_path}")

# # Load the video file
# video = VideoFileClip(video_file_path)

# # Process video and ensure frames_with_boxes is populated
# frames_with_boxes = process_video(video_file_path, model, lambda frame, boxes: condition_func(frame, boxes, threshold=0.5), fps=30)

# # Verify the FPS of the video
# video_fps = video.fps
# if video_fps is None:
#     video_fps = 30  # Use default fps if not specified

# # Compile video segments based on combined timestamps and frames with boxes
# compile_video_segments_with_boxes(video, frames_with_boxes, combined_timestamps, compiled_video_path, fps=v


# In[ ]:





# # WITHOUT BB

# # matching segments json

# In[ ]:


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


# In[ ]:


common_timestamps


# In[ ]:





# # pick out these segments from video

# In[ ]:


import json
import cv2

# Load the JSON file with timestamps
with open("/kaggle/working/matching_timestamps.json", "r") as file:
    timestamps = json.load(file)

# Path to the input video file
# video_file_path = "/kaggle/input/sample-cricket-video-clips/Sample Cricket Video Clips/Sample Cricket Video Clips/DC vs KXIP IPL 2nd match highlights HD 2020 ipl2020.mp4"
video_file_path = "/kaggle/input/7-min-long-video-test/y2mate.com - 6  6  6  Shahid Afridi vs Chris Woakes  Pakistan vs England  2nd T20I 2015  PCB  MA2A_1080pFH.mp4"
# Open the video file
cap = cv2.VideoCapture(video_file_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# Path for the output video file
output_file_path = "/kaggle/working/final_compiled_video.mp4"
out = cv2.VideoWriter(output_file_path, fourcc, fps, (width, height))

# Process each timestamp
for ts in timestamps:
    start_time = ts["start"]
    end_time = ts["end"]

    # Convert time to frame number
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    # Set the start frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

# Release everything if job is finished
cap.release()
out.release()

print(f"Final compiled video saved to {output_file_path}")


# In[ ]:





# In[ ]:


# !pip install moviepy


# In[ ]:


len(combined_highlights)


# In[ ]:


import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def combine_highlights(video_highlights, text_events):
    combined_highlights = []

    for video_highlight in video_highlights:
        video_start = video_highlight['start']
        video_end = video_highlight['end']

        for event in text_events:
            event_start = event['start']
            event_end = event['end']

            # Check for overlap
            if (video_start <= event_end and video_end >= event_start):
                overlap_start = max(video_start, event_start)
                overlap_end = min(video_end, event_end)

                # Calculate intervals
                video_interval = video_end - video_start
                event_interval = event_end - event_start

                # Determine the largest interval
                if video_interval >= event_interval:
                    new_segment = {
                        'start_time': video_start,
                        'end_time': video_end,
                        'confidence': video_highlight.get('confidence', 0.0)
                    }
                else:
                    new_segment = {
                        'start_time': event_start,
                        'end_time': event_end,
                        'confidence': event.get('confidence', 0.0)
                    }
                
                # Check for duplicates
                if new_segment not in combined_highlights:
                    combined_highlights.append(new_segment)

    return combined_highlights

# Load JSON files
video_highlights_file = '/kaggle/input/new-yolo-times/final_timestamps.json'
text_events_file = '/kaggle/working/extracted_events_with_spacy.json'

video_highlights = load_json(video_highlights_file)
text_events = load_json(text_events_file)

# Combine highlights and events
combined_highlights = combine_highlights(video_highlights, text_events)

# Save combined highlights to JSON file
combined_highlights_file = 'combined_highlights_unique.json'
with open(combined_highlights_file, 'w') as f:
    json.dump(combined_highlights, f, indent=4)

print(f"Combined highlights saved to {combined_highlights_file}")


# In[ ]:


len(combined_highlights)


# In[ ]:





# # trying more things for finding the combined highlights 

# In[ ]:


import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def find_best_match(video_highlights, text_events):
    combined_highlights = []

    for video_highlight in video_highlights:
        video_start = video_highlight['start']
        video_end = video_highlight['end']
        best_match = None
        best_score = float('inf')

        for event in text_events:
            event_start = event['start']
            event_end = event['end']

            # Calculate overlap duration
            overlap_start = max(video_start, event_start)
            overlap_end = min(video_end, event_end)
            overlap_duration = max(0, overlap_end - overlap_start)

            # Ensure valid interval
            if overlap_start >= overlap_end:
                continue

            # Calculate proximity score
            proximity_score = abs(video_start - event_start) + abs(video_end - event_end)

            # Calculate total score (considering overlap and proximity)
            total_score = proximity_score - overlap_duration

            if total_score < best_score:
                best_score = total_score
                best_match = {
                    'start_time': overlap_start,
                    'end_time': overlap_end,
                    'confidence': max(video_highlight.get('confidence', 0.0), event.get('confidence', 0.0))
                }

        if best_match and best_match not in combined_highlights:
            combined_highlights.append(best_match)

    return combined_highlights

# Load JSON files
video_highlights_file = '/kaggle/input/new-yolo-times/final_timestamps.json'
text_events_file = '/kaggle/working/extracted_events_with_spacy.json'

video_highlights = load_json(video_highlights_file)
text_events = load_json(text_events_file)

# Find best matches
combined_highlights = find_best_match(video_highlights, text_events)

# Save combined highlights to JSON file
combined_highlights_file = 'combined_highlights_optim.json'
with open(combined_highlights_file, 'w') as f:
    json.dump(combined_highlights, f, indent=4)

print(f"Combined highlights saved to {combined_highlights_file}")


# In[ ]:


combined_highlights


# In[ ]:


len(combined_highlights)


# # same as above but it sets text as base

# In[ ]:


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


# In[ ]:


combined_highlights


# In[ ]:





# # add +2 second before and after actual segment

# In[ ]:


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


# In[ ]:


combined_highlights


# In[ ]:





# # generate video

# In[ ]:





# In[ ]:


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


# In[ ]:




