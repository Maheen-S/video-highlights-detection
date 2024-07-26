#!/usr/bin/env python
# coding: utf-8



# In[3]:

import cv2
import json
import torch
import ffmpeg
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import noisereduce as nr
import soundfile as sf
from transformers import pipeline
import spacy
import time
import os
import tempfile
import string

# In[4]:
# video_path = "/kaggle/input/7-min-long-video-test/y2mate.com - 6  6  6  Shahid Afridi vs Chris Woakes  Pakistan vs England  2nd T20I 2015  PCB  MA2A_1080pFH.mp4"

# In[5]:


# Load YOLO model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
t1 = time.time()
nlp = spacy.load("en_core_web_sm")
t2 =time.time()

print(f"Time taken by nlp = spacy.load(en_core_web_sm) {t2 - t1}")
  

# In[8]:

t1 =time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

t2 =time.time()

print(f"Time taken by pipe = pipeline automatic-speech-recognition {t2 - t1}")

audio_path = "extractedAudiofinal.wav"


# In[9]:


keywords = [
    'six', 'four', 'boundary', 'shot', 'drive', 'pull', 'hook', 'sweep', 'reverse sweep',
    'cover drive', 'cut shot', 'flick', 'straight drive', 'lofted shot', 'single', 'double', 'triple', 'run',
    'wicket', 'out', 'bowled', 'lbw', 'catch', 'stumped', 'run-out', 'appeal', 'no-ball', 'wide', 'yorker',
    'bouncer', 'swing', 'spin', 'googly', 'off-cutter', 'leg-cutter', 'full-toss', 'catch', 'dropped', 'fielded',
    'boundary save', 'direct hit', 'misfield', 'century', 'fifty', 'hat-trick', 'partnership', 'maiden over', 'over',
    'session', 'innings', 'powerplay','review', 'DRS', 'umpire', 'captain', 'run rate'
]


# In[10]:


# Step 1: from the obj detection.py file
# 1) load yolo model
# 2)  Saving time stamps too, this will return he video_highlights-7-min.json file

def process_video(video_path, output_video_path, model, condition_func, json_output_path, fps=30):

    video_path = str(video_path)
    print(type(video_path))

    # # Check if the video path exists and is a valid string
    # if not isinstance(video_path, str):
    #     raise ValueError("The video_path should be a string.")
    #     video_path = str(video_path)
    
    # if not os.path.exists(video_path):
    #     raise FileNotFoundError(f"The file at {video_path} does not exist.")
    
    # if not isinstance(video_path, str):
    # video_path = str(video_path)

    
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()

    if not success:
        print("Failed to open video")
        return

    # Video properties
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    highlights = []
    in_highlight_segment = False
    segment_start_time = None
    max_confidence = 0.0

    while success:
        # YOLO detection
        results = model(frame)
        boxes = results.xyxy[0].cpu().numpy()  # Convert bounding boxes to numpy array

        # Analyze if frame should be included
        if condition_func(frame, boxes):
            current_time = vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current time in seconds

            if not in_highlight_segment:
                # Start of a new highlight segment
                in_highlight_segment = True
                segment_start_time = current_time
                max_confidence = max(box[4] for box in boxes)  # Highest confidence score in this segment

            # Update max confidence if this frame has higher confidence detections
            max_confidence = max(max_confidence, max(box[4] for box in boxes))

            # Draw bounding boxes and labels on the frame
            for box in boxes:
                x1, y1, x2, y2, conf, cls = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4], int(box[5])
                label = model.names[int(cls)]
                confidence = box[4]

                # Draw the bounding box
                frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Prepare the label text
                label_text = f'{label} {confidence:.2f}'
                label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                label_ymin = max(y1, label_size[1] + 10)

                # Draw the label background
                frame = cv2.rectangle(frame, (x1, label_ymin - label_size[1] - 10),
                                      (x1 + label_size[0], label_ymin), (0, 255, 0), cv2.FILLED)

                # Put the label text above the bounding box
                frame = cv2.putText(frame, label_text, (x1, label_ymin - 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Write frame to output video
            out.write(frame)

        else:
            if in_highlight_segment:
                # End of the current highlight segment
                segment_end_time = vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current time in seconds
                highlights.append({
                    'start_time': segment_start_time,
                    'end_time': segment_end_time,
                    'confidence': float(max_confidence)
                })
                in_highlight_segment = False
                max_confidence = 0.0

        # Read next frame
        success, frame = vidcap.read()
        frame_count += 1

    # Check if the last segment was not closed
    if in_highlight_segment:
        segment_end_time = vidcap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        highlights.append({
            'start_time': segment_start_time,
            'end_time': segment_end_time,
            'confidence': float(max_confidence)  # Ensure confidence is a float
        })

    vidcap.release()
    out.release()

    # Save highlights to JSON file
    with open(json_output_path, 'w') as f:
        json.dump(highlights, f, indent=4)

    print(f"Processed {frame_count} frames.")
    print(f"Highlights saved to {json_output_path}")

def condition_func(frame, boxes):
    return any(box[4] > 0.8 and box[4] < 0.95 for box in boxes)

def obj_detection_main(video_path, model):
    # Parameters
    output_video_path = 'final_7_min.mp4'
    json_output_path_yolo = 'video_highlights_times.json'
    fps = 30

#     # Load YOLO model
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Process the video
    t1 = time.time()
    process_video(video_path, output_video_path, model, condition_func, json_output_path_yolo, fps)
    t2 =time.time()

    print(f"Time taken to process video in obj detection def {t2 - t1}")


# In[14]:


#from the text file:
# 3) extract audio from video
# extract audio from video
def extract_audio_from_video(video_path, audio_output_path):

    t1 =time.time()
    # # probe = ffmpeg.probe(video_path)
    # # audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
    # if not check_audio_stream(video_path):
    #     raise RuntimeError("No audio stream found in the video file.")
    # try:
    #     ffmpeg.input(video_path).output(audio_output_path).run()
    # except ffmpeg.Error as e:
    #     print(f"ffmpeg error: {e.stderr.decode()}")
    #     raise

    ffmpeg.input(video_path).output(audio_output_path).run()

    t2 =time.time()

    print(f"Time taken to extract_audio_from_video {t2 - t1}")

# output_path = "output_audio1.wav"
# extract_audio_from_video(video_path , output_path)


# In[33]:

# 4) denoise the audio

def denoise(audio_path, output_path):

    t1 = time.time()

    # Load your audio file
    y, sr = librosa.load(audio_path, sr=16000)

    # Apply noise reduction
    reduced_noise = nr.reduce_noise(y=y, sr=sr)

    # Save the denoised audio
    # output_path = "/kaggle/working/output_audio_denoised.wav"
    sf.write(output_path, reduced_noise, sr)

    t2 =time.time()

    print(f"Time taken to denoise {t2 - t1}")


# In[35]:

# 5) apply Whisper with the 30 second divisions for timestamps


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
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_chunk_path = temp_file.name

    # Export the chunk to the temporary file
    chunk.export(temp_chunk_path, format="wav")
    
    # Convert the AudioSegment to a numpy array
    chunk_np = audiosegment_to_np(chunk)
    
    # Perform the transcription
    transcription = pipe({"array": chunk_np, "sampling_rate": sample_rate}, return_timestamps=True)

    # Adjust timestamps
    for item in transcription["chunks"]:
        item["timestamp"] = (
            item["timestamp"][0] + start_time_s,
            item["timestamp"][1] + start_time_s
        )

    # Clean up the temporary file
    os.remove(temp_chunk_path)

    return transcription["chunks"]

# def transcribe_chunk(chunk, start_time_s, pipe, sample_rate):
#     chunk.export("/tmp/temp_chunk.wav", format="wav")
#     chunk_np = audiosegment_to_np(chunk)
#     transcription = pipe({"array": chunk_np, "sampling_rate": sample_rate}, return_timestamps=True)

#     for item in transcription["chunks"]:
#         item["timestamp"] = (
#             item["timestamp"][0] + start_time_s,
#             item["timestamp"][1] + start_time_s
#         )
#     return transcription["chunks"]

def whisper_main(pipe, audio_path):
    t1 =time.time()
    # Load the Whisper model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     pipe = pipeline("automatic-speech-recognition", model="openai/whisper-base", device=device)

    # Split the audio into chunks
#     audio_path = "/kaggle/working/output_audio_denoised.wav"
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
    with open("final_transcription.json", "w") as f:
        json.dump(final_transcription, f, indent=4)

    t2 =time.time()

    print(f"Time taken by whisper_main {t2 - t1}")


# In[36]:


# 6) Extract events with timestamps using spaCy

# # Load the JSON file
# json_file_path = "/kaggle/working/final_transcription.json"
# with open(json_file_path, "r") as file:
#     transcription_with_timestamps = json.load(file)

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

def spacy_main(keywords, json_file_path_transription, output_json_path ):
    t1 =time.time()

    # json_file_path_transription = "/kaggle/working/final_transcription.json"
    with open(json_file_path_transription, "r") as file:
        transcription_with_timestamps = json.load(file)

    events_with_timestamps = extract_events_with_spacy(transcription_with_timestamps, keywords)

    t2 =time.time()

    print(f"Time taken to extract_events_with_spacy {t2 - t1}")

    # Save the extracted events to a JSON file
    output_json_path = "extracted_events_with_spacy.json"
    with open(output_json_path, "w") as json_file:
        json.dump(events_with_timestamps, json_file, indent=4)

    print(f"Extracted events with timestamps saved to {output_json_path}")


# 8) Pick out the matching_timestamps segments from video and Sets text as base [Spacy]
def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def align_video_to_text(text_events, video_highlights):
    t1 = time.time()
    combined_highlights = []

    for event in text_events:
        event_start = event['start']
        event_end = event['end']
        best_match = None
        best_score = float('inf')

        for video_highlight in video_highlights:
            video_start = video_highlight['start_time']
            video_end = video_highlight['end_time']

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

        t2 =time.time()

        print(f"Time taken to do the overlap thing {t2 - t1}")

    return combined_highlights


# In[51]:

# 9) generate video

def create_final_video(input_video_path, combined_highlights, output_video_path):
    t1 = time.time()

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

    t2 =time.time()

    print(f"Time taken to create final video {t2 - t1}")


# In[52]:

def main_running_code(video_path):
    t1 = time.time()
    
    # 1) obj detection -> saves .json of yolo 
    model_path = 'C:\\Users\\User\\Desktop\\video-highlights-detection\\video_highlight_detection\\yolov5s.pt'
    # model_path = 'C:\\Users\\User\\Desktop\\video-highlights-app\\yolov5s.pt'
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

    obj_detection_main(video_path, model)


    # 2) extract audio + denoise it 
    audio_output_path = "extracted_audio1.wav"
    extract_audio_from_video(video_path, audio_output_path)

    output_path_denoised = "denoised_audio1.wav"
    denoise(audio_output_path, output_path_denoised)

    # 3) Whisper on denoised audio to extract text + save txt in a .json file
    whisper_main(pipe, output_path_denoised)

    # 4) Extract events using spacy
    json_file_path_transription = "final_transcription.json"
    output_json_path = "extracted_events_with_spacy.json"
    spacy_main(keywords, json_file_path_transription, output_json_path )

    # 5) overlapping regions
    text_events_file = output_json_path
    video_highlights_file = 'video_highlights_times.json'

    text_events = load_json(text_events_file)
    video_highlights = load_json(video_highlights_file)

    # Align video highlights to text events
    combined_highlights = align_video_to_text(text_events, video_highlights)

    # Save combined highlights to JSON file
    combined_highlights_file = 'combined_highlights.json'
    with open(combined_highlights_file, 'w') as f:
        json.dump(combined_highlights, f, indent=4)

    print(f"Combined highlights saved to {combined_highlights_file}")

    # 6) final video
    final_video_path = "final_video1.mp4"
    create_final_video(video_path, combined_highlights, final_video_path)

    print(f"Final highlights video saved to {final_video_path}")

    t2 =time.time()

    print(f"Time taken by MAIN {t2 - t1}")
