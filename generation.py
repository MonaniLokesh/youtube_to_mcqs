import streamlit as st
import re
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOllama

# Function to remove timestamps from text
def remove_timestamps(text):
  timestamp_pattern = r'\[\d+:\d+:\d+\.\d+ --> \d+:\d+:\d+\.\d+\]'
  return re.sub(timestamp_pattern, '', text)

# Function to summarize text using transformers pipeline
def summarize_text(text):
  summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
  max_chunk_size = 512
  chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
  summaries = []
  for chunk in chunks:
    summary = summarizer(chunk, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, temperature=1.0, do_sample=True)
    summaries.append(summary[0]['summary_text'])
  return ' '.join(summaries)

# Function to get YouTube transcript
def get_youtube_transcript(video_url):
  video_id = video_url.split("v=")[1]
  transcript = YouTubeTranscriptApi.get_transcript(video_id)
  return transcript

# Function to format transcript
def format_transcript(transcript):
  formatted_transcript = ""
  for entry in transcript:
    start_time = entry["start"]
    end_time = entry["start"] + entry["duration"]
    formatted_transcript += f"[{format_time(start_time)} --> {format_time(end_time)}] {entry['text']}\n"
  return formatted_transcript

# Function to format time
def format_time(seconds):
  minutes, seconds = divmod(seconds, 60)
  hours, minutes = divmod(minutes, 60)
  return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):05.2f}"

# Streamlit app
st.title("YouTube Video Summarizer & MCQ Generator")

# User input: YouTube URL
youtube_url = st.text_input("Enter YouTube Video URL:")

# Process input when 'Summarize' button is clicked
if st.button("Summarize"):
  if youtube_url:
    # Get YouTube transcript
    transcript = get_youtube_transcript(youtube_url)
    formatted_transcript = format_transcript(transcript)
  
    # Remove timestamps
    text_without_timestamps = remove_timestamps(formatted_transcript)
  
    # Summarize text
    summary = summarize_text(text_without_timestamps)
  
    # Ollama Integration for MCQ Generation
    model = "llama2"  # Or any other Ollama model you prefer
    message = summary  # Assuming summary holds the summarized text
    prompt_template = ChatPromptTemplate.from_messages( [
        ("system","You are a helpful assistant that generates 5 mcqs question based on "+ message),
    ])

    message_dict = {"message": summary}  # Create a dictionary with the "message" key
    formatted_prompt = prompt_template.format_messages(summary=summary)
    ollama = ChatOllama(model=model)
    response = ollama(formatted_prompt)
    mcqs = response.content
  
    # Display the summarized text and MCQs
    st.header("Summarized Text:")
    st.write(summary)
    st.header("Multiple Choice Questions:")
    st.write(mcqs)
  else:
    st.warning("Please enter a valid YouTube Video URL.")
