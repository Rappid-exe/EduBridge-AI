import pandas as pd
import json
import os
from pydub import AudioSegment
import speech_recognition as sr

def load_data(file_path):
    """
    Load data from a CSV or JSON file.
    """
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = pd.DataFrame(json.load(f))
    else:
        raise ValueError("Only CSV and JSON files are supported.")
    return data

def clean_text_data(data, text_column):
    """
    Clean text data by removing extra whitespace, special characters, and converting text to lowercase.
    """
    data[text_column] = data[text_column].str.strip()  # Remove extra whitespace
    data[text_column] = data[text_column].str.replace('[^a-zA-Z0-9\s]', '', regex=True)  # Remove special characters
    data[text_column] = data[text_column].str.lower()  # Convert text to lowercase
    return data

def convert_audio_to_text(audio_path):
    """
    Convert an audio file to text using speech recognition.
    """
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(audio_path)
    
    # Convert audio to WAV if needed (speech_recognition prefers WAV format)
    if not audio_path.endswith('.wav'):
        wav_path = audio_path.rsplit('.', 1)[0] + '.wav'
        audio.export(wav_path, format='wav')
        audio_path = wav_path

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = ""
        except sr.RequestError:
            text = ""
    
    return text

def process_audio_data(data, audio_column, text_column='transcription'):
    """
    Process audio files in the dataset and add transcriptions.
    """
    if text_column not in data.columns:
        data[text_column] = ""
    
    for i, audio_file in enumerate(data[audio_column]):
        if pd.notnull(audio_file):  # Check if the field is not empty
            text = convert_audio_to_text(audio_file)
            data.at[i, text_column] = text
    
    return data

def save_clean_data(data, output_path):
    """
    Save the cleaned data to a CSV file.
    """
    data.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Data saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Load and clean the data
    file_path = "input_file.csv"  # or input_file.json
    data = load_data(file_path)
    
    # Clean text data
    text_column = "text"  # Replace with the name of the text column
    data = clean_text_data(data, text_column)
    
    # Process audio files and add transcriptions
    audio_column = "audio_path"  # Replace with the name of the audio file column
    data = process_audio_data(data, audio_column)
    
    # Save the cleaned data
    output_path = "cleaned_data.csv"
    save_clean_data(data, output_path)
