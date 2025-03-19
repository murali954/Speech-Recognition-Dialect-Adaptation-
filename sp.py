from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import numpy as np
from jiwer import wer

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

def transcribe_audio(audio_input):
    if isinstance(audio_input, str):
        audio, rate = librosa.load(audio_input, sr=16000)
    else:
        audio = audio_input
    
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
    
    transcription = processor.decode(predicted_ids[0])
    return transcription

if __name__ == "__main__":
    input_type = input("Enter 'file' to use audio file or 'text' to input audio data: ").strip().lower()
    if input_type == 'file':
        file_path = input("Enter audio file path: ").strip()
        transcription = transcribe_audio(file_path)
    else:
        audio_data = np.array([float(x) for x in input("Enter audio data as space-separated values: ").strip().split()])
        transcription = transcribe_audio(audio_data)
    
    print("Transcription:", transcription)
    
    ground_truth = input("Enter ground truth transcription (optional): ").strip()
    if ground_truth:
        error_rate = wer(ground_truth, transcription)
        print("Word Error Rate (WER):", error_rate)
