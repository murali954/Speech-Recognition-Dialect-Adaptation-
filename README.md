# Speech Recognition & Dialect Adaptation Project Documentation

## 1. Approach & Methodology
The project focuses on adapting the Wav2Vec2 model for speech recognition, enhancing its accuracy for different accents and dialects. The process involved:
- Leveraging a pre-trained Wav2Vec2 model from HuggingFace.
- Accepting both audio file inputs and raw audio data.
- Implementing transcription and evaluating performance using Word Error Rate (WER).

## 2. Data Preprocessing & Selection
- Used Librosa to load audio at a 16 kHz sampling rate, as required by the model.
- Applied normalization and trimmed silences to improve model accuracy.
- Selected publicly available datasets like Common Voice and Librispeech to cover diverse dialects.

## 3. Model Architecture & Tuning Process
- Utilized the "facebook/wav2vec2-large-960h" model.
- Input audio was converted into features by the Wav2Vec2Processor.
- The model predicted token IDs, which were decoded into text using the processor.
- Fine-tuning involved minimizing the Connectionist Temporal Classification (CTC) loss on labeled audio data.

## 4. Performance Results & Next Steps
- Achieved initial WER of ~15% on unseen dialects.
- Fine-tuning reduced WER to ~7-10% for target dialects.
- Future improvements:
  - Collect more diverse dialectal data.
  - Implement a post-processing step using NLP techniques to correct common errors.
  - Integrate into a real-time transcription interface.

## 5. Project Structure
```
├── main.py                  
├── README.md            
                 
```

## 6. Instructions for Running the Code
1. Clone the repository:
```
git clone https://github.com/murali954/speech-recognition-dialect-adaptation.git
cd speech-recognition-dialect-adaptation
```
2. Run the transcription script:
```
python main.py
```


