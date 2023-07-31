import torch
import torchaudio
import sounddevice as sd
import soundfile as sf
from model_utils import load_model_and_tokenizer, get_model, get_tokenizer

# Global variables
freq = 16000

def record_audio(filename):
    recording = sd.rec(int(10 * freq), samplerate=freq, channels=2, dtype='int16')
    print("Recording in progress. Press Enter to stop recording.")
    input()
    sd.wait()
    sf.write(filename, recording, freq)

def tokenize_audio(audio_path, tokenizer, target_sample_rate=16000):
    waveform = preprocess_audio(audio_path, target_sample_rate)
    input_values = tokenizer(waveform.squeeze().numpy(), return_tensors="pt").input_values
    return input_values

def preprocess_audio(audio_path, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
    waveform = resampler(waveform)
    return waveform

def transcribe_audio(audio_path, target_sample_rate=16000):
    load_model_and_tokenizer()  # Load the model and tokenizer once
    
    model = get_model()
    tokenizer = get_tokenizer()

    input_values = tokenize_audio(audio_path, tokenizer, target_sample_rate)
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def main():
    print("Press Enter to start recording...")
    input()
    print("Recording started. Press Enter again to stop recording.")
    
    file_name = "./Rec/file.wav"
    record_audio(file_name)
    
    print(f"Recording saved as '{file_name}'.")
    
    transcription = transcribe_audio(file_name)
    print(transcription)

if __name__ == "__main__":
    main()
