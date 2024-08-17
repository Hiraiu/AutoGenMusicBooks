from transformers import AutoFeatureExtractor, ClapModel
import torch
import librosa
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

def get_audio_features_from_wav(file_path):
    # Load the pre-trained CLAP model and feature extractor
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
    feature_extractor = AutoFeatureExtractor.from_pretrained("laion/clap-htsat-unfused")

    # Load the audio file using librosa
    audio_path = file_path
    audio_data, sample_rate = librosa.load(audio_path, sr=48000)  # Ensure the sample rate matches the model's expected rate

    # Convert the audio data to a PyTorch tensor
    audio_tensor = torch.tensor(audio_data)

    # Extract features from the audio tensor
    inputs = feature_extractor(audio_tensor, return_tensors="pt", sampling_rate=sample_rate)
    audio_features = model.get_audio_features(**inputs)
    return audio_features
