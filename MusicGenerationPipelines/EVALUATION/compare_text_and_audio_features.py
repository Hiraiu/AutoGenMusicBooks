from EVALUATION.get_audio_features_from_wav import get_audio_features_from_wav
from EVALUATION.get_text_features import get_text_features
import torch

def evaluate_similarity(texts, wav_file):
    if type(texts) == str:
        texts = [texts]

    text_features = get_text_features(texts)
    audio_features = get_audio_features_from_wav(wav_file)

    print("\t\tText: ", texts)
    print("\t\tAudio: ", wav_file[-100:])

    similarity = torch.nn.functional.cosine_similarity(text_features, audio_features)
    similarity_values = similarity.tolist()

    return similarity_values
