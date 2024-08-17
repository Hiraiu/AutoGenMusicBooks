import librosa
from transformers import AutoProcessor, ClapModel
import torch

# Load audio data from the "cat_purring.wav" file
audio_file_path = "EVALUATION/cat_purring.wav"
audio_sample, _ = librosa.load(audio_file_path, sr=None)

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

input_text = ["purring", "meowing", "barking", "fan"]

inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities

print(probs)
print()
print("Most likely label:", probs.argmax().item(), "-->", input_text[probs.argmax().item()])
print()
print("Label probabilities:")
for i, prob in enumerate(probs[0]):
    print(f"Label {i}: {prob.item()}")
print()