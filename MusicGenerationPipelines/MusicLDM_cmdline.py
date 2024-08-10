import time
import argparse
import scipy.io.wavfile

from diffusers import MusicLDMPipeline
from accelerate import Accelerator

def main():
    parser = argparse.ArgumentParser(description="Generate music based on a text prompt")
    parser.add_argument('prompt', type=str, help='Text prompt for generating music')
    parser.add_argument('duration', type=float, help='Duration of the generated music in seconds')
    parser.add_argument('path', type=str, help='Path to save the generated music')

    args = parser.parse_args()

    # Initialize the accelerator
    accelerator = Accelerator()
    device = accelerator.device

    start = time.time()

    repo_id = "ucsd-reach/musicldm"
    pipe = MusicLDMPipeline.from_pretrained(repo_id)

    pipe = pipe.to(device)

    print(f"Generating music with prompt: {args.prompt}, duration: {args.duration}s, output path: {args.path}")
    audio = pipe(args.prompt, num_inference_steps=50, audio_length_in_s=args.duration).audios[0]

    # Save the audio sample as a .wav file
    scipy.io.wavfile.write(args.path, rate=16000, data=audio)

    end = time.time()
    print(f"Time taken: {end - start} seconds")

if __name__ == "__main__":
    main()
