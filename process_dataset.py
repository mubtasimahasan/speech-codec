from datasets import load_dataset
import librosa
import numpy as np
import soundfile as sf
import argparse
import os
import torch

# Function to process audio
def process_audio(example, segment_size):
    audio = example["audio"]["array"]
    sr = example["audio"]["sampling_rate"]

    # Resample to 16kHz if necessary
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Process audio to ensure it is 3 seconds long
    segment_size_samples = int(segment_size * sr)
    # print("before+++++++++> audio.shape", audio.shape)
    if len(audio) > segment_size_samples:
        # Randomly select a starting point for the audio segment
        max_start_index = len(audio) - segment_size_samples
        start_index = np.random.randint(0, max_start_index + 1)
        end_index = start_index + segment_size_samples
        # Slice the audio segment
        audio = audio[start_index:end_index]
        
    else:
        # Pad the audio to 3 seconds if it's shorter
        padding_length = segment_size_samples - len(audio)
        audio = np.pad(audio, (0, padding_length), 'constant')
        
    # print("after---------> audio.shape", audio.shape)    
    # Save processed audio
    output_path = os.path.join(output_dir, f"processed_{example['audio']['path'].split('/')[-1]}")
    sf.write(output_path, audio, sr)

    return {"path": output_path, "sampling_rate": sr}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and resample audio files from a dataset.")
    parser.add_argument("--dataset", type=str, default="zachary24/librispeech_train_clean_100", help="Name of the dataset to load (e.g., 'PolyAI/minds14').")
    parser.add_argument("--split", type=str, default="train", help="Subset of the dataset to use (e.g., 'en-US').")
    parser.add_argument("--ratio", type=float, default=0.00616, help="Ratio of the dataset to use (e.g., 0.5).")
    parser.add_argument("--segment_length", type=float, default=3.0, help="Length of each audio segment in seconds.")
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset, split=args.split)
    dataset = dataset.select(range(int(len(dataset) * args.ratio)))
    # dataset = load_dataset("zachary24/librispeech_train_clean_100", split="train")
    # dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
    # dataset = dataset.select(range(int(len(dataset) * 0.235)))
    print(dataset)

    # Directory to save the processed audio files
    output_dir = "processed_dataset"
    os.makedirs(output_dir, exist_ok=True)

    # Apply processing to the dataset
    processed_dataset = dataset.map(lambda example: process_audio(example, args.segment_length), remove_columns=dataset.column_names)
    processed_dataset.save_to_disk(output_dir)
