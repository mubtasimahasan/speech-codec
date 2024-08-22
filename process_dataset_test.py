# test the data loading functions
import os
import torch
from torch.utils.data import DataLoader
from meldataset import CustomDataset, build_dataloader, collate

def test_custom_dataset(data_dir="processed_dataset"):
    # Check if the dataset is initialized correctly
    dataset = CustomDataset(data_dir=data_dir)
    num_files = len([fname for fname in os.listdir(data_dir) if fname.endswith('.flac')])
    assert len(dataset) == num_files, f"Expected {num_files} files, but got {len(dataset)}"

    # Check if we can load a few samples
    for i in range(min(10, len(dataset))):
        wave, mel = dataset[i]
        assert isinstance(wave, torch.Tensor), "Waveform should be a torch.Tensor"
        assert isinstance(mel, torch.Tensor), "Mel spectrogram should be a torch.Tensor"
        assert wave.dim() == 1, "Waveform should be 1D"
        assert mel.dim() == 2, "Mel spectrogram should be 2D"
        assert wave.size(0) == 16000 * 3, "Waveform length should be 24000 * 3 samples (3 seconds)"
        print(f"Sample {i} loaded successfully")

def test_dataloader(data_dir="processed_dataset"):
    # Build the dataloader
    dataloader = build_dataloader(batch_size=4, num_workers=0)

    # Check if we can iterate through a batch
    for i, (waves, mels, wave_lengths, mel_lengths) in enumerate(dataloader):
        assert waves.size(0) == 4, "Batch size should be 4"
        assert mels.size(0) == 4, "Batch size should be 4"
        assert wave_lengths.size(0) == 4, "Batch size should be 4"
        assert mel_lengths.size(0) == 4, "Batch size should be 4"
        print(f"Batch {i} loaded successfully")
        if i >= 2:  # Check only the first 3 batches
            break

if __name__ == "__main__":
    test_custom_dataset()
    test_dataloader()