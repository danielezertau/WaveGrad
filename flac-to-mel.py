import sys
import torchaudio
import torch


if __name__ == '__main__':
    flac_file = sys.argv[1]
    output_file = sys.argv[2]
    audio, sample_rate = torchaudio.load(flac_file)

    transform = torchaudio.transforms.MelSpectrogram(n_mels=80, sample_rate=sample_rate)
    mel = transform(audio).numpy()
    torch.save(mel, output_file)
