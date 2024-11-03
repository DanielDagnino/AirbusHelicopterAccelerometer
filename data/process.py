#!/usr/bin/env python
import argparse
import sys

import h5py
import librosa
import numpy as np
from path import Path
from tqdm import tqdm


def extract_df(fn_df):
    with h5py.File(fn_df, 'r') as h5_file:
        print("Keys: ", list(h5_file.keys()))
        dftrain_group = h5_file[Path(fn_df).stem]

        print("Members of 'df...':", list(dftrain_group.keys()))
        for key in dftrain_group.keys():
            data = dftrain_group[key][:]
            print(f"Data in '{key}':", data.shape)

        df = dftrain_group['block0_values'][:]
    return df


def process(dir_out, win_length, n_fft, hop_length, eps=1.e-9):
    dir_out = Path(dir_out).expanduser()
    for fn_df in ['dfvalid.h5', 'dftrain.h5']:
        fn_df = Path(fn_df)
        df = extract_df(fn_df)

        # Fourier transform
        spectrograms = []
        for idx, waveform in tqdm(enumerate(df), total=len(df)):
            waveform = waveform / (np.max(np.abs(waveform)) + eps)
            spectrogram = np.abs(librosa.stft(
                waveform,
                n_fft=n_fft,
                hop_length=hop_length,
                window="hann",
                win_length=win_length,
            ))
            spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
            if idx == 0:
                print(f'spectrogram.shape = {spectrogram.shape}')

            spectrograms.append(spectrogram)
        spectrograms = np.array(spectrograms)
        spectrograms = np.transpose(spectrograms, (0, 2, 1))
        print(f'spectrograms.shape = {spectrograms.shape}')

        # Spectrogram to N_2D-matrices
        n_samples, n_twind, n_feats = spectrograms.shape
        spectrograms = spectrograms.reshape(n_samples, -1, n_feats, n_feats)
        n_samples, n_wind, n_feats, n_feats = spectrograms.shape
        spectrograms = spectrograms.reshape(-1, n_feats, n_feats)

        # Save matrix in separate files (too much to upload a single huge file in the dataloader)
        dir_out_split = dir_out / f'{fn_df.stem}_{n_fft}_{win_length}_{hop_length}-{n_samples}-{n_wind}'
        dir_out_split.makedirs_p()
        np.save(dir_out / f'{fn_df.stem}_{n_fft}_{win_length}_{hop_length}-{n_samples}-{n_wind}.npy', spectrograms)
        for idx, spectrogram in enumerate(spectrograms):
            fn_out_df = dir_out_split / f'{idx}.npy'
            np.save(fn_out_df, spectrogram)


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Airbus Helicopter Accelerometer Data Processing Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dir_out", type=str, default="~/MyTmp/AirbusHelicopterAccelerometer/data-db",
        help="Output directory for processed data."
    )
    parser.add_argument(
        "--n_fft", type=int, default=127,
        help="Number of FFT components for STFT."
    )
    parser.add_argument(
        "--win_length", type=int, default=126,
        help="Length of each STFT window."
    )
    parser.add_argument(
        "--hop_length", type=int, default=8,
        help="Number of samples between successive frames."
    )
    args = parser.parse_args(args)
    args = vars(args)
    process(**args)


if __name__ == "__main__":
    main(sys.argv[1:])
