import time
from multiprocessing import Pool
import numpy as np
import tqdm
from tqdm import tqdm
from pathlib import Path
from utils import encode_params
import soundfile as sf
from feature_extraction import create_mel_filters, generate_mfccs

mfcc_params = {
    'n_filters': 40,
    'n_ceps': 12,
    'n_fft': 512
}


def load_file(file):
    return np.load(file)


def process_file(file):
    audio, sample_rate = sf.read(file)

    filters = create_mel_filters(mfcc_params['n_filters'], mfcc_params['n_fft'], sample_rate)
    mfccs = generate_mfccs(audio, sample_rate, filters, mfcc_params['n_ceps'])

    mfcc_path = Path(f'mfccs/{encode_params(mfcc_params)}/training/{file.stem}.npy')
    np.save(mfcc_path, mfccs)

    return mfccs


def main():
    mfcc_params_hash = encode_params(mfcc_params)
    mfcc_dir = Path(f'mfccs/{mfcc_params_hash}/training/')
    audio_dir = Path('audio/training/')

    data = []

    if mfcc_dir.exists():
        print(f'Loading pre-generated NFCCs from disk, param hash: {mfcc_params_hash}')

        files = sorted(mfcc_dir.glob('*.npy'))
        n_files = len(list(files))

        with Pool() as pool:
            data = list(tqdm(pool.imap(load_file, files), total=n_files))

        print(f'Loaded {len(data)} MFCCs from disk')

    else:
        print(f'Generating MFCCs, param hash: {mfcc_params_hash}')
        mfcc_dir.mkdir(parents=True, exist_ok=True)
        audio_files = sorted(audio_dir.glob('*.wav'))

        with Pool() as pool:
            data = list(tqdm(pool.imap(process_file, audio_files), total=len(audio_files)))


if __name__ == '__main__':
    main()
