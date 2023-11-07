import numpy as np
from keras.optimizers.legacy import Adam
from tqdm import tqdm
from pathlib import Path
from neural_net import train_model, test_model, build_layers
from utils import encode_params
import soundfile as sf
from feature_extraction import create_mel_filters, generate_mfccs


def generate_mfccs_from_file(path, mfcc_dir, mfcc_params):
    audio, sample_rate = sf.read(path)

    filters = create_mel_filters(40, 512, sample_rate)
    data = generate_mfccs(audio, sample_rate, filters, mfcc_params['n_ceps'])

    mfcc_path = Path(f'{mfcc_dir}/{path.stem}.npy')
    np.save(mfcc_path, data, allow_pickle=True)

    label = path.stem.split('-')[0]

    return data, label


def load_mfcc_from_file(path):
    data = np.load(path)
    label = path.stem.split('-')[0]

    return data, label


def load_mfccs(mfcc_dir, mfcc_params, mfcc_params_hash, audio_dir):
    if mfcc_dir.exists():
        files = sorted(mfcc_dir.glob('*.npy'))
        n_files = len(list(files))

        if n_files == 0:
            print("No *.npy files found in", mfcc_dir)
            exit(1)

        if __name__ == '__main__':
            loaded_data = [load_mfcc_from_file(file) for file in tqdm(files, total=n_files)]
        else:
            loaded_data = [load_mfcc_from_file(file) for file in files]

        data, labels = zip(*loaded_data)

        return np.array(data), np.array(labels)
    else:
        mfcc_dir.mkdir(parents=True, exist_ok=True)
        audio_files = sorted(audio_dir.glob('*.wav'))

        if __name__ == '__main__':
            generated_data = [generate_mfccs_from_file(file, mfcc_dir, mfcc_params) for file in
                              tqdm(audio_files, total=len(audio_files))]
        else:
            generated_data = [generate_mfccs_from_file(file, mfcc_dir, mfcc_params) for file in audio_files]

        data, labels = zip(*generated_data)

        return np.array(data), np.array(labels)


def load_model(path, dnn_params, training_data, training_labels):
    if path.exists():
        model = build_layers(20, training_data[0].shape, 3)
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=dnn_params['learning_rate']))
        model.load_weights(path)
        return model
    else:
        model = train_model(training_data, training_labels, dnn_params, verbose=1 if __name__ == '__main__' else 0)
        model.save_weights(path)
        return model


def run_test(mfcc_params, dnn_params):
    mfcc_params_hash = encode_params(mfcc_params)

    mfcc_training_dir = Path(f'./mfccs/{mfcc_params_hash}/training/')
    mfcc_testing_dir = Path(f'./mfccs/{mfcc_params_hash}/testing/')

    training_audio_dir = Path('./audio/training/')
    testing_audio_dir = Path('./audio/testing/')

    training_data, training_labels = load_mfccs(mfcc_training_dir, mfcc_params, mfcc_params_hash, training_audio_dir)
    testing_data, testing_labels = load_mfccs(mfcc_testing_dir, mfcc_params, mfcc_params_hash, testing_audio_dir)

    dnn_params_hash = encode_params(dnn_params)
    dnn_path = Path(f'./models/{mfcc_params_hash}-{dnn_params_hash}.h5')

    model = load_model(dnn_path, dnn_params, training_data, training_labels)

    return test_model(model, testing_data, testing_labels)


if __name__ == '__main__':
    mfcc_params = {
        'n_ceps': 12
    }

    dnn_params = {
        'epochs': 100,
        'batch_size': 16,
        'learning_rate': 0.001
    }

    accuracy = run_test(mfcc_params, dnn_params)

    print(f'Accuracy: {accuracy * 100}%')
