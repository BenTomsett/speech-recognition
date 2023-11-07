import csv
import itertools
import time

from tqdm import tqdm

from end_to_end_test import run_test


def main():
    mfcc_params_options = {
        'n_ceps': [13, 26, 39],
    }

    dnn_params_options = {
        'epochs': [10, 50, 100],
        'batch_size': [16, 32, 64],
        'learning_rate': [0.001, 0.01, 0.1]
    }

    filename = f'results-{int(time.time())}.csv'

    start = time.time()

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)

        all_params = list(itertools.product(
            mfcc_params_options['n_ceps'],
            dnn_params_options['epochs'],
            dnn_params_options['batch_size'],
            dnn_params_options['learning_rate']
        ))

        for params in tqdm(all_params, desc='Running tests'):
            current_mfcc_params = {
                'n_ceps': params[0],
            }

            current_dnn_params = {
                'epochs': params[1],
                'batch_size': params[2],
                'learning_rate': params[3]
            }

            try:
                accuracy = run_test(current_mfcc_params, current_dnn_params)
                writer.writerow([accuracy, current_mfcc_params, current_dnn_params])
            except Exception as e:
                print(f'Test failed')
                print('Params: ', current_mfcc_params, current_dnn_params)
                print('Error: ', e)

    end = time.time()

    print(f'Time taken: {end - start}s')


if __name__ == '__main__':
    main()
