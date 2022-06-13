import io
import os
import json
import sys
import numpy as np
import csv
import argparse
import random
import logging
from logging import INFO, DEBUG, NOTSET
from logging import StreamHandler, FileHandler, Formatter

import torch
from torch.utils.data import Dataset


def transform_nop(src, tgt):
    return (src, tgt)


def _append_data(dataset, src, tgt, transform, c):
    item = transform(src, tgt)
    if c < 5:
        logging.info(f'{src} -> {tgt}')
        if isinstance(item, tuple) and (item[0] != src or item[1] != tgt):
            logging.info(f' => {item[0]} -> {item[1]}')
    dataset.append(item)


def _loading_dataset(hparams, files=None, transform=transform_nop):
    if files is None:
        files = hparams.files
    column = hparams.column
    target_column = hparams.target_column
    dataset = []
    if target_column == -1:
        for file in files:
            logging.info(f'loading {file}')
            if file.endswith('.csv') or file.endswith('.tsv'):
                sep = ',' if file.endswith('.csv') else '\t'
                with io.open(file, encoding=hparams.encoding) as f:
                    reader = csv.reader(f, delimiter=sep)
                    for c, row in enumerate(reader):
                        if column >= len(row):
                            continue
                        _append_data(dataset, row[column], None, transform, c)
            elif file.endswith('.jsonl'):
                with io.open(file, encoding=hparams.encoding) as f:
                    for c, line in enumerate(f.readlines()):
                        data = json.loads(line)
                        if column not in data:
                            continue
                        _append_data(dataset, data[column], None, transform, c)
            else:
                with io.open(file, encoding=hparams.encoding) as f:
                    for c, line in enumerate(f.readlines()):
                        line = line.rstrip('\n')
                        _append_data(dataset, line, None, transform, c)
    else:
        for file in files:
            logging.info(f'loading {file}')
            if file.endswith('.csv') or file.endswith('.tsv'):
                sep = ',' if file.endswith('.csv') else '\t'
                with io.open(file, encoding=hparams.encoding) as f:
                    reader = csv.reader(f, delimiter=sep)
                    for c, row in enumerate(reader):
                        if column < len(row) and target_column < len(row):
                            src = row[column]
                            tgt = row[target_column]
                            _append_data(dataset, src, tgt, transform, c)
            elif file.endswith('.jsonl'):
                with io.open(file, encoding=hparams.encoding) as f:
                    for c, line in enumerate(f.readlines()):
                        data = json.loads(line)
                        if column in data and target_column in data:
                            _append_data(
                                dataset, data[column], data[target_column], transform, c)
            else:
                with io.open(file, encoding=hparams.encoding) as f:
                    for c, line in enumerate(f.readlines()):
                        d = line.rstrip('\n')
                        _append_data(
                            dataset, d, d, transform, c)
    logging.info(f'loaded {len(dataset)} dataset')
    if hparams.testing:
        return dataset[:20]
    return dataset

# MULTITASKING_TRANSFORM


class TSVDataset(Dataset):
    def __init__(self, hparams, dataset):
        self.hparams = hparams
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def test_and_save(self, generate, filename, max=None):
        if self.hparams.testing:
            max = 10
        with open(filename, 'w') as f:
            c = 0
            if max is not None:
                random.shuffle(self.dataset)
            for src, tgt in self.dataset:
                gen = generate(src)
                print(f'{src}\t{gen}\t{tgt}', file=f)
                f.flush()
                if c % 10 == 0:
                    logging.info(f'{src}\t{gen}\t{tgt}')
                c += 1
                if max is not None and c > max:
                    break


def load_DataSet(hparams, transform=transform_nop):
    train_data = _loading_dataset(hparams, None, transform)
    return TSVDataset(hparams, train_data)


def load_TrainTestDataSet(hparams, transform=transform_nop):
    train_files = []
    test_files = []
    for file in hparams.files:
        if '_train.' in file:
            train_files.append(file)
        elif '_test.' in file:
            test_files.append(file)
        else:
            train_files.append(file)
    train_data = _loading_dataset(hparams, train_files, transform)
    test_data = _loading_dataset(hparams, test_files, transform)
    return TSVDataset(hparams, train_data), TSVDataset(hparams, test_data)

# argparse


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _add_arguments(parser, args_dict):
    for key in args_dict:
        option_name = f'--{key}'
        default = args_dict[key]
        if isinstance(default, bool):
            if default == False:
                parser.add_argument(
                    option_name, action='store_true', default=default)
            elif default == True:
                parser.add_argument(
                    option_name, action='store_false', default=default)
        elif isinstance(default, int):
            parser.add_argument(option_name, type=int, default=default)
        elif isinstance(default, float):
            parser.add_argument(option_name, type=float, default=default)
        elif isinstance(default, str):
            parser.add_argument(option_name, default=default)


DEFAULT_SETUP = dict(
    model_name_or_path='google/mt5-small',
    tokenizer_name_or_path='google/mt5-small',
    additional_tokens='<b> </b> <nl> <e0> <e1> <e2> <e3> <e4> <e5> <e6> <e7> <e8> <e9>',
    seed=42,
    encoding='utf_8',
    column=0, target_column=1,
    max_length=128,
    target_max_length=128,
    progress_bar=False,
    # unsupervised training option
    masking=False,
    masking_ratio=0.35,
    masking_style='denoising_objective',
    output_dir='model',  # path to save the checkpoints
    # da
    da_choice=1.0,
    da_shuffle=1.0,
    bos_token='',
)



def _setup_logger(hparams):
    if not os.path.isdir(hparams.output_dir):
        os.makedirs(hparams.output_dir)

    log_file = f'{hparams.output_dir}/log_{hparams.project}.txt'

    # ストリームハンドラの設定
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(Formatter("%(message)s"))

    # ファイルハンドラの設定
    file_handler = FileHandler(log_file)

    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(
        Formatter(
            "%(asctime)s@ %(name)s [%(levelname)s] %(funcName)s: %(message)s")
    )
    # ルートロガーの設定
    logging.basicConfig(level=NOTSET, handlers=[stream_handler, file_handler])
    logging.info(f'PyTorch: {torch.__version__}')
    logging.info(f'hparams: {hparams}')


def parse_hparams(setups={}, Tokenizer=None):
    init_dict = DEFAULT_SETUP.copy()
    init_dict.update(setups)
    parser = argparse.ArgumentParser(description='train_data.py')
    parser.add_argument('files', nargs='+', help='files')
    parser.add_argument('--project', type=str,
                        default='test', help='project name')
    _add_arguments(parser, init_dict)
    # parser.add_argument('-q', '--quantize', action='store_true',
    #                     help='quantize model')
    hparams = parser.parse_args()
    hparams.output_dir = f'./{hparams.project}'
    _setup_logger(hparams)

    if hparams.project == 'test':
        print('***** TEST PROJECT *****')
        hparams.testing = True
        hparams.max_epochs = 3
    else:
        hparams.testing = False

    if hparams.masking or hparams.target_column == -1:
        hparams.masking = True
        hparams.target_column = -1

    if hparams.additional_tokens == '':
        hparams.additional_tokens = []
    else:
        hparams.additional_tokens = hparams.additional_tokens.split()

    if Tokenizer is not None:
        # if not hasattr(hparams, 'use_fast_tokenizer'):
        #     hparams.use_fast_tokenizer = False
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        hparams.tokenizer = Tokenizer.from_pretrained(
            hparams.tokenizer_name_or_path, is_fast=True)
        hparams.vocab_size = hparams.tokenizer.vocab_size
        if len(hparams.additional_tokens) > 0:
            hparams.tokenizer.add_tokens(hparams.additional_tokens)
            hparams.vocab_size += len(hparams.additional_tokens)
        logging.info(
            f'vocab_size: {hparams.tokenizer.vocab_size} {hparams.vocab_size}')

    _set_seed(hparams.seed)
    return hparams


def _main():
    hparams = parse_hparams()
    print(hparams)
    load_TrainTestDataSet(hparams)


if __name__ == '__main__':
    _main()
