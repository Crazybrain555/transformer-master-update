"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""

import matplotlib.pyplot as plt
import re


def read(name, path):
    full_path = f'{path}/{name}'
    with open(full_path, 'r') as f:
        file = f.read()
        file = re.sub('\\[', '', file)
        file = re.sub('\\]', '', file)

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw(mode, path='./result'):
    if mode == 'loss':
        train = read('train_loss.txt', path)
        test = read('test_loss.txt', path)
        plt.plot(train, 'r', label='train')
        plt.plot(test, 'b', label='validation')
        plt.legend(loc='lower left')

    elif mode == 'bleu':
        bleu = read('bleu.txt', path)
        plt.plot(bleu, 'b', label='bleu score')
        plt.legend(loc='lower right')

    plt.xlabel('epoch')
    plt.ylabel(mode)
    plt.title(f'training result for {path}')
    plt.grid(True, which='both', axis='both')
    plt.show()


if __name__ == '__main__':
    model_paths = [
        './result/model3'
        ]

    for path in model_paths:
        draw(mode='loss', path=path)
        draw(mode='bleu', path=path)
