"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import os
from torchtext.datasets import Multi30k
# train_iter,valid_iter, test_iter = Multi30k(split=('train', 'valid','test'))
# train_iter,valid_iter, test_iter = Multi30k()
from torchtext.datasets import multi30k, Multi30k
from util.new_Multi30k import newMulti30k

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from collections import Counter
from torchtext.vocab import vocab
import random
import torch
from torchtext.data.functional import to_map_style_dataset

class Dataloader:

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.init_token = init_token
        self.eos_token = eos_token
        print('dataset initializing start')

    def make_dataset(self):
        # Update URLs to point to data stored by user
        # multi30k.URL["train"] = "https://github.com/neychev/small_DL_repo/raw/master/datasets/Multi30k/training.tar.gz"
        # multi30k.URL[
        #     "valid"] = "https://github.com/neychev/small_DL_repo/raw/master/datasets/Multi30k/validation.tar.gz"

        # Update hash since there is a discrepancy between user hosted test split and that of the test split in the original dataset
        multi30k.MD5["test"] = "96ba7b2bfb42a087f2f9b7cdeb5ef1f270afdc894054b9a74698b9357995141b"
        #
        # train_data = Multi30k(split='train',language_pair=self.ext)
        # valid_data = Multi30k(split='valid',language_pair=self.ext)



        test_data = newMulti30k(root=os.path.join(os.path.abspath((os.getcwd())), 'data'),
                                split='test',language_pair=self.ext)
        train_data= newMulti30k(root=os.path.join(os.path.abspath((os.getcwd())), 'data'),
                                split='train',language_pair=self.ext)
        valid_data = newMulti30k(root=os.path.join(os.path.abspath((os.getcwd())), 'data'),
                                split='val',language_pair=self.ext)


        # Thesolution is touse torchtext.data.functional.to_map_style_dataset(iter_data)(official
        # doc) toconvertyouriterable - styledatasettomap - styledataset.UserWarning: Some child DataPipes are
        # not exhausted when __iter__ is called. We are resetting the buffer and each child DataPipe will read from the start again.
        #   warnings.warn("Some child DataPipes are not exhausted when __iter__ is called. We are resetting "


        train_data = [(src.lower(), trg.lower()) for src, trg in to_map_style_dataset(train_data)[:]]
        valid_data = [(src.lower(), trg.lower()) for src, trg in to_map_style_dataset(valid_data)[:]]
        test_data = [(src.lower(), trg.lower()) for src, trg in to_map_style_dataset(test_data)[:]]
        # train_data = [(src.lower(), trg.lower()) for src, trg in to_map_style_dataset(train_data)[:1000]]
        # valid_data = [(src.lower(), trg.lower()) for src, trg in to_map_style_dataset(valid_data)[:1000]]
        # test_data = [(src.lower(), trg.lower()) for src, trg in to_map_style_dataset(test_data)[:-1]]

        # print(next(iter(data_test)))

        return train_data, valid_data, test_data

    def build_vocab(self, train_data, min_freq):
        # self.source.build_vocab(train_data, min_freq=min_freq)
        # self.target.build_vocab(train_data, min_freq=min_freq)
        source_counter = Counter()
        target_counter = Counter()

        if self.ext == ('de', 'en'):
            for (source, target) in train_data:
                source_counter.update(self.tokenize_de(source))
                target_counter.update(self.tokenize_en(target))
            self.source_vocab = vocab(source_counter, min_freq=min_freq, specials=('<unk>', self.init_token,
                                                                                   self.eos_token, '<pad>'))
            self.target_vocab = vocab(target_counter, min_freq=min_freq, specials=('<unk>', self.init_token,
                                                                                   self.eos_token, '<pad>'))
            #set .set_default_index as the index of <unk> token
            self.source_vocab.set_default_index(self.source_vocab['<unk>'])
            self.target_vocab.set_default_index(self.target_vocab['<unk>'])


        elif self.ext == ('en', 'de'):
            for (source, target) in train_data:
                source_counter.update(self.tokenize_en(source))
                target_counter.update(self.tokenize_de(target))
            self.source_vocab = vocab(source_counter, min_freq=min_freq, specials=('<unk>', self.init_token,
                                                                                   self.eos_token, '<pad>'))
            self.target_vocab = vocab(target_counter, min_freq=min_freq, specials=('<unk>', self.init_token,
                                                                                   self.eos_token, '<pad>'))
            #set .set_default_index as the index of <unk> token
            self.source_vocab.set_default_index(self.source_vocab['<unk>'])
            self.target_vocab.set_default_index(self.target_vocab['<unk>'])

    def make_iter(self, train, validate, test, batch_size, device):
        # Simplifying the tokenization method selection
        tokenize_source = self.tokenize_de if 'de' in self.ext[0] else self.tokenize_en
        tokenize_target = self.tokenize_en if 'en' in self.ext[1] else self.tokenize_de

        source_text_transform = lambda x: (
                [self.source_vocab[self.init_token]] + [self.source_vocab[token] for token in tokenize_source(x)] + [
            self.source_vocab[self.eos_token]])

        target_text_transform = lambda x: (
                [self.target_vocab[self.init_token]] + [self.target_vocab[token] for token in tokenize_target(x)] + [
            self.target_vocab[self.eos_token]])

        def collate_batch(batch):
            source_list, target_list = [], []
            for (_source, _target) in batch:
                sourceprocessed_text = torch.tensor(source_text_transform(_source))
                source_list.append(sourceprocessed_text)

                targetprocessed_text = torch.tensor(target_text_transform(_target))
                target_list.append(targetprocessed_text)

            source_list = pad_sequence(source_list, padding_value=self.source_vocab['<pad>'])
            target_list = pad_sequence(target_list, padding_value=self.target_vocab['<pad>'])
            return source_list, target_list

        def batch_sampler(data):
            if self.ext == ('de', 'en'):
                indices = [(i, len(self.tokenize_de(x[0]))) for i, x in enumerate(data)]  # the length of de is enough to sort
            elif self.ext == ('en', 'de'):
                indices = [(i, len(self.tokenize_en(x[0]))) for i, x in enumerate(data)]
            else:
                raise ValueError('language pair not supported')

            random.shuffle(indices)
            pooled_indices = []
            # create pool of indices with similar lengths
            for i in range(0, len(indices), batch_size * 100):
                pooled_indices.extend(sorted(indices[i:i + batch_size * 100], key=lambda x: x[1]))

            pooled_indices = [x[0] for x in pooled_indices]
            # yield indices for current batch
            for i in range(0, len(pooled_indices), batch_size):
                yield pooled_indices[i:i + batch_size]

        train_iterator = DataLoader(train, batch_sampler=batch_sampler(train), collate_fn=collate_batch)
        valid_iterator = DataLoader(validate, batch_sampler=batch_sampler(validate), collate_fn=collate_batch)
        test_iterator = DataLoader(test, batch_sampler=batch_sampler(test), collate_fn=collate_batch)

        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator


if __name__ == '__main__':
    import torch
    from tokenizer import Tokenizer

    # GPU device setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model parameter setting
    batch_size = 128
    max_len = 256
    d_model = 512
    n_layers = 6
    n_heads = 8
    ffn_hidden = 2048
    drop_prob = 0.1

    # optimizer parameter setting
    init_lr = 1e-5
    factor = 0.9
    adam_eps = 5e-9
    patience = 10
    warmup = 100
    epoch = 1000
    clip = 1.0
    weight_decay = 5e-4
    inf = float('inf')

    tokenizer = Tokenizer()
    loader = Dataloader(ext=('en', 'de'),
                        tokenize_en=tokenizer.tokenize_en,
                        tokenize_de=tokenizer.tokenize_de,
                        init_token='<sos>',
                        eos_token='<eos>')

    train, valid, test = loader.make_dataset()
    loader.build_vocab(train_data=train, min_freq=2)
    train_iter, valid_iter, test_iter = loader.make_iter(train, valid, test,
                                                         batch_size=batch_size,
                                                         device=device)



    src_pad_idx = loader.source_vocab.get_stoi()['<pad>']
    trg_pad_idx = loader.target_vocab.get_stoi()['<pad>']
    trg_sos_idx = loader.target_vocab.get_stoi()['<sos>']

    enc_voc_size = len(loader.source_vocab)
    dec_voc_size = len(loader.target_vocab)

