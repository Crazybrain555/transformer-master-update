"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
from conf import *
from util.data_loader import Dataloader
from util.tokenizer import Tokenizer
import torch

tokenizer = Tokenizer()
loader = Dataloader(ext=('en', 'de'),
                    tokenize_en=tokenizer.tokenize_en,
                    tokenize_de=tokenizer.tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>')

train_data, valid_data, test_data = loader.make_dataset()
loader.build_vocab(train_data=train_data, min_freq=2)
train_iter, valid_iter, test_iter = loader.make_iter(train_data, valid_data, test_data,
                                                     batch_size=batch_size,
                                                     device=device)


src_pad_idx = loader.source_vocab.get_stoi()['<pad>']
trg_pad_idx = loader.target_vocab.get_stoi()['<pad>']
trg_sos_idx = loader.target_vocab.get_stoi()['<sos>']

enc_voc_size = len(loader.source_vocab)
dec_voc_size = len(loader.target_vocab)
