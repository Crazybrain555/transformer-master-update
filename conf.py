"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import torch

# GPU device setting
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model parameter setting
batch_size = 128
max_len = 100
d_model = 512
n_layers = 6
n_heads = 8
ffn_hidden = 2048
drop_prob = 0.1

# optimizer parameter setting
init_lr = 1e-4
factor = 0.9
adam_eps = 5e-9
patience = 10
warmup = 100
epoch = 1000
clip = 1.0
weight_decay = 5e-5
inf = float('inf')


# # model parameter setting
# batch_size = 128
# max_len = 256
# d_model = 512
# n_layers = 6
# n_heads = 8
# ffn_hidden = 2048
# drop_prob = 0.1
#
# # optimizer parameter setting
# init_lr = 1e-5
# factor = 0.9
# adam_eps = 5e-9
# patience = 10
# warmup = 100
# epoch = 1000
# clip = 1.0
# weight_decay = 0
# inf = float('inf')



# batch_size = 128
# max_len = 256
# d_model = 512
# n_layers = 6
# n_heads = 8
# ffn_hidden = 2048
# drop_prob = 0.1
# init_lr = 0.1
# factor = 0.9
# patience = 10
# warmup = 100
# adam_eps = 5e-9
# epoch = 1000
# clip = 1
# weight_decay = 5e-4