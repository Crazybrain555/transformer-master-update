"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""
import math
import time

from torch import nn, optim
from torch.optim import Adam

import data
from data import *
from models.model.transformer import Transformer
from util.bleu_old import idx_to_word, get_bleu
from util.epoch_timer import epoch_time
from tqdm import tqdm
import os
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def initialize_weights(m):
#     if hasattr(m, 'weight') and m.weight.dim() > 1:
#         nn.init.kaiming_uniform(m.weight.data)
def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):  # 这里是在检查每一个子模块，而不是整个模型
            nn.init.kaiming_uniform_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight.data, mean=0, std=0.01)





model = Transformer(src_pad_idx=src_pad_idx,
                    trg_pad_idx=trg_pad_idx,
                    trg_sos_idx=trg_sos_idx,
                    d_model=d_model,
                    enc_voc_size=enc_voc_size,
                    dec_voc_size=dec_voc_size,
                    max_len=max_len,
                    ffn_hidden=ffn_hidden,
                    n_head=n_heads,
                    n_layers=n_layers,
                    drop_prob=drop_prob,
                    device=device).to(device)

print(f'The model has {count_parameters(model):,} trainable parameters')
model.apply(initialize_weights)
optimizer = Adam(params=model.parameters(),
                 lr=init_lr,
                 weight_decay=weight_decay,
                 eps=adam_eps)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                 verbose=True,
                                                 factor=factor,
                                                 patience=patience)

criterion = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)
# def label_smoothed_nll_loss(lprobs, target, eps):
#     nll_loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1))
#     nll_loss = nll_loss.squeeze(-1)
#     smooth_loss = -lprobs.mean(dim=-1)
#     loss = (1.0 - eps) * nll_loss + eps * smooth_loss
#     return loss.sum()


def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0

    total_batches = len(iterator.dataset) // batch_size+1


    for i, batch in enumerate(iterator):
        src = batch[0].transpose(0,1).to(device)       #   from Sequence Length x Batch Size to Batch Size x Sequence Length
        trg = batch[1].transpose(0,1).to(device)

        optimizer.zero_grad()
        output = model(src, trg[:, :-1])
        output_reshape = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        loss = criterion(output_reshape, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()
        print('step :', round(((i+1) / total_batches) * 100, 2), '% , loss :', loss.item())

        # #print最后一组batch样例的数据，检查是否数据是不是不是random还是固定的
        # src_words = batch[0].transpose(0, 1)
        # print('src_words :', src_words)
        # print('src_words len :', batch[0].shape)

    return epoch_loss / total_batches


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    total_batches = len(iterator.dataset) // batch_size+1


    with torch.no_grad():
        for i, batch in enumerate(iterator):
            # src = batch.src
            # trg = batch.trg

            src = batch[0].transpose(0, 1).to(
                device)  # from Sequence Length x Batch Size to Batch Size x Sequence Length
            trg = batch[1].transpose(0, 1).to(device)

            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch[1].transpose(0, 1)[j], loader.target_vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target_vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    # return epoch_loss / len(iterator), batch_bleu
    return epoch_loss / total_batches, batch_bleu


#建立test_iter的函数
def test(model, iterator, criterion, path='result/model3'):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    total_bleu = 0
    total_batches = len(iterator.dataset) // batch_size + 1

    if not os.path.exists(path):
        os.makedirs(path)

    log_file_path = os.path.join(path, 'test_log.txt')

    with open(log_file_path, 'w',encoding='utf-8') as log_file:
        with torch.no_grad():
            for i, batch in enumerate(iterator):
                src = batch[0].transpose(0, 1).to(device)
                trg = batch[1].transpose(0, 1).to(device)

                output = model(src, trg[:, :-1])
                output_reshape = output.contiguous().view(-1, output.shape[-1])
                trg = trg[:, 1:].contiguous().view(-1)

                loss = criterion(output_reshape, trg)
                epoch_loss += loss.item()

                for j in range(batch_size):
                    try:
                        src_words = idx_to_word(batch[0].transpose(0, 1)[j], loader.source_vocab)
                        trg_words = idx_to_word(batch[1].transpose(0, 1)[j], loader.target_vocab)
                        output_words = output[j].max(dim=1)[1]
                        output_words = idx_to_word(output_words, loader.target_vocab)
                        bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                        batch_bleu.append(bleu)

                        log_file.write(f'Source: {src_words}\n')
                        log_file.write(f'Target: {trg_words}\n')
                        log_file.write(f'Predicted Target: {output_words}\n')
                        log_file.write(f'BLEU Score: {bleu}\n\n')

                    except Exception as e:
                        pass
                        # print(f"An error occurred: {e}")
                        # log_file.write(f"An error occurred: {e}\n")


                total_bleu = sum(batch_bleu) / len(batch_bleu)


        log_file.write(f'Total Epoch Loss: {epoch_loss / total_batches}\n')
        log_file.write(f'Total BLEU Score: {total_bleu}\n')

        log_file.flush()  # 最后一次刷新缓冲区

        return epoch_loss / total_batches, total_bleu


def run(total_epoch, best_loss,model_name='model4'):
    train_losses, test_losses, bleus = [], [], []

    # Check if 'result' directory exists, if not create it
    if not os.path.exists('result'):
        os.makedirs('result')


    #读取之前的模型 路径在saved\{model_name}\model-xx.pt 其中xx是之前保存的最好的loss的，取XX值最小的那一个文件
    #如果没有之前的模型，就从头开始训练

    if os.path.exists('saved/{0}'.format(model_name)):
        if len(os.listdir('saved/{0}'.format(model_name))) == 0:
            model_path = None
        else:
            model_path = os.listdir('saved/{0}'.format(model_name))
            model_path = [os.path.join('saved/{0}'.format(model_name), path) for path in model_path]
            # 去掉文件夹
            model_path = [path for path in model_path if os.path.isfile(path)]
            model_path = sorted(model_path, key=lambda x: int(x.split('-')[-1].split('.')[0]))
            model_path = model_path[0]
            model.load_state_dict(torch.load(model_path))
            print('load model from', model_path)
    else:
        os.makedirs('saved/{0}'.format(model_name))



    for step in tqdm(range(total_epoch)):
        start_time = time.time()
        # make sure train_iter and valid_iter are re-generated for each epoch

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




        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss, bleu = evaluate(model, valid_iter, criterion)

        end_time = time.time()


        if step > warmup:
            scheduler.step((train_loss+valid_loss))

        train_losses.append(train_loss)
        test_losses.append(valid_loss)
        bleus.append(bleu)
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_loss:
            best_loss = valid_loss

            torch.save(model.state_dict(), 'saved/{0}/model-{1}.pt'.format(model_name,valid_loss))



        def save_results_to_path(value, metric_name, path='result/{0}'.format(model_name)):
            file_path = os.path.join(path, metric_name + '.txt')
            if os.path.exists(file_path):
                # 读取之前的数据
                with open(file_path, 'r') as f:
                    metric_list = eval(f.read())

                metric_list.append(value)

                with open(file_path, 'w') as f:
                    f.write(str(metric_list))
            else:
                #创建路径文件
                if not os.path.exists(path):
                    os.makedirs(path)
                with open(file_path, 'w') as f:
                    f.write(str([value]))

        # 使用方法：
        save_results_to_path(train_loss, 'train_loss')
        save_results_to_path(bleu, 'bleu')
        save_results_to_path(valid_loss, 'test_loss')




        print(f'Epoch: {step + 1} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\tVal Loss: {valid_loss:.3f} |  Val PPL: {math.exp(valid_loss):7.3f}')
        print(f'\tBLEU Score: {bleu:.3f}')


        #只有最后一个epoch 才调用test函数
        # Call the test function here
        if step == total_epoch-1:
            test_epoch_loss, test_total_bleu = test(model, test_iter, criterion, path='result/{0}'.format(model_name))
            print(f'\tTest Loss: {test_epoch_loss:.3f} |  Test PPL: {math.exp(test_epoch_loss):7.3f}')
            print(f'\tTest BLEU Score: {test_total_bleu:.3f}')






    #调取test_iter 然后打入日志存放在 path='result/model3'，打印每一个source target 和predicted target 最后 打印blue score和total bluescore


if __name__ == '__main__':
    run(total_epoch=epoch, best_loss=inf)
