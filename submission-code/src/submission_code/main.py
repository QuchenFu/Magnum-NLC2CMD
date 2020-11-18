# This part of the code was adapted from the following open source code:
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/seq2seq_transformer/seq2seq_transformer.py

import torch
import torch.optim as optim
from model.helper import load_checkpoint, translate_sentence
from torchtext.data import Field, TabularDataset
from nl2bash.bashlint.data_tools import bash_tokenizer
from nl2bash.nlp_tools import tokenizer
from model.Transformer import Transformer
from onmt.translate.translator import build_translator
from argparse import Namespace

def tokenize_eng(text):
    return tokenizer.ner_tokenizer(text)[0]


def tokenize_bash(text):
    return bash_tokenizer(text)




opt = Namespace(models=['src/model_step_2000.pt'], n_best=1, alpha=0.0, batch_type='sents', beam_size=5, beta=-0.0, block_ngram_repeat=0, coverage_penalty='none', data_type='text', dump_beam='', fp32=True, gpu=-1, ignore_when_blocking=[], length_penalty='none', max_length=100, max_sent_length=None, min_length=0, output='/dev/null', phrase_table='', random_sampling_temp=1.0, random_sampling_topk=1, ratio=-0.0, replace_unk=False, report_align=False, report_time=False, seed=829, stepwise_penalty=False, tgt=None, verbose=False, tgt_prefix=None)
translator = build_translator(opt, report_score=False)
# sentence="copi loadabl kernel modul _FILE to driver in modul directori match current kernel"
# trg="cp File $( uname -r )"

#
#
# english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
# bash = Field(tokenize=tokenize_bash, lower=True, init_token="<sos>", eos_token="<eos>")
# fields = {"English": ("eng", english), "Bash": ("bash", bash)}
#
# train_data = TabularDataset(
#     path="src/submission_code/train.json", format="json",
#     fields=fields
# )
# english.build_vocab(train_data, max_size=10000, min_freq=2)
# bash.build_vocab(train_data, max_size=10000, min_freq=2)


def predict(invocations, result_cnt=5):
    # t0=time.time()
    # device = torch.device("cpu")
    # learning_rate = 1e-4
    # # Model hyperparameters
    # src_vocab_size = len(english.vocab)
    # trg_vocab_size = len(bash.vocab)
    # embedding_size = 256
    # num_heads = 8
    # num_encoder_layers = 10
    # num_decoder_layers = 10
    # dropout = 0.10
    # max_len = 70
    # forward_expansion = 2048
    # src_pad_idx = english.vocab.stoi["<pad>"]
    #
    # model = Transformer(
    #     embedding_size,
    #     src_vocab_size,
    #     trg_vocab_size,
    #     src_pad_idx,
    #     num_heads,
    #     num_encoder_layers,
    #     num_decoder_layers,
    #     forward_expansion,
    #     dropout,
    #     max_len,
    #     device,
    # ).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # load_checkpoint(torch.load("src/200_all_data_cmd10_batch200_max_len70_10layer.tar", map_location='cpu'), model, optimizer)


    """
    Function called by the evaluation script to interface the participants model
    `predict` function accepts the natural language invocations as input, and returns
    the predicted commands along with confidences as output. For each invocation,
    `result_cnt` number of predicted commands are expected to be returned.

    Args:
        1. invocations : `list (str)` : list of `n_batch` (default 16) natural language invocations
        2. result_cnt : `int` : number of predicted commands to return for each invocation

    Returns:
        1. commands : `list [ list (str) ]` : a list of list of strings of shape (n_batch, result_cnt)
        2. confidences: `list[ list (float) ]` : confidences corresponding to the predicted commands
                                                 confidence values should be between 0.0 and 1.0.
                                                 Shape: (n_batch, result_cnt)
    """

    n_batch = len(invocations)

    # `commands` and `confidences` have shape (n_batch, result_cnt)
    commands = [
        [''] * result_cnt
        for _ in range(n_batch)
    ]
    cf=[1.0]
    cf.append(0)
    cf.append(0)
    cf.append(0)
    cf.append(0)
    confidences = [
        cf
        for _ in range(n_batch)
    ]

    ################################################################################################
    #     Participants should add their codes to fill predict `commands` and `confidences` here    #
    ################################################################################################
    for idx, inv in enumerate(invocations):

        # prediction = translate_sentence(model, inv, english, bash, device, max_length=30)[:-1]
        # temp = " ".join(prediction)
        new_inv=tokenize_eng(inv)
        new_inv=' '.join(new_inv)
        translated = translator.translate([new_inv], batch_size=1)
        temp=translated[1][0][0]
        top_commands=[temp]*5
        print(top_commands)
        # For testing evalAI docker push, just fill top command - just need to check
        # if tellina imports work correctly right now
        for i in range(result_cnt):
            commands[idx][i] = top_commands[i]

    ################################################################################################
    #                               Participant code block ends                                    #
    ################################################################################################

    return commands, confidences
