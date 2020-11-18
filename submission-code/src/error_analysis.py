
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import Field, BucketIterator, TabularDataset
from model.Transformer import Transformer
from nl2bash.bashlint.data_tools import bash_tokenizer
from nl2bash.nlp_tools import tokenizer
from model.helper import translate_sentence, save_checkpoint, load_checkpoint, competition_metric, bleu


# wandb.init(project="magnum")
# config = wandb.config

def tokenize_eng(text):
    return tokenizer.ner_tokenizer(text)[0]


def tokenize_bash(text):
    return bash_tokenizer(text)

english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
bash = Field(tokenize=tokenize_bash, lower=True, init_token="<sos>", eos_token="<eos>")
fields = {"English": ("eng", english), "Bash": ("bash", bash)}

train_data = TabularDataset(
    path="/tmp/pycharm_project_78/submission-code/src/submission_code/train.json", format="json",
    fields=fields
)
english.build_vocab(train_data, max_size=10000, min_freq=2)
bash.build_vocab(train_data, max_size=10000, min_freq=2)





device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


learning_rate = 1e-4

# Model hyperparameters
src_vocab_size = len(english.vocab)
trg_vocab_size = len(bash.vocab)
embedding_size = 256
num_heads = 8
num_encoder_layers = 8
num_decoder_layers = 8
dropout = 0.10
max_len = 100
forward_expansion = 2048
src_pad_idx = english.vocab.stoi["<pad>"]

model = Transformer(
    embedding_size,
    src_vocab_size,
    trg_vocab_size,
    src_pad_idx,
    num_heads,
    num_encoder_layers,
    num_decoder_layers,
    forward_expansion,
    dropout,
    max_len,
    device,
).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
load_checkpoint(torch.load("/tmp/pycharm_project_78/submission-code/src/my_checkpoint.pth.tar", map_location='cpu'), model, optimizer)


with open("/tmp/pycharm_project_78/submission-code/src/submission_code/test.json") as f:
    td = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
td = [x.strip() for x in td]

real_score, flagscore, utilscore, all = competition_metric(td, model, english, bash, device, True)
# wandb.log({"real_score": real_score})
# wandb.log({"bleu_score": bleu_score})
# print(bleu_score)
print(real_score)
print(flagscore)
print(utilscore)
print(all)