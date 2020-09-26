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

wandb.init(project="magnum")
config = wandb.config

def tokenize_eng(text):
    return tokenizer.ner_tokenizer(text)[0]


def tokenize_bash(text):
    return bash_tokenizer(text)


english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
bash = Field(tokenize=tokenize_bash, lower=True, init_token="<sos>", eos_token="<eos>")
fields = {"English": ("eng", english), "Bash": ("bash", bash)}

train_data, test_data = TabularDataset.splits(
    path="", train="/tmp/pycharm_project_78/submission-code/src/submission_code/train.json", test="/tmp/pycharm_project_78/submission-code/src/submission_code/test.json", format="json", fields=fields
)
english.build_vocab(train_data, max_size=10000, min_freq=2)
bash.build_vocab(train_data, max_size=10000, min_freq=2)


# We're ready to define everything we need for training our Seq2Seq model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_model = False
save_model = True

# Training hyperparameters
num_epochs = 1
learning_rate = 1e-4
batch_size = 128

config.learning_rate = learning_rate
config.batch_size = batch_size

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
log_interval = 30

config.embedding_size = embedding_size
config.num_encoder_layers = num_encoder_layers
config.num_decoder_layers = num_decoder_layers
config.log_interval = log_interval

# Tensorboard to get nice loss plot
writer = SummaryWriter("runs/loss_plot")
step = 0

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    device=device
)

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

wandb.watch(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=10, verbose=True
)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

sentence = "Add \"prefix_\" to every non-blank line in \"a.txt\""

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    model.eval()
    translated_sentence = translate_sentence(
        model, sentence, english, bash, device, max_length=30
    )

    print(f"Translated example sentence: \n {translated_sentence}")
    if epoch > 0 and epoch % config.log_interval == 0:
        # score = my_metric(test_data, model, english, bash, device, True)
        bleu_score = bleu(test_data, model, english, bash, device, False)
        with open("/tmp/pycharm_project_78/submission-code/src/submission_code/test.json") as f:
            test_data = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        test_data = [x.strip() for x in test_data]
        real_score = competition_metric(test_data, model, english, bash, device, False)
        wandb.log({"real_score": real_score})
        wandb.log({"bleu_score": bleu_score})
        print(bleu_score)
    model.train()
    losses = []

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.eng.to(device)
        target = batch.bash.to(device)

        # Forward prop
        output = model(inp_data, target[:-1, :])
        output = output.reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()

        loss = criterion(output, target)
        losses.append(loss.item())

        loss.backward()
        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        wandb.log({"loss": loss})
        step += 1

    mean_loss = sum(losses) / len(losses)
    scheduler.step(mean_loss)
    wandb.log({"mean_loss": mean_loss})
