
import torch
import torch.optim as optim
from model.helper import load_checkpoint, translate_sentence
from torchtext.data import Field, TabularDataset
from nl2bash.bashlint.data_tools import bash_tokenizer
from nl2bash.nlp_tools import tokenizer
from model.Transformer import Transformer

def tokenize_eng(text):
    return tokenizer.ner_tokenizer(text)[0]


def tokenize_bash(text):
    return bash_tokenizer(text)

def predict(invocations, result_cnt=5):

    english = Field(tokenize=tokenize_eng, lower=True, init_token="<sos>", eos_token="<eos>")
    bash = Field(tokenize=tokenize_bash, lower=True, init_token="<sos>", eos_token="<eos>")
    fields = {"English": ("eng", english), "Bash": ("bash", bash)}
    train_data, test_data = TabularDataset.splits(
        path="", train="/tmp/pycharm_project_78/submission-code/src/submission_code/train.json",
        test="/tmp/pycharm_project_78/submission-code/src/submission_code/test.json", format="json",
        fields=fields
    )
    english.build_vocab(train_data, max_size=10000, min_freq=2)
    bash.build_vocab(train_data, max_size=10000, min_freq=2)

    # We're ready to define everything we need for training our Seq2Seq model
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = True
    save_model = False

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
    if load_model:
        load_checkpoint(torch.load("/tmp/pycharm_project_78/submission-code/src/my_checkpoint.pth.tar", map_location='cpu'), model, optimizer)


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
    cf=[1.0] * (result_cnt-1)
    cf.append(0)
    confidences = [
        cf
        for _ in range(n_batch)
    ]

    ################################################################################################
    #     Participants should add their codes to fill predict `commands` and `confidences` here    #
    ################################################################################################
    for idx, inv in enumerate(invocations):

        # Call the translate method to retrieve translations and scores
        prediction = translate_sentence(model, inv, english, bash, device, max_length=30)[:-1]
        temp = " ".join(prediction)
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
