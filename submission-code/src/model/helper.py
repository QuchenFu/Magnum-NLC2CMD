import torch
from nl2bash.nlp_tools import tokenizer
import nltk
import os
from matplotlib import pyplot as plt
import numpy as np
from nl2bash.metric import metric_utils
import json
from pathlib import Path

smoothing = nltk.translate.bleu_score.SmoothingFunction()


def translate_sentence(model, sentence, english, bash, device, max_length=30):
    if type(sentence) == str:
        tokens = [token.lower() for token in tokenizer.ner_tokenizer(sentence)[0]]
    else:
        tokens = [token.lower() for token in sentence]
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, english.init_token)
    tokens.append(english.eos_token)

    # Go through each english token and convert to an index
    text_to_indices = [english.vocab.stoi[token] for token in tokens]
    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)
    outputs = [bash.vocab.stoi["<sos>"]]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)

        with torch.no_grad():
            output = model(sentence_tensor, trg_tensor)

        best_guess = output.argmax(2)[-1, :].item()
        outputs.append(best_guess)

        if best_guess == bash.vocab.stoi["<eos>"]:
            break

    translated_sentence = [bash.vocab.itos[idx] for idx in outputs]
    # remove start token
    return translated_sentence[1:]



def save_checkpoint(state, filename="100_epoch_my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    # print(os.path.dirname(os.path.realpath(__file__)))
    path=os.path.join(Path('./').parent, 'submission_code/'+filename)
    # print(path)
    torch.save(state, path)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def bleu(data, model, english, bash, device, detail):
    averagebleu = 0
    bleus=[]
    for example in data:
        targets = []
        src = example.eng
        trg = example.bash
        prediction = translate_sentence(model, src, english, bash, device, max_length=50)
        prediction = prediction[:-1]  # remove <eos> token
        targets.append(trg)
        bleu = nltk.translate.bleu_score.sentence_bleu(targets, prediction,
                                                       smoothing_function=smoothing.method1, auto_reweigh=True)
        if detail:
            print(bleu)
            print(prediction)
            print(trg)
        averagebleu = averagebleu + bleu
        bleus.append(bleu)
    if detail:
        bins = np.arange(0, 1, 0.01) # fixed bin size
        plt.xlim([min(bleus), max(bleus)])
        plt.hist(bleus, bins=bins, alpha=0.5)
        plt.show()
        plt.savefig('bleu.png')
    return averagebleu/len(data)

def competition_metric(data, model, english, bash, device, detail):
    average_score = 0
    scores=[]
    for example in data:
        ob = json.loads(example)
        src = ob["English"]
        trg = ob["Bash"]
        prediction = translate_sentence(model, src, english, bash, device, max_length=50)
        prediction = prediction[:-1]  # remove <eos> token
        prediction=' '.join(prediction)
        predicted_confidence = 1.0
        metric_val = max(metric_utils.compute_metric(prediction, predicted_confidence, trg), 0)
        if detail:
            print(metric_val)
            print(prediction)
            print(trg)
        average_score = average_score + metric_val
        scores.append(metric_val)
    if detail:
        bins = np.arange(-1, 1, 0.1) # fixed bin size
        plt.xlim([min(scores), max(scores)])
        plt.hist(scores, bins=bins, alpha=0.5)
        plt.show()
        plt.savefig('score.png')
    return average_score/len(data)