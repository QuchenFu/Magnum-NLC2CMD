import torch
from nl2bash.nlp_tools import tokenizer
import nltk

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



def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
