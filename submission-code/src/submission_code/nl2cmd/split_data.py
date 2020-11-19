import json
import random
from nl2bash.bashlint.data_tools import bash_tokenizer
from nl2bash.nlp_tools2 import tokenizer


def correct_format():
    data = {}
    with open('data/nl2bash-data.json') as f:
        raw_data = json.load(f)
    for i in  range(1, len(raw_data.keys())+1):
        data[str(i)] = raw_data[str(i)]
        data[str(i)]['cmd'] = [raw_data[str(i)]['cmd']]
    with open('data/data-correct-format.json', 'w') as f:
        json.dump(data, f)

def split_train_test():
    rand_seed = 94726
    random.seed(rand_seed)
    train_data, test_data = {}, {}
    with open('data/data-correct-format.json', 'r') as f:
        data = json.load(f)
    all_index = [i for i in range(1, len(data.keys())+1)]
    random.shuffle(all_index)
    for i in all_index[:int(len(all_index)*0.8)]:
        train_data[str(i)] = data[str(i)]
    for j in all_index[int(len(all_index)*0.8):]:
        test_data[str(j)] = data[str(j)]
    with open('data/train_data.json', 'w') as f:
        json.dump(train_data, f)
    with open('data/test_data.json', 'w') as f:
        json.dump(test_data, f)

def preprocess_data_opennmt_transformer():
    def tokenize_eng(text):
        return tokenizer.ner_tokenizer(text)[0]

    def tokenize_bash(text):
        return bash_tokenizer(text)

    for split  in ['train', 'test']:
        file_path = 'data/{}_data.json'.format(split)
        with open(file_path, 'r') as f:
            data = json.load(f)
        english_txt = []
        bash_txt = []
        for i in data:
            english_txt.append(data[i]['invocation'])
            bash_txt.append(data[i]['cmd'][0])

        processed_cmd = []
        processed_nl = []

        for cmd, nl in zip(bash_txt, english_txt):
            processed_cmd.append(' '.join(tokenize_bash(cmd)))
            processed_nl.append(' '.join(tokenize_eng(nl)))

        with open('data/cmds_proccess_{}.txt'.format(split), 'w') as outF:
            for line in processed_cmd:
                outF.write(line)
                outF.write("\n")

        with open('data/invocations_proccess_{}.txt'.format(split), 'w') as outF:
            for line in processed_nl:
                outF.write(line)
                outF.write("\n")

    print('Data saved in data/cmds_proccess.txt and data/invocations_proccess.txt')
    print("--------------Preprocess Complete---------")


if __name__ == '__main__':
    correct_format()
    split_train_test()
    preprocess_data_opennmt_transformer()


