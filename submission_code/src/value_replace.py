from submission_code.nl2cmd.nl2bash.nlp_tools import tokenizer
from onmt.translate.translator import build_translator
from argparse import Namespace
import math

opt = Namespace(models=[
    # 'src/submission_code/nl2cmd/run/model_step_2100.pt',
    # 'src/submission_code/nl2cmd/run2/model_step_2100.pt',
    # 'src/submission_code/nl2cmd/run3/model_step_2000.pt',
    '/tmp/pycharm_project_78/submission_code/src/submission_code/model_step_2000.pt'
], n_best=5,
    avg_raw_probs=False,
    alpha=0.0, batch_type='sents', beam_size=5,
    beta=-0.0, block_ngram_repeat=0, coverage_penalty='none', data_type='text', dump_beam='', fp32=True,
    gpu=-1, ignore_when_blocking=[], length_penalty='none', max_length=100, max_sent_length=None,
    min_length=0, output='/dev/null', phrase_table='', random_sampling_temp=1.0, random_sampling_topk=1,
    ratio=-0.0, replace_unk=True, report_align=False, report_time=False, seed=829, stepwise_penalty=False,
    tgt=None, verbose=False, tgt_prefix=None)
translator = build_translator(opt, report_score=False)


def tokenize_eng(text):
    return tokenizer.ner_tokenizer(text)

inv="Archive \"/path/to/files/source\" to \"user@remoteip:/path/to/files/destination\" via ssh on port 2121"
my_inv="archiv _FILE to _FILE via ssh on port _NUMBER"

new_inv = tokenize_eng(inv)[0]
print(new_inv)
new_inv = ' '.join(new_inv)
translated = translator.translate([new_inv], batch_size=1)
commands = translated[1][0][0]
print(commands)
print(tokenize_eng(inv)[1][0])