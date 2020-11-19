from onmt.translate.translator import build_translator
from argparse import Namespace
import metric_utils
import numpy as np

opt = Namespace(models=['run/model_step_2000.pt'], n_best=1, alpha=0.0, batch_type='sents', beam_size=5, beta=-0.0, block_ngram_repeat=0, coverage_penalty='none', data_type='text', dump_beam='', fp32=True, gpu=-1, ignore_when_blocking=[], length_penalty='none', max_length=100, max_sent_length=None, min_length=0, output='/dev/null', phrase_table='', random_sampling_temp=1.0, random_sampling_topk=1, ratio=-0.0, replace_unk=False, report_align=False, report_time=False, seed=829, stepwise_penalty=False, tgt=None, verbose=False, tgt_prefix=None)
translator = build_translator(opt, report_score=False)

english_txt = open("/tmp/pycharm_project_240/nl2cmd/invocations_proccess.txt", encoding="utf8").read().split("\n")
bash_txt = open("/tmp/pycharm_project_240/nl2cmd/cmds_proccess.txt", encoding="utf8").read().split("\n")
scores=[]
for i in range(100):
    sentence=english_txt[i]
    trg=bash_txt[i]
    translated = translator.translate([sentence], batch_size=1)
    prediction=translated[1][0][0]
    score=metric_utils.compute_metric(prediction, 1.0, trg)
    if score<0:
        score=score/5
    scores.append(score)
    print(score)
print(np.mean(scores))

