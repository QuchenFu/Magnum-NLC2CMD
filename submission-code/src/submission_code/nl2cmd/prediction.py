from onmt.translate.translator import build_translator
from argparse import Namespace
import metric_utils
import numpy as np


from matplotlib import pyplot as plt
import math

opt = Namespace(models=['/tmp/pycharm_project_78/submission-code/src/submission_code/model_step_2000.pt'], n_best=5, alpha=0.0, batch_type='sents', beam_size=5, beta=-0.0, block_ngram_repeat=0, coverage_penalty='none', data_type='text', dump_beam='', fp32=True, gpu=-1, ignore_when_blocking=[], length_penalty='none', max_length=100, max_sent_length=None, min_length=0, output='/dev/null', phrase_table='', random_sampling_temp=1.0, random_sampling_topk=1, ratio=-0.0, replace_unk=False, report_align=False, report_time=False, seed=829, stepwise_penalty=False, tgt=None, verbose=False, tgt_prefix=None)
translator = build_translator(opt, report_score=False)

english_txt = open("/tmp/pycharm_project_78/submission-code/src/submission_code/nl2cmd/invocations_proccess.txt", encoding="utf8").read().split("\n")
bash_txt = open("/tmp/pycharm_project_78/submission-code/src/submission_code/nl2cmd/cmds_proccess.txt", encoding="utf8").read().split("\n")
scores=[]
good_weights=[]
bad_weights=[]
k=0
for i in range(10346):
    sentence=english_txt[i]
    trg=bash_txt[i]
    translated = translator.translate([sentence], batch_size=1)
    weight=translated[0][0][0].item()
    prediction=translated[1][0][0]
    predictions=[]
    scs=[]
    for j in range(5):
        p=translated[1][0][j]
        scs.append(metric_utils.compute_metric(p, 1.0, trg))
        predictions.append(p)
    if scs[0]<0 and np.max(scs[1:4])>0:
        k=k+1
    score=metric_utils.compute_metric(prediction, 1.0, trg)
    # if score<0:
    #     score=score/5
    scores.append(score)
    if score>0:
        good_weights.append(math.exp(weight))
    else:
        bad_weights.append(math.exp(weight))
# bins = np.arange(-1, 1.1, 0.0999)  # fixed bin size
# plt.xlim([min(scores), max(scores) + 0.1])
# plt.hist(scores, bins=bins, alpha=0.5)
# plt.show()
bins = np.arange(0, 1, 0.1)  # fixed bin size
plt.xlim([min(good_weights), max(good_weights) + 0.1])
plt.hist(good_weights, bins=bins, alpha=0.5)
plt.show()

bins = np.arange(0, 1, 0.1)  # fixed bin size
plt.xlim([min(bad_weights), max(bad_weights) + 0.1])
plt.hist(bad_weights, bins=bins, alpha=0.5)
plt.show()
#
# print(k)
# print(np.mean(scores))

