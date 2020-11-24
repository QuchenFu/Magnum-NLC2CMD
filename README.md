# <img src="https://www.magnum.io/img/magnum.png" width="24" height="24"> Magnum-NLC2CMD 

<img src="https://evalai.s3.amazonaws.com/media/logos/4c055dbb-a30a-4aa1-b86b-33dd76940e14.jpg" align="right"
     alt="Magnum logo" height="150">

Magnum-NLC2CMD is the winning team' solution for the **[NeurIPS 2020 NLC2CMD challenge]**. The NLC2CMD Competition challenges you to build an algorithm that can translate an English description (𝑛𝑙𝑐) of a command line task to its corresponding command line syntax (𝑐). The model achieved a 0.53 score in Accuracy Track on the open **[Leaderboard]**. The  **[tellina]** model was the previous SOTA which was used as the baseline.
<p align="left">
<img width="650" alt="Screen Shot 2020-11-23 at 3 38 13 PM" src="https://user-images.githubusercontent.com/31392274/100018358-f34fa600-2da1-11eb-94c6-b848c774aca9.png">
</p>

[NeurIPS 2020 NLC2CMD challenge]: http://nlc2cmd.us-east.mybluemix.net/#/
[leaderboard]: https://eval.ai/web/challenges/challenge-page/674/leaderboard/1831
[tellina]: https://github.com/IBM/clai/tree/master/clai/server/plugins/tellina

## Requirements
<details><summary>Show details</summary>
<p>

* numpy
* six
* nltk
* experiment-impact-tracker
* scikit-learn
* pandas
* flake8==3.8.3
* spacy==2.3.0
* tb-nightly==2.3.0a20200621
* tensorboard-plugin-wit==1.6.0.post3
* torch==1.6.0
* torchtext==0.4.0
* torchvision==0.7.0
* tqdm==4.46.1
* OpenNMT-py==2.0.0rc2

</p>
</details>

## How it works

### Environment
1. Create a virtual environment with python3.6 installed(`virtualenv`).
2. use `pip3 install -r requirements.txt` to install the two requirements files
(`submission_code/src/submission_code/requirements.txt`, `submission_code/src/requirements.txt`)

### Data pre-processing
1. We have processed data in `submission_code/src/submission_code/nl2cmd/data`.
2. You can also download the Original raw data [here](https://ibm.ent.box.com/v/nl2bash-data)
3. `cd submission_code/src/submission_code/nl2cmd/`
4. `python3 OpenNMT_data_process.py`
5. `onmt_build_vocab -config nl2cmd.yaml -n_sample 10347 --src_vocab_threshold 2 --tgt_vocab_threshold 2`

### Train

1. `cd submission_code/src/submission_code/nl2cmd`
2. `onmt_train -config nl2cmd.yaml`

### Inference

1. `cd submission_code/src/submission_code/nl2cmd`
2. `onmt_translate -model run/model_step_2000.pt -src invocations_proccess.txt -output pred_2000.txt -gpu 0 -verbose`

### Evaluate

1. `python3 evaluate.py --annotation_filepath submission-code/src/submission_code/nl2cmd/data/test_data.json --params_filepath submission-code/configs/core/evaluation_params.json --output_folderpath submission-code/logs`

### Local test

1. `sh submission_code/BuildDockerImage.sh`
2. `python3 test_locally`

## Metrics

### Accuracy metric

𝑆𝑐𝑜𝑟𝑒(𝐴(𝑛𝑙𝑐))=max𝑝∈𝐴(𝑛𝑙𝑐)𝑆(𝑝) if ∃𝑝∈𝐴(𝑛𝑙𝑐) such that 𝑆(𝑝)>0; 
𝑆𝑐𝑜𝑟𝑒(𝐴(𝑛𝑙𝑐))=1|𝐴(𝑛𝑙𝑐)|∑𝑝∈𝐴(𝑛𝑙𝑐)𝑆(𝑝) otherwise.

### Reproduce

1. Train multiple models by modify random seed in nl2cmd.yaml, you should also modify the `save_model` to avoid overwrite existing models.
2. Hand pick the best performed ones on local test set and put their directories in the main.py

## reference

* [Open-NMT](https://github.com/OpenNMT/OpenNMT-py)
* [Bashlex](https://github.com/idank/bashlex)
* [clai](https://github.com/IBM/clai)
* [Tellina](https://github.com/TellinaTool/nl2bash)
* [Training Tips for the Transformer Model](https://ufal.mff.cuni.cz/pbml/110/art-popel-bojar.pdf)