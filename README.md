# NLP_erc
In this project, I have to implement emotion classification in the conversation. This is the final project of NLP in NYCU, and the competition is held on Kaggle.

## Inference
If you just want to predict the result, you can click [this link](https://colab.research.google.com/drive/1aPiXwO73QrMLHYFmq2_QCWQ7cPKXJfIQ?usp=sharing).

## Download Dataset
If you don't git clone to my GitHub, you can use this code to get the dataset.
```
os.chdir("/content")
!mkdir -p dataset
os.chdir("/content/dataset")
!gdown --id '1-iAzlcS8XBLd0cC3_iQzhJlBVF1KiLSn' --output fixed_valid.csv
!gdown --id '145NjbOX9J9v7t6q7onGIXRmuo5mhiDPN' --output fixed_train.csv
!gdown --id '14PWtoDy1fN-wkRGcZFPDeGtSsRi-2w5g' --output fixed_test.csv
```

## Data Directory
```text
+- content
    +- dataset
        fixed_train.csv
        fixed_test.csv
        fixed_valid.csv
    ERC_dataset.py
    model.py
    test.py
    train.py
    utils.py
```

## Train
**For CoMPM.**

Argument
- pretrained: type of model (CoM and PM) (default: roberta-large)
- initial: initial weights of the model (pretrained or scratch) (default: pretrained)
- cls: label class (emotion or sentiment) (default: emotion)
- dataset: only use NLP dataset
- sample: ratio of the number of the train dataset (default: 1.0)
- freeze: Whether to learn the PM or not
```
!python3 train.py --initial pretrained --cls emotion --dataset NLP --freeze
```

## Test
Use the model.bin from [Google Drive](https://drive.google.com/uc?id=1tN8WCNEXM8fhf9uda-kCfx4Ved_7dNx9).
The final result will be in pred.csv
```
os.chdir("/content/NLP_erc/")
!gdown --id '1tN8WCNEXM8fhf9uda-kCfx4Ved_7dNx9' --output model.zip

!apt-get install unzi
!unzip -q 'model.zip' -d model
```
```
!python3 test.py --initial pretained
```

## Reference
**Thanks a lot to this author.**
https://github.com/rungjoo/CoMPM.git

## Citation

```bibtex
@article{lee2021compm,
  title={CoMPM: Context Modeling with Speaker's Pre-trained Memory Tracking for Emotion Recognition in Conversation},
  author={Lee, Joosung and Lee, Wooin},
  journal={arXiv preprint arXiv:2108.11626},
  year={2021}
}
```
