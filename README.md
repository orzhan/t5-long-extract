# t5-long-extract
The code for T5-LONG-EXTRACT system submitted to [FNS-2021 Shared Task](http://wp.lancs.ac.uk/cfie/fns2021/)


# Training

Unpack training data to current directory.

Run `source install.sh` to install requirements.

Run `python prepare.sh` to prepare training data.

Run `source train.sh` to train the model.

# Running the model

Unpack test data to current directory.

Run `python eval.sh` to prepare the data, run the model and generate summaries based on the predictions.

Generated summaries will be placed in verification/system directory.

# Model weights

https://huggingface.co/orzhan/t5-long-extract

# Colab demo

https://colab.research.google.com/drive/1cJiDGYrmmlQ3Ei5ZcitRJLMcw_WA5RW-?usp=sharing

# Citation

If you use this codebase, or otherwise found our work valuable, please cite:
```
@inproceedings{orzhenovskii-2021-t5,
    title = "T5-{LONG}-{EXTRACT} at {FNS}-2021 Shared Task",
    author = "Orzhenovskii, Mikhail",
    booktitle = "Proceedings of the 3rd Financial Narrative Processing Workshop",
    month = "15-16 " # sep,
    year = "2021",
    address = "Lancaster, United Kingdom",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.fnp-1.12",
    pages = "67--69",
}
```