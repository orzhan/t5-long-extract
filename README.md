# t5-long-extract
The code for T5-LONG-EXTRACT system submitted to [FNS-2021 Shared Task](http://wp.lancs.ac.uk/cfie/fns2021/)


# Training

Unpack training data to current directory.

Run `source install.sh` to install requirements.

Run `python prepare.sh` to prepare training data.

Run `source train.sh` to train the model.

# Running the model

Unpack test data to curreny directory.

Run `python eval.sh` to prepare the data, run the model and generate summaries based on the predictions.

Generated summaries will be placed in verification/system directory.