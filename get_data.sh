#!/bin/bash

mkdir raw_data/

# Daily Dialog
wget -c http://yanran.li/files/ijcnlp_dailydialog.zip
unzip -o ijcnlp_dailydialog.zip -d raw_data/
rm ijcnlp_dailydialog.zip

unzip -o raw_data/ijcnlp_dailydialog/test.zip -d raw_data/ijcnlp_dailydialog/
unzip -o raw_data/ijcnlp_dailydialog/train.zip -d raw_data/ijcnlp_dailydialog/
unzip -o raw_data/ijcnlp_dailydialog/validation.zip -d raw_data/ijcnlp_dailydialog/

# Cleaned DD

wget https://github.com/yq-wen/overlapping-datasets/releases/download/v0.1/cleaned_dd.zip
unzip cleaned_dd.zip
mkdir -p raw_data/ijcnlp_dailydialog_cc/train/
mkdir -p raw_data/ijcnlp_dailydialog_cc/validation/
mkdir -p raw_data/ijcnlp_dailydialog_cc/test/
mv cleaned_dd/dialogs/train_dialogs.txt raw_data/ijcnlp_dailydialog_cc/train/dialogues_train.txt
mv cleaned_dd/dialogs/valid_dialogs.txt raw_data/ijcnlp_dailydialog_cc/validation/dialogues_validation.txt
mv cleaned_dd/dialogs/test_dialogs.txt raw_data/ijcnlp_dailydialog_cc/test/dialogues_test.txt
rm -rf cleaned_dd
rm cleaned_dd.zip


# DSTC7
# wget -c http://parl.ai/downloads/dstc7/dstc7_v2.tgz
# mkdir -p raw_data/dstc7-ubuntu
# tar -xvf dstc7_v2.tgz \
#         -C ./raw_data/dstc7-ubuntu \
#         ubuntu_dev_subtask_1.json \
#         ubuntu_test_subtask_1.json \
#         ubuntu_responses_subtask_1.tsv \
#         ubuntu_train_subtask_1.json
# rm dstc7_v2.tgz

