### Sync local run checkpoints and files to google drive
RUN_NAME=resnet50_bin

rm -r runs/${RUN_NAME}/__pycache__
rm -r runs/${RUN_NAME}/wandb

python upload_to_drive.py \
--token token.json \
--dir runs/${RUN_NAME} \
--parent 1EVkOIO9qQYrkePASr7Gp4MbftAbD_lJT

