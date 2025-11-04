### Sync local run checkpoints and files to google drive
RUN_NAME=eff_s_firstrun

rm -r runs/${RUN_NAME}/__pycache__
rm -r runs/${RUN_NAME}/wandb

python upload_to_drive.py \
--token token.json \
--dir runs/${RUN_NAME} \
--parent 1EVkOIO9qQYrkePASr7Gp4MbftAbD_lJT

RUN_NAME=eff_s_lr1e3_wd0.05

rm -r runs/${RUN_NAME}/__pycache__
rm -r runs/${RUN_NAME}/wandb

python upload_to_drive.py \
--token token.json \
--dir runs/${RUN_NAME} \
--parent 1EVkOIO9qQYrkePASr7Gp4MbftAbD_lJT

RUN_NAME=eff_s_lr1e4_wd0.05

rm -r runs/${RUN_NAME}/__pycache__
rm -r runs/${RUN_NAME}/wandb

python upload_to_drive.py \
--token token.json \
--dir runs/${RUN_NAME} \
--parent 1EVkOIO9qQYrkePASr7Gp4MbftAbD_lJT

RUN_NAME=eff_s_lr1e4_wd1e5

rm -r runs/${RUN_NAME}/__pycache__
rm -r runs/${RUN_NAME}/wandb

python upload_to_drive.py \
--token token.json \
--dir runs/${RUN_NAME} \
--parent 1EVkOIO9qQYrkePASr7Gp4MbftAbD_lJT