### Sync local run checkpoints and files to google drive
RUN_NAME=vit_cleaned_data

rm -r runs/${RUN_NAME}/__pycache__
rm -r runs/${RUN_NAME}/wandb

python upload_to_drive.py \
--token token.json \
--dir runs/${RUN_NAME} \
--parent 1JTyIP_maAXurSFMcDDruJRVgLdS_Z7Pe

