RUN_NAME=effnet_v2_s_prod_statue_coslr
CHKPT_NAME=model_100.pth

# test script to save predictions as pickle file
python src/predict.py \
--checkpoint runs/${RUN_NAME}/checkpoints/${CHKPT_NAME} \
--config runs/${RUN_NAME}/config.py

# save classification report and heatmap of confusion matrix
python src/evaluate.py \
--pkl_path runs/${RUN_NAME}/predictions.pkl \
--save_dir runs/${RUN_NAME}/infer/

## visualize top N worst predictions as individual files
# model predicted human as non humans
python src/viz_predictions.py \
--pkl_path runs/${RUN_NAME}/predictions.pkl \
--output_dir runs/${RUN_NAME}/infer/human_as_nonhuman/ \
--top_n 100 \
--mistake_type human_as_nonhuman

# model predicted non human as human
python src/viz_predictions.py \
--pkl_path runs/${RUN_NAME}/predictions.pkl \
--output_dir runs/${RUN_NAME}/infer/nonhuman_as_human/ \
--top_n 100 \
--mistake_type nonhuman_as_human

