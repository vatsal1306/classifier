RUN_NAME=eff_s_lr1e4_wd0.05
CHKPT_NAME=model_400.pth

# test script to save predictions as pickle file
python src/predict.py \
--checkpoint runs/${RUN_NAME}/checkpoints/${CHKPT_NAME} \
--config runs/${RUN_NAME}/config.py

# save classification report and heatmap of confusion matrix
python src/evaluate.py \
--pkl_path runs/${RUN_NAME}/predictions.pkl \
--save_dir runs/${RUN_NAME}/infer/

## visualize top N worst predictions as individual files
python src/viz_predictions.py \
--pkl_path runs/${RUN_NAME}/predictions.pkl \
--output_dir runs/${RUN_NAME}/infer/incorrect_preds/ \
--top_n 1000


# infer on 25K inference set (human)
python src/inference.py \
-i dataset/infer_data/ \
-o runs/${RUN_NAME}/infer/infer_data/ \
-c runs/${RUN_NAME}/checkpoints/${CHKPT_NAME} \
-m efficientnet_v2_s \
-b 128


## save grids
python save_grids.py \
-i runs/${RUN_NAME}/infer/incorrect_preds/ \
-o runs/${RUN_NAME}/infer/grids/incorrect_preds \
--img_size 752 512 \
--num_workers 32


python save_grids.py \
-i runs/${RUN_NAME}/infer/infer_data/ \
-o runs/${RUN_NAME}/infer/grids/infer_data \
--img_size 732 512 \
--num_workers 8
