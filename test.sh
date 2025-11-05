RUN_NAME=resnet50_bin
CHKPT_NAME=model_400.pth
BEST_CHKPT_NAME=model_best.pth


# test script to save predictions as pickle file
python src/predict.py \
--checkpoint runs/${RUN_NAME}/checkpoints/${CHKPT_NAME} \
--config runs/${RUN_NAME}/config.py \
--output predictions_last.pkl

# predict for best model
python src/predict.py \
--checkpoint runs/${RUN_NAME}/checkpoints/${BEST_CHKPT_NAME} \
--config runs/${RUN_NAME}/config.py \
--output predictions_best.pkl



# save classification report and heatmap of confusion matrix
python src/evaluate.py \
--pkl_path runs/${RUN_NAME}/predictions_last.pkl \
--save_dir runs/${RUN_NAME}/infer_last/

# save for best model
python src/evaluate.py \
--pkl_path runs/${RUN_NAME}/predictions_best.pkl \
--save_dir runs/${RUN_NAME}/infer_best/



## visualize top N worst predictions as individual files
# model predicted human as non humans
python src/viz_predictions.py \
--pkl_path runs/${RUN_NAME}/predictions_last.pkl \
--output_dir runs/${RUN_NAME}/infer_last/not_safe_as_safe/ \
--top_n 100 \
--mistake_type not_safe_as_safe

# model predicted non human as human
python src/viz_predictions.py \
--pkl_path runs/${RUN_NAME}/predictions_last.pkl \
--output_dir runs/${RUN_NAME}/infer_last/safe_as_not_safe/ \
--top_n 100 \
--mistake_type safe_as_not_safe

# for best model
python src/viz_predictions.py \
--pkl_path runs/${RUN_NAME}/predictions_best.pkl \
--output_dir runs/${RUN_NAME}/infer_best/not_safe_as_safe/ \
--top_n 100 \
--mistake_type not_safe_as_safe

python src/viz_predictions.py \
--pkl_path runs/${RUN_NAME}/predictions_best.pkl \
--output_dir runs/${RUN_NAME}/infer_best/safe_as_not_safe/ \
--top_n 100 \
--mistake_type safe_as_not_safe



# Test on inference data
python src/inference.py \
-i dataset/infer_data_resized/ \
-o runs/${RUN_NAME}/infer_last/infer_data/ \
-c runs/${RUN_NAME}/checkpoints/${CHKPT_NAME} \
-m resnet50 \
-b 8 \
-w 16

# for best model
python src/inference.py \
-i dataset/infer_data_resized/ \
-o runs/${RUN_NAME}/infer_best/infer_data/ \
-c runs/${RUN_NAME}/checkpoints/${BEST_CHKPT_NAME} \
-m resnet50 \
-b 8 \
-w 16



# save grids
python save_grids.py \
-i runs/${RUN_NAME}/infer_last/not_safe_as_safe/ \
-o runs/${RUN_NAME}/infer_last/grids/not_safe_as_safe \
--img_size 712 512 \
--num_workers 16

python save_grids.py \
-i runs/${RUN_NAME}/infer_last/safe_as_not_safe/ \
-o runs/${RUN_NAME}/infer_last/grids/safe_as_not_safe \
--img_size 712 512 \
--num_workers 16


python save_grids.py \
-i runs/${RUN_NAME}/infer_last/infer_data/ \
-o runs/${RUN_NAME}/infer_last/grids/infer_data \
--img_size 732 512 \
--num_workers 16


# for best model
python save_grids.py \
-i runs/${RUN_NAME}/infer_best/not_safe_as_safe/ \
-o runs/${RUN_NAME}/infer_best/grids/not_safe_as_safe \
--img_size 712 512 \
--num_workers 16

python save_grids.py \
-i runs/${RUN_NAME}/infer_best/safe_as_not_safe/ \
-o runs/${RUN_NAME}/infer_best/grids/safe_as_not_safe \
--img_size 712 512 \
--num_workers 16


python save_grids.py \
-i runs/${RUN_NAME}/infer_best/infer_data/ \
-o runs/${RUN_NAME}/infer_best/grids/infer_data \
--img_size 732 512 \
--num_workers 16