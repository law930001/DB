cd ../
CUDA_VISIBLE_DEVICES=0 \
python eval.py experiments/seg_detector/ic15_efficient_thre.yaml \
--resume /root/Storage/DB_v100/model/5004/model_epoch_1200_minibatch_300000 \
--polygon \
--box_thresh 0.7
# --thresh 0.4
