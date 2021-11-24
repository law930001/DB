cd ../
CUDA_VISIBLE_DEVICES=0 \
python eval.py experiments/seg_detector/ic15_efficient_thre.yaml \
--resume /root/Storage/DB_v100/workspace/SegDetectorModel-seg_detector/efficentnet_b7/L1BalanceCELoss/model_exp32/final \
--polygon \
--box_thresh 0.7
# --thresh 0.3
