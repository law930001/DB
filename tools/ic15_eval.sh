cd ../
CUDA_VISIBLE_DEVICES=0 \
python eval.py experiments/seg_detector/tt_efficient_thre.yaml \
--resume /root/Storage/DB_v100/model/5004/final \
--polygon \
--box_thresh 0.7
# --thresh 0.3
