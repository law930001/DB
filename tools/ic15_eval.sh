cd ../
CUDA_VISIBLE_DEVICES=0 \
python eval.py experiments/seg_detector/ic17_hrnet48_thre.yaml \
--resume /root/Storage/DB_v100/model/5002/model_epoch_1123_minibatch_1152000 \
--polygon \
--box_thresh 0.7
# --thresh 0.3
