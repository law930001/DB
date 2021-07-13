cd ../
CUDA_VISIBLE_DEVICES=0 \
python eval.py experiments/seg_detector/ic15_resnet152_deform_thre.yaml \
--resume /root/Storage/DB_v100/model/5002/model_epoch_2272_minibatch_150000 \
--polygon \
--box_thresh 0.7
# --thresh 0.4
