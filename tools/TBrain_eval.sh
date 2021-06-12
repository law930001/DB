cd ../
python eval.py experiments/seg_detector/TBrain_resnet152_deform_thre.yaml \
--resume /root/Storage/DB_v100/model/model_epoch_1016_minibatch_312000 \
--polygon \
--box_thresh 0.6
