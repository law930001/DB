cd ../
python eval.py experiments/seg_detector/TBrain_resnet152_deform_thre.yaml \
--resume model/model_epoch_1123_minibatch_345000 \
--polygon \
--box_thresh 0.6
