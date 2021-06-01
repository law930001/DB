cd ../
python eval.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml \
--resume model/totaltext_resnet50 \
--polygon \
--box_thresh 0.6
