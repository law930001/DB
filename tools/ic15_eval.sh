cd ../
CUDA_VISIBLE_DEVICES=0 \
python eval.py experiments/seg_detector/ic15_resnet152_interd2v2.yaml \
--resume /root/Storage/DB_v100/model/jung/model_epoch_1183_minibatch_56000_ic15interd2v2_8853.pth \
--polygon \
--box_thresh 0.7
# --thresh 0.3
