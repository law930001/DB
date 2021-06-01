cd ../
python train.py experiments/seg_detector/totaltext_resnet50_deform_thre.yaml \
--num_gpus 1 \
--num_workers 0 \
--batch_size 5 \
--resume model/pre-trained-model-synthtext-resnet50