cd ../
CUDA_VISIBLE_DEVICES=0 \
python train.py experiments/seg_detector/synthtext_resnet152_deform_thre.yaml \
--num_gpus 1 \
--num_workers 10 \
--batch_size 13