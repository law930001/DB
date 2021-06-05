cd ../
CUDA_VISIBLE_DEVICES=2 \
python train.py experiments/seg_detector/synthtext_hrnet48_deform_thre.yaml \
--num_gpus 1 \
--num_workers 10 \
--batch_size 6