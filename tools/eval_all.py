import os
import subprocess
from natsort import natsorted

def main():

    model_root = '/root/Storage/DB_v100/workspace/SegDetectorModel-seg_detector/efficentnet_b7/L1BalanceCELoss/model_exp22/'

    os.chdir('/root/Storage/DB_v100')

    model_list = natsorted(os.listdir(model_root))
    if 'final' in model_list:
        model_list.remove('final')
        model_list.append('final')

    # print(model_list)

    try:
        for model in reversed(model_list):
            print(model)

            subprocess.call(['python', 'eval.py', 'experiments/seg_detector/tt_efficient_thre.yaml',
                '--resume', model_root + model,
                '--polygon', '--box_thresh', '0.6'])
    except:
        print('error')


if __name__ == '__main__':
    main()