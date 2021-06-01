from os import listdir
from os.path import isfile, join
import json

root = "/root/Storage/datasets/T-Brain/PublicTestDataset/"

def from_json():
    for json_file in listdir(root + 'json'):
        print(json_file)
        with open(root + 'json/' + json_file) as file:
            data = json.load(file)
            with open(root + 'train_gts/' + json_file.replace('.json', '.jpg.txt'), 'w') as gt_file:
                for shapes in data['shapes']:
                    data = ','.join(str(int(x)) + ',' + str(int(y)) for x, y in shapes['points']) + ',null\n'
                    gt_file.write(data)
                    print(data)

def make_list_txt():
    with open(root + 'test_list.txt', 'w') as file:
        for i in range(1,1001):
            file.write('img_{}.jpg\n'.format(i))

def to_one_list():
    with open('/root/Storage/DB_v100/submission.csv', 'w') as sub_file:
        for i in range(1, 1001):
            print(i)
            with open('/root/Storage/DB_v100/results/res_img_' + str(i) + '.txt', 'r') as res_file:
                for line in res_file.readlines():
                    sub_line = str(i) + ',' + line
                    sub_file.write(sub_line)


            # break

if __name__ == '__main__':
    to_one_list()
    print("finish")
