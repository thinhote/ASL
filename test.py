from datasets.nslt_dataset import NSLT

root = {'word': '../../data/WLASL2000'}
split = 'train'
json_path = 'preprocess/nslt_2000.json'

ds = NSLT(json_path, split, root, 'rgb', None)
print("Số mẫu trong ds:", len(ds))