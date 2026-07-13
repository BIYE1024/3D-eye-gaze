import pickle, sys
sys.path.insert(0, '/workspace')
import helperfunctions.CurriculumLib as CurLib

DS_sel = pickle.load(open('/workspace/cur_objs/dataset_selections.pkl', 'rb'))
AllDS = CurLib.readArchives('/workspace/Datasets/MasterKey')

datasets, subsets = CurLib.listDatasets(AllDS)
print(f'Datasets in MasterKey: {datasets}')
print(f'Subsets: {subsets}')
print(f'Total entries: {len(AllDS["pupil_loc"])}')

for split_name, split in [('train', DS_sel['train']), ('test', DS_sel['test'])]:
    if 'OVM6211' in split:
        cond = CurLib.selSubset(AllDS, split['OVM6211'])
        print(f'{split_name} OVM6211: {len(cond["pupil_loc"])} entries')

# Quick test: generate file list from OVM6211 train data
cond_train = CurLib.selSubset(AllDS, DS_sel['train']['OVM6211'])
dataDiv = CurLib.generate_fileList(cond_train, mode='vanilla', notest=False)
print(f'File list generated: {len(dataDiv.arch)} unique archives')

# Test DataLoader
args_dummy = {
    'net_ellseg_head': False, 'net_rend_head': True, 'net_simply_head': False,
    'loss_w_rend_pred_2_gt_edge': 0, 'loss_w_rend_gt_2_pred': 0,
    'loss_w_rend_pred_2_gt': 0, 'train_data_percentage': 1.0,
    'frames': 4, 'scale_bound_eye': 'version_0',
}
from helperfunctions.CurriculumLib import DataLoader_riteyes
loader = DataLoader_riteyes(dataDiv, '/workspace/Datasets/All', 'train',
                            True, (240, 320), scale=False, num_frames=4, args=args_dummy)
print(f'Train loader: {len(loader)} batches')
sample = loader[0]
print(f'Sample image shape: {sample["image"].shape}')
print(f'Sample gaze: {sample.get("gaze_vector", "N/A")}')
print('Data pipeline OK!')
