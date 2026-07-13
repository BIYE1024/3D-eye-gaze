import pickle, numpy as np, sys
sys.path.insert(0, '/workspace')
import helperfunctions.CurriculumLib as CurLib
from helperfunctions.helperfunctions import simple_string

DS_sel = pickle.load(open('/workspace/cur_objs/dataset_selections.pkl', 'rb'))
AllDS = CurLib.readArchives('/workspace/Datasets/MasterKey')

train_ss = simple_string(DS_sel['train']['TEyeD'])
test_ss  = simple_string(DS_sel['test']['TEyeD'])
print('train sample:', train_ss[:3])
print('test sample:', test_ss[:3])

subsets = list(CurLib.listDatasets(AllDS)[1])
for s in subsets:
    temp = 'Dikablis_SS' if 'Dikablis' in s else s
    ss = simple_string(temp)
    print(f'{s} -> temp={temp} -> ss={ss}')
    print(f'  in train: {ss in train_ss}')
    print(f'  in test:  {ss in test_ss}')

cond_train = CurLib.selSubset(AllDS, DS_sel['train']['TEyeD'])
print(f'\nAfter train selSubset: {len(cond_train["pupil_loc"])}')

cond_test = CurLib.selSubset(AllDS, DS_sel['test']['TEyeD'])
print(f'After test selSubset: {len(cond_test["pupil_loc"])}')
