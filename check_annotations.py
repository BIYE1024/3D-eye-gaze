import json, os
path = r'C:\Users\BY\Downloads\Model-aware_3D_Eye_Gaze-main\Model-aware_3D_Eye_Gaze-main\images\0\annotations.json'
with open(path) as f:
    data = json.load(f)

good = sum(1 for v in data.values() if v.get('pupil_ellipse') and v.get('iris_ellipse'))
only_p = sum(1 for v in data.values() if v.get('pupil_ellipse') and not v.get('iris_ellipse'))
only_i = sum(1 for v in data.values() if not v.get('pupil_ellipse') and v.get('iris_ellipse'))
none   = sum(1 for v in data.values() if not v.get('pupil_ellipse') and not v.get('iris_ellipse'))

print(f'Total: {len(data)} annotated')
print(f'Complete (pupil+iris): {good}')
print(f'Pupil only: {only_p}')
print(f'Iris only: {only_i}')
print(f'Skipped: {none}')
print()

for name, ann in list(data.items())[:8]:
    p = ann.get('pupil_ellipse')
    i = ann.get('iris_ellipse')
    ps = [f'{x:.1f}' for x in p] if p else 'MISSING'
    ii = [f'{x:.1f}' for x in i] if i else 'MISSING'
    print(f'{name}')
    print(f'  pupil: {ps}')
    print(f'  iris:  {ii}')
