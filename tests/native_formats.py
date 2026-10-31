"""Actual public pickle layout and explicitly measured-velocity extension fixture."""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mdot_radar.convert import convert, point_returns
from mdot_radar.protocol import main as protocol

root=Path('runs/native-check');root.mkdir(parents=True,exist_ok=True)
frames=[np.array([[0.,1.+i*.01,0.,2.],[0.,2.+i*.01,0.,3.]]) for i in range(5)]
try:
    point_returns(frames,10)
except ValueError:
    pass
else:
    raise AssertionError('xyz+intensity must not be treated as Doppler')
points=point_returns(frames,10,proxy=True)
np.testing.assert_allclose(points[:,0],.1,atol=1e-6)
with (root/'trial.pkl').open('wb') as stream:
    pickle.dump(frames,stream)
fields=['file','id','subject','pair','angle','label','split']
with (root/'metadata.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for subject,split in enumerate(['train','val','test']):
        for angle in [-30,0,30]:
            writer.writerow(dict(file='trial.pkl',id=f'{subject}:{angle}',subject=subject,pair=subject,angle=angle,label=0,split=split))
convert(root/'metadata.csv',root/'converted',10,proxy=True)
sys.argv=['protocol','--input',str(root/'converted/manifest.json'),'--output',str(root/'one_view.json'),'--training-angles','0']
protocol()
rows=json.loads((root/'one_view.json').read_text())
assert len([r for r in rows if r['split']=='train'])==1
assert len([r for r in rows if r['split']=='test'])==3
assert all(r['velocity_source']=='range_rate_proxy' for r in rows)
measured=[np.column_stack([frame,np.full(len(frame),-.7)]) for frame in frames]
np.testing.assert_allclose(point_returns(measured,10,velocity_column=4)[:,0],-.7)
print('PASS: native pickle, measured Doppler/proxy distinction, raw conversion and frozen few-view split')
