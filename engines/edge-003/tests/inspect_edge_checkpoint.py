import os, json
from safetensors import safe_open
from glob import glob
import torch
p=os.environ['EVO_MODEL']
print('DEVICE',torch.cuda.get_device_name(), 'MODEL',p,flush=True)
c=json.load(open(p+'/config.json'))
print('CONFIG',json.dumps(c),flush=True)
for f in sorted(glob(p+'/*.safetensors')):
 with safe_open(f,framework='pt',device='cpu') as s:
  keys=[k for k in s.keys() if 'experts' in k]
  if keys:
   for k in keys[:3]: print(k,s.get_slice(k).get_shape(),s.get_slice(k).get_dtype(),flush=True)
   for k in keys[-3:]: print(k,s.get_slice(k).get_shape(),s.get_slice(k).get_dtype(),flush=True)
