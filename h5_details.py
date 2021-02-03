import os
os.chdir('/ubc/cs/research/shield/projects/aditya10/RefSeg/')
import sys
sys.path.append('/ubc/cs/research/shield/projects/aditya10/python-packages/')
import h5py
import numpy as np

dataset = 'unc'
type = 'train'

prefix = dataset+'_'+type
print("Reading hdf5 file...")
f = h5py.File('/ubc/cs/research/shield/datasets/refer/data/'+dataset+'/'+dataset+'_'+type+'.hdf5','r')
print(len(f.keys()))

# Load batch from file
batch_id = 1
batch = {
    "text_batch": np.array(f[prefix+'_'+str(batch_id)]["text_batch"]),
    "im_batch": np.array(f[prefix+'_'+str(batch_id)]["im_batch"]),
    "sent_batch": np.array(f[prefix+'_'+str(batch_id)]["sent_batch"]),
    "mask_batch": np.array(f[prefix+'_'+str(batch_id)]["mask_batch"]),
}

print(batch["text_batch"], batch["im_batch"], batch["sent_batch"][()], batch["mask_batch"])