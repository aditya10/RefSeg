from __future__ import print_function

import numpy as np
import os
import threading
import Queue as queue
import h5py

def run_prefetch(prefetch_queue, folder_name, prefix, num_batch, shuffle):
    n_batch_prefetch = 0
    fetch_order = np.arange(num_batch)

    # Reading from a h5 file
    print("Reading hdf5 file...")
    f = h5py.File(folder_name+prefix+'.hdf5','r')

    while True:
        # Shuffle the batch order for every epoch
        if n_batch_prefetch == 0 and shuffle:
            fetch_order = np.random.permutation(num_batch)

        # Load batch from file
        batch_id = fetch_order[n_batch_prefetch]
        batch = {
            "text_batch": np.array(f[prefix+'_'+str(batch_id)]["text_batch"]),
            "im_batch": np.array(f[prefix+'_'+str(batch_id)]["im_batch"]),
            "sent_batch": np.array(f[prefix+'_'+str(batch_id)]["sent_batch"]),
            "mask_batch": np.array(f[prefix+'_'+str(batch_id)]["mask_batch"]),
        }

        # add loaded batch to fetchqing queue
        prefetch_queue.put(batch, block=True)

        # Move to next batch
        n_batch_prefetch = (n_batch_prefetch + 1) % num_batch

class DataReader:
    def __init__(self, folder_name, prefix, shuffle=True, prefetch_num=8):
        self.folder_name = folder_name
        self.prefix = prefix
        self.shuffle = shuffle
        self.prefetch_num = prefetch_num

        self.n_batch = 0
        self.n_epoch = 0

        # Read h5 file to get number of batches to process
        f = h5py.File(folder_name+prefix+'.hdf5','r')
        num_batch = len(f.keys())

        if num_batch > 0:
            print('found %d batches under %s with prefix "%s"' % (num_batch, folder_name, prefix))
        else:
            raise RuntimeError('no batches under %s with prefix "%s"' % (folder_name, prefix))
        self.num_batch = num_batch

        # Start prefetching thread
        self.prefetch_queue = queue.Queue(maxsize=prefetch_num)
        self.prefetch_thread = threading.Thread(target=run_prefetch,
            args=(self.prefetch_queue, self.folder_name, self.prefix,
                  self.num_batch, self.shuffle))
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()

    def read_batch(self, is_log = True):
        if is_log:
            print('data reader: epoch = %d, batch = %d / %d' % (self.n_epoch, self.n_batch, self.num_batch))

        # Get a batch from the prefetching queue
        if self.prefetch_queue.empty():
            print('data reader: waiting for file input (IO is slow)...')
        batch = self.prefetch_queue.get(block=True)
        self.n_batch = (self.n_batch + 1) % self.num_batch
        self.n_epoch += (self.n_batch == 0)
        return batch
