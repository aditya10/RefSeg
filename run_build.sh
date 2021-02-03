python build_batches.py -d unc -t train
python build_batches.py -d unc -t val
python build_batches.py -d unc -t testA
python build_batches.py -d unc -t testB
python build_batches.py -d unc+ -t train
python build_batches.py -d unc+ -t val
python build_batches.py -d unc+ -t testA
python build_batches.py -d unc+ -t testB

# Expected results:
# Saved file: unc_train.hdf5 with n_batch = 120623
# Saved file: unc_val.hdf5 with n_batch = 10833
# Saved file: unc_testA.hdf5 with n_batch = 5656
# Saved file: unc_testB.hdf5 with n_batch = 5094
# Saved file: unc+_train.hdf5 with n_batch = 120190
# Saved file: unc+_val.hdf5 with n_batch = 10757
# Saved file: unc+_testA.hdf5 with n_batch = 5725
# Saved file: unc+_testB.hdf5 with n_batch = 4888

