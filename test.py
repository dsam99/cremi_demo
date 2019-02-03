import h5py

# reading h5 file
f = h5py.File("volume.h5")

# converting file to list of lists of slices (list of integers)
dset = f['slices']
d = dset[::].tolist()
