
import os
import h5py

model_path = os.path.join("..", "..", "pretrained")

with h5py.File(os.path.join(model_path, "ActinModelZoo.hdf5"), "r") as file:
    group = file["Ilastik"]
    with h5py.File(os.path.join(model_path, "ACTIN.ilp"), "w") as outfile:
        for key, values in group.items():
            outfile.copy(values, outfile, key)

with h5py.File(os.path.join(model_path, "ACTIN.ilp"), "r") as file:
    print(file.keys())
