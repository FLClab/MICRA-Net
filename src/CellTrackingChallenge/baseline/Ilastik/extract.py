
import os
import h5py

model_path = os.path.join("..", "..", "MICRA-Net", "models")

with h5py.File(os.path.join(model_path, "CTCModelZoo.hdf5"), "r") as file:
    group = file["Ilastik-FS"]
    with h5py.File(os.path.join(model_path, "CTC.ilp"), "w") as outfile:
        for key, values in group.items():
            outfile.copy(values, outfile, key)

with h5py.File(os.path.join(model_path, "CTC.ilp"), "r") as file:
    print(file.keys())
