
import os

src = os.path.join("/home-local", "MICRA-Net")
for folder in ["Actin", "CellTrackingChallenge", "EM", "MNIST", "PVivax"]:
    dst = os.path.join(".", folder, "MICRA-Net")
    try:
        os.symlink(src, dst, target_is_directory=True)
    except FileExistsError:
        os.remove(dst)
        os.symlink(src, dst, target_is_directory=True)
