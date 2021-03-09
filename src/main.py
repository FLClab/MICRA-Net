
import os

root = os.path.expanduser("~")

######################################################################
######################################################################
# Downloads the folder from the server
######################################################################
######################################################################

from torchvision.datasets.utils import download_and_extract_archive
download_and_extract_archive(
    "https://s3.valeria.science/flclab-micranet/MICRA-Net.tar.gz",
    os.path.join(root, "Downloads"),
    extract_root=os.path.join(root, "Downloads", "MICRA-Net")
)

######################################################################
######################################################################
# Creates symlinks to downloaded folder
######################################################################
######################################################################

# Sets path to the MICRA-Net folder which contains datasets and models
src = os.path.join(root, "Downloads", "MICRA-Net")
for folder in ["Actin", "CellTrackingChallenge", "EM", "MNIST", "PVivax"]:
    dst = os.path.join(".", folder, "MICRA-Net")
    try:
        os.symlink(src, dst, target_is_directory=True)
    except FileExistsError:
        os.remove(dst)
        os.symlink(src, dst, target_is_directory=True)
