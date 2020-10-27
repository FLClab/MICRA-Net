
import numpy
import json
import os, glob

from skimage import io, measure
from matplotlib import pyplot

def get_total_perimeter():

    path = "/home-local/Multilabel-Proteins-Actin/Segmentation/precise"
    image_names = glob.glob(os.path.join(path, "*.tif"))

    perimeters = []
    for image_name in image_names:

        image = io.imread(image_name)
        image = numpy.transpose(image, axes=(2, 0, 1))[-1]

        labeled = measure.label(image > 0)
        regionprops = measure.regionprops(labeled)

        perimeter = 0
        for rprop in regionprops:
            perimeter += rprop.perimeter
        perimeters.append(perimeter)
    return numpy.mean(perimeters)

if __name__ == "__main__":

    output = {
        "total_perimeter" : get_total_perimeter()
    }
    json.dump(output, open("stats.json", "w"), indent=4, sort_keys=True)
