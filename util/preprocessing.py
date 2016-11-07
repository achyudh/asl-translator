from skimage.feature import hog, daisy, local_binary_pattern
from skimage import color, io, exposure
import matplotlib.pyplot as plt


def generate_hog_features(filename):
    input_image = io.imread(filename)
    gray_image = color.rgb2gray(input_image)
    # 87% for orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1)
    fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualise=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    return hog_image_rescaled


def generate_daisy_features(filename):
    image_arr = io.imread(filename, as_grey=True)
    return daisy(image_arr, step=8, radius=30, rings=3, histograms=6, orientations=8)


def generate_lbp_features(filename):
    radius = 3
    n_points = 10 * radius
    image_arr = io.imread(filename, as_grey=True)
    return local_binary_pattern(image_arr, P=n_points, R=radius)


def save_hog_image_comparison(filename):
    input_image = io.imread(filename)
    gray_image = color.rgb2gray(input_image)
    out_filename = "hog/" + filename

    # 87% for orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1)
    fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualise=True)
    # io.imsave("hog/" + filename, hog_image)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(gray_image, cmap=plt.cm.gray)
    ax1.set_title('Input image')
    ax1.set_adjustable('box-forced')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    ax1.set_adjustable('box-forced')
    plt.savefig(out_filename)
    plt.close()

    return hog_image

