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


def image_to_hog_features(input_image):
    gray_image = color.rgb2gray(input_image)
    # 87% for orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1)
    fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualise=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
    return hog_image_rescaled


def hog_gen(image, path=0):
    if(path != 0 and image == 0):
        image = io.imread(path)
    hog_image = image_to_hog_features(image)
    return hog_image


def generate_daisy_features(filename):
    image_arr = io.imread(filename, as_grey=True)
    return daisy(image_arr, step=8, radius=30, rings=3, histograms=4, orientations=4)


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


def percentage_overlap(lx1,ly1,rx1,ry1,lx2,ly2,rx2,ry2):
    x_overlap = max(0,min([rx1,rx2]) - max([lx1,lx2]))
    y_overlap = max(0,min([ry1,ry2]) - max([ly1,ly2]))
    overlap_area = x_overlap*y_overlap
    box_1_area = (rx1-lx1)*(ry1-ly1)
    box_2_area = (rx2-lx2)*(ry2-ly2)
    return overlap_area/float(box_1_area + box_2_area - overlap_area)
