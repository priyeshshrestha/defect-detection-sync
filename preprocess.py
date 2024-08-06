import numpy as np
import cv2
import math
import scipy.signal
from util import crop_image
from pyaml_env import parse_config
config = parse_config('config/config.yaml')

CROP_SIZE = config["processor"]["crop_size"]
SMCU_WINDOW = config["processor"]["preprocess"]["smcu_window"] # Window size after orange extraction
PIXEL_THRESHOLD = config["processor"]["preprocess"]["smcu_pixel_threshold"] # Correlation similarity threshold
WHITE_THRESHOLD = int('300')
WHITE_SIZE_THRES = int('100')
BIG_SIZE_THRES = int('300')

crop_center = int(CROP_SIZE/2) 
lowerBound = (5, 20, 40)
upperBound = (42, 235,255)
defect_check_window_size = int(5)
defect_and_edge_overlap_window_size = int(15)


def crop_image(image, window):
    window1 = math.floor(window/2)
    window2 = math.ceil(window/2)
    w = image.shape[1]
    x1 = y1 = int((w/2) - window1)
    x2 = y2 = int((w/2) + window2)
    cropped_image = image[y1:y2, x1:x2]
    return cropped_image

def crop_to_smaller(image1, image2):
    height1, width1, _ = image1.shape
    height2, width2, _ = image2.shape

    if width1 * height1 < width2 * height2:
        # Crop image2 to match image1
        cropped_image = image2[0:height1, 0:width1]
        return image1, cropped_image
    else:
        # Crop image1 to match image2
        cropped_image = image1[0:height2, 0:width2]
        return cropped_image, image2


def check_pixel(image):
    rows, cols = image.shape[:2]
    pixels = []
    for i in range(rows):
        for j in range(cols):
            k = image[i, j]
            pixels.append(k)
    pixels = np.array(pixels)
    boole = np.isin(pixels, [0, 0, 0])
    return ([False, False, False] == boole).all(1).any()


def cross_image(im1, im2):
    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    im1_gray = np.sum(im1.astype('float'), axis=2)
    im2_gray = np.sum(im2.astype('float'), axis=2)
    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)
    # calculate the correlation image; note the flipping of onw of the images
    return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1, ::-1], mode='same')


def align_image(coord, img_mas, img_sus):
    crop_sus = img_sus[abs(coord[0]-crop_center):(coord[0] + crop_center),
                       abs(coord[1]-crop_center):(coord[1]+crop_center)]
    w, h, d = img_mas.shape
    crop_mas = img_mas[abs(int(w/2) - crop_center):(int(w/2) + crop_center),
                       abs(int(h/2) - crop_center):(int(h/2) + crop_center)]
    return crop_mas, crop_sus


def get_color(image, lowerBound, upperBound, window):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    mask = cv2.GaussianBlur(mask, (1, 1), 0)
    output = cv2.bitwise_and(image, image, mask=mask)
    window_img = crop_image(output, SMCU_WINDOW)
    pixel_average = np.average(window_img)
    return window_img, pixel_average, output


def get_orange(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerBound, upperBound)
    mask = cv2.GaussianBlur(mask, (1, 1), 0)
    output = cv2.bitwise_and(image, image, mask=mask)
    window_img = crop_image(output, SMCU_WINDOW)
    pixel_average = np.average(window_img)
    return window_img, pixel_average, output


def filter_exception(sus, mas):
    defect = False
    # white parts detection code
    lowerBound = (20, 0, 150)
    upperBound = (85, 255, 255)
    window_su, pixel_su, sus_color = get_color(
        sus, lowerBound, upperBound, window=30)
    if (check_pixel(window_su)) == True:
        # exception detection code
        lowerBound = (0, 0, 80)
        upperBound = (255, 255, 210)
        grayA = cv2.cvtColor(mas, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(sus, cv2.COLOR_BGR2GRAY)
        sub = cv2.subtract(grayB, grayA)
        sub = cv2.cvtColor(sub, cv2.COLOR_GRAY2RGB)
        window_sus, pixel_sus, sus_color = get_color(
            sub, lowerBound, upperBound, window=10)
        defect = True if (check_pixel(window_sus)) == True else False
    return defect

def center_image_with_padding(image):

    num_channels = image.shape[2] if len(image.shape) == 3 else 1
    
    if num_channels == 1:
        new_image = np.zeros((100, 100), dtype=np.uint8)
    else:
        new_image = np.zeros((100, 100, num_channels), dtype=np.uint8)
    
    x_offset = (100 - image.shape[1]) // 2
    y_offset = (100 - image.shape[0]) // 2
    
    if num_channels == 1:
        new_image[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image
    else:
        new_image[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1], :] = image
    
    return new_image
    


def image_subtraction(background_image, current_frame):

    background_gray = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    background_gray = cv2.GaussianBlur(background_gray, (5, 5), 0)
    current_frame_gray = cv2.GaussianBlur(current_frame_gray, (5, 5), 0)

    abs_diff = cv2.absdiff(background_gray, current_frame_gray)

    _, thresholded = cv2.threshold(abs_diff, 25, 255, cv2.THRESH_BINARY)

    return thresholded

def preprocess_orange(img_sus, img_mas):
    defect = False
    # img_mas = cv2.imread(img_mas)
    # img_sus = cv2.imread(os.path.join(image_path))
    # imagename = os.path.basename(image_path)
    # shift correction
    corr_img = cross_image(img_sus, img_mas)
    coord = np.unravel_index(np.argmax(corr_img), corr_img.shape)
    mas, sus = align_image(coord, img_mas, img_sus)
    main_img = np.zeros((im_width, im_width*2, 3), dtype=np.uint8)
    if sus is None or mas is None:
        defect = False
    else:
        if sus.size == 0 or sus.shape != (100, 100, 3) or mas.size == 0 or mas.shape != (100, 100, 3):
            defect = False
        else:
            # orange conversion and window crop
            window_sus, pixel_sus, sus_org = get_orange(sus)
            window_mas, pixel_mas, mas_org = get_orange(mas)
            # similarity of window check
            if (check_pixel(window_mas)) == False and (check_pixel(window_sus)) == False:
                defect = filter_exception(sus, mas)
            else:
                corr_img = cross_image(window_sus, window_mas)
                coord = np.unravel_index(np.argmax(corr_img), corr_img.shape)
                # filter defects
                if (abs((SMCU_WINDOW/2) - coord[0]) > PIXEL_THRESHOLD or abs((SMCU_WINDOW/2) - coord[1]) > PIXEL_THRESHOLD):
                    defect = True
                else:
                    value_diff = abs(pixel_sus - pixel_mas)
                    if value_diff > 60:
                        defect = True
                    else:
                        defect = False
            im_width, im_height,_= mas.shape
            defect_im = center_image_with_padding(window_sus)
            main_img[0:im_height, 0:im_width] = mas                  
            main_img[0:im_height, im_width:im_width*2] = defect_im

    return defect, main_img

def preprocess_big(sample_image, master_image):

    num_white_pixels = 0
    defect = False
    corr_img = cross_image(sample_image, master_image)
    coord = np.unravel_index(np.argmax(corr_img), corr_img.shape)
    master_image_, sample_image_ = align_image(coord, master_image, sample_image)
    main_img = np.zeros((im_width, im_width*2, 3), dtype=np.uint8)


    if sample_image_.size == 0 or master_image_.shape == 0:
            defect = False
    else:

        if master_image_.shape[:2] != sample_image_.shape[:2]:
            master_image_, sample_image_ = crop_to_smaller(master_image_, sample_image_)
        master_image_gray = cv2.cvtColor(master_image, cv2.COLOR_BGR2GRAY)
        master_edges = cv2.Canny(master_image_gray, 20, 200)

        dilation_kernel_1 = np.ones((5, 5), np.uint8)  # this one is perfect
        dilated_master_edges_1 = cv2.dilate(
            master_edges, dilation_kernel_1, iterations=1)  # perfect
        
        thresh = image_subtraction(
            master_image_, sample_image_)
        
        center_x = thresh.shape[1] // 2 - 5
        center_y = thresh.shape[0] // 2 - 5

        cropped_roi = thresh[center_y:center_y + 10, center_x:center_x+ 10]
        

        num_white_pixels = cv2.countNonZero(thresh)

        if num_white_pixels >= BIG_SIZE_THRES:
            defect = True
    
        im_width, im_height,_= master_image_.shape
        defect_im = center_image_with_padding(thresh)
        main_img[0:im_height, 0:im_width] = master_image_                  
        main_img[0:im_height, im_width:im_width*2] = defect_im
    
    return defect, main_img

def preprocess_white(sample_image, master_image):

    num_white_pixels = 0
    defect = False
    corr_img = cross_image(sample_image, master_image)
    coord = np.unravel_index(np.argmax(corr_img), corr_img.shape)
    master_image_, sample_image_ = align_image(coord, master_image, sample_image)
    main_img = np.zeros((im_width, im_width*2, 3), dtype=np.uint8)


    if sample_image_.size == 0 or master_image_.shape == 0:
            defect = False
    else:

        if master_image_.shape[:2] != sample_image_.shape[:2]:
            master_image_, sample_image_ = crop_to_smaller(master_image_, sample_image_)
        master_image_gray = cv2.cvtColor(master_image, cv2.COLOR_BGR2GRAY)
        master_edges = cv2.Canny(master_image_gray, 20, 200)

        dilation_kernel_1 = np.ones((5, 5), np.uint8)  # this one is perfect
        dilated_master_edges_1 = cv2.dilate(
            master_edges, dilation_kernel_1, iterations=1)  # perfect
        
        thresh = image_subtraction(
            master_image_, sample_image_)
        
        center_x = thresh.shape[1] // 2 - 5
        center_y = thresh.shape[0] // 2 - 5

        cropped_roi = thresh[center_y:center_y + 10, center_x:center_x+ 10]
        

        num_white_pixels = cv2.countNonZero(thresh)

        if num_white_pixels >= WHITE_SIZE_THRES:
            defect = True
    
        im_width, im_height,_= master_image_.shape
        defect_im = center_image_with_padding(thresh)
        main_img[0:im_height, 0:im_width] = master_image_                  
        main_img[0:im_height, im_width:im_width*2] = defect_im

    return defect, main_img
