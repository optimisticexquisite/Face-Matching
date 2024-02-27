import cv2
import numpy as np

def adjust_brightness_contrast(img, clip_hist_percent=1):
    """
    Automatically adjust brightness and contrast of an image.
    clip_hist_percent (1 to 100): is the percent of the histogram to clip off the tails.
    The lower the value, the more aggressive the adjustment.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = [float(hist[0])]
    for i in range(1, hist_size):
        accumulator.append(accumulator[i -1] + float(hist[i]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    '''
    The function transform the source image using the following formula:
    result = saturate(src * alpha + beta)
    '''
    auto_result = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return auto_result

def simple_white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    img = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return img

def preprocess_image(img, brightness_contrast=True, white_balance=True):
    if brightness_contrast:
        img = adjust_brightness_contrast(img)
    if white_balance:
        img = simple_white_balance(img)
    return img