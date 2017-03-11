import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def absolute_sobel_threshold(image, orient='x', thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    sobel_kernel=11
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    derivative = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel) if orient == 'x' else cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(derivative)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def magnitude_threshold(image, thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    sobel_kernel=11
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)    
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*gradmag/np.max(gradmag))
    # 5) Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return binary_output


def direction_threshold(image, thresh=(0.7, 1.3)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    sobel_kernel=7
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx, abs_sobely = np.absolute(sobelx), np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    sobel_atan = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    binary_output = np.zeros_like(sobel_atan)
    binary_output[(sobel_atan >= thresh[0]) & (sobel_atan <= thresh[1])] = 1
    return binary_output


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(170, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def r_select(img, thresh=(220, 255)):
    # 1) Convert to HLS color space
    # Assuming BGR because life is too short
    R = img[:,:,0]
    # 2) Apply a threshold to the R channel
    binary_output = np.zeros_like(R)
    binary_output[(R > thresh[0]) & (R <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output


def clamp_255(x):
    return x if x <= 255 else 255

def clamp_157(x):
    return x if x <= (np.pi/2 - 0.01)  else np.pi/2


def test_absolute_sobel(image, orientation, kernel, windows, window_size):
    return [absolute_sobel_threshold(image, orientation, kernel, (i, clamp_255(i+window_size))) for i in windows]


def test_magnitude(image, kernel, windows, window_size):
    return [magnitude_threshold(image, kernel, (i, clamp_255(i+window_size))) for i in windows]


def test_direction(image, kernel, windows, window_size, max_size):
    binaries = []
    for j, i in enumerate(windows):
        if j < max_size:
            binaries.append(direction_threshold(image, kernel, (i, clamp_255(i+window_size))))
    return binaries


def frange(x, y, jump):
  p = []
  while x <= y:
    p.append(x)
    x += jump
  return p


def search_common_values_brutally(force=False):
    """ This is an exhaustive search, it generates a ton of images
        The idea is to use these images to empirically find proper values to select
        thresholds and kernel sizes.
        force: rerun all them things
    """
    kernels = [7,9,11]
    windows = [80, 100, 120]
    radians = [0.3, 0.5, 0.7]
    offsets = [10, 20] # 0 offset is as good as useless
    offsets_radians = [np.pi/2 - 0.3, np.pi/2 - 0.1]
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    dir_to_images = os.path.join(location, 'test_images')
    dir_to_output_images = os.path.relpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../output_images/sobel/'))
    filenames = os.listdir(dir_to_images)
    print('Saving to folders:', filenames)
    for fname in filenames:
        output_image_folder = os.path.join(dir_to_output_images, os.path.splitext(fname)[0])
        if not os.path.isdir(output_image_folder):
            os.mkdir(output_image_folder)
    for fname in filenames:
        path_image = os.path.join(dir_to_images, fname)
        img = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_RGB2BGR)
        for kernel in kernels:
            for window_size, radians_size in zip(windows, radians):
                for offset, offset_radians in zip(offsets, offsets_radians):
                    filename = 'kernel_{kernel}_window_size{size}_o-{offset}.png'.format(kernel=kernel,
                                                                                     size=window_size,
                                                                                     offset=offset)
                    output_image_filename = os.path.join(dir_to_output_images, os.path.splitext(fname)[0],filename)
                    reprocess = not os.path.isfile(output_image_filename) or force
                    if reprocess:
                        windows_thresholds = range(0+offset, 255, window_size)
                        radians_thresholds = frange(-np.pi/2+offset_radians, np.pi/2, radians_size)
                        binaries_x_sobel = test_absolute_sobel(img, 'x', kernel, windows_thresholds, window_size)
                        binaries_y_sobel = test_absolute_sobel(img, 'y', kernel, windows_thresholds, window_size)
                        binaries_mag = test_magnitude(img, kernel, windows_thresholds, window_size)
                        binaries_radians = test_direction(img, kernel, radians_thresholds, radians_size, len(binaries_x_sobel))
                        combined_ = []
                        for ((gradx, grady),(mag_binary, dir_binary)) in zip(zip(binaries_x_sobel, binaries_y_sobel), zip(binaries_mag, binaries_radians)):
                            combined = np.zeros_like(dir_binary)
                            combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
                            combined_.append(combined)
                        cols = len(binaries_x_sobel) + 1
                        rows = 5
                        fig, axes = plt.subplots(rows, cols, figsize=(len(binaries_x_sobel)*4, rows*4))
                        for i in range(rows):
                            axes[i][0].imshow(img)
                            axes[i][0].set_title('ORIGINAL')
                            axes[i][0].axis('off')
                        images = binaries_x_sobel + binaries_y_sobel + binaries_mag + binaries_radians + combined_
                        j = 1
                        for i, im in enumerate(images):
                            if j == cols:
                                j = 1
                            r = i // (cols - 1)
                            ax = axes[r, j]
                            j += 1
                            if r == 3:
                                nx = round(radians_thresholds[i % len(radians_thresholds)], 2)
                                ny = round(clamp_157(nx+radians_size), 2)
                                nx = "{0:.2f}".format(nx)
                                ny = "{0:.2f}".format(ny)
                                label = 'kernel: {kernel}\nthreshold: ({x},{y})'.format(kernel = kernel,
                                                                                        x=nx,
                                                                                        y=ny)
                            elif r == 4:
                                label = 'combined'
                            else:
                                nx = windows_thresholds[i % len(windows_thresholds)]
                                ny = clamp_255(nx+window_size)
                                label = 'kernel: {kernel}\nthreshold: ({x},{y})'.format(kernel = kernel,
                                                                                        x=nx,
                                                                                        y=ny)
                            ax.imshow(im, cmap='gray')
                            ax.set_title(label)
                            ax.axis('off')
                        plt.subplots_adjust(wspace=0.01, hspace=0.5, top=0.95, bottom=0.0, left=0.0, right=1.0)
                        output_image_filename = os.path.join(dir_to_output_images, os.path.splitext(fname)[0],filename)
                        plt.savefig(output_image_filename)
                        plt.clf()
        cv2.destroyAllWindows()
                    

def search_common_values_less_brutally():
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    dir_to_images = os.path.join(location, 'test_images')
    dir_to_output_images = os.path.relpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../output_images/specific_sobel/'))
    filenames = os.listdir(dir_to_images)
    thresholds = [
        (30, 80),
        (30, 100),
        (30, 110),
        (30, 130),
        (20, 80),
        (20, 100),
        (20, 110),
        (20, 130)
    ]
    for fname in filenames:
        path_image = os.path.join(dir_to_images, fname)
        img = mpimg.imread(path_image) 
        dir_binary = direction_threshold(img)
        hls_binary = hls_select(img)
        r_binary = r_select(img)
        cols = 9
        rows = 7
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        for i in range(rows):
            axes[i][0].imshow(img)
            axes[i][0].set_title('ORIGINAL')
            axes[i][0].axis('off')
        def set_axis(row, col, img, title):
            axes[row][col].imshow(img, cmap='gray')
            axes[row][col].set_title(title)
            axes[row][col].axis('off')
        for c, thresh in enumerate(thresholds, 1):    
            x_binary = absolute_sobel_threshold(img, 'x', thresh=thresh)
            y_binary = absolute_sobel_threshold(img, 'y', thresh=thresh)
            mag_binary = magnitude_threshold(img, thresh=thresh)
            combined = np.zeros_like(dir_binary)
            combined[((x_binary == 1) & (y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (hls_binary == 1) | (r_binary == 1)] = 1
            set_axis(0,c, x_binary, 'ABS X - {thresh}'.format(thresh=thresh))
            set_axis(1,c, y_binary, 'ABS Y - {thresh}'.format(thresh=thresh))
            set_axis(2,c, mag_binary, 'MAGNITUDE - {thresh}'.format(thresh=thresh))
            set_axis(3,c, dir_binary, 'DIRECTION')
            set_axis(4,c, hls_binary, 'S CHANNEL')
            set_axis(5,c, r_binary, 'R CHANNEL')
            set_axis(6,c, combined, 'COMBINED')
        plt.subplots_adjust(wspace=0.01, hspace=0.5, top=0.95, bottom=0.0, left=0.0, right=1.0)
        output_image_filename = os.path.join(dir_to_output_images, os.path.splitext(fname)[0])
        plt.savefig(output_image_filename)
        plt.clf()


def birds_eye(image_in_rgb):
    img_size = (image_in_rgb.shape[1], image_in_rgb.shape[0])
    offset = 0
    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image_in_rgb, M, img_size)
    return warped, M


def thresholds(img, debug=False):
    height, width = img.shape[0], img.shape[1]
    area = width*height
    R_L = np.mean(img[:,:,0])
    G_L = np.mean(img[:,:,1])
    B_L = np.mean(img[:,:,2])
    perceived_brightness = 0.2126 * R_L + 0.7152 * G_L + 0.0722 * B_L
    
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB) 
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(13,13))
    
    luv_u, luv_v = luv[:,:,1], luv[:,:,2]
    yuv_u, yuv_v = yuv[:,:,1], yuv[:,:,2]
    lab_b = lab[:,:,2]
    hls_s = hls[:,:,2]
    rgb_r = img[:,:,0]
    discard_rgb, discard_xyz = False, False
    if perceived_brightness < 65:
        luv_u_binary = np.zeros_like(luv_u)
        luv_u_binary[(luv_u > np.max(luv_u) - 5)] = 1

        luv_v_binary = np.zeros_like(luv_v)
        luv_v_binary[(luv_v > 200)] = 1

        yuv_u_binary = np.zeros_like(yuv_u)

        yuv_v_binary = np.zeros_like(yuv_v)
        yuv_v_binary[(yuv_v >= 0) & (yuv_v <= 110)] = 1

        lab_b_binary = np.zeros_like(lab_b)
        lab_b_binary[(lab_b >= 160)] = 1

        hls_s_binary = np.zeros_like(hls_s)
        hls_s_binary[(hls_s > 15) & (hls_s <= 24)] = 1

        rgb_r_binary = np.zeros_like(rgb_r)
        rgb_r_binary[(rgb_r == 22) | (rgb_r >= 200)] = 1

        full_mask = np.zeros_like(rgb_r)
        full_mask[(luv_u_binary == 1) | 
                  (luv_v_binary == 1) |
                  (lab_b_binary == 1) |
                  (rgb_r_binary == 1)] = 1
        
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_GRADIENT, kernel)
        full_mask[(yuv_v_binary == 1) |
                  (hls_s_binary == 1)] = 1
    else:

        luv_u_binary = np.zeros_like(luv_u)
        luv_u_binary[(luv_u >= 113)] = 1

        luv_v_binary = np.zeros_like(luv_v)
        luv_v_binary[(luv_v >= 180)] = 1

        yuv_u_binary = np.zeros_like(yuv_u)
        yuv_u_binary[(yuv_u >= 148)] = 1

        yuv_v_binary = np.zeros_like(yuv_v)
        yuv_v_binary[(yuv_v <= 95)] = 1

        lab_b_avg = int(np.mean(lab_b))
        lab_boffset = (np.max(lab_b) - lab_b_avg) // 2
        lab_b_binary = np.zeros_like(lab_b)
        lab_b_binary[(lab_b >= (lab_b_avg + lab_boffset))] = 1

        hls_s_binary = np.zeros_like(hls_s)
        rgb_r_binary = np.zeros_like(rgb_r)

        full_mask = np.zeros_like(rgb_r)

        if perceived_brightness < 160:        
            
            hls_s_avg_offset = int(np.mean(hls_s)) * 4.5 
            hls_s_binary[(hls_s >= hls_s_avg_offset)] = 1
            yuv_v_sum = round((np.asarray(yuv_v_binary).sum()/area) * 1000, 3)
            hls_s_sum = round((np.asarray(hls_s_binary).sum()/area) * 1000, 3)
            luv_v_sum = round((np.asarray(luv_u_binary).sum()/area) * 1000, 3)
            discard_hls = yuv_v_sum > 1 and luv_v_sum > 0 and (hls_s_sum - int(yuv_v_sum) >= 17) and (hls_s_sum - int(luv_v_sum) >= 20)

            rgb_r_binary[(rgb_r >= 210)] = 1
            discard_rgb = np.asarray(rgb_r_binary).sum()/area > 0.1

            full_mask[(luv_u_binary == 1) | 
                    (luv_v_binary == 1) |
                    (yuv_u_binary == 1) |
                    (yuv_v_binary == 1) |
                    (lab_b_binary == 1)] = 1
            
            full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_GRADIENT, kernel)

            if not discard_hls:
                full_mask[(hls_s_binary == 1)] = 1
        else:
            hls_s_binary[(hls_s >= 200) & (hls_s <= 254)] = 1

            rgb_r_binary[(rgb_r >= 245) & (hls_s <= 250)] = 1
            discard_rgb = np.asarray(rgb_r_binary).sum()/area > 0.1
            
            full_mask[(luv_u_binary == 1) | 
                      (luv_v_binary == 1) |
                      (yuv_u_binary == 1) |
                      (yuv_v_binary == 1) |
                      (lab_b_binary == 1) |
                      (hls_s_binary == 1)] = 1
            
            full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_GRADIENT, kernel)
                 
        if not discard_rgb:
            full_mask[(rgb_r_binary == 1)] = 1  
    if debug:
        return full_mask, (luv_u_binary, luv_v_binary, yuv_u_binary, yuv_v_binary, lab_b_binary, hls_s_binary, rgb_r_binary)
    else:
        return full_mask

def pipeline(image_in_rgb, debug=False):
    """ Experiments
    """
    img, M = birds_eye(image_in_rgb)
    return thresholds(img, debug)


def pipeline_test():
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    dir_to_images = os.path.join(location, 'test_images')
    dir_to_output_images = os.path.relpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../output_images/third_pipeline_test/'))
    filenames = os.listdir(dir_to_images)
    def set_axis(row, col, img, title):
        axes[row][col].imshow(img, cmap='gray')
        axes[row][col].set_title(title)
        axes[row][col].axis('off')
    for j in range(0, len(filenames), 4):
        rows, cols = 4, 9
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        for i, fname in enumerate(filenames[j:j+4]):
            path_image = os.path.join(dir_to_images, fname)
            img = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_BGR2RGB)
            masked, components = pipeline(img, True)
            
            set_axis(i, 0, img, 'ORIGINAL')
            set_axis(i, 1, components[0], 'LUV U')
            set_axis(i, 2, components[1], 'LUV V')
            set_axis(i, 3, components[2], 'YUV U') 
            set_axis(i, 4, components[3], 'YUV V')
            set_axis(i, 5, components[4], 'LAB B')
            set_axis(i, 6, components[5], 'HLS S')
            set_axis(i, 7, components[6], 'RGB R')
            set_axis(i, 8, masked, 'FULL MASK')
        plt.subplots_adjust(wspace=0.01, hspace=0.5, top=0.95, bottom=0.01, left=0.0, right=1.0)
        filename = 'batch_image_{begin}-{end}.png'.format(begin=str(j+1), end=str(j+4))
        output_image_filename = os.path.join(dir_to_output_images, os.path.splitext(filename)[0])
        plt.savefig(output_image_filename)
        plt.clf()


def new_color_spaces_tests():
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    dir_to_images = os.path.join(location, 'test_images')
    dir_to_output_images = os.path.relpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../output_images/color_space_tracking/'))
    filenames = os.listdir(dir_to_images)
    def set_axis(row, col, img, title):
        axes[row][col].imshow(img, cmap='gray')
        axes[row][col].set_title(title)
        axes[row][col].axis('off')
    brightness = []
    images = []
    for fname in filenames:
        path_image = os.path.join(dir_to_images, fname)
        img = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_BGR2RGB)
        y, x = img.shape[0], img.shape[1]
        img = img[y//2+80:y, 0:x]
        R_L = np.mean(img[:,:,0])
        G_L = np.mean(img[:,:,1])
        B_L = np.mean(img[:,:,2])
        Y = 0.2126 * R_L + 0.7152 * G_L + 0.0722 * B_L
        images.append((img, Y))
    images = sorted(images, key=lambda x: x[1])
    k = 0
    for j in range(0, len(filenames), 4):
        rows, cols = 4, 8
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*1.5))
        for i, fname in enumerate(filenames[j:j+4]):
            img, Y = images[k]
            k += 1
            """ COLOR SPACES TO TEST 
                RGB2Luv, RGB2XYZ, RGB2YCrCb, RGB2LAB, RGB2HLS 
            """
            luv = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
            xyz = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
            yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            
            
            l1 = luv[:,:,1]
            l2 = luv[:,:,2]
            yz = xyz[:,:,1]
            yu = yuv[:,:,1]
            uv = yuv[:,:,2]
            ri = img[:,:,0]
            bl = lab[:,:,2]
            hs = hls[:,:,2]

            set_axis(i, 0, l1, 
                     'LUV U {Luminance} - {min}-{max} | {AVG}'.format(Luminance=round(Y,0), 
                                                                      AVG=round(np.mean(l1), 0), 
                                                                      min=np.min(l1),
                                                                      max=np.max(l1)))
            set_axis(i, 1, l2, 'LUV V - {min}-{max} | {AVG}'.format(AVG=round(np.mean(l2), 0), 
                                                                    min=np.min(l2),
                                                                    max=np.max(l2)))
            set_axis(i, 2, yz, 'XYZ Y- {min}-{max} | {AVG}'.format(AVG=round(np.mean(yz), 0), 
                                                                    min=np.min(yz),
                                                                    max=np.max(yz)))
            set_axis(i, 3, yu, 'YUV U- {min}-{max} | {AVG}'.format(AVG=round(np.mean(yu), 0), 
                                                                    min=np.min(yu),
                                                                    max=np.max(yu)))
            set_axis(i, 4, uv, 'YUV V -{min}-{max} | {AVG}'.format(AVG=round(np.mean(uv), 0), 
                                                                   min=np.min(uv),
                                                                   max=np.max(uv)))
            set_axis(i, 5, ri, 'R- {min}-{max} | {AVG}'.format(AVG=round(np.mean(ri), 0), 
                                                                    min=np.min(ri),
                                                                    max=np.max(ri)))
            set_axis(i, 6, bl, 'LAB B- {min}-{max} | {AVG}'.format(AVG=round(np.mean(bl), 0), 
                                                                    min=np.min(bl),
                                                                    max=np.max(bl)))
            set_axis(i, 7, hs, 'HSL S- {min}-{max} | {AVG}'.format(AVG=round(np.mean(hs), 0), 
                                                                    min=np.min(hs),
                                                                    max=np.max(hs)))

        plt.subplots_adjust(wspace=0.01, hspace=0.2, top=0.95, bottom=0.01, left=0.0, right=1.0)
        filename = 'batch_image_{begin}-{end}.png'.format(begin=str(j+1), end=str(j+4))
        output_image_filename = os.path.join(dir_to_output_images, os.path.splitext(filename)[0])
        plt.savefig(output_image_filename)
        plt.clf()
        


if __name__ == '__main__':
    #search_common_values_brutally()
    #search_common_values_less_brutally()
    #new_color_spaces_tests()
    pipeline_test()
