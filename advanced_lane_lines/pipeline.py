import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from calibration import calibrate


def birds_eye(image_in_rgb):
    img_size = (image_in_rgb.shape[1], image_in_rgb.shape[0])
    offset = 0
    src = np.float32 ([
            [220, 651],
            [350, 577],
            [828, 577],
            [921, 651]
        ])
    dst = np.float32 ([
            [220, 651],
            [220, 577],
            [921, 577],
            [921, 651]
        ])
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


def window_search(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    lanes_mask = np.copy (binary_warped)
    filtered_lanes_mask = np.zeros_like (lanes_mask)
    histogram = np.sum(lanes_mask[int(lanes_mask.shape[0]/4):,:], axis=0)
    plt.plot(histogram)

    plt.imshow(binary_warped, cmap='gray')
    plt.show()



def warp_test():
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    dir_to_images = os.path.join(location, 'test_images')
    dir_to_output_images = os.path.relpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../output_images/warp_test/'))
    filenames = os.listdir(dir_to_images)
    def set_axis(row, col, img, title):
        axes[row][col].imshow(img, cmap='gray')
        axes[row][col].set_title(title)
        axes[row][col].axis('off')
    for j in range(0, 8, 4):
        rows, cols = 4, 3
        #fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
        for i, fname in enumerate(filenames[j:j+4]):
            path_image = os.path.join(dir_to_images, fname)
            # Pipeline
            img = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_BGR2RGB)
            undistort = calibrate()
            undistorted_image = undistort(img)
            warped, M = birds_eye(img)
            masked = thresholds(warped, False)
            window_search(masked)
            
        #    set_axis(i, 0, img, 'ORIGINAL')
        #    set_axis(i, 1, undistorted_image, 'UNDISTORTED')
        #    set_axis(i, 2, masked, 'WARPED + THRESHOLD')
        #plt.subplots_adjust(wspace=0.01, hspace=0.5, top=0.95, bottom=0.01, left=0.0, right=1.0)
        #filename = 'batch_image_{begin}-{end}.png'.format(begin=str(j+1), end=str(j+4))
        #output_image_filename = os.path.join(dir_to_output_images, os.path.splitext(filename)[0])
        #plt.savefig(output_image_filename)
        #plt.clf()


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


if __name__ == '__main__':
    #pipeline_test()
    warp_test()