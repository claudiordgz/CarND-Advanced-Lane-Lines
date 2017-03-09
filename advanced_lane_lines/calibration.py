import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import pickle

def run_calibration(pickle_file, images_expression, force=False, debug=False):
    if not os.path.isfile(pickle_file) or force:
        chessboards = glob.glob(images_expression)
        x, y = 9, 6
        objp = np.zeros((y*x,3), np.float32)
        objp[:,:2] = np.mgrid[0:x, 0:y].T.reshape(-1,2)
        objpoints, imgpoints = [], [] 
        n = 20
        if debug:
            rows = 4
            cols = n // rows
            fig, axes = plt.subplots(rows, cols)
        img_size = None
        for i, filename in enumerate(chessboards):
            chess_img = cv2.imread(filename)
            if not img_size:
                img_size = (chess_img.shape[1], chess_img.shape[0])
            gray = cv2.cvtColor(chess_img, cv2.COLOR_BGR2GRAY) 
            ret, corners = cv2.findChessboardCorners(gray, (x, y), None)
            if debug:
                ax = axes[i // cols, i % cols]
            if ret == True:
                cv2.drawChessboardCorners(chess_img, (x,y), corners, ret)
                objpoints.append(objp)
                imgpoints.append(corners)
                if debug:
                    ax.imshow(chess_img)
                    ax.set_title('FOUND')
                    ax.axis('off')
            else:
                if debug:
                    ax.imshow(chess_img)
                    ax.set_title('FAIL')
                    ax.axis('off')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)    
        if debug:    
            plt.subplots_adjust(wspace=0.0, hspace=0.5, top=0.95, bottom=0.0, left=0.0, right=1.0)
            plt.show()
        pickle.dump((mtx, dist), open(pickle_file, 'wb'))
    else:
        mtx, dist = pickle.load(open(pickle_file, 'rb'))
    return mtx, dist

def calibrate(force=False, DEBUG=False):
    """ We only need to do this once, it saves the calibration to a pickle file
        force: resave everything
        DEBUG: will output a grid of images of the chessboard
        returns: a function to undistort images
    """
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    images_exp = os.path.join(location, 'camera_cal') + '/calibration*.jpg'
    pickle_f = os.path.join(location, 'dist_pickle.p')
    mtx, dist = run_calibration(pickle_f, images_exp, force, DEBUG)
    
    def undistort_image(image):
        """ Undistort image after calibration
            image: a cv2.imread image
            returns: undistorted image
        """
        return cv2.undistort(image, mtx, dist, None, mtx)    
    return undistort_image

def test_undistort_base(undistort, force=False):
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    distorted_image = os.path.join(location, 'camera_cal') + '/calibration2.jpg'
    dir_to_output_images = os.path.relpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../output_images/'))
    filename = '{name}.png'.format(name='calibration_test_chessboard')
    output_image_filename = os.path.join(dir_to_output_images,filename)
    if not os.path.isfile(output_image_filename) or force:
        img = cv2.imread(distorted_image)
        dst = undistort(img)
        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax1.axis('off')
        ax2.imshow(dst)
        ax2.set_title('Undistorted Image')
        ax2.axis('off')
        plt.subplots_adjust(wspace=0.0, hspace=0.5, top=0.95, bottom=0.0, left=0.0, right=1.0)
        plt.savefig(output_image_filename)
        plt.clf()


def test_undistort_test_images(undistort, force=False):
    location = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
    dir_to_images = os.path.join(location, 'test_images')
    dir_to_output_images = os.path.relpath(os.path.join(os.getcwd(), os.path.dirname(__file__), '../output_images/calibration/'))
    filenames = os.listdir(dir_to_images)
    pickle_file = os.path.join(location, 'undistorted_images_pickle.p')
    if not os.path.isfile(pickle_file) or force:
        test_images, undistorted_test_images = [], []
        for image_filepath in filenames:
            path_image = os.path.join(dir_to_images, image_filepath)
            filename = '{name}.png'.format(name=os.path.splitext(image_filepath)[0])
            output_image_filename = os.path.join(dir_to_output_images,filename)
            img = cv2.cvtColor(cv2.imread(path_image), cv2.COLOR_RGB2BGR)
            dst = undistort(img)
            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(img)
            ax1.set_title('Original')
            ax1.axis('off')
            ax2.imshow(dst)
            ax2.set_title('Undistorted')
            ax2.axis('off')
            plt.subplots_adjust(wspace=0.01, hspace=0.5, top=0.95, bottom=0.0, left=0.0, right=1.0)
            plt.savefig(output_image_filename)
            plt.clf()
            test_images.append(img)
            undistorted_test_images.append(dst)
        pickle.dump((test_images, undistorted_test_images), open(pickle_file, 'wb'))
    else:
        test_images, undistorted_test_images = pickle.load(open(pickle_file, 'rb'))
    return test_images, undistorted_test_images


def transform_to_birds_eye(img):
    img_size = (img.shape[1], img.shape[0])
    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M


def test_calibration():
    undistort = calibrate()
    test_undistort_base(undistort)
    test_images, undistorted_imgs = test_undistort_test_images(undistort)
    for src, im in zip(test_images, undistorted_imgs):
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 6))
        undistorted = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        warped, M = transform_to_birds_eye(undistorted)
        warped = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
        ax1.imshow(src)
        ax1.set_title('Sauce')
        ax2.imshow(im)
        ax2.set_title('Undistorted')
        ax3.imshow(warped)
        ax3.set_title('Undistorted/Warped')
        plt.subplots_adjust(wspace=0.02, hspace=0.5, top=0.95, bottom=0.55, left=0.0, right=1.0)
        plt.show()


if __name__ == '__main__':
    test_calibration()