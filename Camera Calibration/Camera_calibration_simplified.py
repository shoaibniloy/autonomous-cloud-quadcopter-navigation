import numpy as np
import cv2 as cv
import glob
import os

def calibrate_camera(image_directory, chessboard_size=(8, 7), square_size=30):
    """
    Perform camera calibration using chessboard images and calculate distances to objects in 3D space.
    
    Parameters:
    - image_directory: Path to the folder containing the calibration images.
    - chessboard_size: Tuple (columns, rows) specifying the number of internal corners.
    - square_size: Size of each square in the chessboard in millimeters.
    """
    # termination criteria for cornerSubPix
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, based on chessboard size and square size
    objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)  # (rows * cols) points
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size  # Scale the points by the square size to get real-world measurements

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real-world space
    imgpoints = []  # 2D points in image plane

    # Get all the images in the folder (supporting multiple image formats)
    image_folder_path = image_directory
    images = glob.glob(os.path.join(image_folder_path, '*.[pj]*[ng]*'))  # Include .jpg, .jpeg, .png files

    if not images:
        print("No images found in the folder. Please check the path and ensure images are present.")
        return

    print(f"Images found: {images}")

    # Process each image
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv.findChessboardCorners(gray, (chessboard_size[0], chessboard_size[1]), None)

        if ret == True:
            objpoints.append(objp)  # Add object points (3D)
            
            # Refine corner positions for better accuracy
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)  # Add image points (2D)
            
            # Draw and display the corners
            cv.drawChessboardCorners(img, (chessboard_size[0], chessboard_size[1]), corners2, ret)
            cv.imshow('Chessboard corners', img)
            cv.waitKey(500)  # Wait for 500ms before moving to the next image

    cv.destroyAllWindows()

    # Check if object points and image points have been collected
    if len(objpoints) > 0 and len(imgpoints) > 0:
        # Perform camera calibration
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        # Print the calibration results
        print("Camera matrix:\n", mtx)
        print("Distortion coefficients:\n", dist)
        print("Rotation vectors (rvecs):\n", rvecs)  # Rotation vectors
        print("Translation vectors (tvecs):\n", tvecs)  # Translation vectors

        # Save the calibration parameters
        np.savez('camera_calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

        # Use one of the calibration images as a test image (e.g., use the first image from the list)
        test_img_path = images[0]  # Use the first image from the list for testing
        print(f"Testing with image: {test_img_path}")  # Print the image being used for testing

        test_img = cv.imread(test_img_path)
        
        # Ensure that the image is loaded properly
        if test_img is None:
            print(f"Error: Unable to read the image {test_img_path}")
            return
        
        h, w = test_img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Undistort the test image
        dst = cv.undistort(test_img, mtx, dist, None, newcameramtx)

        # Crop the image using the ROI
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        cv.imwrite('calibresult.png', dst)

        cv.imshow("Undistorted Image", dst)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Demonstrate real-world distance measurement
        measure_distance(mtx, dist, rvecs[0], tvecs[0], test_img)

    else:
        print("No valid calibration images with detected corners were found.")

def measure_distance(camera_matrix, distortion_coeffs, rvec, tvec, image):
    """
    Measure the distance to a point in the image.
    
    Parameters:
    - camera_matrix: Camera matrix obtained from calibration.
    - distortion_coeffs: Distortion coefficients from calibration.
    - rvec: Rotation vector obtained from calibration.
    - tvec: Translation vector obtained from calibration.
    - image: The image from which distance is measured.
    """
    # Assume you are measuring the distance to a point in the image (e.g., a chessboard corner)
    point = np.array([[100, 100]], dtype=np.float32)  # Example image point (this should be a detected point)

    # Use the camera matrix and the rotation and translation vectors to calculate the 3D position
    projected_points, _ = cv.projectPoints(point, rvec, tvec, camera_matrix, distortion_coeffs)

    # The distance to the object can be calculated as the Euclidean distance between the camera and the object in 3D space
    distance = np.linalg.norm(tvec)  # This gives the distance from the camera to the object along the Z-axis

    print(f"Distance to object: {distance:.2f} mm")

# Usage Example
if __name__ == "__main__":
    # Call the calibration function with the directory of images, chessboard size, and square size
    calibrate_camera('/home/shoaib/Drone/YOLO+LLM/cameracalibration', chessboard_size=(8, 7), square_size=30)
