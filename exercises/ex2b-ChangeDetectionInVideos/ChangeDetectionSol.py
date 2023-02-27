
import time
import cv2
import numpy as np
from skimage import color, io, measure, img_as_ubyte
from skimage.measure import profile_line
from skimage.transform import rescale, resize
import skimage as sk
import matplotlib.pyplot as plt
import numpy as np
import pydicom as dicom


def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)


def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # Get first image
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()

    # Transform image to gray scale and then to float, so we can do some processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # To keep track of frames per second
    start_time = time.time()
    n_frames = 0
    stop = False
    T = 50
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Transform image to gray scale and then to float, so we can do some processing
        new_frame_gray = cv2.cvtColor(
            new_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Compute difference image
        dif_img = np.abs(new_frame_gray - frame_gray)

        # Threshold the difference in images.
        # mask = sum(dif_img[0])/len(dif_img[0]) > 5000
        mask = dif_img > 50
        threshold_img = dif_img.copy()
        # dif_img[mask] = 255
        threshold_img[threshold_img > T] = 255
        threshold_img[threshold_img < T] = 0
        if cv2.waitKey(1) == ord('e'):
            print("difference")

        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Put the FPS on the new_frame
        str_out = f"fps: {fps}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # Calculate percentage of thresholded pixels
        F = ((sum(sum(threshold_img))/255)/(480*640))
        if F > 0.05:
            str_Warning = "Warning!"
            cv2.putText(new_frame, str_Warning, (200, 200), font, 1, 255, 1)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 1200, 10)
        show_in_moved_window(
            'threshold', threshold_img.astype(np.uint8), 0, 10)
        show_in_moved_window('Difference image',
                             dif_img.astype(np.uint8), 600, 10)

        # Old frame is updated
        alpha = 0.95

        frame_gray = alpha*new_frame_gray+(1-alpha)*threshold_img
        # frame_gray = new_frame_gray

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()
