import tensorflow as tf
import cv2 as cv
import os
import uuid


ANC_PATH = '../data/anchor/'
POS_PATH = '../data/positive/'
NEG_PATH = '../data/negative/'
PATHS = {
ord('a') : ANC_PATH,
ord('p') : POS_PATH
}



def capture_anchor_and_positive(camera=1, filename=None, ext_key=None):
    cap = cv.VideoCapture(camera)
    while cap.isOpened():
        ret, frame = cap.read()
        (x0, y0, size) = (120, 200, 250)
        frame = frame[x0:x0+size, y0:y0+size,:]
        key = cv.waitKey(1) & 0xFF
        if key == ord('a') or key == ord('p') or key == ext_key:
            if filename == None:
                filename = uuid.uuid1()
            imgname = os.path.join(PATHS[key], '{}.jpg'.format(filename))
            cv.imwrite(imgname, frame)
        elif key == ord('q'):
            break          
        cv.imshow('Image Collection', frame) 
    cap.release()
    cv.destroyAllWindows()

def load_negative_images(ROOT):
    for dir in os.listdir(ROOT):
        for file in os.listdir(os.path.join(ROOT, dir)):
            EX_PATH = os.path.join(ROOT, dir, file)
            NEW_PATH = os.path.join(NEG_PATH, file)
            os.replace(EX_PATH, NEW_PATH)




if __name__ == '__main__':
    capture_anchor_and_positive()
