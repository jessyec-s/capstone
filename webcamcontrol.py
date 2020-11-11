import cv2

def identify_object(img):
    img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    mask1=cv2.inRange(img_hsv,(0,50,20),(5,255,255))
    mask2=cv2.inRange(img_hsv,(175,50,20),(180,255,255))

    mask=cv2.bitwise_or(mask1,mask2)
    cropped=cv2.bitwise_and(img,img,mask=mask)
    return mask,cropped

def display_feed():
    cv2.namedWindow("preview")
    vid_capture = cv2.VideoCapture(0)

    if vid_capture.isOpened():
        rval,frame = vid_capture.read()
    else:
        rval=false

    while rval:
        mask,cropped=identify_object(frame)
        cv2.imshow("preview",frame)
        cv2.imshow("cropped",cropped)
        rval,frame =vid_capture.read()
        key = cv2.waitKey(20)
        if key == 27: # exit on esc
            break
    cv2.destroyWindow("preview")
    cv2.destroyWindow("cropped")