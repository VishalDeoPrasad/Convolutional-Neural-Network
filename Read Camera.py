import cv2
import sys

cap = cv2.VideoCapture(1)
if cap.isOpened() == False:
    print("Problem with Opening Camera:")
    sys.exit()
    

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        print("Problem with loading Frame:")
        sys.exit()
    else:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
