import cv2
import sys

cap = cv2.VideoCapture(1)
if cap.isOpened() == False:
    print("Problem with Opening Camera:")
    sys.exit()

# downscale the Video using new width and height
down_width = 300
down_height = 200
down_points = (down_width, down_height)
 
# upscale the Video using new width and height
up_width = 1200
up_height = 800
up_points = (up_width, up_height)
    
while(cap.isOpened()):
    ret, org_frame = cap.read()
    if ret == False:
        print("Problem with loading Frame:")
        sys.exit()
    else:
        #resize_frame = cv2.resize(frame, (0,0), fx=0.6, fy=0.6)
        print(org_frame.shape)
        resized_down = cv2.resize(org_frame, down_points, interpolation= cv2.INTER_LINEAR)
        resized_up = cv2.resize(org_frame, up_points, interpolation= cv2.INTER_LINEAR)
        cv2.imshow('frame', org_frame)
        cv2.imshow("Downscaled Video Frame", resized_down)
        cv2.imshow("Upscaled Video Frame", resized_up)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
