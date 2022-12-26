import cv2

img = cv2.imread("opencv-logo.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
resize_img = cv2.resize(img, (0,0), fx=0.6, fy=0.6)

cv2.imshow("image", resize_img)
cv2.imshow("Gray Image", gray_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
