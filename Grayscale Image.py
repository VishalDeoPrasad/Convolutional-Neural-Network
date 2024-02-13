import cv2

img = cv2.imread("opencv-logo.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("image", img)
cv2.imshow("Gray Image", gray_img)

# this is to hold the window
cv2.waitKey(0)
cv2.destroyAllWindows()
