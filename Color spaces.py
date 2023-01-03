import cv2

img = cv2.imread("opencv.png")
if img is not None:
    b,g,r = cv2.split(img)
    cv2.imshow("image", img)
    cv2.imshow("B", b)
    cv2.imshow("G", g)
    cv2.imshow("R", r)
else:
    print("Image is not loading!")

cv2.waitKey(0)
cv2.destroyAllWindows()