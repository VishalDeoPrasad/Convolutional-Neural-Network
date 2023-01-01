import cv2

org_img = cv2.imread("opencv-logo.png")
#resize_img = cv2.resize(img, (0,0), fx=0.6, fy=0.6)

# downscale the image using new width and height
down_width = 300
down_height = 200
down_points = (down_width, down_height)
resized_down = cv2.resize(org_img, down_points, interpolation= cv2.INTER_LINEAR)
 
# upscale the image using new width and height
up_width = 600
up_height = 400
up_points = (up_width, up_height)
resized_up = cv2.resize(org_img, up_points, interpolation= cv2.INTER_LINEAR)

cv2.imshow("Original Image", org_img)
cv2.imshow("Downscale image", resized_down)
cv2.imshow("Upscale Image", resized_up)

cv2.waitKey(0)
cv2.destroyAllWindows()
