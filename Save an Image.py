import cv2
import os

# Image path
image_path = r'C:\Users\visha\OneDrive\Scaler Academy\Computer-Vision\opencv.png'

# Image directory
target_path = r'C:\Users\visha\OneDrive\Scaler Academy\Computer-Vision'

img = cv2.imread(image_path)

# Change the current directory
# to specified directory
os.chdir(target_path)

print("Before saving image:")
print(os.listdir(target_path))

# Saving the image
cv2.imwrite('newImg.png', img)

# List files and directories
print("After saving image:")
print(os.listdir(target_path))

