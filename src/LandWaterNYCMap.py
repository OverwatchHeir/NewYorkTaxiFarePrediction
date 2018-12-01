import cv2

## Read
img = cv2.imread("NYCMap.png")

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, (90,0,0),(110,255,255))


## save
cv2.imwrite("NYCMapMask.png", mask)