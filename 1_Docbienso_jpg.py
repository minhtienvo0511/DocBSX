# LOAD THU VIEN VA MODUL CAN THIET

from __future__ import print_function

import cv2
import pytesseract

import enum as np
#DOC HINH ANH - TACH HINH ANH NHAN DIEN
img = cv2.imread('5.jpg')
img_need_aligned = cv2.imread('38.jpg')
MAX_FEATURES = 500
GOOD_MATCH_PERCENT = 0.15


im1Gray = cv2.cvtColor(img_need_aligned, cv2.COLOR_BGR2GRAY)
im2Gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect ORB features and compute descriptors.
orb = cv2.ORB_create(MAX_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
matches = matcher.match(descriptors1, descriptors2, None)
# Sort matches by score
matches.sort(key=lambda x: x.distance, reverse=False)
# Remove not so good matches
numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
matches = matches[:numGoodMatches]
# Draw top matches
imMatches = cv2.drawMatches(img, keypoints1, img_need_aligned, keypoints2, matches, None)
cv2.imwrite("5.jpg", imMatches)
# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)
for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
# Find homography
h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
# Use homography
height, width, channels = img_need_aligned.shape
im1Reg = cv2.warpPerspective(img, h,(width, height))
if __name__ == '__main__':
# Read reference image
refFilename = "form.jpg"
print("Reading reference image : ", refFilename)
imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
# Read image to be aligned
imFilename = "scanned-form.jpg"
print("Reading image to align : ", imFilename);
im = cv2.imread(imFilename, cv2.IMREAD_COLOR)
print("Aligning images ...")
# Registered image will be resotred in imReg.
# The estimated homography will be stored in h.
imReg, h = alignImages(im, imReference)
# Write aligned image to disk.
outFilename = "38.jpg"
print("Saving aligned image : ", outFilename);
cv2.imwrite(outFilename, imReg)
# Print estimated homography
print("Estimated homography : \n", h)

cv2.imshow('Image',img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
contours,h = cv2.findContours(thresh,1,2)
largest_rectangle = [0,0]
for cnt in contours:
    lenght = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, lenght, True)
    if len(approx)==4:
        area = cv2.contourArea(cnt)
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
x,y,w,h = cv2.boundingRect(largest_rectangle[1])

#image=img[y:y+h+500, x:x+w+500]
#cv2.drawContours(img,[largest_rectangle[1]],0,(0,255,0),6)
image=img[40:600 ,40:800]
cropped = img[40:600 ,40:800]
cv2.imshow('Dinh vi bien so xe', img)

cv2.drawContours(img,[largest_rectangle[1]],0,(255.255,255),18)

#DOC HINH ANH CHUYEN THANH FILE TEXT
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imshow('Crop', thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 255 - opening
data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
print("Bien so xe la:")
print(data)
cv2.waitKey()
# LOAD THU VIEN VA MODUL CAN THIET
import cv2
import pytesseract
#DOC HINH ANH - TACH HINH ANH NHAN DIEN
img = cv2.imread('1.jpg')
cv2.startWindowThread()
cv2.namedWindow("Image")
cv2.imshow("Image", img)
#cv2.imshow('Image', img)
cv2.waitKey()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
contours,h = cv2.findContours(thresh,1,2)
largest_rectangle = [0,0]
for cnt in contours:
    lenght = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, lenght, True)
    if len(approx)==4:
        area = cv2.contourArea(cnt)
        if area > largest_rectangle[0]:
            largest_rectangle = [cv2.contourArea(cnt), cnt, approx]
x,y,w,h = cv2.boundingRect(largest_rectangle[1])

#image=img[y:y+h+h, x:x+w+w]
image=img[40:400 ,20:700]
cv2.drawContours(img,[largest_rectangle[1]],0,(0,255,0),8)

#cropped = img[y:y+h+h, x:x+w+w]
cropped = img[40:400, 20:700]
cv2.imshow('Dinh vi bien so xe', img)

cv2.drawContours(img,[largest_rectangle[0]],0,(255,255,255),18)

#DOC HINH ANH CHUYEN THANH FILE TEXT
pytesseract.pytesseract.tesseract_cmd = 'C:\Program Files\Tesseract-OCR\\tesseract.exe'
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (3,3), 0)
thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imshow('Crop', thresh)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
invert = 500- opening
data = pytesseract.image_to_string(invert, lang='eng', config='--psm 6')
print("Bien so xe la:")
print(data)
cv2.waitKey()
