import cv2
IMG_PATH = ".\Dataset\FallDown\JPEGImages"
ANT_PATH = ".\Dataset\FallDown\Annotations"
img = cv2.imread(f'{IMG_PATH}\people(1).jpg')

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#获取像素值
px = img[100, 100]
print(px)

#修改像素值
img[100, 100] = [255, 255, 255]
print(img[100, 100])

#获取图像属性
print(img.shape)
print(img.size)
print(img.dtype)

#拆分和合并图像通道
b, g, r = cv2.split(img)
img = cv2.merge((b,g,r))

