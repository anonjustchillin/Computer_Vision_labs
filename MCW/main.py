from img_operations import *
from img_processing import *
from detection import shi_tomasi_compare

file1 = './images/1.png' # google maps audi front
file2 = './images/2.png' # google maps audi side
file3 = './images/3.jpeg' # regular img audi

img1 = cv2.imread(file1)
img2 = cv2.imread(file2)
img3 = cv2.imread(file3)

print("Shape 1: ", img1.shape)
print("Shape 2: ", img2.shape)
print("Shape 3: ", img3.shape)
print("-----------------------")

show_img(img1, "Фото 1")
show_img(img2, "Фото 2")
show_img(img3, "Фото 3")

# x, y, rect_w, rect_h
crop1 = img1[100:300, 100:550]
crop2 = resize_img(img2, crop1.shape[:2])
crop3 = resize_img(img3, crop1.shape[:2])

show_img(crop1, "Фото 1")
show_img(crop2, "Фото 2")
show_img(crop3, "Фото 2")

proc_img1 = process_img(crop1)
proc_img2 = process_img(crop2)
proc_img3 = process_img(crop3)

show_img(proc_img1, "Фото 1 (після обробки)")
show_img(proc_img2, "Фото 2 (після обробки)")
show_img(proc_img3, "Фото 3 (після обробки)")

print()

# shi tomasi
params = [
        {'maxCorners': 1000, 'qualityLevel': 0.01, 'minDistance': 2},
        {'maxCorners': 2000, 'qualityLevel': 0.003, 'minDistance': 2},
        {'maxCorners': 2500, 'qualityLevel': 0.0025, 'minDistance': 1}
]

print("1 vs 2")
for i, param in enumerate(params):
    print(param)
    shi_tomasi_compare(proc_img1, proc_img2, **param)
    print("-----------------------")

print()

print("1 vs 3")
for i, param in enumerate(params):
    print(param)
    shi_tomasi_compare(proc_img1, proc_img3, **param)
    print("-----------------------")