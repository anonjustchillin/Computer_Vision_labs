import cv2
from lab3.img_processing import process_image

file_bing = './bing_img.png'
file_atlas = './atlas_img.png'


## Atlas image processing and building detection
img_atlas = cv2.imread(file_atlas)
process_image(img_atlas, "livingatlas2", 6, 260, 150, 200, 70, 0.8, 180, False)

## Bing image processing and building detection
img_bing = cv2.imread(file_bing)
process_image(img_bing, "Bing", 3, 300, 170, 300, 100, 1.5, 210)
