from cluster_image import process_image
from img_operations import *

file_bing = 'bing_img.png'
file_atlas = 'atlas_img.png'


## Atlas image processing and building detection
img_atlas = cv2.imread(file_atlas)
process_image(img_atlas, "livingatlas2")

## Bing image processing and building detection
img_bing = cv2.imread(file_bing)
process_image(img_bing, "Bing")
