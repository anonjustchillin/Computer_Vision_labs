import cv2
import numpy as np
import os, glob, shutil
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def show_img(img, title):
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()


np.random.seed(0)

img_dir = "photos/"
cluster_dir = "clustered_photos/"

# список файлів
file_list = glob.glob(os.path.join(img_dir, '*.jpg'))
file_list.sort()

# список фото
img_list = [cv2.imread(file_list[i]) for i in range(len(file_list))]
N = len(file_list)
print(f"N = {N}")

# greyscale
bw_img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_list]
for i in range(0, N):
    img = img_list[i]
    show_img(img, f"{i}.jpg")

    grey = bw_img_list[i]
    show_img(grey, f"{i}.jpg (greyscale)")

sift = cv2.SIFT.create()
kps = []
des = []
for img in bw_img_list:
    kp, d = sift.detectAndCompute(img, None)
    kps.append(kp)
    des.append(d)

img_sift = []
for i in range(0, N):
    img_sift.append(cv2.drawKeypoints(bw_img_list[i], kps[i], 0, (255, 0, 0),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
    show_img(img_sift[i], f"{i} photo with SIFT keypoints")


# переводимо усі дескриптори в один масив
descriptors = [np.array(d) for d in des]
all_descriptors = []
for img_des in descriptors:
    for d in img_des:
        all_descriptors.append(d)
all_descriptors = np.stack(all_descriptors)
print(f"all_descriptors.shape: {all_descriptors.shape}")
print()

# kmeans
k=800
codebook, variance = kmeans(all_descriptors, k, 1)

# vector quantization
visual_words = []
for d in des:
    img_visual_words, dist = vq(d, codebook)
    visual_words.append(img_visual_words)
print(f"codebook.shape: {codebook.shape}")
print(f"visual words for img 0: {visual_words[0][:5]}, {len(visual_words[0])}")
print()

freq_vectors = []
for vw in visual_words:
    img_freq_vector = np.zeros(k)
    for w in vw:
        img_freq_vector[w] += 1
    freq_vectors.append(img_freq_vector)
freq_vectors = np.stack(freq_vectors)
print(f"freq_vectors.shape: {freq_vectors.shape}")
print(f"first 10 freq_vectors for img 0: {freq_vectors[0][:10]}")
print()
plt.bar(list(range(k)), freq_vectors[0])
plt.show()

df = np.sum(freq_vectors>0, axis=0)
print(f"df.shape and first 5 df: {df.shape}, {df[:10]}")
print()

idf = np.log(N/ df)
print(f"idf.shape and first 5 idf: {idf.shape}, {idf[:10]}")
print()

tfidf = freq_vectors * idf
print(f"tfidf.shape and first 5 tfidf: {tfidf.shape}, {tfidf[0][:10]}")
print()

plt.bar(list(range(k)), tfidf[0])
plt.show()

###################

max_clusters = 10
clusters = 3
optim_clusters = clusters
optim_silhouette = 0

while clusters <= max_clusters:
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(tfidf)
    labels = kmeans.labels_

    silhouette_avg = silhouette_score(tfidf, labels)
    print(f"Num of clusters = {clusters}\nAverage silhouette score = {silhouette_avg}")
    print("-----------------------")

    if optim_silhouette < silhouette_avg:
        optim_silhouette = silhouette_avg
        optim_clusters = clusters
    clusters += 1

print(f"Optimal num of clusters = {optim_clusters}\nAverage silhouette score = {optim_silhouette}")
kmeans = KMeans(n_clusters=optim_clusters, random_state=0).fit(tfidf)
labels = kmeans.labels_
print(labels)

for i, m in enumerate(labels):
    shutil.copy(file_list[i], cluster_dir + str(m) + "_" + str(i+1) + ".jpg")