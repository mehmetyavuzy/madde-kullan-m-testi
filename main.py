import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def image_similarity(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score = ssim(gray1, gray2)
    return score

width = 224
height = 224

def find_most_similar_image(target_image, image_list):
    target_image = cv2.resize(target_image, (width, height))  # Hedef görüntüyü hedef boyutlara yeniden boyutlandırın

    max_score = -1
    most_similar_image = None

    for image in image_list:
        image = cv2.resize(image, (width, height))  # Sistemdeki resmi hedef boyutlara yeniden boyutlandırın

        similarity_score = image_similarity(target_image, image)
        if similarity_score > max_score:
            max_score = similarity_score
            most_similar_image = image

    return most_similar_image
# Sistemdeki resimleri yükleyin
image1 = cv2.imread('resim1.jpg')
image2 = cv2.imread('resim2.jpg')
image3 = cv2.imread('resim3.jpg')

# Anlık göz görüntüsünü alın
video_capture = cv2.VideoCapture(0)
ret, target_image = video_capture.read()

# Sistemdeki resimlerle benzerlik hesapla
image_list = [image1, image2, image3]
most_similar_image = find_most_similar_image(target_image, image_list)

# Sonuçları görüntüle
cv2.imshow('Target Image', target_image)
cv2.imshow('Most Similar Image', most_similar_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
