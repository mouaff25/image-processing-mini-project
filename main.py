import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import scipy
import os

INPUT_IMAGE = "./images/kitchen_objects.jpg"
OUTPUT_FOLDER = "./output/"

image = cv2.imread(INPUT_IMAGE)
height, width = image.shape[:2]
n_pixels = height * width

print(f"Hauteur: {height}")
print(f"Largeur: {width}")
print(f"Number de pixels: {n_pixels}")


print("Création des histogrammes")
chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("Histogramme de Couleur")
plt.xlabel("Bacs")
plt.ylabel("Nombre de pixels")
for chan, color in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.savefig(os.path.join(OUTPUT_FOLDER, "histogramme_couleur.png"))


print("Creation des images de chaque canal")
def save_image_channels(image):
    blue_image = np.zeros(image.shape, dtype="uint8")
    blue_image[:, :, 2] = image[:, :, 0]
    green_image = np.zeros(image.shape, dtype="uint8")
    green_image[:, :, 1] = image[:, :, 1]
    red_image = np.zeros(image.shape, dtype="uint8")
    red_image[:, :, 0] = image[:, :, 2]

    _, ax = plt.subplots(1, 3)
    ax[0].imshow(blue_image)
    ax[0].set_title("Canal bleu")
    ax[1].imshow(green_image)
    ax[1].set_title("Canal vert")
    ax[2].imshow(red_image)
    ax[2].set_title("Canal rouge")
    plt.savefig(os.path.join(OUTPUT_FOLDER, "image_canal.png"))

save_image_channels(image)


print("Création d'image en niveaux de gris par moyenne pondérée")
coeff_r = 0.299
coeff_g = 0.587
coeff_b = 0.114
grayscale_image = (
    coeff_r * image[..., 0] + coeff_g * image[..., 1] + coeff_b * image[..., 2] # type: ignore
)
grayscale_image = grayscale_image.astype(np.uint8)

print("Création d'image en niveaux de gris par l'algorithme PCA")
pca = PCA(n_components=1)
pca.fit(image.reshape(n_pixels, 3))
pca_image = pca.transform(image.reshape(n_pixels, 3))
pca_image = pca_image.reshape(height, width)
pca_image = (
    (pca_image - pca_image.min()) / (pca_image.max() - pca_image.min()) * 255
).astype(np.uint8)
pca_image = 255 - pca_image

print("Création de comparaison entre les deux images en niveaux de gris")
fig, axes = plt.subplots(1, 2)
axes[0].imshow(grayscale_image, cmap="gray")
axes[0].set_title("Moyenne Pondérée")
axes[1].imshow(pca_image, cmap="gray")
axes[1].set_title("PCA")
plt.savefig(os.path.join(OUTPUT_FOLDER, "image_grayscale.png"))

print("Création d'histogramme de l'image en niveaux de gris")
colors = ("y", "m")

plt.figure()
plt.title("Histogramme de Couleur")
plt.xlabel("Bacs")
plt.ylabel("Nombre de pixels")
for i, g_image in enumerate((grayscale_image, pca_image)):
    hist = cv2.calcHist([g_image], [0], None, [256], [0, 256])
    plt.plot(hist, color=colors[i])
    plt.xlim([0, 256])
plt.legend(["Moyenne Pondérée", "PCA"])
plt.savefig(os.path.join(OUTPUT_FOLDER, "histogramme_grayscale.png"))


print("Binarisation de l'image en niveau de gris")
gaussian_image = cv2.GaussianBlur(pca_image, (11, 11), 0)
_, binary_image = cv2.threshold(gaussian_image, 100, 255, cv2.THRESH_BINARY_INV)
plt.figure()
plt.imshow(binary_image, cmap="gray")
plt.title("Image Binaire")
plt.savefig(os.path.join(OUTPUT_FOLDER, "image_binaire.png"))

print("Binarisation adaptative de l'image en niveau de gris")
gaussian_image = cv2.GaussianBlur(pca_image, (35, 35), 0)
binary_image = cv2.adaptiveThreshold(
    gaussian_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 1001, 4
)
plt.figure()
plt.imshow(binary_image, cmap="gray")
plt.title("Image Binaire")
plt.savefig(os.path.join(OUTPUT_FOLDER, "image_binaire_adaptative.png"))


print("Élimination des Bruits avec une Bordure")
border = 100
black_border_image = binary_image.copy()
black_border_image[:, :border] = 0
black_border_image[:, -border:] = 0
black_border_image[:border, :] = 0
black_border_image[-border:, :] = 0
plt.figure()
plt.imshow(black_border_image, cmap="gray")
plt.title("Image Binaire avec Bordure")
plt.savefig(os.path.join(OUTPUT_FOLDER, "image_binaire_bordure.png"))

print("Élimination des Bruits avec une Ouverture")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
opened_image = cv2.morphologyEx(
    black_border_image, cv2.MORPH_OPEN, kernel, iterations=2
)
plt.figure()
plt.imshow(opened_image, cmap="gray")
plt.title("Image Ouverte")
plt.savefig(os.path.join(OUTPUT_FOLDER, "image_ouverte.png"))

print("Élimination des Bruits avec une Fermeture")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (501, 501))
closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel, iterations=1)
plt.figure()
plt.imshow(closed_image, cmap="gray")
plt.title("Image Fermée")
plt.savefig(os.path.join(OUTPUT_FOLDER, "image_fermee.png"))

print("Réinitialisation de la bordure")
reset_border_image = closed_image.copy()
border = 100
reset_border_image[:border, :] = 0
reset_border_image[-border:, :] = 0
reset_border_image[:, :border] = 0
reset_border_image[:, -border:] = 0
plt.figure()
plt.imshow(reset_border_image, cmap="gray")
plt.title("Image fermée avec bordure réinitialisée")
plt.savefig(os.path.join(OUTPUT_FOLDER, "image_fermee_bordure.png"))

print("Trouver les contours avec sobel")
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

sobel_x_image = scipy.signal.convolve2d(reset_border_image, sobel_x, mode="same")
sobel_y_image = scipy.signal.convolve2d(reset_border_image, sobel_y, mode="same")

sobel_image = np.sqrt(sobel_x_image**2 + sobel_y_image**2)
sobel_image = (sobel_image / sobel_image.max() * 255).astype(np.uint8)
plt.figure()
plt.imshow(sobel_image, cmap="gray")
plt.title("Image Sobel")
plt.savefig(os.path.join(OUTPUT_FOLDER, "image_sobel.png"))

print("Amélioration de l'image Sobel avec le gradient")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
gradient_image = cv2.morphologyEx(sobel_image, cv2.MORPH_GRADIENT, kernel, iterations=1)
plt.figure()
plt.imshow(gradient_image, cmap="gray")
plt.title("Image Gradient")
plt.savefig(os.path.join(OUTPUT_FOLDER, "image_gradient.png"))

print("Dessiner les contours")
contours, hierarchy = cv2.findContours(
    gradient_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 11)
rgb_image = cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(rgb_image)
plt.title("Image avec Contours")
plt.savefig(os.path.join(OUTPUT_FOLDER, "image_contours.png"))

print("Calcul du nombre de pixels par objet")
objects = ["fraidoux", "oranghe", "bouchon", "planche"]
for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    print(f"{objects[i]}: {area} pixels")

print("Dessin des boîtes englobantes")
image_with_boxes = image.copy()

objects = ["fraidoux", "orange", "bouchon", "planche"]

for index, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 11)
    cv2.putText(
        image_with_boxes,
        objects[index],
        (x, y + h + 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        7,
        (0, 255, 0),
        11,
    )

rgb_image = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(rgb_image)
plt.title("Image avec Bounding Boxes")
plt.savefig(os.path.join(OUTPUT_FOLDER, "image_boxes.png"))