import imageio
from matplotlib import pyplot as plt
import sys
from PIL import Image
import numpy as np
from keras.models import load_model

arguments = len(sys.argv) - 1
if (arguments != 1):
    print (sys.argv[0] + ": Incorrect number of arguments")
    sys.exit()

path = sys.argv[1]
img = Image.open(path)
pix = np.array(img)
dim = pix.shape

rgbArray = []

R = pix[:,:,0]
G = pix[:,:,1]
B = pix[:,:,2]

raI = dim[0]//9 + 1
raJ = dim[1]//9 + 1

imi = 0
ima = 0
jmi = 0
jma = 0

for i in range(9):
    if (i == dim[0]%9):
        raI -= 1
    imi = ima
    ima += raI
    for j in range(9):
        if (j == dim[1] % 9):
            raJ -= 1
        jmi = jma
        jma += raJ
        rArray = np.zeros((raI, raJ), 'uint8')
        gArray = np.zeros((raI, raJ), 'uint8')
        bArray = np.zeros((raI, raJ), 'uint8')
        tripleArray = np.zeros((raI, raJ, 3), 'uint8')
        for ii in range(imi, ima):
            for jj in range(jmi, jma):
                rArray[ii - imi][jj - jmi] = R[ii][jj]
                gArray[ii - imi][jj - jmi] = G[ii][jj]
                bArray[ii - imi][jj - jmi] = B[ii][jj]
        tripleArray[..., 0] = rArray
        tripleArray[..., 1] = gArray
        tripleArray[..., 2] = bArray
        rgbArray.append(tripleArray)
    jmi = 0
    jma = 0
    raJ = dim[1] // 9 + 1

for i in range(81):
    newsize = (28, 28)
    img2 = Image.fromarray(rgbArray[i])
    img2 = img2.resize(newsize, Image.LANCZOS)
    img2.save("./Results/" + str(i+1)+".jpg")

gray = []

for i in range(81):
    im = imageio.imread("./Results/" + str(i+1)+".jpg")
    #im = imageio.imread("https://i.imgur.com/a3Rql9C.png")
    gray.append(np.dot(im[...,:3], [0.299, 0.587, 0.114]))
    """
    if i < 5:
        plt.imshow(gray[i], cmap=plt.get_cmap('gray'))
        plt.show()
     """

img_rows, img_cols = 28, 28
# reshape the image
for i in range(81):
    gray[i] = gray[i].reshape(1, img_rows, img_cols, 1)

# normalize image
for i in range(81):
    #gray[i] = gray[i].astype(np.float)
    gray[i] /= 255

# load the model
model = load_model("model.model")

# predict digit
solution = np.zeros((9, 9), 'uint8')
focus = 3
cut = 0.85
for i in range(9):
    for j in range(9):
        counter = 0
        for p in range(img_rows):
            for k in range(img_cols):
                if(p < focus) or (p >= img_rows - focus):
                    gray[9 * i + j][0][p][k][0] = 1.0
                elif (k < focus) or (k >= img_cols - focus):
                    gray[9 * i + j][0][p][k][0] = 1.0
                elif gray[9 * i + j][0][p][k][0] < cut:
                    gray[9 * i + j][0][p][k][0] = 0.0
                    counter += 1
                elif gray[9 * i + j][0][p][k][0] >= cut:
                    gray[9 * i + j][0][p][k][0] = 1.0
                gray[9 * i + j][0][p][k][0] = 1 - gray[9 * i + j][0][p][k][0]
        if counter >= 10:
            prediction = model.predict(gray[9 * i + j])
            solution[i][j] = prediction.argmax()
        else:
            solution[i][j] = 0
        """
        if 9*i+j < 5:
            plt.imshow(gray[9*i+j].reshape(img_rows, img_cols)*255, cmap=plt.get_cmap('gray'),  vmin=0, vmax=255)
            plt.show()
        """

print(solution)

while 1:
    pos = len(path)-1
    while path[pos] != '.':
        pos -= 1
    pos -= 1
    correct = 0
    j = 80
    for i in range(81):
        if path[pos] == str(solution[j//9][j % 9]):
            correct += 1
        j -= 1
        pos -= 1
    break
print("Accuracy: " + str(correct) + "/81 = " + str(float(correct)/81))

