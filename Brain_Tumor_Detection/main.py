import numpy as np
from tkinter import *
from tqdm import tqdm
import cv2
import os
import shutil
import itertools
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
from plotly import tools

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16, preprocess_input
from keras import layers
from keras.models import Model, Sequential
#me
from PIL import Image, ImageEnhance, ImageTk
import tkinter.filedialog
window=Tk()
# Title of our window
window.title('Brain tumor detection')
# window size
window.geometry('1100x720')
#window color
window.configure(bg='#76f5a0')
# Here we fixed our window to be invariable
window.resizable(False, False)
# init_notebook_mode(connected=True)
RANDOM_SEED = 123
IMG_PATH = 'brain_tumor_dataset/'
# split the data by train/val/test
"""
for CLASS in os.listdir(IMG_PATH):
    if not CLASS.startswith('.'):
        IMG_NUM = len(os.listdir(IMG_PATH + CLASS))
        for (n, FILE_NAME) in enumerate(os.listdir(IMG_PATH + CLASS)):
            img = IMG_PATH + CLASS + '/' + FILE_NAME
            if n < 5:
                shutil.copy(img, 'TEST/' + CLASS.upper() + '/' + FILE_NAME)
            elif n < 0.8*IMG_NUM:
                shutil.copy(img, 'TRAIN/'+ CLASS.upper() + '/' + FILE_NAME)
            else:
                shutil.copy(img, 'VAL/'+ CLASS.upper() + '/' + FILE_NAME)
"""


# helper functions
def load_data(dir_path, img_size=(100, 100)):
    """
    Load resized images as np.arrays to workspace
    """
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            labels[i] = path
            for file in os.listdir(dir_path + path):
                if not file.startswith('.'):
                    img = cv2.imread(dir_path + path + '/' + file)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X, dtype=object)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    cm = np.round(cm, 2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


TRAIN_DIR = 'TRAIN/'
TEST_DIR = 'TEST/'
VAL_DIR = 'VAL/'
IMG_SIZE = (224, 224)

# use predefined function to load the image data into workspace
X_train, y_train, labels = load_data(TRAIN_DIR, IMG_SIZE)
X_test, y_test, _ = load_data(TEST_DIR, IMG_SIZE)
X_val, y_val, _ = load_data(VAL_DIR, IMG_SIZE)

#vesualisation de la distribution de donnees
y = dict()
y[0] = []
y[1] = []
for set_name in (y_train, y_val, y_test):
    y[0].append(np.sum(set_name == 0))
    y[1].append(np.sum(set_name == 1))

trace0 = go.Bar(
    x=['Train Set', 'Validation Set', 'Test Set'],
    y=y[0],
    name='No',
    marker=dict(color='#33cc33'),
    opacity=0.7
)
trace1 = go.Bar(
    x=['Train Set', 'Validation Set', 'Test Set'],
    y=y[1],
    name='Yes',
    marker=dict(color='#ff3300'),
    opacity=0.7
)
data = [trace0, trace1]
layout = go.Layout(
    title='Count of classes in each set',
    xaxis={'title': 'Set'},
    yaxis={'title': 'Count'}
)
# fig = go.Figure(data, layout)
# iplot(fig)
# plt.plot(data, layout)


def plot_samples(X, y, labels_dict, n=50):
    """
    Creates a gridplot for desired number of images (n) from the specified set
    """
    for index in range(len(labels_dict)):
        imgs = X[np.argwhere(y == index)][:n]
        j = 10
        i = int(n / j)


        # graph_sg = FigureCanvasTkAgg(plt.figure(figsize=(15, 6)), master=window)
        c = 1
        for img in imgs:
            plt.subplot(i, j, c)
            plt.imshow(img[0])

            plt.xticks([])
            plt.yticks([])
            c += 1
        plt.suptitle('Tumor: {}'.format(labels_dict[index]))
        plt.show()


# plot_samples(X_train, y_train, labels, 30)
#
# RATIO_LIST = []
# for set in (X_train, X_test, X_val):
#     for img in set:
#         RATIO_LIST.append(img.shape[1] / img.shape[0])
#
# plt.hist(RATIO_LIST,master=window)
# plt.title('Distribution of Image Ratios')
# plt.xlabel('Ratio Value')
# plt.ylabel('Count')
# plt.show()


def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS,
                  extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new,dtype=object)

#here an example of crop image
# img = cv2.imread('brain_tumor_dataset/yes/Y108.jpg')
# img = cv2.resize(
#     img,
#     dsize=IMG_SIZE,
#     interpolation=cv2.INTER_CUBIC
# )
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
#
# # threshold the image, then perform a series of erosions +
# # dilations to remove any small regions of noise
# thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
# thresh = cv2.erode(thresh, None, iterations=2)
# thresh = cv2.dilate(thresh, None, iterations=2)
#
# # find contours in thresholded image, then grab the largest one
# cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = imutils.grab_contours(cnts)
# c = max(cnts, key=cv2.contourArea)
#
# # find the extreme points
# extLeft = tuple(c[c[:, :, 0].argmin()][0])
# extRight = tuple(c[c[:, :, 0].argmax()][0])
# extTop = tuple(c[c[:, :, 1].argmin()][0])
# extBot = tuple(c[c[:, :, 1].argmax()][0])
# # add contour on the image
# img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)
#
# # add extreme points
# img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
# img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
# img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
# img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)
#
# # crop
# ADD_PIXELS = 0
# new_img = img[extTop[1] - ADD_PIXELS:extBot[1] + ADD_PIXELS, extLeft[0] - ADD_PIXELS:extRight[0] + ADD_PIXELS].copy()
#
# plt.figure(figsize=(15, 6))
# plt.subplot(141)
# plt.imshow(img)
# plt.xticks([])
# plt.yticks([])
# plt.title('Step 1. Get the original image')
# plt.subplot(142)
# plt.imshow(img_cnt)
# plt.xticks([])
# plt.yticks([])
# plt.title('Step 2. Find the biggest contour')
# plt.subplot(143)
# plt.imshow(img_pnt)
# plt.xticks([])
# plt.yticks([])
# plt.title('Step 3. Find the extreme points')
# plt.subplot(144)
# plt.imshow(new_img)
# plt.xticks([])
# plt.yticks([])
# plt.title('Step 4. Crop the image')
# plt.show()

# apply this for each set
X_train_crop = crop_imgs(set_name=X_train)
X_val_crop = crop_imgs(set_name=X_val)
X_test_crop = crop_imgs(set_name=X_test)


# plot_samples(X_train_crop, y_train, labels, 30)
def save_new_images(x_set, y_set, folder_name):
    i = 0
    for (img, imclass) in zip(x_set, y_set):
        if imclass == 0:
            cv2.imwrite(folder_name + 'NO/' + str(i) + '.jpg', img)
        else:
            cv2.imwrite(folder_name + 'YES/' + str(i) + '.jpg', img)
        i += 1

# saving new images to the folder
# !mkdir
# TRAIN_CROP
# TEST_CROP
# VAL_CROP
# TRAIN_CROP / YES
# TRAIN_CROP / NO
# TEST_CROP / YES
# TEST_CROP / NO
# VAL_CROP / YES
# VAL_CROP / NO

save_new_images(X_train_crop, y_train, folder_name='TRAIN_CROP/')
save_new_images(X_val_crop, y_val, folder_name='VAL_CROP/')
save_new_images(X_test_crop, y_test, folder_name='TEST_CROP/')


def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-16 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)


X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)
X_test_prep = preprocess_imgs(set_name=X_test_crop, img_size=IMG_SIZE)
X_val_prep = preprocess_imgs(set_name=X_val_crop, img_size=IMG_SIZE)

# plot_samples(X_train_prep, y_train, labels, 30)

# set the paramters we want to change randomly
demo_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    rescale=1. / 255,
    shear_range=0.05,
    brightness_range=[0.1, 1.5],
    horizontal_flip=True,
    vertical_flip=True
)
#########check this
if os.path.exists('preview'):
    shutil.rmtree('preview')
os.mkdir('preview')
x = X_train_crop[0]
x = x.reshape((1,) + x.shape)
#
i = 0
for batch in demo_datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix='aug_img', save_format='jpg'):
    i += 1
    if i > 20:
        break

plt.imshow(X_train_crop[0])
plt.xticks([])
plt.yticks([])
plt.title('Original Image')
plt.show()

plt.figure(figsize=(15, 6))
i = 1
for img in os.listdir('preview/'):
    img = cv2.cv2.imread('preview/' + img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.subplot(3, 7, i)
    # plt.imshow(img)
    # plt.xticks([])
    # plt.yticks([])
    i += 1
    if i > 3 * 7:
        break
# plt.suptitle('Augemented Images')
# plt.show()
#######################check it

TRAIN_DIR = 'TRAIN_CROP/'
VAL_DIR = 'VAL_CROP/'

train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    brightness_range=[0.5, 1.5],
    horizontal_flip=True,
    vertical_flip=True,
    preprocessing_function=preprocess_input
)

test_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='binary',
    seed=RANDOM_SEED
)

validation_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    color_mode='rgb',
    target_size=IMG_SIZE,
    batch_size=16,
    class_mode='binary',
    seed=RANDOM_SEED
)

# load base model
vgg16_weight_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model = VGG16(
    weights=vgg16_weight_path,
    include_top=False,
    input_shape=IMG_SIZE + (3,)
)

NUM_CLASSES = 1

model = Sequential()
model.add(base_model)
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))
model.layers[0].trainable = False

model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

model.summary()

# EPOCHS = 20
EPOCHS = 3

# history = model.fit_generator(
history = model.fit(
    train_generator,
    # steps_per_epoch=50,
    steps_per_epoch=len(X_train)//32,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=25
)

# plot model performance
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, EPOCHS + 1)


fig=plt.figure(figsize=(8.3, 3),dpi=100)

fig.add_subplot(121).plot(epochs_range, acc, label='Train Set')

fig.add_subplot(122).plot(epochs_range, loss, label='Train Set')

graph_sg = FigureCanvasTkAgg(fig,window)
graph_sg.get_tk_widget().place(x=235,y=295)
xlabel = Label(window, text="Epoches",bg='white',fg='black',font='Times')
xlabel.place(x=235,y=585, width=830, height=25, rely=0.01)
xlabel1 = Label(window, text="Model Accuracy                                                "
                             "Model loss"\
                ,bg='white',fg='black',font='Times')
xlabel1.place(x=235,y=285, width=830, height=25, rely=0.01)
graph_sg.draw()


# validate on val set
predictions = model.predict(X_val_prep)
predictions = [1 if x > 0.7 else 0 for x in predictions]
accuracy = accuracy_score(y_val, predictions)
vte = accuracy_score(y_val, predictions)
print('Test Accuracy = %.2f' % accuracy)

confusion_mtx = confusion_matrix(y_val, predictions)
cm = plot_confusion_matrix(confusion_mtx, classes=list(labels.items()), normalize=False)

# validate on test set
predictions = model.predict(X_test_prep)
predictions = [1 if x > 0.7 else 0 for x in predictions]
accuracy = accuracy_score(y_test, predictions)
vva = accuracy_score(y_test, predictions)
print('Validation Accuracy = %.2f' % accuracy)

confusion_mtx = confusion_matrix(y_test, predictions)
cm = plot_confusion_matrix(confusion_mtx, classes=list(labels.items()), normalize=False)

ind_list = np.argwhere((y_test == predictions) == False)[:, -1]
if ind_list.size == 0:
    print('There are no missclassified images.')
else:
    print('There are  missclassified images.')

############################Interface####################
titlebel = Label(window,text='Brain Tumor Detection Application',font=2,bg='#acf5a0',fg='black')
titlebel.place(y=30,width=1100,height=50,anchor='w')
def select_image():
    global myFiles, panel,res

    browseFile = tkinter.filedialog.askopenfilename()
    myFiles = browseFile
    res = cv2.imread(browseFile)
    if len(browseFile) > 0:
        img = Image.open(browseFile)
        img = img.resize((300, 200),Image.ANTIALIAS )
        img = ImageTk.PhotoImage(img)
        panel = Label(window, image=img,background='#76f5a0',bg='#76f5a0')
        inFileLbl = Label(window, text="Original image ")
        inFileLbl.place(x=230,y=50,width=300,height=30,rely=0.01)
        panel.image = img
        panel.place(x=228,y=65,rely=0.03,height=200)
#la bare de menu
helpLf2 = LabelFrame(window, text="   Menu  ", bg='#d0eaff',fg='#000000',border=10\
                     ,borderwidth=1,font=1)
helpLf2.place(x=20,y=57,width=200,height=400)
v=0.9
hel0 = Label(window,text='The accuracy of the\n model is : '+str(accuracy),font=2)
hel0.place(x=20,y=550,width=200,height=130,anchor='w')
####
btn1 = Button(helpLf2, text='Select an image', width=20, command=select_image,bg='#76f5a0',font='Times')
btn1.place(x=20,y=70,width=150,height=40,anchor='w')
btn2 = Button(helpLf2, text='Detect tumor', width=20, bg='#76f5a0',font='Times',command=lambda :detect_tomur(res))
btn2.place(x=20,y=140,width=150,height=40,anchor='w')
btn3 = Button(helpLf2, text='Clear', width=20, command=lambda :delet_image(0),bg='#76f5a0',font='Times')
btn3.place(x=20,y=210,width=150,height=40,anchor='w')
btn3 = Button(helpLf2, text='Quit', width=20, command=quit,bg='#76f5a0',font='Times')
btn3.place(x=20,y=280,width=150,height=40,anchor='w')
#interpretation
helpLf1 = LabelFrame(window, text="   The result  ", bg='#76f5a0',fg='black',border=10\
                     ,borderwidth=6,font=1)
helpLf1.place(x=20,y=620,width=1050,height=80)
v=''
hel = Label(helpLf1,text=v,bg='#d0eaff',font='Times')
hel.place(x=70,y=20,width=900,height=40,anchor='w')
brilabel = Label(window, text="Brightness",fg='black',font='Times')
brilabel.place(x=850,y=70, width=200, height=25, rely=0.01)
brights = Scale(window, from_=-255,to=255,orient=HORIZONTAL)
brights.place(x=850,y=100,width=200)
conlabel = Label(window, text="contrasts",fg='black',font='Times')
conlabel.place(x=850,y=145, width=200, height=30, rely=0.01)
contrasts = Scale(window, from_=-127,to=127,orient=HORIZONTAL)
contrasts.place(x=850,y=180,width=200)
btn4 = Button(window, text='apply', width=20,font='Times' ,command=lambda:funcBrightContrast(myFiles))
btn4.place(x=880,y=260,width=150,height=40,anchor='w')
def funcBrightContrast(img):
    global res,panel1
    if  not (not img):
        print('valider2')
        img = cv2.imread(img)
        bright=brights.get()
        contrast=contrasts.get()
        img = apply_brightness_contrast(img, bright, contrast)
        res=img
        img = Image.fromarray(img)
        img = img.resize((300, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = Label(window, image=img, background='#76f5a0', bg='#76f5a0')
        inF = Label(window, text="Modified image")
        inF.place(x=540, y=50, width=300, height=30, rely=0.01)
        panel1 = Label(window, image=img, background='#76f5a0', bg='#76f5a0')
        panel1.image = img
        panel1.place(x=538, y=65, rely=0.03, height=200)
#
def delet_image(v):
    if panel:
        panel.destroy()
    if panel1:
        panel1.destroy()


def detect_tomur(img):
    global v
    x=[]
    x.append(img)
    fX_test_crop = crop_imgs(set_name=x)
    fX_test_crop = np.uint8(fX_test_crop)
    fX_test_prep = preprocess_imgs(set_name=fX_test_crop, img_size=IMG_SIZE)
    predictions = model.predict(fX_test_prep)
    print(predictions)
    predictions = [1 if x > 0.7 else 0 for x in predictions]
    print(predictions)
    if predictions==[1]:
        v='The current brain have a tumor'
        hel = Label(helpLf1, text=v, bg='#d0eaff', font='Times')
        hel.place(x=70, y=20, width=900, height=40, anchor='w')
    else:
        v='The current brain have not a tumor'
        hel = Label(helpLf1, text=v, bg='#d0eaff', font='Times')
        hel.place(x=70, y=20, width=900, height=40, anchor='w')
def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    brightness = map(brightness, 0, 510, -255, 255)
    contrast = map(contrast, 0, 254, -127, 127)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

def map(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)
window.mainloop()


