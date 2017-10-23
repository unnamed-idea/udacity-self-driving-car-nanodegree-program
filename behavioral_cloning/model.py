# import the libraries
import numpy as np
import csv
import cv2
from matplotlib import pyplot as plt
from random import sample
import random

from keras.models import Sequential
from keras.layers import Flatten,Dense,Dropout,Lambda
from keras.layers.advanced_activations import ELU
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.regularizers import l2


print('imported')

# load the data

lines=[]
s_angles=[]
images=[]
with open('driving_log.csv') as mfile:
	reader=csv.reader(mfile)
	for line in reader:
		lines.append(line)

for line in lines:
	s_angles.append(float(line[3]))
	imname=line[0].split('\\')[-1]
	path='IMG\\'+imname
	img=cv2.imread(imname)
	images.append(img)
    
x_train=np.array(images)
y_train=np.array(s_angles)

# show a sample image
inp_img=x_train[-1]
inp_img=inp_img[...,::-1] #bgr 2 rgb

print('# of frames:',len(x_train ))
plt.imshow(inp_img)
plt.title('sample raw input image')
plt.savefig('rawinp.jpg')
plt.show()

#preprocess the data
def preprocess(imageset):
	processed=[]
	for img in imageset:
		#crop
		cropped=img[70:140,:,:]
		#blur
		blurred=cv2.GaussianBlur(cropped,(3,3),0)
		

		img_yuv = cv2.cvtColor(blurred, cv2.COLOR_BGR2YUV)

    	# equalize the histogram of the Y channel
		img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    
		processed.append(img_yuv)
	return processed

x_train_p=preprocess(x_train)

#seperate 0 angle data and turning data since we reduce the size of
# 0 angle data
x_train_p_zeros=[]
x_train_p_turns=[]
y_train_zeros=[]
y_train_turns=[]

for i in range(len(y_train)):
    crit=0.2 #if absolute angle is <0.2 add that to 0 angle data
    			#otherwise add that to turning data 
    
    #if it is straight driving data, append it with crit/2 probability (10% in this case)
    	#ie most of straight angle data will not be included in training set
    if abs(y_train[i])<crit:
        if random.random()<crit/2:
            x_train_p_zeros.append(x_train_p[i])
            y_train_zeros.append(y_train[i])
    else:
        x_train_p_turns.append(x_train_p[i])
        y_train_turns.append(y_train[i])

# show a preprocessed image
showimg=(cv2.cvtColor(x_train_p[0],cv2.COLOR_YUV2BGR))
showimg=showimg[...,::-1]
plt.imshow(showimg)
plt.title('preprocessed image')
plt.savefig('preprocessed.jpg')
plt.show()

#show histogram before selected sampling
num_bins = 10
avg_samples_per_bin = len(y_train)/num_bins
hist, bins = np.histogram(y_train, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(y_train), np.max(y_train)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')

plt.title('Histogram of Initial steering angles')
plt.savefig('histogramofangles')
plt.savefig('initialhist.jpg')
plt.show()

# augment the data: flip the image and and reverse the angle 
def flip(imageset,angleset): #flip the image and reverse the steering angle
    new_img_set=[]
    new_ang_set=[]
    for i in range(len(imageset)):
        newimg=cv2.flip(imageset[i],1)
        new_img_set.append(newimg)
        new_ang_set.append(-angleset[i])
    return new_img_set,new_ang_set

def brightnes_adjust(imageset,angleset):
    new_img_set=[]
    new_ang_set=[]
    
    #this function is no longer a necessity
    return new_img_set,new_ang_set

new_flipped_imset,new_angset=flip(x_train_p_turns,y_train_turns)

#concatenate a-old and augmented data
x_train_f=np.concatenate((x_train_p_zeros,x_train_p_turns,new_flipped_imset),axis=0)
y_train_f=np.concatenate((y_train_zeros,y_train_turns,new_angset),axis=0)

print('final length of the input set:',len(x_train_f),len(y_train_f))

#show histogram after selected sampling
num_bins = 10
avg_samples_per_bin = len(y_train_f)/num_bins
hist, bins = np.histogram(y_train_f, num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.plot((np.min(y_train_f), np.max(y_train_f)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
plt.title('Histogram of steering angles after Augmentation and Sampling')
plt.savefig('histogramofangles')
plt.show()

#create model

model = Sequential()
#I use ELU activation, and used l2 regularizer, dropouts somehow reduced the performance

#normalize
model.add(Lambda(lambda x: x/255 - 0.5,input_shape=(70,320,3)))

#5*5 and 3*3 Conv layers from Nvidia's paper
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3, border_mode='valid', W_regularizer=l2(0.001)))
model.add(ELU())

# Flatten layer
model.add(Flatten())

# Below is different fron Nvidia's model. I construct bigger layers, since I did not include left and right images, 
#I want the layers to capture the function throughly

model.add(Dense(150, W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Dense(70, W_regularizer=l2(0.001)))
model.add(ELU())

model.add(Dense(10, W_regularizer=l2(0.001)))
model.add(ELU())


#train the model
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(x_train_f,y_train_f,validation_split=0.2,shuffle=True,nb_epoch=25)#25 epoch is found by trial and error


#save the model
model.save('model.h5')

print('saved')