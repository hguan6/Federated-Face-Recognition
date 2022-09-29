# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 23:19:03 2022

@author: rajan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 20:21:21 2022

@author: rajan
"""

import face_recognition


# Example codes to play with

known_image = face_recognition.load_image_file("biden.jpg")
unknown_image = face_recognition.load_image_file("unknownb.jpg")
my_image = face_recognition.load_image_file("rajan.jpeg")

biden_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
my_encoding = face_recognition.face_encodings(my_image)[0]

results = face_recognition.compare_faces([biden_encoding], unknown_encoding)

print(results)

print(len(biden_encoding))
print(len(unknown_encoding))



l = []
l.append(known_image)
l.append(unknown_image)



for i in l:
    enc = face_recognition.face_encodings(i)[0]
    res.append(enc)
    
print(len(res), len(res[0]), len(res[1]))




# Main coded starts here


    

# Load all image names from folder
# variable "name" will now have the filename of all images in the folder

import os
from os import listdir

name = []
 
# get the path/directory
folder_dir = "Subset/"
for images in os.listdir(folder_dir):
 
    # check if the image ends with jpg
    if (images.endswith(".jpg")):
        name.append(images)
        
type(name[0])


# Load all images from image names list using face_encodings 


imgs = []

for i in name:
    known_image = face_recognition.load_image_file(i)
    imgs.append(known_image)

len(imgs)




# Obtain the encodings for all images in the list of names
# res will have the 128 dim vector for each image

res = []
c = 0
for i in imgs:
    c+=1
    try:
        enc = face_recognition.face_encodings(i)[0]
        res.append(enc)
    except:
        print(c)
        
    
len(res)


    
    
    
    
# Save file    
# Save "res" 
        
import pickle

with open("test", "wb") as fp:   
    pickle.dump(res, fp)
    

import numpy as np

np.save('imgs.npy', imgs, allow_pickle=True)




# The remaining part is to load separate images (test set) to check 
# how good the predictions are



# check for existence

result = face_recognition.face_distance(res, my_encoding)

len(result)

result[554]

result[369], result[651]

max(result), min(result)

for i in range(len(results)):
    if results[i] == True:
        print(i)
print("No match")









#False positive test


names = []
 
# get the path/directory
folder_dir = "Test set/"
for images in os.listdir(folder_dir):
 
    # check if the image ends with png
    if (images.endswith(".jpg")):
        names.append(images)
        
type(name[0])


# Load all images from image names list using face_ecodings 

imgs1 = []

for i in names:
    known_image = face_recognition.load_image_file(i)
    imgs1.append(known_image)

len(imgs1)




# Obtain the encodings for all images in the list of names

res1 = []
c = 0
for i in imgs1:
    c+=1
    try:
        enc = face_recognition.face_encodings(i)[0]
        res1.append(enc)
    except:
        print(c)
        
    
len(res1)

#Save the test set image encodings

import pickle

with open("test_set", "wb") as fp:   
    pickle.dump(res1, fp)



# Check matches

lis = {}

lis1 = {}
k = 0

for i in res1:
    k+=1
    c = 0
    results = face_recognition.face_distance(res, i)
    
    for j in range(len(results)):
        if results[j] < 0.5:
            c+=1
    lis1[k] = c
        
lis1



lis

tot = sum(lis.values())
avg = tot/43

print(avg)

    