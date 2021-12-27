#!/usr/bin/env python
# coding: utf-8

# In[26]:


import cv2


# In[27]:


img = cv2.imread('C:/Users/lavik/Downloads/face.jpg')


# In[28]:


img.shape


# In[9]:


img[0]


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


plt.imshow(img)


# In[20]:


while True:
    cv2.imshow('result',img)
    #27--ASCII of escape
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()


# In[21]:


haar_data = cv2.CascadeClassifier('C:/Users/lavik/OneDrive/Desktop/haarcascade_frontalface_default.xml')


# In[22]:


haar_data.detectMultiScale(img)


# In[23]:


#cv2.rectangle(img,(x,y),(w,h),(b,g,r),border_thickness)


# In[38]:


while True:
    faces = haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
    cv2.imshow('result',img)
    #27--ASCII of escape
    if cv2.waitKey(2) == 27:
        break
cv2.destroyAllWindows()


# In[53]:


capture = cv2.VideoCapture(0)
data = []
while True:
    flag, img = capture.read()
    if flag:
        faces = haar_data.detectMultiScale(img)
        for x,y,w,h in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
            face = img[y:y+h, x:x+w, :]
            face = cv2.resize(face,(50,50))
            print(len(data))
            if len(data)<400:
                data.append(face)
        cv2.imshow('result',img)
        #27--ASCII of escape
        if cv2.waitKey(2) == 27 or len(data) >= 200:
            break

capture.release()
cv2.destroyAllWindows()
        


# In[51]:


import numpy as np


# In[52]:


np.save('without_mask.npy',data)


# In[54]:


np.save('with_mask.npy',data)


# In[55]:


plt.imshow(data[0])


# In[ ]:




