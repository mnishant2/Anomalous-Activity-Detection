# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 03:42:08 2017

@author: Asus
"""

import numpy as np
import cv2 
import cPickle
from math import*
from skimage import measure
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.models import model_from_json
from sklearn.decomposition import PCA
from keras.models import load_model
from Tkinter import *
import tkFileDialog
import winsound
f=open('C:/Users/Asus/Downloads/pcalik.save','rb')
##address where principle component analysis parameters extracted from training data is stored 
pca=cPickle.load(f)
f.close()
model=load_model("C:/Users/Asus/Downloads/model_activity_400_7.h5")
##address to where trained parameters of our model is stored
path=''

actiondictionary={0:'running',1:'boxing',2:'jogging',3:'walking',4:'waving',5:'side-walking',6:'jumping'}
##below function creates a window from where we choose the file to run 
def select_video():
    global path
    path=tkFileDialog.askopenfilename()
    root.destroy()
root = Tk()
btn = Button(root, text="Select a Video", command=select_video)
btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
root.mainloop(0)
#################################################

#####################################
def get_frame(cap):
    ret,frame=cap.read()
    if ret:
      frame = cv2.resize(frame, (320,240),interpolation=cv2.INTER_CUBIC)
    return ret, frame
######################################
    
#########################################
def main_process(frame,x,y,w,h,mosaic,count_frame):
    temp=np.zeros((frame.shape),dtype='uint8')
    temp1=np.zeros((frame.shape),dtype='uint8')
    edge=cv2.Canny(frame[y:y+h+1,x:x+w+1],100,100)
    con_list=[]
    temp[y:y+h+1,x:x+w+1]=edge
    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            if temp[i,j]!=0:
                con_list.append([j,i])
    con_list=np.reshape(con_list,(-1,1,2))
    a=[]
    a.append(con_list)
    im1,contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(temp1,contours,-1,(255,255,255),-1)
    cv2.drawContours(temp,a,0,(255,255,255),-1)
    im2,contours, hierarchy = cv2.findContours(temp.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    maxi=0
    for i in range(len(contours)):
          area=cv2.contourArea(contours[i])
          if area>maxi:
             maxi=area
             ind=i
    if maxi>0:
       cnt = contours[ind]
       M = cv2.moments(cnt)
       cx = int(M['m10']/M['m00'])
       cy = int(M['m01']/M['m00'])
       leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
       rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
       topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
       bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
       full_specs=[[cx,cy],leftmost,rightmost,topmost,bottommost]
    #cv2.drawContours(temp,contours,ind,(128,128,128),2)
    #print frame.shape
    temp1=cv2.resize(temp1,(int(temp.shape[1]*0.5),int(temp.shape[0]*0.5)), interpolation=cv2.INTER_CUBIC)
    #temp=cv2.resize(temp,(int(temp.shape[1]*0.5),int(temp.shape[0]*0.5)), interpolation=cv2.INTER_CUBIC)
    #cv2.namedWindow('ghj',0)
    
    #medge=cv2.Canny(temp,100,100)
    #mosaic=cv2.bitwise_or(mosaic,temp)
    #cv2.imshow('frameg',medge)
    #cv2.imshow('ghj',temp)
    #cv2.imshow('framef1',edge)
 
    if maxi<2500:
        count_frame=0
        mosaic=[]
        temp=np.zeros((frame.shape),dtype='uint8')
        return count_frame,mosaic,temp
    else:
        count_frame+=1
        mosaic.append([temp1,full_specs])
        return count_frame,mosaic,temp
#############################################

        
############################################
def ml_wala_part(mosaic,canvas):  

    kmpl=np.zeros((120,160*5),dtype='uint8')
    index_start=len(mosaic)-5
    l=0
    for i in range(index_start,len(mosaic)):
        kmpl[:,l*160:(l+1)*160]=mosaic[i][0]
        #cv2.rectangle(kmpl,(l*160,0),(160,120),(255,255,255),2)
        l=l+1
    armra=[]
    armra.append([mosaic[i][1] for i in range(index_start,len(mosaic))])
    cent=np.asarray(armra)
#            print cent.shape
    cent=cent.reshape(len(cent),cent.shape[1]*cent.shape[2]*cent.shape[3])
#            print cent.shape
    x=cent[:,0::10]
    y=cent[:,1::10]
    x=(x[:,1:]-x[:,:-1])**2
    y=(y[:,1:]-y[:,:-1])**2
    feature=x+y
#            print feature
    feature=np.reshape(feature,(-1,4))
            
            
    specs=np.reshape(armra,(-1,50))
    kmpl_lis=np.reshape(kmpl,(-1,96000))
    test_dat=np.concatenate((feature,specs,kmpl_lis),axis=1)
    test_dat=pca.transform(test_dat)
    vote.append(model.predict_classes(test_dat,1,0)[0])
    print vote
    
    predicted=max(set(vote),key=vote.count)
    
    
    text=actiondictionary[predicted]
    #print model.predict_proba(test_dat,1,0)
    
#######-----------------NOTE-------------##################
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas,text,(500,60), font, 2,(0,255,255),2)
    
    
    canvas[120:240,:,0]=kmpl
    canvas[120:240,:,1]=kmpl
    canvas[120:240,:,2]=kmpl
    return predicted
###################################################################    
    
    
    
    
    #chrmara.append(kmpl)
    #gl_c+=1
    #cv2.imwrite(tret,kmpl)
    #cv2.imshow("kmpl",kmpl)
    #print 'kinjara'
    #return gl_c



########################################################   
play_video=False
close_video=False
pause_video=False
############################################

#############################################
def button_control(event,x,y,flags,param):
    global play_video,close_video,pause_video
    #print event
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if x>=0 and y>=240 and x<=50 and y<=260 :
            play_video=True
           
        if x>=740 and y>=240 and x<=800 and y<=260:
            close_video=True
        if x>=370 and y>=240 and x<=430 and y<=260:
            pause_video=True
##################################################
            
###################################################
def main_code(path):
 
 cap = cv2.VideoCapture(path)

 
 canvas=np.zeros((260,800,3),dtype='uint8')
 cv2.namedWindow("GUI",0)
 cv2.setMouseCallback('GUI',button_control)
 cv2.rectangle(canvas,(0,240),(50,260),(128,128,128),-1)
 cv2.rectangle(canvas,(740,240),(800,260),(128,128,128),-1)
 font = cv2.FONT_HERSHEY_SIMPLEX
 cv2.putText(canvas,'RUN',(5,255), font, 0.5,(0,0,0),2)
 cv2.putText(canvas,'CLOSE',(745,255), font, 0.5,(0,0,0),2)
 cv2.imshow("GUI",canvas)
 fgbg = cv2.createBackgroundSubtractorMOG2(history=500,detectShadows=False,varThreshold=20)
 flag=False
 count_frame=0
 retv, frame1 = get_frame(cap)
 mosaic=[]
 frame=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
 frame = cv2.GaussianBlur(frame,(5,5), 0)
 prevframe=fgbg.apply(image=frame)
 get_out=False
 kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
 while 1:
    cv2.waitKey(1)
    global play_video,close_video,pause_video,video_finished
    if play_video:
     
     cv2.rectangle(canvas,(370,240),(430,260),(128,128,128),-1) 
     cv2.imshow("GUI",canvas)
     cv2.putText(canvas,'PAUSE',(375,255), font, 0.5,(0,0,0),2)
     while(1):
      e1 = cv2.getTickCount()
      if close_video:
          play_video=False
          break;
      if pause_video:
          cv2.waitKey(0)
          pause_video=False
      retv, frame1 = get_frame(cap)
      if not retv:
          play_video=False
          close_video=True
          video_finished=True
          break
      frame=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
      frame = cv2.GaussianBlur(frame,(5,5), 0)
      
      #cv2.imshow('frame1',frame)
      fgmask = fgbg.apply(image=frame,learningRate=0.009)
      #cv2.imshow('frameinf',fgmask)   
#      kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
      im=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
      #im=cv2.dilate(im,kernel)
      im3,contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      maxi=0
      ind=-1
      xl=[]
      yl=[]
      xwl=[]
      yhl=[]
      for i in range(len(contours)):
          area=cv2.contourArea(contours[i])
          if area>50:
             x,y,w,h = cv2.boundingRect(contours[i])
             xl.append(x)
             yl.append(y)
             xwl.append(x+w)
             yhl.append(y+h)
             cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2) 
             maxi=area
             ind=i
      area_box=20000
      if maxi>50:
         xmax=np.sort(xl)[0]
         ymax=np.sort(yl)[0]
         xmin=np.sort(xwl)[-1]
         ymin=np.sort(yhl)[-1]
         area_box=(xmin-xmax)*(ymin-ymax) 
      if area_box<10000:
         change=True         
         while change:
               change=False
               thresh_s=50
               if ymax-thresh_s>0:
                  temp=frame[ymax-thresh_s:ymin+1,xmax:xmin+1]
                 
                  
                  edge=cv2.Canny(temp,100,100)
                
                  im4,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                  #cv2.drawContours(temp, contours, -1, (0,128,0), 2)
                  
                  kl=thresh_s
                  lm=0
                  if len(contours)>0:
                     pt=ymax
                     
                     
                     for i in range(len(contours)):
                      if cv2.contourArea(contours[i])>0:
                         tet=np.sort(contours[i],axis=0)[0][0][1]
                        
                         if tet<kl:
                            kl=tet
                            lm=1
                     if lm==0 :
                         change=False
                     else:
                         ymax=ymax-thresh_s+kl        
                  else:
                      change=False
                  
         change=True         
         while change:
               change=False
               thresh_s=50
               if ymin+thresh_s<frame.shape[1]:
                  temp=frame[ymax:ymin+thresh_s+1,xmax:xmin+1]
                  
                  
                  edge=cv2.Canny(temp,100,100)
                  
                  
                  im5,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                  #cv2.drawContours(temp, contours, -1, (0,128,0), 2)
                  
                  kl=ymin-ymax
                 
                  lm=0
                  if len(contours)>0:
                
                     
                     for i in range(len(contours)):
                      if cv2.contourArea(contours[i])>0:
                         tet=np.sort(contours[i],axis=0)[-1][0][1]
                        
                         if tet>kl:
                            kl=tet
                            lm=1 
                     if lm==0 :
                         change=False
                     else:
                         
                         ymin=ymax+kl 
                         
                  else:
                      change=False
         
         change=True         
         while change:
               change=False
               thresh_s=50
               if xmax-thresh_s>0:
                  temp=frame[ymax:ymin+1,xmax-thresh_s:xmin+1]
                 
                  
                  edge=cv2.Canny(temp,100,100)
                  
                  
                  im6,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                  #cv2.drawContours(temp, contours, -1, (0,128,0), 2)
                  
                  kl=xmax-thresh_s
                 
                  lm=0
                  if len(contours)>0:
                    
                     
                     for i in range(len(contours)):
                      if cv2.contourArea(contours[i])>0:
                         tet=np.sort(contours[i],axis=0)[0][0][0]
                         
                         if tet<kl:
                            kl=tet
                            lm=1 
                     if lm==0 :
                         change=False
                     else:
                         xmax=xmax-thresh_s+kl 
                         
                  else:
                      change=False   
        
         change=True         
         while change:
               change=False
               thresh_s=50
               if xmax+thresh_s<frame.shape[0]:
                  temp=frame[ymax:ymin+1,xmax:xmin+thresh_s+1]
                  #print xmin
                  #print temp.shape
                  
                  edge=cv2.Canny(temp,100,100)
                  #cv2.imshow("tem[",edge)
                  
                  im7,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                  #cv2.drawContours(temp, contours, -1, (0,128,0), 2)
                  #cv2.imshow("tem[",temp)
                  kl=xmin-xmax
                  #print kl
                  lm=0
                  if len(contours)>0:
                    # print 'cont'
                     
                     for i in range(len(contours)):
                      if cv2.contourArea(contours[i])>0:
                         tet=np.sort(contours[i],axis=0)[-1][0][0]
                        # print tet
                         if tet>kl:
                            kl=tet
                            lm=1 
                     if lm==0 :
                         change=False
                     else:
                         xmin=xmax+kl 
                         
                  else:
                      change=False       
      if maxi>0:
          cv2.rectangle(frame1,(xmax,ymax),(xmin,ymin),(255,0,0),2)
          flag=True
      else:
          count_frame=0
          flag=False
          mosaic=[]
          mosaic_count=0
      if flag==True:
         count_frame,mosaic,tempers=main_process(frame,xmax,ymax,xmin-xmax,ymin-ymax,mosaic,count_frame)
         mira=cv2.resize(tempers,(160,120))
         canvas[0:120,320:480,0]=mira
         canvas[0:120,320:480,1]=mira
         canvas[0:120,320:480,2]=mira
         if len(mosaic)>4:
           predicted= ml_wala_part(mosaic,canvas)
           if predicted==1 or predicted==5:
            winsound.Beep(500,50)
            cv2.rectangle(frame1,(xmax,ymax),(xmin,ymin),(0,0,255),2)
      
      #cv2.imshow('frame2',frame1)
      canvas[0:120,0:160]=cv2.resize(frame1,(160,120))
      mira=cv2.resize(im,(160,120))
      canvas[0:120,160:320,0]=mira
      canvas[0:120,160:320,1]=mira
      canvas[0:120,160:320,2]=mira
      e2 = cv2.getTickCount()
      time = (e2 - e1)/ cv2.getTickFrequency()
      time = float("{0:.3f}".format(time))
      
      cv2.putText(canvas,'TIME ELAPSED:',(100,255), font, 0.5,(114,128,250),2)
      cv2.putText(canvas,str(time)+'s',(220,255), font, 0.6,(255,0,0),2)
      cv2.imshow("GUI",canvas) 
      canvas[0:240,:,:]=0
      canvas[210:260,210:350]=0
      #cv2.imshow('frame',im)
      
      k = cv2.waitKey(1) & 0xff
      if k == 27:
         play_video=False
         close_video=True
         video_finished=True
         break
         
    elif close_video:
        break;
 


video_finished=False
############################################################################
while 1: 
  
  if len(path)>0:
   # print path
       global play_video,close_video,pause_video,video_finished
       play_video=False
       close_video=False
       pause_video=False
       video_finished=False
       vote=[]
       main_code(path)
       cv2.destroyAllWindows()
       path=''
  if video_finished:
       root = Tk()
       btn = Button(root, text="Select a Video", command=select_video)
       btn.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
       root.mainloop(0)
  if len(path)==0  :
     break