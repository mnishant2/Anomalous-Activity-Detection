# -*- coding: utf-8 -*-
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

from matplotlib import pyplot as plt
import os
imdir='C:\\Users\\Asus\\desktop\\TEST'
#plt.axis("off")
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
txt='D:/cnn/side/daria_side.avi'
txt1='C:\\Users\\Asus/Downloads/z02.avi'
txt2='D:/cnn/jack/eli_jack.avi'
txt3='D:/cnn/box/person01_boxing_d1_uncomp.avi'
txt4='D:/cnn/wave/person01_handwaving_d1_uncomp.avi'
txt5='D:/cnn/jog/person01_jogging_d1_uncomp.avi'
txt6='D:/cnn/walk/person01_walking_d1_uncomp.avi'
actiondictionary={0:'running',1:'boxing',2:'jogging',3:'walking',4:'waving',5:'side-walking',6:'jumping-jack'}
for root, dirs, filenames in os.walk(imdir):
    for f in filenames:
        k = cv2.waitKey(1) & 0xff
        if k == 113:
          break
        armra=[]
        chrmara=[]
        vote=[]
        gl_c=0
        fullpath = os.path.join(root, f)
        #print fullpath
        cap = cv2.VideoCapture(fullpath)
#        cap = cv2.VideoCapture(txt1)
        
        f=open('pcalik.save','rb')
        pca=cPickle.load(f)
        f.close()
        model=load_model("model_activity_400_7.h5")
        #json_file = open('model_activity_1.json', 'r')
        #loaded_model= json_file.read()
        #json_file.close()
        #model = model_from_json(loaded_model_json)
        ### load weights into new model
        #model.load_weights("model_activity_1.h5")
        #gs1 = gridspec.GridSpec(4, 4)
        #gs1.update(wspace=0.025, hspace=0.05)
#        print cap.grab()
        def get_frame():
            
            ret, frame = cap.read()
            if ret:
                (h, w) = frame.shape[:2]
                w1=640.
                x=float(w1/w)
                frame = cv2.resize(frame, (int(160),int(120)),interpolation=cv2.INTER_CUBIC)
                frame = cv2.resize(frame, (int(320),int(240)),interpolation=cv2.INTER_CUBIC)
            return ret, frame
        def main_process(frame,x,y,w,h,mosaic,count_frame):
            temp=np.zeros((frame.shape),dtype='uint8')
            temp1=np.zeros((frame.shape),dtype='uint8')
           
           
          
            edge=cv2.Canny(frame[y:y+h+1,x:x+w+1],100,100)
        #            cv2.resize(edge, (int(320),int(240)),interpolation=cv2.INTER_CUBIC)
            con_list=[]
#            print 'lele1'
            temp[y:y+h+1,x:x+w+1]=edge
            for i in range(temp.shape[0]):
                for j in range(temp.shape[1]):
                    if temp[i,j]!=0:
                        con_list.append([j,i])
#            print 'lele'
            con_list=np.reshape(con_list,(-1,1,2))
            a=[]
            a.append(con_list)
            im1,contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(temp1,contours,-1,(255,255,255),-1)
            cv2.drawContours(temp,a,0,(255,255,255),-1)
            im1,contours1, hierarchy = cv2.findContours(temp.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            maxi=0
            for i in range(len(contours1)):
                  area=cv2.contourArea(contours1[i])
                  if area>maxi:
                     maxi=area
                     ind=i
            if maxi>0:
               cnt = contours1[ind]
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
            edge1=cv2.Canny(temp,100,100)
            temp1=cv2.resize(temp1,(int(temp.shape[1]*0.5),int(temp.shape[0]*0.5)), interpolation=cv2.INTER_CUBIC)
            #temp=cv2.resize(temp,(int(temp.shape[1]*0.5),int(temp.shape[0]*0.5)), interpolation=cv2.INTER_CUBIC)
        #    cv2.namedWindow('ghj',0)
#            print temp1.shape
            
            #medge=cv2.Canny(temp,100,100)
            #mosaic=cv2.bitwise_or(mosaic,temp)
            #cv2.imshow('frameg',medge)
        #    ax1=plt.subplot2grid((3,3),(0,0))
        #    ax1.imshow(temp)
        #    cv2.imshow('ghj',temp)
            ax2=plt.subplot2grid((4,4),(0,3),rowspan=2)
            ax2.imshow(edge,cmap='gray')
            ax2.set_title("original")
            ax2.axis("off")
            
        #    cv2.imshow('framef1',edge)
            cv2.imshow('woah',edge1)
#            print 'maxi area'
#            print maxi
            if maxi<2000:
                count_frame=0
                mosaic=[]
#                print 'mari'
                return count_frame,mosaic
            else:
                count_frame+=1
#                print 'nahi mari'
                mosaic.append([temp1,full_specs])
                return count_frame,mosaic
        def ml_wala_part(mosaic,vote):  
            kmpl=np.zeros((120,160*5),dtype='uint8')
            index_start=len(mosaic)-5
            l=0
            
            for i in range(index_start,len(mosaic)):
                kmpl[:,l*160:(l+1)*160]=mosaic[i][0]
                #cv2.rectangle(kmpl,(l*160,0),(160,120),(255,255,255),2)
                l=l+1
#            print l
#            print count_frame
#            print index_start
            
            
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
        #    test_dat=np.concatenate((kmpl_lis,specs),axis=1)
            test_dat=pca.transform(test_dat)
#            print model.predict_proba(test_dat,1,0)
            vote.append(model.predict_classes(test_dat,1,0)[0])
            
        #    vote.append(model.predict_classes(test_dat,1,0) for index_start in range(7))
#            print vote     
#            print model.predict_classes(test_dat,1,0)
               #     
               #chrmara.append(kmpl)
               #gl_c+=1
               #cv2.imwrite(tret,kmpl)
            
            ax2=plt.subplot2grid((4,4),(2,0),rowspan=2,colspan=4)
            ax2.imshow(kmpl,cmap='gray')
            ax2.set_title("features passed")
            ax2.axis("off")
            
            
        #    cv2.imshow("kmpl",kmpl)
            predicted=max(set(vote),key=vote.count)
            
                #print 'kinjara'
            return predicted,vote
        
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500,detectShadows=False,varThreshold=20)
        flag=False
        count_frame=0
        retv, frame1 = get_frame()
        mosaic=[]
        frame=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame,(5,5), 0)
        prevframe=fgbg.apply(image=frame)
        while(1):
              retv, frame1 = get_frame()
              if not retv:
                  break
                    
              frame=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
              frame = cv2.GaussianBlur(frame,(5,5), 0)
              kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
              
              ax2=plt.subplot2grid((4,4),(0,0),rowspan=2)
              ax2.imshow(frame,cmap='gray')
              ax2.set_title("original")
              ax2.axis("off")
              
        #      cv2.imshow('frame1',frame)
              fgmask = fgbg.apply(image=frame,learningRate=0.009)
              cv2.imshow('frameinf',fgmask)   
              im=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
              #im=cv2.dilate(im,kernel)
              im1,contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
#              print maxi
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
                        
                          im1,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                          
                          
                          im1,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                          
                          
                          im1,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
#                          print xmin
#                          print temp.shape
                          
                          edge=cv2.Canny(temp,100,100)
                          cv2.imshow("tem[",edge)
                          
                          im1,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                          #cv2.drawContours(temp, contours, -1, (0,128,0), 2)
                          cv2.imshow("tem[",temp)
                          kl=xmin-xmax
#                          print kl
                          lm=0
                          if len(contours)>0:
#                             print 'cont'
                             
                             for i in range(len(contours)):
                              if cv2.contourArea(contours[i])>0:
                                 tet=np.sort(contours[i],axis=0)[-1][0][0]
#                                 print tet
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
                  cv2.rectangle(frame1,(xmax,ymax),(xmin,ymin),(0,0,255),2)
                  flag=True
              else:
                  count_frame=0
                  flag=False
                  mosaic=[]
                  mosaic_count=0
              if flag==True:
                 count_frame,mosaic=main_process(frame,xmax,ymax,xmin-xmax,ymin-ymax,mosaic,count_frame)
#                 print 'kel'
                 if len(mosaic)>4:
                    predicted,vote=ml_wala_part(mosaic,vote)
                    print predicted
                    text=actiondictionary[predicted]
                    print text
                    font=cv2.FONT_HERSHEY_SIMPLEX
                    xt=frame1.shape[0]/2
                    yt=frame1.shape[1]/2
                    cv2.putText(frame1,text,(xt,yt),font,1,(255,255,255),2,cv2.LINE_AA)
                    if predicted==1 or predicted==5:
                        cv2.rectangle(frame1,(xmax,ymax),(xmin,ymin),(255,0,0),2)
                    else:
                        cv2.rectangle(frame1,(xmax,ymax),(xmin,ymin),(0,0,255),2)
        #                        import winsound
        #                        winsound.Beep(300,2000)
              ax2=plt.subplot2grid((4,4),(0,1),rowspan=2)
              ax2.imshow(frame1,cmap='gray')
              ax2.set_title("Detected")
              ax2.axis("off")   
              
        #      cv2.imshow('frame2',frame1)
        #      both=np.hstack((frame1,im))
        #      cv2.imshow('imx',both)
              ax2=plt.subplot2grid((4,4),(0,2),rowspan=2)
              ax2.imshow(im,cmap='gray')
              ax2.set_title("BGS image")
              ax2.axis("off")
              
               
        #      cv2.imshow('frame',im)
              out.write(frame1)
              k = cv2.waitKey(1) & 0xff
              if k == 27:
                 break

       
    
cap.release()
out.release()
cv2.destroyAllWindows()
