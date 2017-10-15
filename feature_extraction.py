import numpy as np
import cv2 
from math import*
from skimage import measure
import pandas as pd
import cPickle
txt='D:/cnn/side/daria_side.avi'
txt1='D:/cnn/run/person01_running_d1_uncomp.avi'
txt2='D:/cnn/jack/eli_jack.avi'
txt3='D:/cnn/box/person01_boxing_d1_uncomp.avi'
txt4='D:/cnn/wave/person01_handwaving_d1_uncomp.avi'
txt5='D:/cnn/jog/person01_jogging_d1_uncomp.avi'
txt6='D:/cnn/walk/person01_walking_d1_uncomp.avi'
#fulltxt=['D:/cnn/jack/daria_jack.avi','D:/cnn/jack/denis_jack.avi','D:/cnn/jack/eli_jack.avi','D:/cnn/jack/ido_jack.avi','D:/cnn/jack/ira_jack.avi','D:/cnn/jack/lena_jack.avi','D:/cnn/jack/lyova_jack.avi','D:/cnn/jack/moshe_jack.avi','D:/cnn/jack/shahar_jack.avi']
#fulltxt=['D:/cnn/side/daria_side.avi','D:/cnn/side/denis_side.avi','D:/cnn/side/eli_side.avi','D:/cnn/side/ido_side.avi','D:/cnn/side/ira_side.avi','D:/cnn/side/lena_side.avi','D:/cnn/side/lyova_side.avi','D:/cnn/side/moshe_side.avi','D:/cnn/side/shahar_side.avi']
armra=[]
chrmara=[]
#print fulltxt[0]
gl_c=0
for kit in range(1,5):
    k = cv2.waitKey(0) & 0xff
    if k == 113:
        break
    for ui in range(1,26):
     prev_gl_c=gl_c
     if ui<=9:
         txt1='C:/Users/Asus/Downloads/boxing/person0'+str(ui)+'_boxing_d' +str(kit)+'_uncomp.avi'
     else:
         txt1='C:/Users/Asus/Downloads/boxing/person'+str(ui)+'_boxing_d' +str(kit)+'_uncomp.avi'
     cap = cv2.VideoCapture(txt1)
     
     
     print cap.grab()
     def get_frame():
        
        ret, frame = cap.read()
        if ret:
            (h, w) = frame.shape[:2]
            w1=640.
            x=float(w1/w)
            frame = cv2.resize(frame, (int(320),int(240)),interpolation=cv2.INTER_CUBIC)
        return ret, frame
     def main_process(frame,x,y,w,h,mosaic,count_frame):
        temp=np.zeros((frame.shape),dtype='uint8')
        temp1=np.zeros((frame.shape),dtype='uint8')
        #for i in range(x,x+w):
            #for j in range(y,y+h):
              #  temp[j,i]=frame[j,i]
       
        
        edge=cv2.Canny(frame[y:y+h+1,x:x+w+1],100,100)
        con_list=[]
        print 'lele1'
        temp[y:y+h+1,x:x+w+1]=edge
        for i in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if temp[i,j]!=0:
                    con_list.append([j,i])
        print 'lele'
        con_list=np.reshape(con_list,(-1,1,2))
        a=[]
        a.append(con_list)
        imu,contours, hierarchy = cv2.findContours(temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #maxi=0
        #ind=0
        #indi=[]
        #for i in range(len(contours)):
        #      area=cv2.contourArea(contours[i])
        #      if len(contours[i])>0:
        #         indi.append(i)
        #      if area>maxi:
        #         maxi=area
        #         ind=i
        #for j in range(len(indi)):         
        cv2.drawContours(temp1,contours,-1,(255,255,255),-1)
        cv2.drawContours(temp,a,0,(255,255,255),-1)
        imv,contours, hierarchy = cv2.findContours(temp.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
        cv2.namedWindow('ghj',0)
        print temp1.shape
        
        #medge=cv2.Canny(temp,100,100)
        #mosaic=cv2.bitwise_or(mosaic,temp)
        #cv2.imshow('frameg',medge)
        cv2.imshow('ghj',temp)
        cv2.imshow('framef1',edge)
        print 'maxi area'
        print maxi
        if maxi<2500:
            count_frame=0
            mosaic=[]
            print 'mari'
            return count_frame,mosaic
        else:
            count_frame+=1
            print 'nahi mari'
            mosaic.append([temp1,full_specs])
            return count_frame,mosaic
     def ml_wala_part(mosaic,gl_c):  
        kmpl=np.zeros((120,160*5),dtype='uint8')
        index_start=len(mosaic)-5
        l=0
        for i in range(index_start,len(mosaic)):
            kmpl[:,l*160:(l+1)*160]=mosaic[i][0]
            #cv2.rectangle(kmpl,(l*160,0),(160,120),(255,255,255),2)
            l=l+1
        print 'pinjara'
        armra.append([mosaic[i][1] for i in range(index_start,len(mosaic))])
        chrmara.append(kmpl)
        gl_c+=1
        tret='D:/cnn/train_pan/jog'+str(gl_c)+'.jpg'
        cv2.imwrite(tret,kmpl)
        cv2.imshow("kmpl",kmpl)
        print 'kinjara'
        return gl_c
    
     fgbg = cv2.createBackgroundSubtractorMOG2(history=500,detectShadows=False,varThreshold=100)
     flag=False
     count_frame=0
     retv, frame1 = get_frame()
     mosaic=[]
     mosaic_count=0
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
          #thresh=cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,5,2) 
          #edge=cv2.Canny(frame,100,100)
          #cv2.imshow('edge',thresh)
          cv2.imshow('frame1',frame)
          fgmask = fgbg.apply(image=frame,learningRate=0.0)
          cv2.imshow('frameinf',fgmask)   
          im=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
          #im=cv2.dilate(im,kernel)
          imk,contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
          print maxi
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
                    
                      im11,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                      
                      
                      iml,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                      
                      
                      imj,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
                      print xmin
                      print temp.shape
                      
                      edge=cv2.Canny(temp,100,100)
                      cv2.imshow("tem[",edge)
                      
                      imz,contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                      #cv2.drawContours(temp, contours, -1, (0,128,0), 2)
                      cv2.imshow("tem[",temp)
                      kl=xmin-xmax
                      print kl
                      lm=0
                      if len(contours)>0:
                         print 'cont'
                         
                         for i in range(len(contours)):
                          if cv2.contourArea(contours[i])>0:
                             tet=np.sort(contours[i],axis=0)[-1][0][0]
                             print tet
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
              #x,y,w,h = cv2.boundingRect(contours[ind])
              #print "pura" 
              #print ymax
              cv2.rectangle(frame1,(xmax,ymax),(xmin,ymin),(0,0,255),2)
              #print "goenka"
              #print xmin
              #print xmax
              #print ymax
              #print ymin
              #print 'area'
              #print (xmin-xmax)*(ymin-ymax)
              #print 'w/h ratio'
              #print (float(xmin-xmax))/(ymin-ymax)
              flag=True
          else:
              count_frame=0
              flag=False
              mosaic=[]
              mosaic_count=0
          if flag==True:
             count_frame,mosaic=main_process(frame,xmax,ymax,xmin-xmax,ymin-ymax,mosaic,count_frame)
             print 'kel'
             if len(mosaic)>4:
                gl_c=ml_wala_part(mosaic,gl_c)
             
          
          cv2.imshow('frame2',frame1)
          
        
           
          cv2.imshow('frame',im)
          k = cv2.waitKey(1) & 0xff
          if k == 27:
             break
     import winsound
     winsound.Beep(300,2000)  
     oi=cv2.waitKey(0)
     if oi==48:
         gl_c=prev_gl_c
         casw=[]
         cwer=[]
         for i in range(gl_c):
             casw.append(armra[i])
             cwer.append(chrmara[i])
         armra=[]
         chrmara=[]
         armra=casw
         chrmara=cwer
import cPickle
f = open('box.save', 'wb')
cPickle.dump(armra, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
f = open('box_kmpl.save', 'wb')
cPickle.dump(chrmara, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()
print gl_c
import winsound
winsound.Beep(300,2000)
cap.release()
cv2.destroyAllWindows()
