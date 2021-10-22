import zipfile
from PIL import Image
import pytesseract
import cv2 as cv
import numpy as np

# loading the face detection classifier
haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'

# the rest is up to you!

def get_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def find_word_in_image(page,img,word):
    preProcessed = get_grayscale(img)
    d = pytesseract.image_to_data(preProcessed)
    
    for n,i in enumerate(d.splitlines()):
        i=i.split('\t')
        if len(i) > 11 and (word in i[11] and n >0):
            print(f"\"{word}\" found in {page}")
            extract_faces(page,img)
            break
            
def extract_faces(page,img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #blur = cv.GaussianBlur(gray,(7,7),cv.BORDER_DEFAULT)
    #threshold, thresh = cv.threshold (gray,80,255, cv.THRESH_BINARY)
    faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor = 1.5, minNeighbors=3)
    if len(faces_rect) > 0: ##If faces are found
        print(f'{len(faces_rect)} faces found in {page}')

        #Context sheet size based on number of faces found
        sheet_height = len(faces_rect)//5
        if len(faces_rect)%5 > 0:
            sheet_height = sheet_height + 1
        sheet = np.zeros((350*sheet_height, 350*5,3),dtype=np.uint8)
        sx=0
        sy=0
        for (x,y,w,h) in faces_rect:
            #Cropping face
            face=img[y:y+h,x:x+w]
            #Resizing face
            face_resized = cv.resize(face,(350,350),interpolation=cv.INTER_CUBIC)
            #Populating sheet
            sheet[sy:sy+350,sx:sx+350] = face_resized
            if sx == 350 * 4:
                sx = 0
                sy = sy + 350
            else:
                sx = sx + 350
        #cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=5)
        #Convert to RGB for PIL display
        sheet = cv.cvtColor(sheet,cv.COLOR_BGR2RGB)
        #cv.imshow('Detected Faces', img)
        Image.fromarray(sheet).show()
    
    else:
        print(f"But no faces found in {page}")
        
try:
    word = input ("Enter word to search : ")
except:
    print("Input error, try again")
    
with zipfile.ZipFile('images.zip','r') as news_zip:
    news_zip.extractall('news_papers')
    
    #word = "Christopher"
    for im in news_zip.namelist():
        img = cv.imread(f'news_papers/{im}')
        print(f'Processing {im}')
        find_word_in_image(im,img,word)
    