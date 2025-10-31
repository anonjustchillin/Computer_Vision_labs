import cv2
from lab4.meanshift import meanShift
from lab4.camshift import camShift
from lab4.kcf import kcf

filename = './crowd.mp4'
#filename = './cup.mp4'
filesize = (1920,1080)
cap = cv2.VideoCapture(filename)

print('Choose the object tracking method:')
print('1 - MeanShift')
print('2 - CamShift')
print('3 - KCF')

print()

x = input('Method: ')

match x:
    case "1":
        print('1 - MeanShift')
        meanShift(cap)
    case "2":
        print('2 - CamShift')
        camShift(cap)
    case "3":
        print('3 - KCF')
        kcf(cap)
    case _:
        print("?")