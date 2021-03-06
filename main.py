import time
import PIL.ImageTk
import cv2
import numpy as np
import pyttsx3
import tkinter
from tkinter import *
from cvzone.HandTrackingModule import HandDetector
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from keras.models import load_model
from PIL import Image

window = tkinter.Tk()
window.title("ASL")

cam = cv2.VideoCapture(0)

canvas_w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
canvas_h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
canvas = tkinter.Canvas(window, width=canvas_w + 400, height=canvas_h, bg="sky blue")
canvas.pack()

text = str("")
character = str("")
count = 0

bg_model = None
resetBackground = 0
segmentor = SelfiSegmentation()
detector = HandDetector(detectionCon=0.8, maxHands=2)

model = load_model('models/weightsFeb28031.00.hdf5')


def button_quit():
    window.quit()


def button_reset():
    global bg_model, resetBackground
    bg_model = None
    resetBackground = 0
    time.sleep(1)


button1 = tkinter.Button(window, text="Quit", command=button_quit)
button1.pack(side=LEFT)
button2 = tkinter.Button(window, text="Reset background", command=button_reset)
button2.pack(side=LEFT)

photo = None
photo2 = None

class_names = ['1', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
               'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'enter', ' ']


def character_pairing(character, text):
    if character == 'enter':
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        text = ""
    elif character == 'del':
        if len(text) > 0:
            text = text[:len(text) - 1]
    elif character is None:
        pass
    else:
        text = text + str(character)
    return text


def predict_image(image):
    image = np.stack((image,) * 3, axis=-1)
    image = cv2.resize(image, (224, 224))
    image = image.reshape(224, 224, 3)
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    predict = model.predict(image)
    if max(predict[0]) >= 0.9:
        # print(class_names[np.argmax(predict[0])], max(predict[0]))
        return class_names[np.argmax(predict[0])]


def remove_background(img):
    global bg_model
    mask = bg_model.apply(img, learningRate=0)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    img_hand = cv2.bitwise_and(img, img, mask=mask)
    img_hand = cv2.blur(img_hand, (9, 9))
    return img_hand


def hand_process(img_hand):
    global photo2
    img_hand = cv2.resize(img_hand, dsize=(400, 400))
    img_hand = cv2.cvtColor(img_hand, cv2.COLOR_BGR2GRAY)
    img_hand = cv2.GaussianBlur(img_hand, (7, 7), 0)
    thresh, img_hand = cv2.threshold(img_hand, 25, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow("Hands", img_hand)
    photo2 = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img_hand))
    canvas.create_image(canvas_w, 0, image=photo2, anchor=tkinter.NW)
    character = predict_image(img_hand)
    return character


def update_cam():
    global photo, canvas, character, resetBackground, count, text, bg_model
    success, img = cam.read()
    img = cv2.bilateralFilter(img, 5, 50, 100)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    hands = detector.findHands(img, draw=False)

    if resetBackground == 0:
        bg_model = cv2.createBackgroundSubtractorMOG2(0, 50)
        resetBackground = 1
    img_hand = remove_background(img)

    old_character = character

    if hands:
        x, y, w, h = hands[0]['bbox']
        if w > h:
            cv2.rectangle(img, (x - 10, int(y + h / 2 - w / 2 - 10)), (x + w + 10, int(y + h / 2 + w / 2 + 10)),
                          (0, 255, 255), 2)
            if (10 < x) & (0 < y + h / 2 - w / 2 - 10):
                img_hand = img_hand[int(y + h / 2 - w / 2 - 10):int(y + h / 2 + w / 2 + 10), x - 10:x + w + 10]
        else:
            cv2.rectangle(img, (int(x + w / 2 - h / 2 - 10), y - 10), (int(x + w / 2 + h / 2 + 10), y + h + 10),
                          (0, 255, 255), 2)
            if (0 < x + w / 2 - h / 2 - 10) & (10 < y):
                img_hand = img_hand[y - 10:y + h + 10, int(x + w / 2 - h / 2 - 10):int(x + w / 2 + h / 2 + 10)]

        character = hand_process(img_hand)

        if character == old_character:
            count += 1
            if count > 5:
                count = 0
                text = character_pairing(character, text)
                display = text + "|"
                print(display)
                canvas.delete("del")
                canvas.create_text(canvas_w, 400, text=display, anchor=tkinter.NW, tags="del")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img))
    canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
    # k = cv2.waitKey(10)
    # if k == ord('q'):  # Quit
    #     button_quit()
    # elif k == ord('r'):  # Reset background
    #     button_reset()
    window.after(15, update_cam)


update_cam()
window.mainloop()
