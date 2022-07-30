import re
import os
import pandas as pd
from paddleocr import PaddleOCR
import speech_recognition as sr
import pyttsx3
import pytesseract

import json
import imutils
import cv2
from flask import Flask, jsonify
import numpy as np
import torch
import uvicorn
from base64 import encodebytes
from deepface import DeepFace
from PIL import Image
from pyzbar.pyzbar import decode
from paddleocr import PaddleOCR
from fuzzywuzzy import fuzz
from fastapi import FastAPI, File, UploadFile, Response
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from starlette.responses import StreamingResponse, FileResponse
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
from keras.preprocessing.image import img_to_array
import tensorflow as tf

# from functools import cacher = sr.Recognizer()

app = FastAPI()

origins = ["*"]
#     "http://localhost.tiangolo.com",#     "https://localhost.tiangolo.com",#     "http://localhost",#     "http://localhost:8080",# ]app.add_middleware(CORSMiddleware,                   allow_origins=origins,                   allow_credentials=False,                   allow_methods=["*"],                   allow_headers=["*"],                   )


# @cachedef detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)


# @cache@app.post("/Aadharcard")
async def analyze_route(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        model = torch.hub.load('ultralytics/yolov5', 'custom',                               path='/home/troondxadmin/amp/pt_files/Aadhar.pt')  # custom model        results = model(img, size=640)
        results.print()
        a = results.pandas().xyxy[0]
        # print(a)        name = a['name']
        s = a.values.tolist()
        h = name.values.tolist()

        for f, j in zip(s, h):
            a = (f[0:4])

            x = int(a[0])
            y = int(a[1])
            h = int(a[3])
            w = int(a[2])
            crop_img = img[y:h, x:w]

            # cv2.imwrite("car_crop.jpg", crop_img)            ocr = PaddleOCR(use_angle_cls=True, lang='en')
            # ocr = PaddleOCR(model_storage_directory='./model')            result = ocr.ocr(crop_img, det=True, cls=False)
            print(result)
            Name = []
            for res in result:
                a = (res[1])
                if a[1] > 0.90:
                    Name.append(a[0])

            print(Name)
            if len(Name) == 2:
                Name = Name[0] + " " + Name[1]
                # print()            else:
                Name = Name[0]
            print(Name)
        result = ocr.ocr(img, det=True, cls=False)
        sd = []
        for res in result:
            a = (res[1])
            if a[1] > 0.90:
                sd.append(a[0])

        sd = str(sd)
        print(sd)
        s = sd.replace(" ", "")

        regex = "[0-9]{12}"        p = re.compile(regex)
        matchs = re.search(p, s)
        Number = matchs.group()
        print(Number)
        # print(Number)        if re.search(r'(\d+/\d+/\d+)', str(result)):
            m = re.search(r'(\d+/\d+/\d+)', str(result))
            date = m.group()
            print(date)
        else:
            regex = "[0-9]{4}"            p = re.compile(regex)
            matchs = re.search(p, s)
            date = matchs.group()
            print(date)

        return {"Success": "true", "Name": Name, "AadharNumber": Number, "DateofBirth": date}
    except Exception as e:
        return {"Success": "false", "Result": "", "message": "Please upload clear Aadhar Card image"}


# @cache@app.post("/Pancard")
async def analyze_route(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        model = torch.hub.load('ultralytics/yolov5', 'custom',                               path='/home/troondxadmin/amp/pt_files/Pancard.pt')  # custom model        results = model(img, size=640)
        results.print()
        a = results.pandas().xyxy[0]

        name = a['name']
        s = a.values.tolist()
        h = name.values.tolist()

        for f, j in zip(s, h):
            a = (f[0:4])

            x = int(a[0])
            y = int(a[1])
            h = int(a[3])
            w = int(a[2])
            crop_img = img[y:h, x:w]

            # cv2.imwrite("car_crop.jpg", crop_img)            ocr = PaddleOCR(use_angle_cls=True, lang='en')

            result = ocr.ocr(crop_img, det=True, cls=False)
            Name = []
            for res in result:
                a = (res[1])
                if a[1] > 0.90:
                    Name.append(a[0])

            if len(Name) == 2:
                Name = Name[-1]

            else:
                Name = Name[0]

        result = ocr.ocr(img, det=True, cls=False)
        sd = []
        for res in result:
            a = (res[1])
            if a[1] > 0.90:
                sd.append(a[0])

        sd = str(sd)
        s = sd.replace(" ", "")

        regex = "[A-Z]{5}[0-9]{4}[A-Z]{1}"        p = re.compile(regex)
        matchs = re.search(p, s)
        Number = matchs.group()
        print(Number)
        if re.search(r'(\d+/\d+/\d+)', str(result)):
            m = re.search(r'(\d+/\d+/\d+)', str(result))
            date = m.group()
            print(date)
        else:
            regex = "[0-9]{4}"            p = re.compile(regex)
            matchs = re.search(p, s)
            date = matchs.group()
            print(date)
        return {"Success": "true", "Name": Name, "PanCardNumber": Number, "DateofBrith": date}
    except Exception as e:
        return {"Success": "false", "Result": "", "message": "Please upload clear Pancard Card image"}


# @cache@app.post('/Covid')
async def analyze_route(file: UploadFile = File(...)):
    try:

        contents = await file.read()
        npimg = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)
        img_size = 100        model = load_model('/home/troondxadmin/aimp/all_model/model-015.model')
        label_dict = {0: 'Negative', 1: 'Positive'}
        img = np.array(image)
        # if (img.ndim == 3):        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)        # else:        # gray = img        gray = img / 255        resized = cv2.resize(gray, (img_size, img_size))
        reshaped = resized.reshape(1, img_size, img_size)
        prediction = model.predict(reshaped)
        result = np.argmax(prediction, axis=1)[0]
        accuracy = float(np.max(prediction, axis=1)[0])
        label = label_dict[result]
        return ({"Success": "True", "Result": label, "Score": accuracy})
    except Exception as e:
        return {"Success": "false", "Result": "", "message": "Please upload clear Covid image"}


# @cache@app.post("/Carnoplate")
async def analyze_route(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        model = torch.hub.load('ultralytics/yolov5', 'custom',                               path='/home/troondxadmin/aimp/all_model/big_car.pt')  # custom model        model.conf = 0.25  # confidence threshold (0-1)        model.iou = 0.45  # NMS IoU threshold (0-1)        results = model(image, size=640)
        results.print()
        a = results.pandas().xyxy[0]
        print(a)
        name = a['name']
        s = a.values.tolist()
        h = name.values.tolist()

        for f, j in zip(s, h):
            a = (f[0:4])
        # print("a",a)        x = int(a[0])
        y = int(a[1])
        h = int(a[3])
        w = int(a[2])
        crop_img = image[y:h, x:w]
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # Incase thresh image is not giving use the below kernel and opening        # Morph open to remove noise and invert image        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))        # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)        # cv2.imwrite("/Volumes/AshokWork/Troondx/opt_images/carsyolo2.jpg", crop_img)        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        result = ocr.ocr(thresh, det=True, cls=False)
        dd = pd.DataFrame(result, columns=["arrays", "accuracy"])
        results = dd['accuracy']
        sd = []
        for x in results:
            y = x[0]
            sd.append(y)

        carno = str(sd)
        return {"Success": "True", "Result": carno}
    except Exception as e:
        return {"Success": "false", "Result": "", "message": "Please upload clear Car image"}


# @cache@app.post("/Facemask_image")
async def analyze_route(file: UploadFile = File(...)):
    try:

        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        for path in os.listdir('/home/troondxadmin/amp/opt_imgs/opt/'):
            if path.endswith('.jpg') or path.endswith('.jpeg') or path.endswith('.png'):
                os.remove("/home/troondxadmin/amp/opt_imgs/opt/" + path)

        prototxtPath = "/home/troondxadmin/aimp/all_model/deploy.prototxt"        weightsPath = "/home/troondxadmin/aimp/all_model/res10_300x300_ssd_iter_140000.caffemodel"        net = cv2.dnn.readNet(prototxtPath, weightsPath)
        model = load_model("/home/troondxadmin/aimp/all_model/mask_detector.model")

        orig = image.copy()
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)

        detections = net.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                (mask, withoutMask) = model.predict(face)[0]

                label = "Mask" if mask > withoutMask else "No Mask"                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                cv2.putText(image, label, (startX, startY - 10),                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 3)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 3)
                #     cv2.resize(640,480)                cv2.imwrite("/home/troondxadmin/amp/opt_imgs/opt/{}".format(file.filename), image)
                return {"Success": "true", "Result": str(label), "filename": str(file.filename)}

        else:
            cv2.imwrite("/home/troondxadmin/amp/opt_imgs/opt/{}".format(file.filename), image)
            return {"Success": "false", "Result": "", "message": "Please upload clear Face image"}


    except Exception as e:
        return {"Success": "false", "Result": "", "message": "Please upload clear Face image"}


# @cache@app.post("/Passport")
async def analyze_route(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        model = torch.hub.load('ultralytics/yolov5', 'custom',                               path='/home/troondxadmin/amp/pt_files/Passport.pt')  # custom model        results = model(img, size=640)
        results.print()
        a = results.pandas().xyxy[0]
        name = a['name']
        s = a.values.tolist()
        h = name.values.tolist()

        for f, j in zip(s, h):
            a = (f[0:4])

            x = int(a[0])
            y = int(a[1])
            h = int(a[3])
            w = int(a[2])
            crop_img = img[y:h, x:w]

            # cv2.imwrite("car_crop.jpg", crop_img)            ocr = PaddleOCR(use_angle_cls=True, lang='en')
            result = ocr.ocr(crop_img, det=True, cls=False)
            dd = pd.DataFrame(result, columns=["arrays", "accuracy"])
            results = dd['accuracy']
            Name = []
            for x in results:
                y = x[0]
                Name.append(y)
            # print(sd)            if len(Name) == 2:
                Name = Name[-1]
                # print(Name[-1])            else:
                Name = Name[0]

        result = ocr.ocr(img, det=True, cls=False)
        dd = pd.DataFrame(result, columns=["arrays", "accuracy"])
        results = dd['accuracy']
        sd = []
        for x in results:
            y = x[0]
            sd.append(y)

        regex = "[A-Z]{1}[0-9]{7}"        # print(str(sd))        p = re.compile(regex)
        matchs = re.search(p, str(sd))
        Number = matchs.group()
        print(Number)

        regex = "[A-Z]{1}[0-9]{7}"        # print(str(sd))        p = re.compile(regex)
        matchs = re.search(p, str(sd))
        Number = matchs.group()
        print(Number)

        date = re.findall(r'(\d+/\d+/\d+)', str(sd))
        print(date)
        dob = date[0]
        start_date = date[1]
        end_date = date[2]
        print(dob, start_date, end_date)
        return {"Success": "true", "Name": Name, "Passportnumber": Number, "Dateofbirth": dob, "Startdate": start_date,                "Enddate": end_date}

    except Exception as e:
        print("Exception Occured", e)
        return {"Success": "false", "Result": "", "message": "Please upload clear Passport image"}


# @cache@app.post("/Chequenumber")
async def analyze_route(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        model = torch.hub.load('ultralytics/yolov5', 'custom',                               path='/home/troondxadmin/amp/pt_files/Cheque.pt')  # custom model        CUDA_VISIBLE_DEVICES = "0"        model.conf = 0.25  # confidence threshold (0-1)        model.iou = 0.45        results = model(img, size=640)
        results.print()

        a = results.pandas().xyxy[0]
        # print(a)        name = a['name']

        s = a.values.tolist()
        h = name.values.tolist()

        for f, j in zip(s, h):
            a = (f[0:4])

            start_point = (int(a[2]), int(a[1]))
            end_point = (int(a[0]), int(a[3]))
            color = (0, 0, 255)
            thickness = 6            fontScale = 3            color = (0, 0, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)

            x = int(a[0])
            y = int(a[1])
            h = int(a[3])
            w = int(a[2])
            crop_img = img[y:h, x:w]
            print(crop_img)

            cv2.imwrite("/home/troondxadmin/amp/car_crop.jpg", crop_img)
            Number = pytesseract.image_to_string(crop_img, lang='mcr')
            print("this is a numaber", Number)
            Number = Number.replace('\n', '')
            Number = Number.replace('\n', '')
            return {"Success": "true", "Chequenumber": Number}
    except Exception as e:
        return {"Success": "false", "Result": "", "message": "Please upload clear cheque image"}


# @cache@app.post("/Drivinglicense")
async def analyze_route(file: UploadFile = File(...)):
    try:

        contents = await file.read()
        # print(type(contents))        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        det = cv2.QRCodeDetector()
        rv, pts = det.detect(img)
        # print(type(rv))        if rv == True:
            detectedBarcodes = decode(img)

            for barcode in detectedBarcodes:
                (x, y, w, h) = barcode.rect
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 5)                res = str.split((barcode.data.decode('utf-8')), ',')
                # print(barcode.data, "\n")                print(res)
                nm = res[0]
                name = nm[5:]
                dno = res[2]
                dlno = dno[5:]
                dob = res[1]
                db = dob[4:]
                print(name, dlno)
                # name = str(res[0])                # nm = name[4:]                return {"Success": "true", "Name": name, "DLNO": dlno, "Dateofbirth": db}
        elif rv == False:
            ocr = PaddleOCR(use_angle_cls=True, lang='en')
            result = ocr.ocr(img, det=True, cls=False)
            dd = pd.DataFrame(result, columns=["arrays", "accuracy"])
            results = dd['accuracy']
            sd = []
            for x in results:
                y = x[0]
                sd.append(y)
                a = str(sd)

            wordstring = '(Name|IAName)$'            for wordline in sd:
                xx = wordline.split()
                if ([w for w in xx if re.search(wordstring, w)]):
                    lineno = sd.index(wordline)
                    textlist = sd[lineno + 1:]
                    Name = textlist[0]
            # print("your Name :", Name)            regex = "[A-Z]{2}[0-9]{2}[A-Z]{1}[0-9]{11}"            p = re.compile(regex)
            matchs = re.search(p, a)
            Number = matchs.group()
            # print(Number)            # print("driving lisence Number :", Number)            date = re.findall(r'(\d+/\d+/\d+)', a)
        # print("your date of birth:", date[1])        return {"Success": "true", "Name": Name, "DLNO": Number, "Dateofbirth": date[1]}
    except Exception as e:
        return {"Success": "false", "Result": "", "message": "Please upload clear Driving License image"}


# @cache@app.post("/Voterid")
async def analyze_route(file: UploadFile = File(...)):
    try:

        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/troondxadmin/amp/pt_files/new_voterid.pt')

        results = model(img, size=640)
        results.print()
        a = results.pandas().xyxy[0]

        name = a['name']
        s = a.values.tolist()
        h = name.values.tolist()

        for f, j in zip(s, h):
            a = (f[0:4])

            x = int(a[0])
            y = int(a[1])
            h = int(a[3])
            w = int(a[2])
            crop_img = img[y:h, x:w]

            # cv2.imwrite("car_crop.jpg", crop_img)            ocr = PaddleOCR(use_angle_cls=True, lang='en')

            result = ocr.ocr(crop_img, det=True, cls=False)
            Name = []
            for res in result:
                a = (res[1])
                if a[1] > 0.90:
                    Name.append(a[0])

            # Name = str(name)            if len(Name) == 2:
                Name = Name[-1]
                print(Name)
            else:
                Name = Name[0]
                print(Name)
        result = ocr.ocr(img, det=True, cls=False)
        sd = []
        for res in result:
            a = (res[1])
            if a[1] > 0.90:
                sd.append(a[0])

        sd = str(sd)
        s = sd.replace(" ", "")

        regex = "[A-Z]{3}[0-9]{7}"        p = re.compile(regex)
        matchs = re.search(p, s)
        Number = matchs.group()
        print(Number)
        return {"Success": "true", "Name": Name, "voterid": Number}
    except Exception as e:
        return {"Success": "false", "Result": "", "message": "Please upload clear Voterid image"}


# @cache@app.post("/yolo")
async def analyze_route(file: UploadFile = File(...)):
    try:

        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        for path in os.listdir('/home/troondxadmin/amp/opt_imgs/opt/'):
            if path.endswith('.jpg') or path.endswith('.jpeg') or path.endswith('.png'):
                os.remove("/home/troondxadmin/amp/opt_imgs/opt/" + path)

        model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # image = [image]        results = model(image, size=640)
        results.print()
        a = results.pandas().xyxy[0]
        if len(a) != 0:

            name = a['name']
            print("i am name", name)
            s = a.values.tolist()
            h = name.values.tolist()
            for f, j in zip(s, h):
                a = (f[0:4])

                start_point = (int(a[2]), int(a[1]))
                end_point = (int(a[0]), int(a[3]))
                color = (0, 255, 255)
                thickness = 30                fontScale = 3                color = (0, 0, 153)
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (50, 50)
                cv2.rectangle(image, start_point, end_point, color, thickness)

                cv2.putText(image, j, (int(a[0]), int(a[1])), cv2.FONT_HERSHEY_SIMPLEX, 7, (153, 0, 0), thickness=10)

                cv2.imwrite("/home/troondxadmin/amp/opt_imgs/opt/{}".format(file.filename), image)
            return {"Success": "true", "filename": str(file.filename)}
        else:
            # cv2.imwrite("/home/troondxadmin/amp/opt_imgs/opt/yolo.jpg", image)            return {"Success": "false", "Result": "Please upload Valid image"}
    except Exception as e:
        print("Exception Occured", e)
        return {"Success": "false", "Result": "", "message": "Please upload clear  image"}


# @cache@app.post("/Liveness")
async def analyze_route(file: UploadFile = File(...)):
    try:

        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # for path in os.listdir('/home/troondxadmin/amp/opt_imgs/opt/'):        # if path.endswith('.jpg') or path.endswith('.jpeg') or path.endswith('.png'):        # os.remove("/home/troondxadmin/amp/opt_imgs/opt/"+path)        height, width, _ = img.shape
        print(height, width)
        if height >= 640 and width >= 640:
            # print(height,width)            face_cascade = cv2.CascadeClassifier(
                "/home/troondxadmin/aimp/all_model/haarcascade_frontalface_default.xml")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(gray)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            print(faces)

            for (x, y, w, h) in faces:
                obj = DeepFace.analyze(img, actions=['age', 'gender', 'race', 'emotion'])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=5)

                lbls = ["Age : " + str(obj['age']), "Gender : " + str(obj['gender']),                        "Emotion : " + str(obj['dominant_emotion']), "Race : " + str(obj["dominant_race"])]
                offset = 35                print("Labels:", lbls)
                for idx, lbl in enumerate(lbls):
                    cv2.putText(img, str(lbl), (x + w, y + offset * idx), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),                                5)
                    # cv2.imwrite("/home/troondxadmin/amp/opt_imgs/opt/{}".format(file.filename), img)                    return {"Success": "True", "Age": str(obj['age']), "Gender": str(obj['gender']),                            "Emotion": str(obj['dominant_emotion']), "Race": str(obj["dominant_race"])}
        else:
            return {"Success": "false", "Result": "Please Upload morethan 640x640 size images"}
    except ValueError:
        return {"Success": "False", "message": "Please upload a image with faces"}


# @cache@app.post("/Speech")
async def analyze_route():
    try:
        with sr.Microphone() as source2:

            r.adjust_for_ambient_noise(source2, duration=0.2)

            print("you speak")
            audio2 = r.listen(source2)

            # Using google to recognize audio            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()
            print("Did you say " + MyText)

            return {"text": MyText}
    except sr.RequestError as e:
        return {"text": "false", "error": e}

    except sr.UnknownValueError as a:
        return {"text": "false", "error": a}


# @cache@app.post("/Text")
async def analyze_route(input: str):
    try:
        engine = pyttsx3.init()
        engine.say(input)
        engine.runAndWait()
        return {"Success": "True"}



    except Exception as e:
        print("Exception Occured", e)
        return {"Success": "false", "Result": "", "message": "Please upload clear Passport image"}


# @cache@app.post("/Stroke")
async def analyze_route(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.fromstring(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        model = load_model("/home/troondxadmin/aimp/all_model/stroke_model/Stroke_best.h5")
        class_names = ['NO_Stroke', 'Stroke']
        img = cv2.resize(img, (180, 180), interpolation=cv2.INTER_AREA)
        img_array = img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        result = class_names[np.argmax(score)]
        print("This image most likely belongs to", result)
        return {"Success": "true", "result": "Your image is most likely belongs to  " + str(result)}
    except Exception as e:
        return {"Success": "false", "Result": "", "message": " image"}


if __name__ == '__main__':
    uvicorn.run('test:app', port=8001, host='0.0.0.0', reload=True, debug=True)
