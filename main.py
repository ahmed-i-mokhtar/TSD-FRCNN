import base64
import os
import numpy as np
import cv2
import torch
import time
import aiofiles

#API Deployment
from typing import Union

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
app = FastAPI()

templates = Jinja2Templates(directory="templates")

device1 =torch.device('cuda:0')
host = ""
#_____________________ classification _________________________
model1 = torch.load("models/TCIS_113class_epoch_11.pth",map_location='cpu' )
model1 = model1.to(device1)
#### best dete best_model__FRCNN_25_11_14
### _________ detection ____________________-------
model = torch.load("models/TCIS_MAPP_det3_4_29.pth",map_location='cpu' )
model=model.to(device1)

def classification_model(model,img,classes_s):
    #cv2.imshow("class",img)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    # add the batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the image to a floating point tensor
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    # send the input to the device and pass the it through the network to
    # get the detections and predictions
    image = image.to(device1)
    model.eval()
    classify = model(image)[0]
    for i in range(0, len(classify["boxes"])):
    
        confidence = classify["scores"][i]

        if confidence > 0.8:
            idx = int(classify["labels"][i])
            label1 = "{}".format(classes_s[idx])
            print("[INFO_class]  {}:{}".format(label1,confidence*100))
            return label1

@app.get("/", response_class=HTMLResponse)
async def read_items(request: Request):
    host = request.client.host
    return templates.TemplateResponse("basic.html", {"request": request, "host":host})

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/uploadImage/", response_class=HTMLResponse)
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    file_names= os.listdir("images")
    print(file_names)
    if len(file_names) > 20:
        for file_name in file_names:
            os.remove(os.path.join("images",file_name))
    async with aiofiles.open("images/"+file.filename, 'wb') as out_file:
        while content := await file.read(1024):  # async read chunk
            await out_file.write(content)  # async write chunk
    run(os.path.join("images",file.filename), file.filename)
    
    
    uploaded_file = open("output/"+file.filename, "rb")

    base64_encoded_image = base64.b64encode(uploaded_file.read()).decode("utf-8")

    return templates.TemplateResponse("output.html", {"request": request,  "myImage": base64_encoded_image, "file_name":file.filename})
    # return templates.TemplateResponse("output.html", {"request": request, "output":os.path.abspath(os.path.join("images",file.filename))})


    # return {"filename": file.filename}

@app.post("/uploadImageAPI/", response_class=HTMLResponse)
async def create_upload_file(request: Request, file: UploadFile = File(...)):
    file_names= os.listdir("images")
    print(file_names)
    if len(file_names) > 20:
        for file_name in file_names:
            os.remove(os.path.join("images",file_name))
    async with aiofiles.open("images/"+file.filename, 'wb') as out_file:
        while content := await file.read(1024):  # async read chunk
            await out_file.write(content)  # async write chunk
    run(os.path.join("images",file.filename), file.filename)
    
    
    # uploaded_file = open("output/"+file.filename, "rb")

    # base64_encoded_image = base64.b64encode(uploaded_file.read()).decode("utf-8")

    return FileResponse(os.path.join("output",file.filename))
    # return templates.TemplateResponse("output.html", {"request": request, "output":os.path.abspath(os.path.join("images",file.filename))})


    # return {"filename": file.filename}










def run(image_path, file_name):



    #print(device1)
    last_file=101

    CLASSES=["nothing","TS"]
    image = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    ### make a copy from orignal image for visulaization
    orig = image.copy()
    orig2 = image.copy()



    def vis_TCI(image,BB,type="image",vis_images_path="/home/alaa/Documents/",label="Roundabout mandatory"):
        """
        :param image:
        :param BBboxcenterpoint:
        :param w:  bbox width in 2d
        :param h:  bbox height in 2d
        :param type:  vis. type (text or image)
        :param vis_images_path: label images folder path
        :param label:
        """
        startX=BB[0]
        startY=BB[1]
        endX=BB[2]
        endY=BB[3]
        if type == "image":
            vis_img=cv2.imread(f"{vis_images_path}{label}.jpg")
            vis_img=cv2.resize(vis_img,(int(endY-startY),int(endX-startX)))
            #cv2.imshow("roundabout",vis_img)
            orig2[startY:startY+vis_img.shape[0], startX-vis_img.shape[1]:startX]=vis_img
            #####
            sub_img = orig[startY:endY, startX:endX]
            ##------ dummy the main sign detected ------
            white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255
            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
            # Putting the image back to its position
            orig2[startY:endY, startX:endX] = res
        elif type == "text":
            ## draw BB around label
            cv2.rectangle(orig2, (startX, startY), (endX, endY),(255,0,0), 2)
            ## text parameters
            y=startY
            font=cv2.FONT_ITALIC
            font_scale=0.9
            font_thickness=1
            textsize = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
            text_w, text_h = textsize
            ##
            cv2.rectangle(orig2, (startX-text_w, y-5), (startX, y+text_h+5), (0,0,0), -1)
            cv2.putText(orig2, label, (startX-text_w, y+text_h),font, font_scale, (255,255,255), 0)
    CLASSES_class=["BG",
            'regulatory--maximum-speed-limit-20' ,
            'regulatory--maximum-speed-limit-30' ,
            'regulatory--maximum-speed-limit-50' ,
            'regulatory--maximum-speed-limit-60' ,
            'regulatory--maximum-speed-limit-70' ,
            'regulatory--maximum-speed-limit-80' ,
            'regulatory--end--maximum-speed-limit-80' ,
            'regulatory--maximum-speed-limit-100' ,
            'regulatory--maximum-speed-limit-120' ,
            'No passing',
            'No passing veh over 3.5 tons',
            'Right-of-way at intersection',
            'Priority road',
            'Yield',
            'Stop',
            'No vehicles',
            'Veh > 3.5 tons prohibited',
            'No entry',
            'General caution',
            'Dangerous curve left',
            'Dangerous curve right',
            'Double curve',
            'Bumpy road',
            'Slippery road',
            'Road narrows on the right',
            'Road work',
            'Traffic signals',
            'Pedestrians',
            'Children crossing',
            'Bicycles crossing',
            'Beware of ice/snow',
            'Wild animals crossing',
            'End speed + passing limits',
            'Turn right ahead',
            'Turn left ahead',
            'Ahead only',
            'Go straight or right',
            'Go straight or left',
            'Keep right',
            'Keep left',
            'Roundabout mandatory',
            'End of no passing',
            'End no passing veh > 3.5 tons',
            "information--end-of-limited-access-road",
            "information--end-of-living-street",
            "information--limited-access-road",
            "information--living-street",
            "information--motorway",
            "information--pedestrians-crossing",
            "regulatory--end-of-priority-road",
            "regulatory--maximum-speed-limit-10",
            "regulatory--maximum-speed-limit-110",
            "regulatory--maximum-speed-limit-15",
            "regulatory--maximum-speed-limit-25",
            "regulatory--maximum-speed-limit-40",
            "regulatory--maximum-speed-limit-45",
            "regulatory--maximum-speed-limit-5",
            "regulatory--maximum-speed-limit-90",
            "regulatory--no-motor-vehicles",
            "regulatory--pass-on-either-side",
            "regulatory--priority-over-oncoming-vehicles",
            "regulatory--turn-right",
            "warning--crossroads-with-priority-to-the-right",
            "warning--double-curve-first-right",
            "warning--junction-with-a-side-road-perpendicular-left",
            "warning--junction-with-a-side-road-perpendicular-right",
            "warning--pedestrians-crossing",
            "warning--road-narrows",
            "warning--road-narrows-left",
            "warning--turn-left",
            "warning--two-way-traffic",
            "TRAFFIC_SIGN_MAIN_PROHIB_VEHICLES_WIDER_3M",
            "complementary--distance",
            "TRAFFIC_SIGN_MAIN_PROHIB_VEHICLES_WIDER_4M",
            "TRAFFIC_SIGN_MAIN_PROHIB_VEHICLES_WIDER_2.5M",
            "TRAFFIC_SIGN_MAIN_PROHIB_VEHICLES_WIDER_3.2M",
            "TRAFFIC_SIGN_MAIN_PROHIB_VEHICLES_WIDER_3.5M",
            "TRAFFIC_SIGN_MAIN_PROHIB_VEHICLES_WIDER_4.2M",
            "TRAFFIC_SIGN_MAIN_PROHIB_VEHICLES_WIDER_4.5M",
            "TRAFFIC_SIGN_MAIN_MIN_SPEED_50",
            "TRAFFIC_SIGN_MAIN_MIN_SPEED_60",
            "TRAFFIC_SIGN_MAIN_MIN_SPEED_70",
            "TRAFFIC_SIGN_MAIN_MIN_SPEED_80",
            "TRAFFIC_SIGN_MAIN_MIN_SPEED_90",
            "TRAFFIC_SIGN_MAIN_MIN_SPEED_100",
            "TRAFFIC_SIGN_MAIN_MIN_SPEED_110",
            "TRAFFIC_SIGN_MAIN_DIR_ARROW_TURN_LEFT_OR_RIGHT",
            "TRAFFIC_SIGN_MAIN_SPEED_LIMIT_010_END",
            "TRAFFIC_SIGN_MAIN_SPEED_LIMIT_020_END",
            "TRAFFIC_SIGN_MAIN_SPEED_LIMIT_030_END",
            "TRAFFIC_SIGN_MAIN_SPEED_LIMIT_040_END",
            "TRAFFIC_SIGN_MAIN_SPEED_LIMIT_050_END",
            "TRAFFIC_SIGN_MAIN_SPEED_LIMIT_060_END",
            "TRAFFIC_SIGN_MAIN_SPEED_LIMIT_070_END",
            "TRAFFIC_SIGN_MAIN_SPEED_LIMIT_080_END",
            "TRAFFIC_SIGN_MAIN_SPEED_LIMIT_100_END",
            "regulatory--no-motor-vehicles-except-motorcycles",
            "regulatory--one-way-left",
            "regulatory--one-way-right",
            "regulatory--give-way-to-oncoming-traffic",
            "information--end-of-motorway",
            "regulatory--one-way-straight",
            "regulatory--shared-path-pedestrians-and-bicycles",
            "regulatory--shared-path-bicycles-and-pedestrians",
            "regulatory--dual-path-bicycles-and-pedestrians",
            "regulatory--dual-path-pedestrians-and-bicycles",
            "regulatory--maximum-speed-limit-led-60",
            "regulatory--maximum-speed-limit-led-80",
            "regulatory--maximum-speed-limit-led-100",
            "information--minimum-speed-40",
            "warning--railroad-crossing-without-barriers",
            "regulatory--u-turn",
            "regulatory--no-stopping",
            "TS"
            ]

    # convert the image from BGR to RGB channel ordering and change the
    # image from channels last to channels first ordering
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    # add the batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the image to a floating point tensor
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    # send the input to the device and pass the it through the network to
    # get the detections and predictionsq
    image = image.to(device1)
    model.eval()
    t=time.time()
    detections = model(image)[0]

    boxes = detections["boxes"].detach().cpu().numpy()
    scores = np.array([detections["scores"].detach().cpu().numpy().tolist()]).T
    labels = np.array([detections["labels"].detach().cpu().numpy().tolist()]).T
    # tracker_input = np.concatenate((boxes, scores.T, labels.T),axis=1)
    # tracker_input = np.array([trk  for trk in tracker_input if trk[4]>0.85])
    # output = tracker.update(dets=tracker_input)

    for i in range(boxes.shape[0]):
        #for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = scores[i][0]
        print("confidence: ", confidence)
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        #print(confidence)
        if confidence > 0.85:
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            # idx = int(detections["labels"][i])
            (startX, startY, endX, endY) = boxes[i].astype(int)

            # display the prediction to our terminal
            # label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            #print("[INFO] {}".format(label))
            # draw the bounding box and label on the image
            cv2.rectangle(orig2, (startX, startY), (endX, endY),
                        (255,0,0), 2 )
            # cv2.putText(orig2, str(t.id),(startX-5, startY-5), cv2.FONT_HERSHEY_SIMPLEX, thickness = 2,fontScale = 2,color=(0,0,255))
            cv2.putText(orig2, CLASSES[int(labels[i])],(startX+5, startY+5), cv2.FONT_HERSHEY_SIMPLEX, thickness = 2,fontScale = 0.5,color=(0,0,0))
            y = startY - 15 if startY - 15 > 15 else startY + 15
            #cv2.putText(orig, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            img=orig[startY-20:endY+20,startX-20:endX+20]
            try:
                label1=classification_model(model1,img,CLASSES_class)
                if label1== None:
                    label1="TS"
                    
            except:
                label1= "TS"
            # label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            # print("[INFO] {}".format(label))
            if label1 != "TS":
                        vis_TCI(image,BB=(startX, startY, endX, endY),type="text",vis_images_path="/home/alaa/Desktop/VIS_TCIS/vis_images/",label=label1)


    size= (orig.shape[1],orig.shape[0])
    orig2= cv2.resize(orig2,(1500,640),interpolation = cv2.INTER_AREA)

    #cv2.imwrite(save_path+name,orig2)
    last_file+=1
    file_names= os.listdir("output")

    cv2.imwrite("output/"+file_name, orig2)
    if len(file_names) > 5:
        for file_name in file_names:
            os.remove(os.path.join("output",file_name))


    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    #     break
