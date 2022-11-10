import cv2
import json
dict = {}
output_ann_path = "/home/amokhtar/teams/continental/mot/TCIS/STSC_ANN/"
output_img_path = "/home/amokhtar/teams/continental/mot/TCIS/STSC_IMGS/"
labels = ['100_SIGN', '70_SIGN', 'STOP', 'PASS_RIGHT_SIDE', 'PRIORITY_ROAD', 'OTHER', 'PASS_EITHER_SIDE', 'GIVE_WAY', '80_SIGN', 'NO_PARKING', '50_SIGN', 'PEDESTRIAN_CROSSING', 'NO_STOPPING_NO_STANDING', '30_SIGN']
labels_map = {
'100_SIGN':'regulatory--maximum-speed-limit-100', 
'110_SIGN':'regulatory--maximum-speed-limit-110', 
'120_SIGN':'regulatory--maximum-speed-limit-120',
'70_SIGN':'regulatory--maximum-speed-limit-70',
'90_SIGN':'regulatory--maximum-speed-limit-90',
'STOP':'Stop', 
'PASS_RIGHT_SIDE':'Keep right', 
'PASS_LEFT_SIDE':'Keep left', 
'PRIORITY_ROAD':'Priority road', 
'OTHER':"TS",
'PASS_EITHER_SIDE':"regulatory--pass-on-either-side", 
'GIVE_WAY':"regulatory--give-way-to-oncoming-traffic", 
'80_SIGN':'regulatory--maximum-speed-limit-80',
'60_SIGN':'regulatory--maximum-speed-limit-60', 
'NO_PARKING':"regulatory--no-stopping", 
'50_SIGN':'regulatory--maximum-speed-limit-70', 
'PEDESTRIAN_CROSSING':"information--pedestrians-crossing", 
'NO_STOPPING_NO_STANDING':"regulatory--no-stopping", 
'30_SIGN':'regulatory--maximum-speed-limit-30'
}

input_path = "/home/amokhtar/Research/SWEDISH-TSD/STSD/"

with open('/home/amokhtar/Research/SWEDISH-TSD/annotations-2.txt') as f:
    lines = f.readlines()
    for line in lines:

    # reads each line and trims of extra the spaces
    # and gives only the valid words
        command, description = line.strip().split(":", 1)
        print(input_path+command)
        image = cv2.imread(input_path+command,cv2.IMREAD_COLOR)
        image_name = command.split(".")[0]
        idx = 0
        if description != "":
            objects = description.split(";")
            for object in objects:
                if object != "":
                    data = object.split(",")
                   
                    if len(data) <6:
                        continue
                    label = str(data[6]).split(" ")[1]
                    if label == 'URDBL':
                        continue
                    
                    output_file = open(output_ann_path+image_name+"_"+str(idx)+".json", "w")

                    xmin = 0 if (int(float(data[3])) - 5)  <=0 else 5
                    xmax = 1500 if int(float(data[1])) >1500 else int(float(data[1]))
                    ymin = 0 if (int(float(data[4])) -5) <=0 else 5
                    ymax = 640 if int(float(data[2])) >640 else int(float(data[2]))

                    image_xmin = 0 if (int(float(data[3])) - 5)  <=0 else int(float(data[3])) -5
                    image_xmax = image.shape[1] if (int(float(data[1])) +5) >=image.shape[1] else int(float(data[1])) +5
                    image_ymin = 0 if (int(float(data[4])) -5) <=0 else int(float(data[4])) -5
                    image_ymax = image.shape[0] if (int(float(data[2])) + 5) >=image.shape[0] else int(float(data[2])) +5
                    if image_name == "1277381830Image000003":
                        x=3

                    output_batch = image[image_ymin:image_ymax, image_xmin:image_xmax]

                                
                    cv2.imwrite(output_img_path+image_name+"_"+str(idx)+".jpg", output_batch)

                    idx+=1
                    data_dict = {
                        "objects":[
                                {
                            "label":labels_map[label],
                            "bbox": {
                            "xmax": int(float(data[1]))-int(float(data[3]))+xmin, 
                            "ymax": int(float(data[2]))-int(float(data[4]))+ymin,
                            "xmin": xmin,
                            "ymin": ymin,
                            }
                        }
                        ]
                    }

                    json.dump(data_dict,output_file)


# print(labels)


    