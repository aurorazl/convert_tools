# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt
import serve_utils as utils
import numpy as np
import requests
import json
from PIL import Image
import logging
import visualization_utils
import serve_utils
from io import BytesIO

def object_detect(input_path="./road.jpg", output_path='./demo.jpg',flag=False):

    img_size = 608
    num_channels = 3
    # image_path = "./docs/images/sample_computer.jpg"
    image_path = input_path
    # original_image = cv2.imread(image_path)
    with open(input_path,"rb") as f:
        a = f.read()
    original_image = cv2.imdecode(np.frombuffer(a, np.uint8), -1)
    # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # image_data = utils.image_preporcess(np.copy(original_image), [img_size, img_size]).astype(np.uint8)# 图片处理成608*608*3
    # print(image_data.shape)

    # image = Image.open(image_path).convert("RGB")
    img_np_arr = np.frombuffer(a, np.uint8)
    inputImg = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)

    # image = Image.open(BytesIO(a)).convert("RGB")
    # (im_width, im_height) = image.size
    # print(im_width, im_height)
    # print(im_width, im_height)
    # image_data =  np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    # plt.imshow(image_data)
    # plt.show()

    yolov3_api = "http://219.133.167.42:30000/endpoints/v3/v1/models/ifs-6418bdee-b3c5-46d8-9b89-32ab3c7eb8a1:predict	"   # 刚刚产生的接口
    # yolov3_api = "http://192.168.1.182:8080/v1/models/test:predict"   # 刚刚产生的接口
    image_data_yolo_list = inputImg[np.newaxis, :].tolist() # 转化为多维数组矩阵
    with open("sample_response2.json", "w") as f:
        f.write(json.dumps({"signature_name": "serving_default","instances":image_data_yolo_list}))
    headers = {"Content-type": "application/json","Host":"ifs-6418bdee-b3c5-46d8-9b89-32ab3c7eb8a1-predictor-default.kfserving-pod.example.com"}
    if not flag:
        print("start requests")
        r = requests.post(yolov3_api, headers=headers,
                          # data=json.dumps({"signature_name": "predict","instances":image_data_yolo_list}))
                          data=json.dumps({"signature_name": "serving_default","instances":image_data_yolo_list}))
                          # json=image_data_yolo_list)
        print(r.status_code)
        print(r.content)
        print("get request done")
        r = r.json() 	#post请求
        with open("sample_response3.json","w") as f:
            f.write(json.dumps(r))

    print("start ")
    with open("sample_response3.json","r") as f:
        r = json.load(f)
    # print(r)
    # print('r',r) # 19, 19, 85 = 30685
    # {'error': 'Input to reshape is a tensor with 18411 values, but the requested shape requires a multiple of 30685\n\t [[{{node pred_multi_scale/Reshape_2}}]]'}
    # 18411 的因子 [3, 17, 19, 51, 57, 323, 361, 969, 1083, 6137]
    # output = np.array(r['predictions'])

    output_dict = r['predictions'][0]
    print(output_dict['detection_scores'])
    print(output_dict['detection_boxes'])
    output_dict['num_detections'] = int(output_dict['num_detections'])
    output_dict['detection_classes'] = np.array([int(class_id) for class_id in output_dict['detection_classes']])
    output_dict['detection_boxes'] = np.array(output_dict['detection_boxes'])
    output_dict['detection_scores'] = np.array(output_dict['detection_scores'])

    category_index = serve_utils.read_class_names2("coco.names")

    import copy
    image_data_save = copy.deepcopy(image_data)
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_data,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks11'),
        use_normalized_coordinates=False,
        line_thickness=5
    )
    Image.fromarray(image_data).show()
    print((image_data_save==image_data).all())
    return
    #   (63, 19, 19, 85)  reduction factor 注：衰减系数以及步长：32  608/32=19      85 = 80类+1可能性+4个坐标
    #   416 x 416 则为 13*13

    output = np.reshape(output, (-1, 85)) # 这一步处理成 22743*85的维度（63*19*19 =22743， 85 = 80类+1可能性+4个坐标,根据自己数据集改）
    # print(output.shape)

    original_image_size = original_image.shape[:2]
    bboxes = utils.postprocess_boxes(output, original_image_size, img_size, 0.001)  # 这一步是将所有可能的预测信息提取出来，主要是三类：类别，可能性，坐标值。
    bboxes = utils.nms(bboxes, 0.001, method='nms')  # 这一步是 将刚刚提取出来的信息进行筛选，返回最好的预测值，同样是三类。
    image = utils.draw_bbox(original_image, bboxes) # 这一步是把结果画到新图上面。
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    import base64
    print(base64.b64encode(image).decode())
    image = Image.fromarray(image)
    image.show()
    # image.save(output_path)  # 保存图片到本地

object_detect = object_detect(input_path="./cats.jpg", output_path='./demo.jpg',flag=False)