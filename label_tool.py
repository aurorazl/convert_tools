# -*- coding: utf-8 -*-
#!/usr/bin/python
import os
import json
import random
import re
import pyprind
import shutil
import argparse
import textwrap
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from xml.etree.ElementTree import tostring
from pycocotools.coco import COCO
from xml.dom import minidom
from DirectoryUtils import cd
import collections
from pycocotools import mask
from skimage import measure
import numpy as np
from shapely.geometry import Polygon
from PIL import Image
from PIL import ImageDraw
import matplotlib.pyplot as plt
import cv2
import utils
import glob
import requests

filename_pattern = re.compile(r"(\S+)\.(xml|json|jpg|txt)")
image_pattern = re.compile(r"(\S+)\.(jpg)")
gen_pattern = re.compile(r"(\S*?)(\d+)\.(xml|json|txt)")
number_pattern = re.compile(r"(\S*?)(\d+)")
category_map = {1:96,2:97,3:49,4:98,5:99}

def get(root, name):
    vars = root.findall(name)
    return vars

def get_value(root, name):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if len(vars) != 1:
        raise NotImplementedError('The size of %s is supposed to be 1, but is %d.'%(name,len(vars)))
    re = vars[0].text
    try:
        return int(float(re))
    except Exception:
        return re

coco = None
def get_class_number(name):
    number = 0
    if name == "work_uniformm":
        name = "work_uniform"
    if name == "othe_hat":
        name = "other_hat"
    try:
        global coco
        if not coco:
            res = requests.get(url="https://apulis-sz-dev-worker01.sigsus.cn/api/labels")
            coco = json.loads(res.content)["categories"]
    except Exception:
        with open(os.path.join("meta.json"), "r") as f:
            coco = json.load(f)["categories"]
    for one in coco:
        if one.get("name")==name:
            number = int(one.get("id",0))
    # if number==0:
    #     print(name)
    return number

def get_class_name(number,categories=None):
    name = None
    if not categories:
        with open(os.path.join("meta.json"), "r") as f:
            coco = json.load(f)
        categories = coco["categories"]
    for one in categories:
        if one.get("id")==number:
            name = one.get("name").replace(" ","_")
    return name

def get_node_or_create(elem,key,new):
    re = get(elem, key)
    if not re or new:
        child = Element(key)
        elem.append(child)
        return child
    else:
        return re[-1]

def add_xml_element(elem,key,val,new=False):
    if(type(key)==str):
        child = Element(key)
        child.text = str(val)
        elem.append(child)
    elif type(key)==list:
        node = elem
        if(len(key))<2:
            return
        for i in key[:-1]:
            node = get_node_or_create(node,i,new)
        child = Element(key[-1])
        child.text = str(val)
        node.append(child)

def check_path_exit_or_raise_exception(*args,**kwargs):
    for i in args:
        if not os.path.exists(i):
            raise Exception("{} not found!!!".format(i))

map_path = {}
pbar = None
def get_image_name_list(src_path):
    src_list = os.listdir(src_path)
    name_list = []
    for i in src_list:
        res = gen_pattern.match(i)
        if res:
            name_list.append(res.groups()[0]+res.groups()[1])
    print("start to process %s picture"%len(name_list))
    global pbar
    pbar = pyprind.ProgBar(len(name_list),monitor=True)
    return name_list

index=0
def one_voc_format_to_json_format(src_path,file_path,image_id):
    file_path = os.path.join(src_path,file_path)
    tree = ET.parse(file_path)
    root = tree.getroot()
    filename = get_value(root, "filename")
    height = get_value(root, "size/height")
    width = get_value(root, "size/width")
    depth = get_value(root, "size/depth")
    json_dict = {"images": [{"file_name": str(image_id)+'.jpg', "height": height, "width": width, "id": image_id,
                             "license": 2, "coco_url": None, "data_captured": None, "flickr_url": None}],
                 "annotations": []
                 }
    for one in get(root, "object"):
        name = get_value(one, "name")
        pose = get_value(one, "pose")
        xmin = get_value(one, "bndbox/xmin")
        ymin = get_value(one, "bndbox/ymin")
        xmax = get_value(one, "bndbox/xmax")
        ymax = get_value(one, "bndbox/ymax")
        global index
        index +=1
        json_dict["annotations"].append(
            {"segmentation": [], "area": (xmax - xmin) * (ymax - ymin), "iscrowd": 0, "image_id": image_id,
             "bbox": [xmin, ymin, xmax - xmin, ymax - ymin], "category_id": get_class_number(name), "id": index}
        )
    return json_dict

def voc_format_to_json_format(src_path,dir_path):
    image_id_list = []
    utils.mkdirs(os.path.join(dir_path,"images"))
    for image_id in get_image_name_list(src_path):
        image_id_list.append(image_id)
        json_dict = one_voc_format_to_json_format(src_path,image_id+".xml",image_id)
        dir_file = os.path.join(dir_path, "images", str(image_id) + ".json")
        with open(dir_file, "w+") as f:
            f.write(json.dumps(json_dict,indent=4, separators=(',', ':')))
        pbar.update()
    with open(os.path.join(dir_path,"list.json"),"w") as f:
        f.write(json.dumps({"ImgIDs":image_id_list},indent=4, separators=(',', ':')))

def one_json_format_to_voc_format(coco_dict,new_image_name,dir_path,categories=None):
    elem = Element("annotation")
    filename = coco_dict.get("images")[0].get("file_name")
    height = coco_dict.get("images")[0].get("height")
    width = coco_dict.get("images")[0].get("width")
    add_xml_element(elem, "folder", os.path.basename(dir_path))
    add_xml_element(elem, "filename", new_image_name)
    add_xml_element(elem, "path", os.path.join(dir_path,new_image_name))
    add_xml_element(elem, ["source", "database"], "Unkonwn")
    add_xml_element(elem, ["size", "height"], height)
    add_xml_element(elem, ["size", "width"], width)
    add_xml_element(elem, ["size", "depth"], 3)
    add_xml_element(elem, "segmented", 0)
    for one in coco_dict.get("annotations"):
        add_xml_element(elem, ["object", "name"], get_class_name(int(one.get("category_id")),categories), True)
        add_xml_element(elem, ["object", "pose"], "Unspecified")
        add_xml_element(elem, ["object", "truncated"], 0)
        add_xml_element(elem, ["object", "difficult"], 0)
        add_xml_element(elem, ["object", "bndbox", "xmin"], int(one.get("bbox")[0]))
        add_xml_element(elem, ["object", "bndbox", "ymin"], int(one.get("bbox")[1]))
        add_xml_element(elem, ["object", "bndbox", "xmax"], int(one.get("bbox")[2] + one.get("bbox")[0]))
        add_xml_element(elem, ["object", "bndbox", "ymax"], int(one.get("bbox")[3] + one.get("bbox")[1]))
    return elem

def json_format_to_voc_format(src_path,dir_path):
    for image_id in get_image_name_list(src_path):
        file_path = os.path.join(src_path, image_id+".json")
        with open(file_path,"r") as f:
            coco_dict = json.load(f)
        elem = one_json_format_to_voc_format(coco_dict,image_id+".jpg",dir_path)
        dir_file = os.path.join(dir_path, str(image_id) + ".xml")
        with open(dir_file,"w+") as f:
            f.write(minidom.parseString(tostring(elem)).toprettyxml().replace('<?xml version="1.0" ?>\n', ""))
        pbar.update()

def merge_voc_dataset_to_json_dataset(voc_anno_path,voc_image_path,json_path,prefix="",args=None):
    args.anno_before_suffix = "xml" if not args.anno_before_suffix else args.anno_before_suffix
    utils.check_anno_image_number(voc_anno_path,voc_image_path,args)
    image_id_list = []
    json_anno_path = os.path.join(json_path,"images")
    utils.mkdirs(json_anno_path)
    new_image_id_list = utils.gen_image_name_list(voc_anno_path, voc_image_path, json_anno_path, prefix, args)
    pbar = pyprind.ProgBar(len(new_image_id_list), monitor=True, title="writing to file")
    for new_image_id, old_image_id in new_image_id_list.items():
        image_id_list.append(new_image_id)
        json_dict = one_voc_format_to_json_format(voc_anno_path,args.anno_before_prefix + str(old_image_id) + ".xml", args.anno_after_prefix + str(new_image_id))
        dir_file = os.path.join(json_path, "images",args.anno_after_prefix + str(new_image_id) + ".json")
        if args and not args.ignore_image:
            shutil.copyfile(os.path.join(voc_image_path,args.image_before_prefix + str(old_image_id) + ".jpg"),os.path.join(json_anno_path,args.image_after_prefix + str(new_image_id) + ".jpg"))
        with open(dir_file, "w+") as f:
            f.write(json.dumps(json_dict,indent=4, separators=(',', ':')))
        pbar.update()
    if os.path.exists(os.path.join(json_path, "list.json")):
        with open(os.path.join(json_path, "list.json"), "r") as f:
            ImgIDs = json.load(f)["ImgIDs"]
    else:
        ImgIDs = []
    ImgIDs.extend(image_id_list)
    with open(os.path.join(json_path,"list.json"),"w") as f:
        f.write(json.dumps({"ImgIDs":ImgIDs},indent=4, separators=(',', ':')))
    shutil.copyfile("./meta.json",os.path.join(json_path,"meta.json"))

def merge_json_dataset_to_voc_dataset(json_path,voc_anno_path,voc_image_path,prefix="",args=None):
    args.anno_before_suffix = "json" if not args.anno_before_suffix else args.anno_before_suffix
    utils.mkdirs(voc_image_path)
    json_anno_path = os.path.join(json_path, "images")
    image_id_list = utils.gen_image_name_list(json_anno_path, json_anno_path, voc_anno_path, prefix, args)
    pbar = pyprind.ProgBar(len(image_id_list), monitor=True, title="writing to file")
    for new_image_id, old_image_id in image_id_list.items():
        if args and not args.ignore_image:
            shutil.copyfile(os.path.join(json_anno_path,args.image_before_prefix + str(old_image_id) + ".jpg"),os.path.join(voc_image_path,args.image_after_prefix + str(new_image_id) + ".jpg"))
        file_path = os.path.join(json_anno_path,args.anno_before_prefix + str(old_image_id) + "." + args.anno_before_suffix)
        with open(file_path,"r") as f:
            coco_dict = json.load(f)
        elem = one_json_format_to_voc_format(coco_dict, args.image_after_prefix + str(new_image_id)+".jpg",voc_image_path)
        dir_file = os.path.join(voc_anno_path, args.anno_after_prefix + str(new_image_id) + ".xml")
        with open(dir_file,"w+") as f:
            f.write(minidom.parseString(tostring(elem)).toprettyxml().replace('<?xml version="1.0" ?>\n', ""))
        pbar.update()

def get_exists_coco_max_num(coco_output_path,coco,prefix):
    max_num = 0
    old_coco = COCO(coco_output_path)
    coco["info"] = old_coco.dataset.get('info', [])
    coco["licenses"] = old_coco.dataset.get('licenses', [])
    coco["type"] =  old_coco.dataset.get('type', "instance")
    coco["categories"] =  old_coco.dataset.get('categories')
    ImgIDs = list(old_coco.imgs.keys())
    pbar = pyprind.ProgBar(len(ImgIDs), monitor=True, title="counting exist coco specified prefix {} number".format(prefix))
    for ImgID in ImgIDs:
        coco["images"].extend(old_coco.loadImgs([ImgID]))
        global index
        old_anno = old_coco.loadAnns(old_coco.getAnnIds(imgIds=[ImgID]))
        for i in old_anno:
            if index < i["id"]:
                index = i["id"]
        coco["annotations"].extend(old_anno)
        res = number_pattern.match(str(ImgID))
        if (res):
            if res.groups()[0] == prefix:
                num = int(res.groups()[1])
                if num > max_num:
                    max_num = num
        pbar.update()
    return max_num

def merge_voc_dataset_to_coco_dataset(voc_anno_path,voc_image_path,coco_output_path,coco_image_path,prefix="",args=None):
    utils.mkdirs(coco_image_path)
    max_num = 0
    if os.path.exists(coco_output_path):
        coco = {"images": [], "annotations": []}
        max_num = get_exists_coco_max_num(coco_output_path,coco,prefix)
    else:
        with open("meta.json","r") as f:
            coco = json.load(f)
            coco["images"]=[]
            coco["annotations"]=[]
    src_list = os.listdir(voc_anno_path)
    pbar = pyprind.ProgBar(len(src_list),monitor=True,title="converting voc to coco")
    for one in src_list:
        if gen_pattern.match(one):
            image_id = os.path.splitext(one)[0]
            if not prefix and max_num==0:
                new_image_id = image_id
            else:
                max_num += 1
                new_image_id = prefix+str(max_num)
            json_dict = one_voc_format_to_json_format(voc_anno_path,one,new_image_id)
            coco["images"].extend(json_dict["images"])
            coco["annotations"].extend(json_dict["annotations"])
            if args and not args.ignore_image:
                shutil.copyfile(os.path.join(voc_image_path, str(image_id) + ".jpg"),os.path.join(coco_image_path,str(new_image_id)+".jpg"))
            pbar.update()
    with open(coco_output_path, "w") as f:
        f.write(json.dumps(coco, indent=4, separators=(',', ':')))

def get_file_name_from_coco(li,image_id):
    for i in li:
        if i.get("id")==image_id:
            return i.get("file_name")

def merge_coco_to_voc_dataset(coco_file_path,coco_image_path,voc_anno_path,voc_image_path,prefix="",args=None):
    utils.mkdirs(voc_image_path)
    coco = COCO(coco_file_path)
    categories = coco.dataset.get('categories')
    ImgIDs = list(coco.imgs.keys())
    max_num = utils.get_dir_path_max_num(voc_anno_path, prefix,args)
    pbar = pyprind.ProgBar(len(ImgIDs), monitor=True, title="converting coco to voc")
    for ImgID in ImgIDs:
        if not prefix and max_num==0:
            new_image_id = str(ImgID)
        else:
            max_num += 1
            new_image_id = prefix+str(max_num)
        json_dict = {}
        json_dict["images"] = coco.loadImgs([ImgID])
        json_dict["annotations"] = coco.loadAnns(coco.getAnnIds(imgIds=[ImgID]))
        old_image_name = get_file_name_from_coco(json_dict["images"], ImgID)
        elem = one_json_format_to_voc_format(json_dict, new_image_id + ".jpg",voc_image_path,categories)
        dir_file = os.path.join(voc_anno_path, new_image_id + ".xml")
        with open(dir_file, "w+") as f:
            f.write(minidom.parseString(tostring(elem)).toprettyxml().replace('<?xml version="1.0" ?>\n', ""))
        if args and not args.ignore_image:
            shutil.copyfile(os.path.join(coco_image_path,old_image_name),os.path.join(voc_image_path,new_image_id+".jpg"))
        pbar.update()

def merge_coco_to_json_dataset(coco_file_path,coco_image_path,json_path,prefix="",args=None):
    json_anno_path = os.path.join(json_path, "images")
    utils.mkdirs(json_anno_path)
    coco = COCO(coco_file_path)
    meta_json_dict = {
        "info": coco.dataset.get('info', []),
        "licenses": coco.dataset.get('licenses', []),
        "type": coco.dataset.get('type', "instance"),
        "categories": coco.dataset.get('categories')}
    with open(os.path.join(json_path, "meta.json") , "w") as f:
        f.write(json.dumps(meta_json_dict,indent=4, separators=(',', ':')))
    ImgIDs = list(coco.imgs.keys())
    global pbar
    pbar = pyprind.ProgBar(len(ImgIDs),monitor=True,title="converting coco to json")
    max_num = utils.get_dir_path_max_num(json_anno_path, prefix,args)
    if os.path.exists(os.path.join(json_path, "list.json")):
        with open(os.path.join(json_path, "list.json"), "r") as f:
            new_image_id_list = json.load(f)["ImgIDs"]
    else:
        new_image_id_list = []
    for ImgID in ImgIDs:
        json_dict = {}
        json_dict["images"] = coco.loadImgs([ImgID])
        if max_num==0 and not prefix:
            new_image_id = json_dict["images"][0]["file_name"].split(".jpg")[0]
        else:
            max_num += 1
            new_image_id = prefix + str(max_num)
        json_dict["images"][0]["id"] = new_image_id
        new_image_id_list.append(new_image_id)
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[ImgID]))
        if args and args.base_category_num != 0:
            for one_anno in annotations:
                one_anno["category_id"] = int(one_anno["category_id"]) + args.base_category_num
        for one_anno in annotations:
            one_anno["image_id"] = new_image_id
            one_anno["id"] = new_image_id
        if args and args.use_category_mapping:
            for one_anno in annotations:
                one_anno["category_id"] = category_map.get(one_anno["category_id"],one_anno["category_id"])
        json_dict["annotations"] = annotations
        img_path = os.path.join(coco_image_path, json_dict["images"][0]["file_name"])
        if args and not args.ignore_image:
            shutil.copyfile(img_path, os.path.join(json_path, "images", "{}.jpg".format(new_image_id)))
        with open(os.path.join(json_path, "images", "{}.json".format(new_image_id)), "w") as f:
            f.write(json.dumps(json_dict, indent=4, separators=(',', ':')))


        pbar.update()
    with open(os.path.join(json_path, "list.json") , "w") as f:
        f.write(json.dumps({"ImgIDs":new_image_id_list},indent=4, separators=(',', ':')))

def merge_json_to_coco_dataset(json_path,coco_file_path,coco_image_path,prefix="",args=None):
    max_num = 0
    utils.mkdirs(coco_image_path)
    if os.path.exists(coco_file_path):
        coco = {"images": [], "annotations": []}
        max_num = get_exists_coco_max_num(coco_file_path,coco,prefix)
    else:
        with open("meta.json","r") as f:
            coco = json.load(f)
            coco["images"] = []
            coco["annotations"] = []
    with open(os.path.join(json_path, "list.json"), "r") as f:
        ImgIDs = json.load(f)["ImgIDs"]
    global pbar
    pbar = pyprind.ProgBar(len(ImgIDs),monitor=True,title="converting json to coco")
    for ImgID in ImgIDs:
        if not prefix and max_num==0:
            new_image_id = str(ImgID)
        else:
            max_num += 1
            new_image_id = prefix + str(max_num)
        with open(os.path.join(json_path, 'images', "{}.json".format(ImgID)), "r") as f:
            json_dict = json.load(f)
        json_dict["images"][0]["file_name"] = "{}.jpg".format(new_image_id)
        json_dict["images"][0]["id"] = new_image_id
        for i in json_dict["annotations"]:
            i["image_id"] = new_image_id
            global index
            i["id"] = index
            index += 1
        coco["images"].extend(json_dict["images"])
        coco["annotations"].extend(json_dict["annotations"])
        source_path = os.path.join(json_path, 'images', "{}.jpg".format(ImgID))
        if args and not args.ignore_image:
            shutil.copyfile(source_path, os.path.join(coco_image_path, "{}.jpg".format(new_image_id)))
        pbar.update()
    with open(coco_file_path, "w") as f:
        f.write(json.dumps(coco, indent=4, separators=(',', ':')))

def copy_dir_by_percent(src_path,dir_path,percent=0,number=0):
    if not percent and not number:
        shutil.copytree(src_path, dir_path)
    else:
        src_list = os.listdir(src_path)
        total_len = len(src_list)
        target = 0
        if percent:
            target = int(total_len*float(percent))
        elif number:
            target = int(number)
        if target:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            for i in src_list[:target]:
                shutil.copyfile(os.path.join(src_path,i),os.path.join(dir_path,i))

def copy_json_by_percent(src_path,dir_path,percent=0,number=0):
    src_image_path = os.path.join(src_path,"images")
    dir_image_path = os.path.join(dir_path,"images")
    utils.mkdirs(dir_image_path)
    if not percent and not number:
        shutil.copytree(src_path, dir_path)
        shutil.copyfile(os.path.join(src_path, "list.json"), os.path.join(dir_path, "list.json"))
        shutil.copyfile(os.path.join(src_path, "meta.json"), os.path.join(dir_path, "meta.json"))
    else:
        src_list = os.listdir(src_image_path)
        total_len = len(src_list)//2
        target = 0
        current=0
        ImgIDs = []
        if percent:
            target = int(total_len * float(percent))
        elif number:
            target = int(number)
        if target:
            if not os.path.exists(dir_image_path):
                os.makedirs(dir_image_path)
            for i in src_list:
                res = gen_pattern.match(i)
                if(res):
                    current +=1
                    if(current>target):
                        break
                    image_id = res.groups()[0]+res.groups()[1]
                    image_name =image_id +".jpg"
                    shutil.copyfile(os.path.join(src_image_path, i), os.path.join(dir_image_path, i))
                    shutil.copyfile(os.path.join(src_image_path,image_name), os.path.join(dir_image_path, image_name))
                    ImgIDs.append(image_id)
            with open(os.path.join(dir_path, "list.json"), "w") as f:
                f.write(json.dumps({"ImgIDs":ImgIDs},indent=4, separators=(',', ':')))
            shutil.copyfile(os.path.join(src_path, "meta.json"), os.path.join(dir_path, "meta.json"))

def copy_voc_by_percent(src_path,dir_path,percent=0,number=0):
    src_image_path = os.path.join(src_path,"JPEGImages")
    src_anno_path = os.path.join(src_path,"Annotations")
    dir_image_path = os.path.join(dir_path, "JPEGImages")
    dir_anno_path = os.path.join(dir_path, "Annotations")
    if not percent and not number:
        shutil.copytree(src_image_path, dir_image_path)
        shutil.copytree(src_anno_path, dir_anno_path)
    else:
        src_list = os.listdir(src_anno_path)
        total_len = len(src_list)
        target = 0
        current=0
        if percent:
            target = int(total_len * float(percent))
        elif number:
            target = int(number)
        if target:
            pbar = pyprind.ProgBar(len(src_list) if target > len(src_list) else target, monitor=True, title="copy coco")
            if not os.path.exists(dir_image_path):
                os.makedirs(dir_image_path)
            if not os.path.exists(dir_anno_path):
                os.makedirs(dir_anno_path)
            for i in src_list:
                res = gen_pattern.match(i)
                if(res):
                    current +=1
                    if(current>target):
                        break
                    image_id = res.groups()[0]+res.groups()[1]
                    image_name =image_id +".jpg"
                    anno_name =image_id +".xml"
                    shutil.copyfile(os.path.join(src_anno_path,anno_name), os.path.join(dir_anno_path, anno_name))
                    shutil.copyfile(os.path.join(src_image_path,image_name), os.path.join(dir_image_path, image_name))
            pbar.update()

def copy_coco_by_percent(from_file_path,from_image_path,to_file_path,to_image_path,percent=0,number=0,args=None):
    if not percent and not number:
        shutil.copytree(from_image_path, to_image_path)
        shutil.copyfile(from_file_path,to_file_path)
    else:
        utils.mkdirs(to_image_path)
        coco = COCO(from_file_path)
        new_coco ={
            "info": coco.dataset.get('info', []),
            "licenses": coco.dataset.get('licenses', []),
            "type": coco.dataset.get('type', "instance"),
            "categories": coco.dataset.get('categories'),
            "images":[],
            "annotations":[]
        }
        ImgIDs = list(coco.imgs.keys())
        total_len = len(ImgIDs) // 2
        target = 0
        current = 0
        if percent:
            target = int(total_len * float(percent))
        elif number:
            target = int(number)
        if target:
            pbar = pyprind.ProgBar(len(ImgIDs) if target>len(ImgIDs)else target, monitor=True, title="copy coco")
            for ImgID in ImgIDs:
                new_images = coco.loadImgs([ImgID])
                new_anno = coco.loadAnns(coco.getAnnIds(imgIds=[ImgID]))
                new_coco["images"].extend(new_images)
                new_coco["annotations"].extend(new_anno)
                shutil.copyfile(os.path.join(from_image_path,new_images[0]["file_name"]), os.path.join(to_image_path,new_images[0]["file_name"]))
                current +=1
                if current>=target:
                    break
                pbar.update()
            with open(to_file_path, "w") as f:
                f.write(json.dumps(new_coco,indent=4, separators=(',', ':')))

def remove_json_by_prefix(json_path,prefix=""):
    json_anno_path = os.path.join(json_path,"images")
    with open(os.path.join(json_path, "list.json"), "r") as f:
        ImgIDs = json.load(f)["ImgIDs"]
    new_images = ImgIDs.copy()
    pbar = pyprind.ProgBar(len(ImgIDs), monitor=True, title="removing coco by prefix {}".format(prefix))
    for ImgID in ImgIDs:
        if str(ImgID).startswith(prefix):
            new_images.remove(ImgID)
            os.remove(os.path.join(json_anno_path,"{}.jpg".format(ImgID)))
            os.remove(os.path.join(json_anno_path,"{}.json".format(ImgID)))
        pbar.update()
    with open(os.path.join(json_path, "list.json") , "w") as f:
        f.write(json.dumps({"ImgIDs":new_images},indent=4, separators=(',', ':')))

def remove_voc_by_prefix(voc_path,prefix=""):
    voc_anno_path = os.path.join(voc_path,"Annotations")
    voc_image_path = os.path.join(voc_path,"JPEGImages")
    src_list = os.listdir(voc_anno_path)
    pbar = pyprind.ProgBar(len(src_list), monitor=True, title="removing coco by prefix {}".format(prefix))
    for i in src_list:
        if i.startswith(prefix):
            os.remove(os.path.join(voc_anno_path,i))
            res = gen_pattern.match(i)
            if res:
                image_id = res.groups()[0]+res.groups()[1]
                os.remove(os.path.join(voc_image_path, "{}.jpg".format(image_id)))
        pbar.update()

def remove_coco_by_prefix(coco_file_path,coco_image_path,prefix=""):
    coco = {"images": [], "annotations": []}
    old_coco = COCO(coco_file_path)
    coco["info"] = old_coco.dataset.get('info', [])
    coco["licenses"] = old_coco.dataset.get('licenses', [])
    coco["type"] = old_coco.dataset.get('type', "instance")
    coco["categories"] = old_coco.dataset.get('categories')
    ImgIDs = list(old_coco.imgs.keys())
    pbar = pyprind.ProgBar(len(ImgIDs), monitor=True, title="removing coco by prefix {}".format(prefix))
    for ImgID in ImgIDs:
        if not str(ImgID).startswith(prefix):
            coco["images"].extend(old_coco.loadImgs([ImgID]))
        else:
            os.remove(os.path.join(coco_image_path, "{}.jpg".format(ImgID)))
        pbar.update()
    with open(coco_file_path, "w") as f:
        f.write(json.dumps(coco, indent=4, separators=(',', ':')))

def annotations_to_voc_xml_file(annotations,width,height,outputfilepath,override=False,privacy_mode=True):
    dir_path = os.path.dirname(outputfilepath)
    xml_name = os.path.basename(outputfilepath)
    image_name = os.path.splitext(xml_name)[0] + ".jpg"
    image_path = os.path.join(dir_path, image_name)
    image_folder = os.path.join(dir_path, 'JPEGImages')
    if privacy_mode:
        image_path = image_name
        image_folder = 'folder'

    elem = Element("annotation")
    add_xml_element(elem, "folder", image_folder)
    add_xml_element(elem, "filename", image_name)
    add_xml_element(elem, "path", image_path)
    add_xml_element(elem, ["source", "database"], "Unkonwn")
    add_xml_element(elem, ["size", "height"], height)
    add_xml_element(elem, ["size", "width"], width)
    add_xml_element(elem, ["size", "depth"], 3)
    add_xml_element(elem, "segmented", 0)
    for annotaton in annotations:
        label,xmin,ymin,xmax,ymax = annotaton
        add_xml_element(elem, ["object", "name"], label, True)
        add_xml_element(elem, ["object", "pose"], "Unspecified")
        add_xml_element(elem, ["object", "truncated"], 0)
        add_xml_element(elem, ["object", "difficult"], 0)
        add_xml_element(elem, ["object", "bndbox", "xmin"], xmin)
        add_xml_element(elem, ["object", "bndbox", "ymin"], ymin)
        add_xml_element(elem, ["object", "bndbox", "xmax"], xmax)
        add_xml_element(elem, ["object", "bndbox", "ymax"], ymax)
    if os.path.exists(outputfilepath):
        if override:
            with open(outputfilepath, "w+") as f:
                f.write(minidom.parseString(tostring(elem)).toprettyxml().replace('<?xml version="1.0" ?>\n', ""))
    else:
        os.makedirs(dir_path,exist_ok=True)
        with open(outputfilepath, "w+") as f:
            f.write(minidom.parseString(tostring(elem)).toprettyxml().replace('<?xml version="1.0" ?>\n', ""))

def count_voc_per_class_and_bbox_numbers(voc_path,prefix=""):
    dir_path = os.path.join(voc_path,"Annotations")
    train_path = os.path.join(voc_path,"ImageSets","Main","train.txt")
    val_path = os.path.join(voc_path,"ImageSets","Main","val.txt")
    test_path = os.path.join(voc_path,"ImageSets","Main","test.txt")
    if os.path.exists(train_path):
        with open(train_path,"r") as f:
            tmp=[]
            for i in f:
                tmp.append(i.strip()+".xml")
        count_file_list(tmp,dir_path,prefix,"counting voc train set")
    if os.path.exists(val_path):
        with open(val_path,"r") as f:
            tmp=[]
            for i in f:
                tmp.append(i.strip()+".xml")
        count_file_list(tmp,dir_path,prefix,"counting voc val set")
    if os.path.exists(test_path):
        with open(test_path,"r") as f:
            tmp=[]
            for i in f:
                tmp.append(i.strip()+".xml")
        count_file_list(tmp,dir_path,prefix,"counting voc test set")
    if not os.path.exists(train_path) and not os.path.exists(val_path) and not os.path.exists(test_path):
        dir_list = os.listdir(dir_path)
        count_file_list(dir_list, dir_path, prefix,"counting voc all set")

def count_file_list(file_name_list,dir_path,prefix,title):
    details = {}
    global pbar
    total_count = 0
    pbar = pyprind.ProgBar(len(file_name_list),title=title)
    for i in file_name_list:
        res = gen_pattern.match(i)
        if res:
            if prefix and  res.groups()[0] != prefix:
                pbar.update()
                continue
            total_count += 1
            json_dict = one_voc_format_to_json_format(dir_path,i, res.groups()[0]+res.groups()[1])
            for one in json_dict.get("annotations"):
                details.setdefault(one["category_id"], {"image_counts":set(), "bbox_count": 0})
                details[one["category_id"]]["bbox_count"] += 1
                details[one["category_id"]]["image_counts"].add(one["image_id"])
        pbar.update()
    for k,v in details.items():
        details[k]["image_counts"]=len(details[k]["image_counts"])
    for j in sorted(details.items(),key=lambda x:x[0]):
        print('{:<20s} {}'.format(get_class_name(j[0]),json.dumps(j[1])))
    print("total images count:",total_count)

def count_json_per_class_and_bbox_numbers(json_path,prefix=""):
    dir_path = os.path.join(json_path,"images")
    dir_list = os.listdir(dir_path)
    details = {}
    global pbar
    pbar = pyprind.ProgBar(len(dir_list),title="counting json")
    for i in dir_list:
        res = gen_pattern.match(i)
        if res:
            if prefix and res.groups()[0] != prefix:
                pbar.update()
                continue
            with open(os.path.join(dir_path,i),'r') as f:
                json_dict = json.load(f)
            for one in json_dict.get("annotations"):
                details.setdefault(one["category_id"], {"image_counts":set(), "bbox_count": 0})
                details[one["category_id"]]["bbox_count"] += 1
                details[one["category_id"]]["image_counts"].add(one["image_id"])
        pbar.update()
    for k,v in details.items():
        details[k]["image_counts"]=len(details[k]["image_counts"])
    for j in sorted(details.items(),key=lambda x:x[0]):
        print('{:<20s} {}'.format(get_class_name(j[0]),json.dumps(j[1])))

def count_coco_per_class_and_bbox_numbers(coco_file_path,prefix=""):
    details = {}
    global pbar
    coco = COCO(coco_file_path)
    ImgIDs = list(coco.imgs.keys())
    global pbar
    pbar = pyprind.ProgBar(len(ImgIDs),title="counting coco")
    for ImgID in ImgIDs:
        json_dict = {}
        json_dict["images"] = coco.loadImgs([ImgID])
        json_dict["annotations"] = coco.loadAnns(coco.getAnnIds(imgIds=[ImgID]))
        image_name = get_file_name_from_coco(json_dict["images"], ImgID)
        res = gen_pattern.match(image_name)
        if res:
            if prefix and res.groups()[0] != prefix:
                pbar.update()
                continue
        for one in json_dict.get("annotations"):
            details.setdefault(one["category_id"], {"image_counts":set(), "bbox_count": 0})
            details[one["category_id"]]["bbox_count"] += 1
            details[one["category_id"]]["image_counts"].add(one["image_id"])
        pbar.update()
    for k,v in details.items():
        details[k]["image_counts"]=len(details[k]["image_counts"])
    for j in sorted(details.items(),key=lambda x:x[0]):
        print('{:<20s} {}'.format(get_class_name(j[0]),json.dumps(j[1])))


def remove_image_zero_prefix(image_path):
    if not os.path.exists(image_path):
        return
    pattern = re.compile(r"0*([1-9]\d*.jpg)")
    with cd(image_path):
        global pbar
        dir_list = os.listdir(image_path)
        pbar = pyprind.ProgBar(len(dir_list), title="remving 0 prefix")
        for one in dir_list:
            ret = pattern.match(one)
            if ret:
                new_image_path = ret.groups()[0]
                os.rename(one,new_image_path)
            pbar.update()

def generate_commit_json(json_path,userId,base_category_num=0):
    with open(os.path.join(json_path, "list.json"), "r") as f:
        ImgIDs = json.load(f)["ImgIDs"]
    global pbar
    commit_json = collections.defaultdict(lambda :{"status":"committed","userId":userId,"categoryIds":[]})
    pbar = pyprind.ProgBar(len(ImgIDs), monitor=True, title="generate commit json")
    json_file_path = os.path.join(json_path,"images")
    for ImgID in ImgIDs:
        with open(os.path.join(json_file_path, "{}.json".format(ImgID)), "r") as f:
            json_dict = json.load(f)
        for i in json_dict["annotations"]:
            if int(i["category_id"])+base_category_num not in commit_json[i["image_id"]]["categoryIds"]:
                commit_json[i["image_id"]]["categoryIds"].append(int(i["category_id"])+base_category_num)
        pbar.update()
    with open(os.path.join(json_path, "commit.json"), "w") as f:
        f.write(json.dumps(commit_json, indent=4, separators=(',', ':')))

def check_coco_image_whether_duplicate(coco_file_path):
    coco = COCO(coco_file_path)
    ImgIDs = list(coco.imgs.keys())
    global pbar
    pbar = pyprind.ProgBar(len(ImgIDs), monitor=True, title="generate commit json")
    already = set()
    for one in ImgIDs:
        if one in already:
            print("warning! {} already exists!".format(one))
        already.add(one)
        pbar.update()

def compress_rle_to_polygon(rle_dict,simply=True):
    contours = measure.find_contours(mask.decode(rle_dict), 0.5)
    # contours,_=cv2.findContours(mask.decode(rle_dict), cv2.RETR_TREE,
    #                             cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        if simply:
            ret = simplify_contour_to_polygon_by_shapely(contour)
            if ret:
                segmentation.append(ret)
        else:
            contour = np.flip(contour, axis=1)
            segmentation.append(contour.ravel().tolist())
    return segmentation[0] if segmentation else segmentation

def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def simplify_contour_to_polygon_by_skimage(contour):
    """
    使用skimage.measure来减少点的数量
    """
    contour = close_contour(contour)
    contour = measure.approximate_polygon(contour, 1)
    contour = np.flip(contour, axis=1)
    segmentation = contour.ravel().tolist()
    return segmentation

def simplify_contour_to_polygon_by_shapely(contour):
    """
    使用shapely来减少点的数量
    """
    contour = np.flip(contour, axis=1)
    poly = Polygon(contour)
    poly = poly.simplify(1.0, preserve_topology=False)
    segmentation = None
    if poly.exterior:
        segmentation = np.array(poly.exterior.coords).ravel().tolist()
    return segmentation

def show_segmentation_on_picture(image_path,segmentation,image_out_path):
    """
    使用pillow将一个标注信息画在图上
    """
    img = Image.open(image_path).convert('RGBA')
    img2 = img.copy()
    draw = ImageDraw.Draw(img2)
    draw.polygon(segmentation, fill="red")
    img3 = Image.blend(img, img2, 0.5)
    img3.save(image_out_path)

def draw_segmentation_point(segmentation1,segmentation2):
    """
    使用matplotlib描出两个标注信息的点
    """
    axis = [(segmentation1[x], segmentation1[x + 1]) for x in range(0, len(segmentation1) - 1, 2)]
    x, y = zip(*axis)
    plt.plot(x, y)
    axis = [(segmentation2[x], segmentation2[x + 1]) for x in range(0, len(segmentation2) - 1, 2)]
    x, y = zip(*axis)
    plt.plot(x, y)
    plt.show()

def show_segmentation_on_picture_by_opencv(image_path,segmentation,segmentation_simply):
    """
    使用cv2将两个标注信息画在图上
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    axis = [(segmentation[x], segmentation[x + 1]) for x in range(0, len(segmentation) - 1, 2)]
    cv2.polylines(img, [np.array([axis], np.int32)], True, (0, 255, 0), thickness=1)
    axis_simply = [(segmentation_simply[x], segmentation_simply[x + 1]) for x in range(0, len(segmentation_simply) - 1, 2)]
    cv2.polylines(axis_simply, [np.array([axis], np.int32)], True, (255, 255, 255), thickness=0.1)
    cv2.imshow('image', img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

def module_predict_segmentation_list_to_json(list_file_path,json_path,base_category_num=0):
    """
    将RLE转为polygon，可指定是否simplify点的数量
    """
    with open(list_file_path, "r") as f:
        segmentation_list = json.load(f)
    json_anno_path = os.path.join(json_path, "images")
    if not os.path.exists(json_anno_path):
        os.makedirs(json_anno_path)
    json_dict = collections.defaultdict(lambda : {"images":{},"annotations":[]})
    global pbar
    pbar = pyprind.ProgBar(len(segmentation_list), monitor=True, title="module_predict_segmentation_list_to_json")
    for one in segmentation_list:
        json_dict[one["image_id"]]["images"] = [{"file_name":str(one["image_id"])+".jpg","id":one["image_id"],"height":one["segmentation"]["size"][1],"width":one["segmentation"]["size"][0]}]
        anno_dict = {"segmentation":compress_rle_to_polygon(one["segmentation"]),"area":int(mask.area(one["segmentation"])),"iscrowd":0,
                                                          "image_id":one["image_id"],"bbox":one["bbox"],
                                                          "category_id":category_map.get(int(one["category_id"])+base_category_num,int(one["category_id"])+base_category_num),
                                                          "id":one["image_id"]
                                                          }
        if "score" in one:
            anno_dict["score"] = one["score"]
        json_dict[one["image_id"]]["annotations"].append(anno_dict)
        pbar.update()
    pbar = pyprind.ProgBar(len(json_dict), monitor=True, title="writing to file")
    for image_id,di in json_dict.items():
        with open(os.path.join(json_path, "images", "{}.json".format(image_id)), "w") as f:
            f.write(json.dumps(di, indent=4, separators=(',', ':')))
        pbar.update()

def generate_image_id_list_for_new_datasets(image_path,out_path):
    src_list = os.listdir(image_path)
    new_image_id_list = []
    for filename in src_list:
        res = image_pattern.match(filename)
        if res:
            image_id = res.groups()[0]
            new_image_id_list.append(image_id)
    with open(os.path.join(out_path, "list.json"), "w") as f:
        f.write(json.dumps({"ImgIDs": new_image_id_list}, indent=4, separators=(',', ':')))

def calculate_dataset_per_category_iou():
    pass

def merge_ocr_to_json(ocr_anno_path,ocr_image_path,json_path,prefix="",args=None):
    args.anno_before_prefix = "gt_img_" if not args.anno_before_prefix else args.anno_before_prefix
    args.anno_after_prefix = "img_" if not args.anno_after_prefix else args.anno_after_prefix
    args.image_before_prefix = "img_" if not args.image_before_prefix else args.image_before_prefix
    args.image_after_prefix = "img_" if not args.image_after_prefix else args.image_after_prefix
    args.anno_before_suffix = "txt" if not args.anno_before_suffix else args.anno_before_suffix
    json_anno_path = os.path.join(json_path, "images")
    utils.mkdirs(json_anno_path)
    utils.check_anno_image_number(ocr_anno_path,ocr_image_path,args)
    image_id_list = []
    name_dict = utils.gen_image_name_list(ocr_anno_path, ocr_image_path, json_anno_path,prefix,args)
    pbar = pyprind.ProgBar(len(name_dict), monitor=True, title="writing to file")
    for new_image_id,old_image_id in name_dict.items():
        image_id_list.append(args.anno_after_prefix + str(new_image_id))
        json_dict = {"images": [{"file_name": args.image_after_prefix + str(new_image_id) + '.jpg', "height": -1, "width": -1, "id": args.anno_after_prefix + str(new_image_id),
                                 "license": 2, "coco_url": None, "data_captured": None, "flickr_url": None}],
                     "annotations": []
                     }
        boxes,text_tags = utils.get_ocr_annotation(os.path.join(ocr_anno_path, "{}{}.{}".format(args.anno_before_prefix, str(old_image_id), args.anno_before_suffix)))
        index = 0
        for box,text_tag in zip(boxes,text_tags):
            index += 1
            json_dict["annotations"].append(
                {"segmentation": box, "area": utils.calculate_quadrangle_area(box), "iscrowd": 0, "image_id": args.image_after_prefix + str(new_image_id),
                 "bbox": [], "category_id": 100, "id": index,"text":text_tag}
            )
        dir_file = os.path.join(json_path, "images",args.anno_after_prefix + str(new_image_id) + ".json")
        if args and not args.ignore_image:
            shutil.copyfile(os.path.join(ocr_image_path, args.image_before_prefix + str(old_image_id) + ".jpg"),os.path.join(json_anno_path,args.image_after_prefix +  str(new_image_id) + ".jpg"))
        with open(dir_file, "w+") as f:
            f.write(json.dumps(json_dict, indent=4, separators=(',', ':')))
        pbar.update()
    if os.path.exists(os.path.join(json_path, "list.json")):
        with open(os.path.join(json_path, "list.json"), "r") as f:
            ImgIDs = json.load(f)["ImgIDs"]
    else:
        ImgIDs = []
    ImgIDs.extend(image_id_list)
    with open(os.path.join(json_path, "list.json"), "w") as f:
        f.write(json.dumps({"ImgIDs": ImgIDs}, indent=4, separators=(',', ':')))

def merge_json_to_ocr(json_path,ocr_out_path,prefix="",args=None):
    args.anno_before_prefix = "img_" if not args.anno_before_prefix else args.anno_before_prefix
    args.anno_after_prefix = "img_" if not args.anno_after_prefix else args.anno_after_prefix
    args.image_before_prefix = "img_" if not args.image_before_prefix else args.image_before_prefix
    args.image_after_prefix = "img_" if not args.image_after_prefix else args.image_after_prefix
    args.anno_before_suffix = "json" if not args.anno_before_suffix else args.anno_before_suffix
    ocr_anno_out_path = os.path.join(ocr_out_path,"gt")
    ocr_image_out_path = os.path.join(ocr_out_path,"img")
    utils.mkdirs(ocr_anno_out_path)
    utils.mkdirs(ocr_image_out_path)
    json_anno_path = os.path.join(json_path, "images")
    image_id_list = utils.gen_image_name_list(json_anno_path, json_anno_path, ocr_anno_out_path, prefix, args)
    pbar = pyprind.ProgBar(len(image_id_list), monitor=True, title="writing to file")
    for new_image_id,old_image_id in image_id_list.items():
        file_path = os.path.join(json_anno_path,"{}{}.json".format(args.anno_before_prefix,old_image_id))
        with open(file_path, "r") as f:
            coco_dict = json.load(f)
        bboxs = []
        for one in coco_dict["annotations"]:
            one["segmentation"].append(one["text"])
            bboxs.append(one["segmentation"])
        dir_file = os.path.join(ocr_anno_out_path,"{}{}.txt".format(args.anno_after_prefix,new_image_id))
        if args and not args.ignore_image:
            shutil.copyfile(os.path.join(json_anno_path,args.image_before_prefix + str(old_image_id) + ".jpg"),os.path.join(ocr_image_out_path,args.image_after_prefix + str(new_image_id) + ".jpg"))
        with open(dir_file, "w+",encoding="utf-8") as f:
            f.write(utils.convert_ocr_annotation_list_to_str(bboxs))
        pbar.update()

def find_coco_dataset_category_ids(coco_anno_file_path):
    category_ids = set()
    coco = COCO(coco_anno_file_path)
    categories = coco.dataset.get('categories')
    ImgIDs = list(coco.imgs.keys())
    pbar = pyprind.ProgBar(len(ImgIDs), monitor=True, title="counting coco")
    for ImgID in ImgIDs:
        anno = coco.loadAnns(coco.getAnnIds(imgIds=[ImgID]))
        for one_anno in anno:
            category_ids.add(one_anno["category_id"])
        pbar.update()
    print(category_ids)

def find_voc_dataset_category_ids(voc_anno_path):
    src_list = glob.glob(os.path.join(voc_anno_path,"*.xml"))
    category_ids = {}
    pbar = pyprind.ProgBar(len(src_list), monitor=True, title="counting voc")
    for file in src_list:
        tree = ET.parse(file)
        root = tree.getroot()
        for one in get(root, "object"):
            name = get_value(one, "name")
            category_ids[name] = get_class_number(name)
        pbar.update()
    print(category_ids)

def find_json_dataset_category_ids(json_anno_path):
    src_list = glob.glob(os.path.join(json_anno_path, "*.json"))
    category_ids = set()
    pbar = pyprind.ProgBar(len(src_list), monitor=True, title="counting json")
    for file in src_list:
        with open(file, "r") as f:
            json_dict = json.load(f)
        for one in json_dict["annotations"]:
            category_ids.add(one["category_id"])
        pbar.update()
    print(category_ids)

def run_command(args, command, nargs, parser):
    if command == "json-to-voc":
        if len(nargs) != 2:
            parser.print_help()
            print("json-to-voc [json_dir] [voc_dir]")
        else:
            json_format_to_voc_format(nargs[0], nargs[1])
    elif command == "voc-to-json":
        if len(nargs) != 2:
            parser.print_help()
            print("voc-to-json [voc_dir] [json_dir]")
        else:
            voc_format_to_json_format(nargs[0], nargs[1])
    elif command == "remove-json":
        if len(nargs) != 1:
            parser.print_help()
            print("\n [--prefix xxx] remove-json [json_dir]")
        else:
            remove_json_by_prefix(nargs[0], prefix=args.prefix)
    elif command == "remove-voc":
        if len(nargs) != 1:
            parser.print_help()
            print("\n [--prefix xxx] remove-voc [voc_dir]")
        else:
            remove_voc_by_prefix(nargs[0], prefix=args.prefix)
    elif command == "remove-coco":
        if len(nargs) != 2:
            parser.print_help()
            print("\n [--prefix xxx] remove-coco [coco_file_path] [coco_image_path]")
        else:
            remove_coco_by_prefix(nargs[0],nargs[1], prefix=args.prefix)
    elif command == "copy":
        if len(nargs) != 2:
            parser.print_help()
            print("\n [--percent 0.1] [--number 100] copy [from_path] [to_path]")
        else:
            copy_dir_by_percent(nargs[0], nargs[1],percent=args.percent,number=args.number)
    elif command == "copy-json":
        if len(nargs) != 2:
            parser.print_help()
            print("\n [--percent 0.1] [--number 100] copy-json [from_path] [to_path]")
        else:
            copy_json_by_percent(nargs[0], nargs[1],percent=args.percent,number=args.number)
    elif command == "copy-voc":
        if len(nargs) != 2:
            parser.print_help()
            print("\n [--percent 0.1] [--number 100] copy-json [from_path] [to_path]")
        else:
            copy_voc_by_percent(nargs[0], nargs[1],percent=args.percent,number=args.number)
    elif command == "copy-coco":
        if len(nargs) != 4:
            parser.print_help()
            print("\n [--percent 0.1] [--number 100] copy-coco [from_file_path] [from_image_path] [to_file_path] [to_image_path]")
        else:
            copy_coco_by_percent(nargs[0], nargs[1], nargs[2], nargs[3],percent=args.percent,number=args.number,args=args)
    elif command == "merge-voc-to-json":
        if len(nargs) != 2:
            parser.print_help()
            print("\n [--prefix xxx] merge-voc-to-json [voc_path] [json_path]")
        else:
            check_path_exit_or_raise_exception(os.path.join(nargs[0],"Annotations"),os.path.join(nargs[0],"JPEGImages"))
            merge_voc_dataset_to_json_dataset(os.path.join(nargs[0],"Annotations"),os.path.join(nargs[0],"JPEGImages"), nargs[1],prefix=args.prefix,args=args)
    elif command == "merge-voc-to-coco":
        if len(nargs) != 3:
            parser.print_help()
            print("\n [--prefix xxx] merge-voc-to-coco [voc_path] [coco_output_file_path] [coco_img_path]")
        else:
            check_path_exit_or_raise_exception(os.path.join(nargs[0],"Annotations"), os.path.join(nargs[0],"JPEGImages"))
            merge_voc_dataset_to_coco_dataset(os.path.join(nargs[0],"Annotations"), os.path.join(nargs[0],"JPEGImages"),nargs[1],nargs[2],prefix=args.prefix,args=args)
    elif command == "merge-coco-to-voc":
        if len(nargs) != 3:
            parser.print_help()
            print("\n [--prefix xxx] merge-coco-to-voc [coco_file_path] [coco_image_path] [voc_path]")
        else:
            check_path_exit_or_raise_exception(nargs[0],nargs[1])
            merge_coco_to_voc_dataset(nargs[0],nargs[1],os.path.join(nargs[2],"Annotations"),os.path.join(nargs[2],"JPEGImages"),prefix=args.prefix,args=args)
    elif command == "merge-json-to-voc":
        if len(nargs) != 2:
            parser.print_help()
            print("\n [--prefix xxx] merge-json-to-voc [json_path] [voc_path]")
        else:
            check_path_exit_or_raise_exception(nargs[0])
            merge_json_dataset_to_voc_dataset(nargs[0],os.path.join(nargs[1],"Annotations"),os.path.join(nargs[1],"JPEGImages"),prefix=args.prefix,args=args)
    elif command == "merge-coco-to-json":
        if len(nargs) != 3:
            parser.print_help()
            print("\n [--prefix xxx] merge-coco-to-json [coco_file_path] [coco_image_path] [json_path]")
        else:
            check_path_exit_or_raise_exception(nargs[0],nargs[1])
            merge_coco_to_json_dataset(nargs[0],nargs[1],nargs[2],prefix=args.prefix,args=args)
    elif command == "merge-json-to-coco":
        if len(nargs) != 3:
            parser.print_help()
            print("\n [--prefix xxx] merge-json-to-coco [json_path] [coco_file_path] [coco_image_path]\n")
        else:
            check_path_exit_or_raise_exception(nargs[0])
            merge_json_to_coco_dataset(nargs[0],nargs[1],nargs[2],prefix=args.prefix,args=args)
    elif command == "count-voc":
        if len(nargs) != 1:
            parser.print_help()
            print("\n [--prefix xxx] count-voc [voc_path]\n")
        else:
            count_voc_per_class_and_bbox_numbers(nargs[0],prefix=args.prefix)
    elif command == "count-json":
        if len(nargs) != 1:
            parser.print_help()
            print("\n [--prefix xxx] count-json [json_path]\n")
        else:
            count_json_per_class_and_bbox_numbers(nargs[0],prefix=args.prefix)
    elif command == "count-coco":
        if len(nargs) != 1:
            parser.print_help()
            print("\n [--prefix xxx] count-coco [coco_file_path]\n")
        else:
            count_coco_per_class_and_bbox_numbers(nargs[0],prefix=args.prefix)
    elif command == "remove_image_zero_prefix":
        if len(nargs) != 1:
            parser.print_help()
            print("\n remove_image_zero_prefix [image_path]\n")
        else:
            remove_image_zero_prefix(nargs[0])
    elif command == "generate_commit_json":
        if len(nargs)!=2:
            parser.print_help()
            print("\n generate_commit_json [json_path] [userId]\n")
        else:
            generate_commit_json(nargs[0],nargs[1],args.base_category_num)
    elif command == "check_coco_image_whether_duplicate":
        if len(nargs)!=1:
            parser.print_help()
            print("\n check_coco_image_whether_duplicate [cooc_file_path]\n")
        else:
            check_coco_image_whether_duplicate(nargs[0])
    elif command == "module_predict_segmentation_list_to_json":
        if len(nargs)!=2:
            parser.print_help()
            print("\n module_predict_segmentation_list_to_json [list_file_path] [json_out_path]\n")
        else:
            module_predict_segmentation_list_to_json(nargs[0],nargs[1],args.base_category_num)
    elif command == "generate_image_id_list_for_new_datasets":
        if len(nargs)!=2:
            parser.print_help()
            print("\n generate_image_id_list_for_new_datasets [image_path] [json_out_path]\n")
        else:
            generate_image_id_list_for_new_datasets(nargs[0],nargs[1])
    elif command == "merge-ocr-to-json":
        if len(nargs)!=3:
            parser.print_help()
            print("\n merge-ocr-to-json [ocr_anno_path] [ocr_image_path] [json_path]\n")
        else:
            merge_ocr_to_json(nargs[0],nargs[1],nargs[2],args.prefix,args)
    elif command == "merge-json-to-ocr":
        if len(nargs)!=2:
            parser.print_help()
            print("\n merge-json-to-ocr [json_path] [ocr_out_path] \n")
        else:
            merge_json_to_ocr(nargs[0],nargs[1],args.prefix,args)
    elif command == "find_coco_dataset_category_ids":
        if len(nargs)!=1:
            parser.print_help()
            print("\n find_coco_dataset_category_ids [coco_anno_path] \n")
        else:
            find_coco_dataset_category_ids(nargs[0])
    elif command == "find_voc_dataset_category_ids":
        if len(nargs)!=1:
            parser.print_help()
            print("\n find_voc_dataset_category_ids [voc_anno_path] \n")
        else:
            find_voc_dataset_category_ids(nargs[0])
    elif command == "find_json_dataset_category_ids":
        if len(nargs)!=1:
            parser.print_help()
            print("\n find_json_dataset_category_ids [json_anno_path] \n")
        else:
            find_json_dataset_category_ids(nargs[0])
    else:
        parser.print_help()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='label_tool.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
    coco format & voc format convert.

    Command:
        copy 
            [from_path] [to_path] : copy on percent
        copy-json
            [from_path] [to_path] : copy json sample on percent
        copy-voc
            [from_path] [to_path] : copy voc sample on percent
        copy-coco
            [from_file_path] [from_image_path] [to_file_path] [to_image_path] : copy coco sample on percent
        remove-voc
            [voc_dir] : remove specify prefix part
        remove-json
            [json_dir] : remove specify prefix part
        remove-coco 
            [coco_file_path] [coco_image_path] : remove specify prefix part
        json-to-voc 
            [coco_dir] [voc_dir] : convert json annotatino to voc annotation
        voc-to-json
            [voc_dir] [coco_dir] : convert voc annotatino to json annotation
        merge-voc-to-json
            [voc_path] [json_path]: merge voc annotatino into json annotation
        merge-json-to-voc
            [json_path] [voc_path]: merge json annotatino into voc annotation
        merge-voc-to-coco
            [voc_path] [coco_output_file_path] [coco_img_path]:merge voc to coco format
        merge-coco-to-voc 
            [coco_file_path] [coco_image_path] [voc_path]:merge coco to voc format
        merge-coco-to-json 
            [coco_file_path] [coco_image_path] [json_path]:merge coco to json format
        merge-json-to-coco 
            [json_path] [coco_file_path] [coco_image_path]:merge json to coco format
      '''))
    parser.add_argument("--prefix", "-p",default="",help="generate file'prefix",action="store")
    parser.add_argument("--anno-before-prefix","-abp", default="",help="before anno prefix",action="store")
    parser.add_argument("--anno-after-prefix", "-aap",default="",help="after anno prefix",action="store")
    parser.add_argument("--anno-before-suffix","-abs", default="",help="before anno suffix",action="store")
    parser.add_argument("--anno-after-suffix", "-aas",default="",help="after anno suffix",action="store")
    parser.add_argument("--image-before-prefix","-ibp", default="",help="before image prefix",action="store")
    parser.add_argument("--image-after-prefix","-iap", default="",help="after image prefix",action="store")
    parser.add_argument("--image-before-suffix","-ibs", default="jpg",help="before image suffix",action="store")
    parser.add_argument("--image-after-suffix", "-ias",default="jpg",help="after image suffix",action="store")
    parser.add_argument("--ignore-image",default=False,help="dont copy image",action="store_true")
    parser.add_argument("--percent",default=0,help="copy file percent",action="store")
    parser.add_argument("--number","-n",default=0,help="copy file numbers",action="store",type=int)
    parser.add_argument("--base_category_num", "-cn",default=0,help="base_category_num",action="store",type=int)
    parser.add_argument("command",help="See above for the list of valid command")
    parser.add_argument('nargs', nargs=argparse.REMAINDER,help="Additional command argument")
    parser.add_argument("--use-category-mapping",default=False,help="use category mapping",action="store_true")
    args = parser.parse_args()
    command = args.command
    nargs = args.nargs
    # 用于category映射为自定义的id
    category_map = {1:96,2:97,3:49,4:98,5:99}
    run_command(args, command, nargs, parser)

