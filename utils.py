import os
import subprocess
import numpy as np
import pyprind
import re
import shutil
import glob
import pyprind
import json

def check_path_exist(path):
    if not os.path.exists(path):
        raise Exception("path {} not found".format(path))

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)

def remove_local_file(path):
    if os.path.exists(path):
        os.remove(path)

def remove_directiry(path):
    if os.path.exists(path):
        os.system("rm -rf %s"%path)

def path_is_file(path):
    if os.path.isfile(path):
        return True
    return False

def scp (identity_file, source, target, user, host, port,verbose = False):
    if port:
        cmd = 'scp -P %s -q -o "StrictHostKeyChecking no" -o "UserKnownHostsFile=/dev/null" -i %s -r "%s" "%s@%s:%s"' % (port,identity_file, source, user, host, target)
    else:
        cmd = 'scp -q -o "StrictHostKeyChecking no" -o "UserKnownHostsFile=/dev/null" -i %s -r "%s" "%s@%s:%s"' % (identity_file, source, user, host, target)
    if verbose:
        print(cmd)
    try:
        output = subprocess.check_output( cmd, shell=True )
    except subprocess.CalledProcessError as e:
        output = "Return code: " + str(e.returncode) + ", output: " + e.output.strip()
    print(output)

def SSH_exec_cmd_with_output(identity_file, user,host,port,cmd, supressWarning = False,verbose=False):
    if len(cmd)==0:
        return ""
    if supressWarning:
        cmd += " 2>/dev/null"
    if port:
        execmd = """ssh -p %s -o "StrictHostKeyChecking no" -o "UserKnownHostsFile=/dev/null" -i %s "%s@%s" "%s" """ % (port,identity_file, user, host, cmd)
    else:
        execmd = """ssh -o "StrictHostKeyChecking no" -o "UserKnownHostsFile=/dev/null" -i %s "%s@%s" "%s" """ % (identity_file, user, host, cmd)
    if verbose:
        print(execmd)
    try:
        output = subprocess.check_output( execmd, shell=True )
    except subprocess.CalledProcessError as e:
        output = "Return code: " + str(e.returncode) + ", output: " + e.output.strip()
    # print output
    return output

def bbox_overlaps(bboxes1, bboxes2, mode='iou'):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1(ndarray): shape (n, 4)
        bboxes2(ndarray): shape (k, 4)
        mode(str): iou (intersection over union) or iof (intersection
            over foreground)
    Returns:
        ious(ndarray): shape (n, k)
    """

    assert mode in ['iou', 'iof']

    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return ious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
        bboxes1[:, 3] - bboxes1[:, 1] + 1)
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
        bboxes2[:, 3] - bboxes2[:, 1] + 1)
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + 1, 0) * np.maximum(
            y_end - y_start + 1, 0)
        if mode == 'iou':
            union = area1[i] + area2 - overlap
        else:
            union = area1[i] if not exchange else area2
        ious[i, :] = overlap / union
    if exchange:
        ious = ious.T
    return ious

def get_dir_path_max_num(dir_path, prefix,args):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    dir_list = os.listdir(dir_path)
    max_num = 0
    pbar = pyprind.ProgBar(len(dir_list), monitor=True,title="counting dir path number for specified prefix: {} ".format(prefix))
    gen_pattern = re.compile(r"(\w*?)(\d+)\.(xml|json|txt)")
    for i in dir_list:
        res = gen_pattern.match(i)
        if res:
            if res.groups()[0] !=args.anno_after_prefix + prefix:
                continue
            num = int(res.groups()[1])
            if num > max_num:
                max_num = num
        pbar.update()
    return max_num

def gen_image_name_list(anno_files_path,image_files_path,json_anno_path,prefix,args):
    src_list = os.listdir(anno_files_path)
    max_num = get_dir_path_max_num(json_anno_path, prefix,args)
    name_dict = {}
    pb = pyprind.ProgBar(len(src_list), monitor=True,title="counting src path specified prefix {} number".format(prefix))
    if args.anno_before_prefix:
        gen_pattern = re.compile(r"({})(\d+)\.(xml|json|txt)".format(args.anno_before_prefix))
    else:
        gen_pattern = re.compile(r"(\w*?)(\d+)\.(xml|json|txt)")
    for one in src_list:
        res = gen_pattern.match(one)
        if res:
            image_id = res.groups()[1]
            # if not check_anno_and_image_both_exist(anno_files_path,image_files_path,image_id,args):
            #     continue
            if  max_num == 0:
                new_image_id = int(image_id)
            else:
                max_num += 1
                new_image_id = max_num
            name_dict[new_image_id] = res.groups()[0] + res.groups()[1]
        pb.update()
    print("start to process %s picture" % len(name_dict))
    return name_dict

def check_anno_and_image_both_exist(anno_files_path,image_files_path,image_id,args):
    if not os.path.exists(os.path.join(image_files_path,"{}{}.{}".format(args.image_before_prefix,image_id,args.image_before_suffix))):
        return False
    return True

def get_ocr_annotation(label_path: str) -> tuple:
    boxes = []
    text_tags = []
    with open(label_path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            params = line.strip().strip('\ufeff').strip('\xef\xbb\xbf').split(',')
            try:
                if len(params)>8:
                    label = params[-1]
                elif len(params)==8:
                    label = ""
                else:
                    label = params[8]
                if label.strip() == "###" or label == "***":
                    continue
                text_tags.append(label)
                # if label == '*' or label == '###':
                x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, params[:8]))
                boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
            except:
                print('load label failed on {}'.format(label_path))
    return boxes,text_tags

def convert_ocr_annotation_list_to_str(bboxs):
    string=""
    for bbox in bboxs:
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4],text_tag] = bbox
        string+=",".join([str(x) for x in [x1, y1, x2, y2, x3, y3, x4, y4, text_tag]])
        string+="\n"
    return string

def calculate_quadrangle_area(bbox):
    coord = np.array(bbox).reshape((4, 2))
    temp_det = 0
    for idx in range(3):
        temp = np.array([coord[idx], coord[idx + 1]])
        temp_det += np.linalg.det(temp)
    temp_det += np.linalg.det(np.array([coord[-1], coord[0]]))
    return temp_det * 0.5

def check_anno_image_number(voc_anno_path,voc_image_path,args):
    anno_list = glob.glob(os.path.join(voc_anno_path,"*.{}".format(args.anno_before_suffix)))
    image_list = glob.glob(os.path.join(voc_image_path,"*.{}".format(args.image_before_suffix)))
    if(len(anno_list)!=len(image_list)):
        raise Exception("anno number not equal image number")

def sort_list_search_int(li, val):
    low = 0
    high = len(li)-1
    count = 0
    while low <= high:
        count +=1
        mid = (low + high) // 2
        # print("mid",mid,li[mid])
        if li[mid] > val:
            high = mid - 1
            # print("high",high)
        elif li[mid] < val:
            low = mid + 1
            # print("low",low)
        else:
            return mid
    else:
        return None

def find_dataset_type(anno_path,image_path):
    check_path_exist(anno_path)
    check_path_exist(image_path)
    if path_is_file(anno_path):
        file_type = os.path.splitext(anno_path)[1]
        if file_type == ".json":
            with open(anno_path, "r") as f:
                di = json.load(f)
            assert type(di) == dict or type(di) == list
            if isinstance(di, dict):
                print("uploading coco annotation file")
                return "coco"
            elif isinstance(di, list):
                print("uploading list annotation file")
                return "list"
        else:
            raise Exception("Only json suffix supported.")
    else:
        src_list = os.listdir(anno_path)
        first_file = src_list[0]
        file_type = os.path.splitext(first_file)[1]
        assert file_type in [".xml", ".txt", ".json"]
        if file_type == ".xml":
            print("uploading voc dataset")
            if anno_path.endswith("/"):
                anno_path = os.path.dirname(anno_path)
            if image_path.endswith("/"):
                image_path = os.path.dirname(image_path)
            path = os.path.dirname(anno_path)
            path2 = os.path.dirname(image_path)
            if (path != path2):
                raise Exception("wrong path.anno path {} not equal with image path {}".format(path, path2))
            return "voc"
        elif file_type == ".txt":
            print("uploading ocr dataset")
            return "ocr"
        elif file_type == ".json":
            print("uploading json dataset")
            return "json"

def find_category_info_by_id(category_list,category_id):
    for one in category_list:
        if one["id"] == category_id:
            return one