# -*- coding: utf-8 -*-
import argparse
import textwrap
import os
import json

import label_tool
import utils
from config import config
from DirectoryUtils import cd


def upload_dataset(image_path, anno_path, project_id, dataset_id, verbose=False, ignore_image=False):
    assert os.path.exists("commit.json")
    assert os.path.exists(anno_path)
    if not ignore_image:
        utils.check_path_exist(image_path)
        utils.check_path_exist("list.json")
        utils.remove_local_file(config["image_tar_name"])
        cmd = "tar zcf %s %s/*.jpg" % (config["image_tar_name"], image_path)
        if verbose:
            print(cmd)
        os.system(cmd)
        utils.check_path_exist(config["image_tar_name"])
        utils.scp(config["identity_file"], config["image_tar_name"], config["nfs_base_path"], config["user"],
                  config["host"], config["port"],verbose)
        utils.scp(config["identity_file"], "list.json", config["nfs_base_path"], config["user"], config["host"],config["port"],
                  verbose)

    utils.remove_local_file(config["json_tar_name"])
    cmd = "tar zcf %s %s/*.json" % (config["json_tar_name"], anno_path)
    if verbose:
        print(cmd)
    os.system(cmd)
    utils.scp(config["identity_file"], config["json_tar_name"], config["nfs_base_path"], config["user"], config["host"],config["port"],
              verbose)
    utils.scp(config["identity_file"], "commit.json", config["nfs_base_path"], config["user"], config["host"],config["port"], verbose)
    utils.scp(config["identity_file"], "category.json", config["nfs_base_path"], config["user"], config["host"], config["port"],verbose)
    target_image_base_path = os.path.join(config["nfs_base_path"], "label/public/tasks", dataset_id)
    target_json_base_path = os.path.join(config["nfs_base_path"], "label/private/tasks", dataset_id, project_id)
    cmd = ""
    if not ignore_image:
        cmd += "rm -rf " + os.path.join(target_image_base_path, "images") + ";"
        cmd += "rm -f " + os.path.join(target_image_base_path, "list.json") + ";"
    cmd += "sudo rm -rf " + os.path.join(config["nfs_base_path"], "label/private/tasks", dataset_id) + ";"
    # cmd += "rm -f " + os.path.join(target_json_base_path, "commit.json") + ";"
    # cmd += "rm -f " + os.path.join(target_json_base_path, "category.json") + ";"
    # cmd += "rm -rf " + os.path.join(target_json_base_path, "images") + ";"
    if not ignore_image:
        cmd += "mkdir -p " + target_image_base_path + ";"
    cmd += "mkdir -p " + target_json_base_path + ";"
    if not ignore_image:
        cmd += "tar zxf %s -C %s" % (
        os.path.join(config["nfs_base_path"], config["image_tar_name"]), target_image_base_path) + ";"
        cmd += "mv %s %s" % (os.path.join(config["nfs_base_path"], "list.json"), target_image_base_path) + ";"
    cmd += "tar zxf %s -C %s" % (
    os.path.join(config["nfs_base_path"], config["json_tar_name"]), target_json_base_path) + ";"
    cmd += "mv %s %s" % (os.path.join(config["nfs_base_path"], "commit.json"), target_json_base_path) + ";"
    cmd += "mv %s %s" % (os.path.join(config["nfs_base_path"], "category.json"), target_json_base_path) + ";"
    utils.SSH_exec_cmd_with_output(config["identity_file"], config["user"], config["host"], config["port"],cmd, verbose=verbose)


def upload_model_predict_result(anno_path, project_id, dataset_id, verbose=False):
    utils.check_path_exist(anno_path)
    cmd = "tar zcf %s %s/*.json" % (config["json_tar_name"], anno_path)
    if verbose:
        print(cmd)
    os.system(cmd)
    utils.scp(config["identity_file"], config["json_tar_name"], config["nfs_base_path"], config["user"], config["host"],config["port"],
              verbose=verbose)
    target_json_base_path = os.path.join(config["nfs_base_path"], "label/private/predict", dataset_id, project_id)
    cmd = ""
    cmd += "rm -rf " + os.path.join(target_json_base_path, "images") + ";"
    cmd += "mkdir -p " + target_json_base_path + ";"
    cmd += "tar zxf %s -C %s" % (
    os.path.join(config["nfs_base_path"], config["json_tar_name"]), target_json_base_path) + ";"
    utils.SSH_exec_cmd_with_output(config["identity_file"], config["user"], config["host"], config["port"], cmd, verbose=verbose)


def upload_dataset_from_coco(coco_anno_path, image_path, project_id, dataset_id, user_id, verbose=False,
                             ignore_image=False, args=None):
    utils.check_path_exist(coco_anno_path)
    utils.check_path_exist(image_path)
    out_json_path = os.path.join("./", "template_for_convert")
    utils.remove_directiry(out_json_path)
    os.system("mkdir %s" % out_json_path)
    label_tool.merge_coco_to_json_dataset(coco_anno_path, image_path, out_json_path, args=args)
    label_tool.generate_commit_json(out_json_path, user_id, args.base_category_num)
    label_tool.find_coco_dataset_category_ids(coco_anno_path,out_json_path)
    with cd(out_json_path):
        upload_dataset("images", "images", project_id, dataset_id, verbose, ignore_image)


def upload_dataset_from_voc(voc_anno_path, voc_image_path,project_id, dataset_id, user_id, verbose=False, ignore_image=False, args=None):
    utils.check_path_exist(voc_anno_path)
    utils.check_path_exist(voc_image_path)
    out_json_path = os.path.join("./", "template_for_convert")
    utils.remove_directiry(out_json_path)
    os.system("mkdir %s" % out_json_path)
    label_tool.merge_voc_dataset_to_json_dataset(voc_anno_path, voc_image_path, out_json_path, args=args)
    label_tool.generate_commit_json(out_json_path, user_id, args.base_category_num)
    label_tool.find_voc_dataset_category_ids(voc_anno_path, out_json_path)
    with cd(out_json_path):
        upload_dataset("images", "images", project_id, dataset_id, verbose, ignore_image)


def upload_dataset_from_ocr(ocr_anno_path, ocr_image_path, project_id, dataset_id, user_id, verbose=False,
                            ignore_image=False, args=None):
    utils.check_path_exist(ocr_anno_path)
    utils.check_path_exist(ocr_image_path)
    out_json_path = os.path.join("./", "template_for_convert")
    utils.remove_directiry(out_json_path)
    os.system("mkdir %s" % out_json_path)
    label_tool.merge_ocr_to_json(ocr_anno_path, ocr_image_path, out_json_path, args=args)
    label_tool.generate_commit_json(out_json_path, user_id, args.base_category_num)
    label_tool.find_ocr_dataset_category_ids(out_json_path)
    with cd(out_json_path):
        upload_dataset("images", "images", project_id, dataset_id, verbose, ignore_image)


def upload_model_predict_result_from_list(list_file_path, project_id, dataset_id, verbose=True, args=None):
    utils.check_path_exist(list_file_path)
    out_json_path = os.path.join("./", "template_for_convert")
    utils.remove_directiry(out_json_path)
    os.system("mkdir %s" % out_json_path)
    label_tool.module_predict_segmentation_list_to_json(list_file_path, out_json_path, args.base_category_num)
    with cd(out_json_path):
        upload_model_predict_result("images", project_id, dataset_id, verbose)


def upload_model_predict_result_from_coco(coco_file_path, project_id, dataset_id, verbose=True, args=None):
    utils.check_path_exist(coco_file_path)
    out_json_path = os.path.join("./", "template_for_convert")
    utils.remove_directiry(out_json_path)
    os.system("mkdir %s" % out_json_path)
    args.ignore_image = True
    label_tool.merge_coco_to_json_dataset(coco_file_path, "", out_json_path, args=args)
    with cd(out_json_path):
        upload_model_predict_result("images", project_id, dataset_id, verbose)


def upload_model_predict_result_from_voc(voc_anno_path, voc_image_path, project_id, dataset_id, verbose=True,
                                         args=None):
    utils.check_path_exist(voc_anno_path)
    out_json_path = os.path.join("./", "template_for_convert")
    utils.remove_directiry(out_json_path)
    os.system("mkdir %s" % out_json_path)
    args.ignore_image = True
    label_tool.merge_voc_dataset_to_json_dataset(voc_anno_path, voc_image_path, out_json_path, args=args)
    with cd(out_json_path):
        upload_model_predict_result("images", project_id, dataset_id, verbose)


def upload_model_predict_result_from_ocr(ocr_anno_path, ocr_image_path, project_id, dataset_id, verbose=True,
                                         args=None):
    utils.check_path_exist(ocr_anno_path)
    utils.check_path_exist(ocr_image_path)
    out_json_path = os.path.join("./", "template_for_convert")
    utils.remove_directiry(out_json_path)
    os.system("mkdir %s" % out_json_path)
    args.ignore_image = True
    label_tool.merge_ocr_to_json(ocr_anno_path, ocr_image_path, out_json_path, args=args)
    with cd(out_json_path):
        upload_model_predict_result("images", project_id, dataset_id, verbose)


def auto_upload_model_predict_result(anno_path, image_path, project_id, dataset_id, verbose=True, args=None):
    dataset_type = utils.find_dataset_type(anno_path,image_path)
    if dataset_type=="coco":
        upload_model_predict_result_from_coco(anno_path, project_id, dataset_id, verbose, args)
    elif dataset_type=="list":
        upload_model_predict_result_from_list(anno_path, project_id, dataset_id, verbose, args)
    elif dataset_type == "voc":
        upload_model_predict_result_from_voc(anno_path, image_path, project_id, dataset_id, verbose, args)
    elif dataset_type == "ocr":
        upload_model_predict_result_from_ocr(anno_path, image_path, project_id, dataset_id, verbose, args)
    elif dataset_type == "json":
        upload_model_predict_result(anno_path, project_id, dataset_id, verbose)
    else:
        raise Exception("not support dataset type")

def auto_upload_dataset(anno_path, image_path, project_id, dataset_id, user_id, verbose=False, ignore_image=False,
                        args=None):
    dataset_type = utils.find_dataset_type(anno_path,image_path)
    if dataset_type=="coco":
        upload_dataset_from_coco(anno_path, image_path, project_id, dataset_id, user_id, verbose, ignore_image,args)
    elif dataset_type == "voc":
        upload_dataset_from_voc(anno_path,image_path, project_id, dataset_id, user_id, verbose, ignore_image, args)
    elif dataset_type == "ocr":
        upload_dataset_from_ocr(anno_path, image_path, project_id, dataset_id, user_id, verbose, ignore_image, args)
    elif dataset_type == "json":
        upload_dataset(image_path, anno_path, project_id, dataset_id, verbose, ignore_image)
    else:
        raise Exception("not support dataset type")

def upload_map_file_from_det_list_gt_coco(project_id, dataset_id,verbose=True):
    utils.check_path_exist("map.json")
    utils.scp(config["identity_file"], "map.json", config["nfs_base_path"], config["user"], config["host"],config["port"],verbose=verbose)
    utils.scp(config["identity_file"], "iou.json", config["nfs_base_path"], config["user"], config["host"],config["port"],verbose=verbose)
    target_json_base_path = os.path.join(config["nfs_base_path"], "label/private/tasks", dataset_id, project_id)
    cmd = ""
    cmd += "rm -rf " + os.path.join(target_json_base_path, "map.json") + ";"
    cmd += "rm -rf " + os.path.join(target_json_base_path, "iou.json") + ";"
    cmd += "mkdir -p " + target_json_base_path + ";"
    cmd += "mv %s %s" % (os.path.join(config["nfs_base_path"], "map.json"), target_json_base_path) + ";"
    cmd += "mv %s %s" % (os.path.join(config["nfs_base_path"], "iou.json"), target_json_base_path) + ";"
    utils.SSH_exec_cmd_with_output(config["identity_file"], config["user"], config["host"],  config["port"],cmd, verbose=verbose)

def upload_map_file(det_anno_path, gt_anno_path,image_path,project_id, dataset_id, verbose=True,args=None):
    out_json_path = os.path.join("./", "template_for_convert")
    utils.remove_directiry(out_json_path)
    os.system("mkdir %s" % out_json_path)
    label_tool.calculate_dataset_map(det_anno_path,gt_anno_path,image_path,out_json_path,args)
    label_tool.module_predict_segmentation_list_to_json(os.path.join(out_json_path,"new_list.json"), out_json_path, args.base_category_num)
    with cd(out_json_path):
        upload_map_file_from_det_list_gt_coco(project_id, dataset_id, verbose)
        upload_model_predict_result("images", project_id, dataset_id, verbose)

def run_command(args, command, nargs, parser):
    if command == "upload_dataset":
        if len(nargs) != 4:
            parser.print_help()
            print("upload_dataset [image_path] [anno_path] [project_id] [dataset_id]")
        else:
            upload_dataset(nargs[0], nargs[1], nargs[2], nargs[3], args.verbose, args.ignore_image)
    elif command == "upload_model_predict_result":
        if len(nargs) != 3:
            parser.print_help()
            print("upload_model_predict_result [anno_path] [project_id] [dataset_id]")
        else:
            upload_model_predict_result(nargs[0], nargs[1], nargs[2], args.verbose)
    elif command == "upload_dataset_from_coco":
        if len(nargs) != 5:
            parser.print_help()
            print("upload_dataset_from_coco [coco_anno_path] [image_path]  [project_id] [dataset_id] [user_id]")
        else:
            upload_dataset_from_coco(nargs[0], nargs[1], nargs[2], nargs[3], nargs[4], args.verbose, args.ignore_image,
                                     args)
    elif command == "upload_dataset_from_voc":
        if len(nargs) != 4:
            parser.print_help()
            print("upload_dataset_from_voc [voc_path] [project_id] [dataset_id] [user_id]")
        else:
            upload_dataset_from_voc(nargs[0], nargs[1], nargs[2], nargs[3], args.verbose, args.ignore_image, args)
    elif command == "upload_dataset_from_ocr":
        if len(nargs) != 5:
            parser.print_help()
            print("upload_dataset_from_ocr [ocr_anno_path] [ocr_image_path] [project_id] [dataset_id] [user_id]")
        else:
            upload_dataset_from_ocr(nargs[0], nargs[1], nargs[2], nargs[3], nargs[4], args.verbose, args.ignore_image,
                                    args)
    elif command == "upload_model_predict_result_from_list":
        if len(nargs) != 3:
            parser.print_help()
            print("upload_model_predict_result_from_list [list_file_path] [project_id] [dataset_id]")
        else:
            upload_model_predict_result_from_list(nargs[0], nargs[1], nargs[2], args.verbose, args)
    elif command == "upload_model_predict_result_from_coco":
        if len(nargs) != 3:
            parser.print_help()
            print("upload_model_predict_result_from_coco [coco_file_path] [project_id] [dataset_id]")
        else:
            upload_model_predict_result_from_coco(nargs[0], nargs[1], nargs[2], args.verbose, args)
    elif command == "upload_model_predict_result_from_voc":
        if len(nargs) != 4:
            parser.print_help()
            print("upload_model_predict_result_from_voc [voc_anno_path] [voc_image_path] [project_id] [dataset_id]")
        else:
            upload_model_predict_result_from_voc(nargs[0], nargs[1], nargs[2], nargs[3], args.verbose, args)
    elif command == "upload_model_predict_result_from_ocr":
        if len(nargs) != 4:
            parser.print_help()
            print("upload_model_predict_result_from_ocr [ocr_anno_path] [ocr_image_path] [project_id] [dataset_id]")
        else:
            upload_model_predict_result_from_ocr(nargs[0], nargs[1], nargs[2], nargs[3], args.verbose, args)
    elif command == "upload-prediction":
        if len(nargs) != 4:
            parser.print_help()
            print("upload-prediction [anno_path] [image_path] [project_id] [dataset_id]")
        else:
            auto_upload_model_predict_result(nargs[0], nargs[1], nargs[2], nargs[3], args.verbose, args)
    elif command == "upload-dataset":
        if len(nargs) != 5:
            parser.print_help()
            print("upload-dataset [anno_path] [image_path] [project_id] [dataset_id] [uid]")
        else:
            auto_upload_dataset(nargs[0], nargs[1], nargs[2], nargs[3], nargs[4], args.verbose, args.ignore_image, args)
    elif command == "upload-prediction-with-map":
        if len(nargs) != 5:
            parser.print_help()
            print("upload-prediction-with-map [det_anno_path] [gt_anno_path] [image_path] [project_id] [dataset_id]")
        else:
            upload_map_file(nargs[0], nargs[1], nargs[2], nargs[3], nargs[4],args.verbose,args)
    else:
        parser.print_help()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='upload.py',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description=textwrap.dedent('''\
    upload tools

    Command:
        upload_dataset 
            [image_path] [anno_path] [project_id] [dataset_id]
      '''))
    parser.add_argument("command",
                        help="See above for the list of valid command")
    parser.add_argument('nargs', nargs=argparse.REMAINDER,
                        help="Additional command argument",
                        )
    parser.add_argument("-v", "--verbose",
                        help="verbose print",
                        action="store_true", default=True)
    parser.add_argument("--ignore-image",
                        default=False,
                        help="dont copy image",
                        action="store_true"
                        )
    parser.add_argument("--use-category-mapping",
                        default=False,
                        help="use category mapping",
                        action="store_true"
                        )
    parser.add_argument("--base_category_num", "-cn",
                        default=0,
                        help="base_category_num",
                        action="store", type=int
                        )
    parser.add_argument("--anno-before-prefix", "-abp", default="", help="before anno prefix", action="store")
    parser.add_argument("--anno-after-prefix", "-aap", default="", help="after anno prefix", action="store")
    parser.add_argument("--anno-before-suffix", "-abs", default="", help="before anno suffix", action="store")
    parser.add_argument("--anno-after-suffix", "-aas", default="", help="after anno suffix", action="store")
    parser.add_argument("--image-before-prefix", "-ibp", default="", help="before image prefix", action="store")
    parser.add_argument("--image-after-prefix", "-iap", default="", help="after image prefix", action="store")
    parser.add_argument("--image-before-suffix", "-ibs", default="jpg", help="before image suffix", action="store")
    parser.add_argument("--image-after-suffix", "-ias", default="jpg", help="after image suffix", action="store")
    parser.add_argument("--ignore-exists", default=True, help="after image suffix", action="store_true")
    args = parser.parse_args()
    command = args.command
    nargs = args.nargs
    os.system("chmod 600 id_rsa")
    run_command(args, command, nargs, parser)
