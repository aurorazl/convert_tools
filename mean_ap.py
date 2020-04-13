from multiprocessing import Pool
import pickle
import numpy as np
from terminaltables import AsciiTable
from class_names import get_classes
from pycocotools.coco import COCO
import json
import collections
import utils
import pyprind

def average_precision(recalls, precisions, mode='area'):
    """Calculate average precision (for single or multiple scales).
    Args:
        recalls (ndarray): shape (num_scales, num_dets) or (num_dets, )
        precisions (ndarray): shape (num_scales, num_dets) or (num_dets, )
        mode (str): 'area' or '11points', 'area' means calculating the area
            under precision-recall curve, '11points' means calculating
            the average precision of recalls at [0, 0.1, ..., 1]
    Returns:
        float or ndarray: calculated average precision
    """
    no_scale = False
    if recalls.ndim == 1:
        no_scale = True
        recalls = recalls[np.newaxis, :]
        precisions = precisions[np.newaxis, :]
    assert recalls.shape == precisions.shape and recalls.ndim == 2
    num_scales = recalls.shape[0]
    ap = np.zeros(num_scales, dtype=np.float32)
    if mode == 'area':
        zeros = np.zeros((num_scales, 1), dtype=recalls.dtype)
        ones = np.ones((num_scales, 1), dtype=recalls.dtype)
        mrec = np.hstack((zeros, recalls, ones))
        mpre = np.hstack((zeros, precisions, zeros))
        for i in range(mpre.shape[1] - 1, 0, -1):
            mpre[:, i - 1] = np.maximum(mpre[:, i - 1], mpre[:, i])
        for i in range(num_scales):
            ind = np.where(mrec[i, 1:] != mrec[i, :-1])[0]
            ap[i] = np.sum(
                (mrec[i, ind + 1] - mrec[i, ind]) * mpre[i, ind + 1])
    elif mode == '11points':
        for i in range(num_scales):
            for thr in np.arange(0, 1 + 1e-3, 0.1):
                precs = precisions[i, recalls[i, :] >= thr]
                prec = precs.max() if precs.size > 0 else 0
                ap[i] += prec
            ap /= 11
    else:
        raise ValueError(
            'Unrecognized mode, only "area" and "11points" are supported')
    if no_scale:
        ap = ap[0]
    return ap


def tpfp_imagenet(det_bboxes,
                  gt_bboxes,
                  gt_bboxes_ignore=None,
                  default_iou_thr=0.5,
                  area_ranges=None):
    """Check if detected bboxes are true positive or false positive.
    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        default_iou_thr (float): IoU threshold to be considered as matched for
            medium and large bboxes (small ones have special rules).
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.
    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp
    # of a certain scale.
    tp = np.zeros((num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((num_scales, num_dets), dtype=np.float32)
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + 1) * (
                det_bboxes[:, 3] - det_bboxes[:, 1] + 1)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp
    ious = bbox_overlaps(det_bboxes, gt_bboxes - 1)
    gt_w = gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1
    gt_h = gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1
    iou_thrs = np.minimum((gt_w * gt_h) / ((gt_w + 10.0) * (gt_h + 10.0)),
                          default_iou_thr)
    # sort all detections by scores in descending order
    sort_inds = np.argsort(-det_bboxes[:, -1])
    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = gt_w * gt_h
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            max_iou = -1
            matched_gt = -1
            # find best overlapped available gt
            for j in range(num_gts):
                # different from PASCAL VOC: allow finding other gts if the
                # best overlaped ones are already matched by other det bboxes
                if gt_covered[j]:
                    continue
                elif ious[i, j] >= iou_thrs[j] and ious[i, j] > max_iou:
                    max_iou = ious[i, j]
                    matched_gt = j
            # there are 4 cases for a det bbox:
            # 1. it matches a gt, tp = 1, fp = 0
            # 2. it matches an ignored gt, tp = 0, fp = 0
            # 3. it matches no gt and within area range, tp = 0, fp = 1
            # 4. it matches no gt but is beyond area range, tp = 0, fp = 0
            if matched_gt >= 0:
                gt_covered[matched_gt] = 1
                if not (gt_ignore_inds[matched_gt]
                        or gt_area_ignore[matched_gt]):
                    tp[k, i] = 1
            elif min_area is None:
                fp[k, i] = 1
            else:
                bbox = det_bboxes[i, :4]
                area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                if area >= min_area and area < max_area:
                    fp[k, i] = 1
    return tp, fp


def tpfp_default(det_bboxes,
                 gt_bboxes,
                 gt_bboxes_ignore=None,
                 iou_thr=0.5,
                 area_ranges=None):
    """Check if detected bboxes are true positive or false positive.
    Args:
        det_bbox (ndarray): Detected bboxes of this image, of shape (m, 5).
        gt_bboxes (ndarray): GT bboxes of this image, of shape (n, 4).
        gt_bboxes_ignore (ndarray): Ignored gt bboxes of this image,
            of shape (k, 4). Default: None
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        area_ranges (list[tuple] | None): Range of bbox areas to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. Default: None.
    Returns:
        tuple[np.ndarray]: (tp, fp) whose elements are 0 and 1. The shape of
            each array is (num_scales, m).
    """
    # an indicator of ignored gts
    gt_ignore_inds = np.concatenate(
        (np.zeros(gt_bboxes.shape[0], dtype=np.bool),
         np.ones(gt_bboxes_ignore.shape[0], dtype=np.bool)))
    # stack gt_bboxes and gt_bboxes_ignore for convenience
    gt_bboxes = np.vstack((gt_bboxes, gt_bboxes_ignore))

    num_dets = det_bboxes.shape[0]
    num_gts = gt_bboxes.shape[0]
    if area_ranges is None:
        area_ranges = [(None, None)]
    num_scales = len(area_ranges)
    thr_nums = len(iou_thr) if isinstance(iou_thr, list) else 1
    # tp and fp are of shape (num_scales, num_gts), each row is tp or fp of
    # a certain scale
    tp = np.zeros((thr_nums,num_scales, num_dets), dtype=np.float32)
    fp = np.zeros((thr_nums,num_scales, num_dets), dtype=np.float32)
    # if there is no gt bboxes in this image, then all det bboxes
    # within area range are false positives
    if gt_bboxes.shape[0] == 0:
        if area_ranges == [(None, None)]:
            fp[...] = 1
        else:
            det_areas = (det_bboxes[:, 2] - det_bboxes[:, 0] + 1) * (
                det_bboxes[:, 3] - det_bboxes[:, 1] + 1)
            for i, (min_area, max_area) in enumerate(area_ranges):
                fp[:,i, (det_areas >= min_area) & (det_areas < max_area)] = 1
        return tp, fp , [0 for _ in range(num_dets)]

    ious = bbox_overlaps(det_bboxes, gt_bboxes)
    # for each det, the max iou with all gts
    ious_max = ious.max(axis=1)
    # for each det, which gt overlaps most with it
    ious_argmax = ious.argmax(axis=1)
    # sort all dets in descending order by scores
    sort_inds = np.argsort(-det_bboxes[:, -1])

    for k, (min_area, max_area) in enumerate(area_ranges):
        gt_covered = np.zeros(num_gts, dtype=bool)
        # if no area range is specified, gt_area_ignore is all False
        if min_area is None:
            gt_area_ignore = np.zeros_like(gt_ignore_inds, dtype=bool)
        else:
            gt_areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
            gt_area_ignore = (gt_areas < min_area) | (gt_areas >= max_area)
        for i in sort_inds:
            for thr_index,one_thr in enumerate(iou_thr):
                if ious_max[i] >= one_thr:
                    matched_gt = ious_argmax[i]
                    if not (gt_ignore_inds[matched_gt]
                            or gt_area_ignore[matched_gt]):
                        if not gt_covered[matched_gt]:
                            gt_covered[matched_gt] = True
                            tp[thr_index,k, i] = 1
                        else:
                            fp[thr_index,k, i] = 1
                    # otherwise ignore this detected bbox, tp = 0, fp = 0
                elif min_area is None:
                    fp[thr_index,k, i] = 1
                else:
                    bbox = det_bboxes[i, :4]
                    area = (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)
                    if area >= min_area and area < max_area:
                        fp[thr_index,k, i] = 1
    return tp, fp , ious_max


def get_cls_results(det_results, annotations, class_id):
    """Get det results and gt information of a certain class.
    Args:
        det_results (list[list]): Same as `eval_map()`.
        annotations (list[dict]): Same as `eval_map()`.
    Returns:
        tuple[list[np.ndarray]]: detected bboxes, gt bboxes, ignored gt bboxes
    """
    cls_dets = [img_res[class_id] for img_res in det_results]
    cls_gts = []
    cls_gts_ignore = []
    for ann in annotations:
        gt_inds = ann['labels'] == (class_id + 1)
        cls_gts.append(ann['bboxes'][gt_inds, :])

        if ann.get('labels_ignore', None) is not None:
            ignore_inds = ann['labels_ignore'] == (class_id + 1)
            cls_gts_ignore.append(ann['bboxes_ignore'][ignore_inds, :])
        else:
            cls_gts_ignore.append(np.empty((0, 4), dtype=np.float32))

    return cls_dets, cls_gts, cls_gts_ignore


def eval_map(det_results,
             annotations,
             scale_ranges=None,
             iou_thr=0.5,
             dataset=None,
             logger=None,
             nproc=4):
    """Evaluate mAP of a dataset.
    Args:
        det_results (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotations (list[dict]): Ground truth annotations where each item of
            the list indicates an image. Keys of annotations are:
                - "bboxes": numpy array of shape (n, 4)
                - "labels": numpy array of shape (n, )
                - "bboxes_ignore" (optional): numpy array of shape (k, 4)
                - "labels_ignore" (optional): numpy array of shape (k, )
        scale_ranges (list[tuple] | None): Range of scales to be evaluated,
            in the format [(min1, max1), (min2, max2), ...]. A range of
            (32, 64) means the area range between (32**2, 64**2).
            Default: None.
        iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        dataset (list[str] | str | None): Dataset name or dataset classes,
            there are minor differences in metrics for different datsets, e.g.
            "voc07", "imagenet_det", etc. Default: None.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
        nproc (int): Processes used for computing TP and FP.
            Default: 4.
    Returns:
        tuple: (mAP, [dict, dict, ...])
    """
    assert len(det_results) == len(annotations)
    iou_thr = iou_thr if isinstance(iou_thr,list) else [iou_thr]
    num_imgs = len(det_results)
    num_scales = len(scale_ranges) if scale_ranges is not None else 1
    num_classes = len(det_results[0])  # positive class num
    area_ranges = ([(rg[0]**2, rg[1]**2) for rg in scale_ranges]
                   if scale_ranges is not None else None)

    pool = Pool(nproc)
    eval_results = {}
    det_iou = []
    print("starting calculate ious for iou_thr={}.............".format(str(iou_thr)))
    for i in range(num_classes):
        # get gt and det bboxes of this class
        cls_dets, cls_gts, cls_gts_ignore = get_cls_results(
            det_results, annotations, i)
        # choose proper function according to datasets to compute tp and fp
        if dataset in ['det', 'vid']:
            tpfp_func = tpfp_imagenet
        else:
            tpfp_func = tpfp_default
        # compute tp and fp for each image with multiple processes
        tpfp = pool.starmap(
            tpfp_func,
            zip(cls_dets, cls_gts, cls_gts_ignore,
                [iou_thr for _ in range(num_imgs)],
                [area_ranges for _ in range(num_imgs)]))
        tp, fp, iou = tuple(zip(*tpfp))
        det_iou.append(iou)
        # calculate gt number of each scale
        # ignored gts or gts beyond the specific scale are not counted
        num_gts = np.zeros(num_scales, dtype=int)
        for j, bbox in enumerate(cls_gts):
            if area_ranges is None:
                num_gts[0] += bbox.shape[0]
            else:
                gt_areas = (bbox[:, 2] - bbox[:, 0] + 1) * (
                    bbox[:, 3] - bbox[:, 1] + 1)
                for k, (min_area, max_area) in enumerate(area_ranges):
                    num_gts[k] += np.sum((gt_areas >= min_area)
                                         & (gt_areas < max_area))
        # sort all det bboxes by score, also sort tp and fp
        cls_dets = np.vstack(cls_dets)
        num_dets = cls_dets.shape[0]
        sort_inds = np.argsort(-cls_dets[:, -1])
        eps = np.finfo(np.float32).eps
        for index,thr in enumerate(iou_thr):
            tp = np.hstack(tp[index])[:, sort_inds]
            fp = np.hstack(fp[index])[:, sort_inds]
            # calculate recall and precision with tp and fp
            tp = np.cumsum(tp, axis=1)
            fp = np.cumsum(fp, axis=1)
            recalls = tp / np.maximum(num_gts[:, np.newaxis], eps)
            precisions = tp / np.maximum((tp + fp), eps)
            # calculate AP
            if scale_ranges is None:
                recalls = recalls[0, :]
                precisions = precisions[0, :]
                num_gts = num_gts.item()
            mode = 'area' if dataset != 'voc07' else '11points'
            ap = average_precision(recalls, precisions, mode)
            eval_results[thr].append({
                'num_gts': num_gts,
                'num_dets': num_dets,
                'recall': recalls,
                'precision': precisions,
                'ap': ap
            })
    if scale_ranges is not None:
        # shape (num_classes, num_scales)
        for index,thr in enumerate(iou_thr):
            all_ap = np.vstack([cls_result['ap'] for cls_result in eval_results[index]])
            all_num_gts = np.vstack(
                [cls_result['num_gts'] for cls_result in eval_results[index]])
            mean_ap = collections.defaultdict(lambda :[])
            for i in range(num_scales):
                if np.any(all_num_gts[:, i] > 0):
                    mean_ap[thr].append(all_ap[all_num_gts[:, i] > 0, i].mean())
                else:
                    mean_ap[thr].append(0.0)
    else:
        mean_ap = collections.defaultdict(lambda :[])
        for index, thr in enumerate(iou_thr):
            aps = []
            for cls_result in eval_results[index]:
                if cls_result['num_gts'] > 0:
                    aps.append(cls_result['ap'])
            mean_ap[thr] = np.array(aps).mean().item() if aps else 0.0
    print("done calculate ious for iou_thr={}.............".format(str(iou_thr)))
    outdata = print_map_summary(
        mean_ap, eval_results, iou_thr, dataset, area_ranges, logger=logger)

    return mean_ap, outdata,det_iou

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

def print_map_summary(mean_ap,
                      results,
                      iou_thrs,
                      dataset=None,
                      scale_ranges=None,
                      logger=None):
    """Print mAP and results of each class.
    A table will be printed to show the gts/dets/recall/AP of each class and
    the mAP.
    Args:
        mean_ap (float): Calculated from `eval_map()`.
        results (list[dict]): Calculated from `eval_map()`.
        dataset (list[str] | str | None): Dataset name or dataset classes.
        scale_ranges (list[tuple] | None): Range of scales to be evaluated.
        logger (logging.Logger | str | None): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.
    """

    if logger == 'silent':
        return

    if isinstance(results[iou_thrs[0]][0]['ap'], np.ndarray):
        num_scales = len(results[iou_thrs[0]][0]['ap'])
    else:
        num_scales = 1

    if scale_ranges is not None:
        assert len(scale_ranges) == num_scales

    num_classes = len(results[iou_thrs[0]])
    thr_nums = len(iou_thrs)
    recalls = np.zeros((thr_nums,num_scales, num_classes), dtype=np.float32)
    aps = np.zeros((thr_nums,num_scales, num_classes), dtype=np.float32)
    num_gts = np.zeros((thr_nums,num_scales, num_classes), dtype=int)
    TOTAL_MAP = []
    for index,one_thr in enumerate(iou_thrs):
        for i, cls_result in enumerate(results[one_thr]):
            if cls_result['recall'].size > 0:
                recalls[index,:, i] = np.array(cls_result['recall'], ndmin=2)[:, -1]
            aps[index,:, i] = cls_result['ap']
            num_gts[index,:, i] = cls_result['num_gts']

        if dataset is None:
            label_names = [str(i) for i in range(1, num_classes + 1)]
        elif isinstance(dataset,str):
            label_names = get_classes(dataset)
        else:
            label_names = dataset

        if not isinstance(mean_ap, list):
            mean_ap = [mean_ap]

        header = ['category', 'gt_nums', 'det_nums', 'recall', 'ap']
        outdata = [['category', 'gt_nums', 'det_nums', 'recall', 'ap']]
        MAP = []
        for i in range(num_scales):
            if scale_ranges is not None:
                print('Scale range {}'.format(scale_ranges[i]))
            table_data = [header]
            for j in range(num_classes):
                one_map = {}
                row_data = [
                    label_names[j], int(num_gts[index,i, j]), int(results[j]['num_dets']),
                    '{:.3f}'.format(float(recalls[index,i, j])), '{:.3f}'.format(float(aps[index,i, j]))
                ]
                one_map["category"] = label_names[j]
                one_map["gt_nums"] = int(num_gts[index,i, j])
                one_map["det_nums"] = int(results[j]['num_dets'])
                one_map["recall"] = float(recalls[index,i, j])
                one_map["ap"] = float(aps[index,i, j])
                table_data.append(row_data)
                MAP.append(one_map)
                outdata.append(row_data)
            table_data.append(['mAP', '', '', '', '{:.3f}'.format(mean_ap[i])])
            table = AsciiTable(table_data)
            table.inner_footing_row_border = True
            print('\n' + table.table)
        TOTAL_MAP.append({"iouThr":one_thr,"data":MAP})
    return TOTAL_MAP

def list_json_to_bbox_list(li):
    tmp = []
    category_ids = set()
    tem = collections.defaultdict(lambda : collections.defaultdict(lambda : []))
    for one in li:
        one["bbox"].append(one["score"])
        tem[one["image_id"]][one["category_id"]].append(one["bbox"])
        category_ids.add(one["category_id"])
    for k,v in tem.items():
        tem1 = []
        for category_id in category_ids:
            tem1.append([category_id,np.array(v.get(category_id,[])).astype(np.float32).reshape(-1,len(category_ids))])
        tmp.append([k,list(map(lambda x:x[1],sorted(tem1,key=lambda x:x[0])))])
        """tem:{"image_id":{"category_id":cls_det,...}}"""
        """tem1: [[[category_id,cls2_det], [category_id,cls1_det], ...], ...]."""
        """tmp: [ [image_id,[cls1_det,cls2_det ...], ...]."""
        """return : [ [cls1_det,cls2_det ...], ...]."""
    return list(map(lambda x:x[1],sorted(tmp,key=lambda x:x[0])))

index = None


def iou_insert_results(li,ious):
    global index
    pbar = pyprind.ProgBar(len(ious), monitor=True, title="iou insert into list")
    for i,category_li in enumerate(ious):
        for j,image_li in enumerate(category_li):
            for k,one in enumerate(image_li):
                pos = int(index[i, j, k])
                li[pos]["iou"] = float(one)
        pbar.update()
    for one in li:
        if "iou" not in one:
            raise Exception("missing one iou")

def _det2list(results):
    bbox_results = []
    for idx in range(len(results)):
        det = results[idx]
        bbox_results.append(det)
    return bbox_results

def _segm2list(results):
    bbox_results = []
    segm_results = []
    for idx in range(len(results)):
        det, seg = results[idx]
        bbox_results.append(det)
        segm_results.append(seg)
    return bbox_results, segm_results

class CocoDataset(object):

    def load_annotations(self,ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        self.img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            self.img_infos.append(info)

    def get_ann_info(self,idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _parse_ann_info(self,img_info, ann_info):
        """Parse bbox and mask annotation.
        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.
        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] == [] or ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, ndmin=2, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, ndmin=2, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def pkl_to_list_json(self,path):
        with open(path, "rb") as f:
            a = pickle.load(f)
        c = []
        for index, one in enumerate(a):
            for id, i in enumerate(one[0]):
                for j in i:
                    d = j.tolist()
                    x1,y1,x2,y2 = d[:4]
                    c.append({"bbox": [x1,y1,x2-x1,y2-y1], "score": d[4], "image_id": self.img_ids[index], "category_id": self.cat_ids[id]})
        return c

    def coco_to_annotation(self,length):
        annotations = []
        pbar = pyprind.ProgBar(length, monitor=True, title="coco to annotation")
        for i in range(length):
            annotations.append(self.get_ann_info(i))
            pbar.update()
        return annotations

    def list_json_to_bbox_list2(self,li):
        tem = collections.defaultdict(lambda : collections.defaultdict(lambda : []))
        tmp = []
        max_num = collections.defaultdict(lambda : 0)
        pbar = pyprind.ProgBar(len(li), monitor=True, title="counting category_id and image_id")
        for one in li:
            max_num["{}__{}".format(str(one["category_id"]),str(one["image_id"]))] += 1
            pbar.update()
        global index
        index = np.zeros((len(self.cat_ids), len(self.img_ids),max((v for k,v in max_num.items()))))
        pbar = pyprind.ProgBar(len(li), monitor=True, title="mapping bbox index according to category_id and image_id")
        for i,one in enumerate(li):
            x = utils.sort_list_search_int(self.cat_ids,one["category_id"])
            # y = utils.sort_list_search_int(image_ids,one["image_id"])
            y = self.img_ids.index(one["image_id"])
            for j,l in enumerate(index[x,y]):
                if l==0:
                    index[x,y,j] = i
                    break
            x_value,y_value,w,h = one["bbox"]
            tem[one["image_id"]][one["category_id"]].append([x_value,y_value,w+x_value,h+y_value,one["score"]])
            pbar.update()
        pbar = pyprind.ProgBar(len(tem), monitor=True, title="sort list by category_id and image_id")
        for k,v in tem.items():
            tem1 = []
            for category_id in self.cat_ids:
                tem1.append([category_id,np.array(v.get(category_id,[])).astype(np.float32).reshape(-1,5)]) # 5 is x1,y1,x2,y2,score
            tmp.append([k,list(map(lambda x:x[1],sorted(tem1,key=lambda x:x[0])))])
            pbar.update()
        return list(map(lambda x:x[1],sorted(tmp,key=lambda x:self.img_ids.index(x[0]))))

def list_json_to_anno_list(li):
    tmp=[]
    for one in li:
        tmp.append({"bboxes":one["bbox"],"labels":[]})

if __name__ == '__main__':
    with open("/data/imagenet/x-ray/cocovis/tianchi/annotations/res0413/xray_test.segm.json") as f:
        results = json.load(f)

    # with open(r"xray_test.pkl", "rb") as f:
    #     a = pickle.load(f)
    # bbox_results,_ = _segm2list(a)
    dataset = CocoDataset()
    dataset.load_annotations("/data/imagenet/x-ray/cocovis/tianchi/annotations/gt_val.json")
    # results = dataset.pkl_to_list_json("/data/imagenet/x-ray/cocovis/tianchi/annotations/res0413/xray_test.pkl")
    bbox_results1 = dataset.list_json_to_bbox_list2(results)
    annotations = dataset.coco_to_annotation(len(bbox_results1))
    # category_id后面有排序，需要image_id对应即可
    _,out,ious = eval_map(bbox_results1,annotations)
    iou_insert_results(results, ious)
    print(out)

