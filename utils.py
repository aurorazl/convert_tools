import os
import subprocess
import numpy as np

def check_path_exist(path):
    if not os.path.exists(path):
        raise Exception("path {} not found".format(path))

def remove_local_file(path):
    if os.path.exists(path):
        os.remove(path)

def remove_directiry(path):
    if os.path.exists(path):
        os.system("rm -rf %s"%path)

def scp (identity_file, source, target, user, host, verbose = False):
    cmd = 'scp -q -o "StrictHostKeyChecking no" -o "UserKnownHostsFile=/dev/null" -i %s -r "%s" "%s@%s:%s"' % (identity_file, source, user, host, target)
    if verbose:
        print(cmd)
    try:
        output = subprocess.check_output( cmd, shell=True )
    except subprocess.CalledProcessError as e:
        output = "Return code: " + str(e.returncode) + ", output: " + e.output.strip()
    print(output)

def SSH_exec_cmd_with_output(identity_file, user,host,cmd, supressWarning = False,verbose=False):
    if len(cmd)==0:
        return ""
    if supressWarning:
        cmd += " 2>/dev/null"
    execmd = """ssh -o "StrictHostKeyChecking no" -o "UserKnownHostsFile=/dev/null" -i %s "%s@%s" "%s" """ % (identity_file, user, host, cmd )
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