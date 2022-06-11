import os
import xml
import os.path as osp
import json
import pathlib
from PIL import Image
import shutil
import numpy as np
from json import JSONEncoder

def load_file(fpath):
    assert os.path.exists(fpath)  #assert() raise-if-not
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines] #str to list
    return records

def clip_coords(boxes, size):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[0] = int(np.clip(boxes[0], 1, size[0]-1))  # x1
    boxes[1] = int(np.clip(boxes[1], 1, size[1]-1))  # y1
    boxes[2] = int(np.clip(boxes[2], 1, size[0]-1))  # x2
    boxes[3] = int(np.clip(boxes[3], 1, size[1]-1))  # y2        
    return boxes

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
def crowdhuman2voc(odgt_path, json_path):
    print(odgt_path, " to ", json_path)    
    records = load_file(odgt_path)
    outdir_images = pathlib.Path(osp.join(json_path, 'images'))
    outdir_annotations = pathlib.Path(osp.join(json_path, 'annotations'))
    outdir_images.mkdir(parents=True, exist_ok=True) 
    outdir_annotations.mkdir(parents=True, exist_ok=True)     
    #预处理
    categories = {}
    #for i in range(len(records)):
    for i in range(10):
        json_dict = {"bboxes":[], "keypoints": []}
        
        file_name = records[i]['ID']
        image_file_name = file_name +'.jpg'
        json_file_name = file_name +'.json'
        gt_box = records[i]['gtboxes']  
        im = Image.open(osp.join("../CrowdHuman/Images", image_file_name))
        for j in range(len(gt_box)):
            bbox = gt_box[j]['fbox']
            x, y, w, h = bbox
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            bbox = [x1, y1, x2, y2]
            bbox = clip_coords(bbox, im.size)
            
            hbox = gt_box[j]['hbox']
            x, y, w, h = hbox
            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h
            hbox = [x1, y1, x2, y2]
            hbox = clip_coords(hbox, im.size)

            x1 = hbox[0]
            y1 = hbox[1]
            x2 = hbox[2]
            y2 = hbox[3]
            #print(x, y, w, h)
            #kpset = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
            kpset = [[x1, y1, 1], [x2, y2, 1]]
            #print(kpset)
            ignore = 0 
            if "ignore" in gt_box[j]['head_attr']:
                ignore = gt_box[j]['head_attr']['ignore']
            if "ignore" in gt_box[j]['extra']:
                ignore = gt_box[j]['extra']['ignore']
            if ( (bbox[0] < bbox[2]) & (bbox[1] < bbox[3]) & (hbox[0] < hbox[2]) & (hbox[1] < hbox[3]) ):
                json_dict['bboxes'].append(bbox)            
                json_dict['keypoints'].append(kpset)
                print(im.size, bbox)
                
        print(f'output file {i}: {j}')
        #print(osp.join(odgt_path, image_file_name))
        #print(osp.join(outdir_images, image_file_name))
        shutil.copyfile(osp.join('../CrowdHuman', 'Images', image_file_name), osp.join(outdir_images, image_file_name))
        with open(osp.join(outdir_annotations, json_file_name), 'w') as f:
            f.write(json.dumps(json_dict))

in_file = "../CrowdHuman/annotation_train.odgt"
out_dir = "train"
crowdhuman2voc(in_file, out_dir)

in_file = "../CrowdHuman/annotation_val.odgt"
out_dir = "test"
crowdhuman2voc(in_file, out_dir)
