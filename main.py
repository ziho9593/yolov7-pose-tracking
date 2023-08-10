import torch
import cv2
import json
import time
import os
import numpy as np
from pathlib import Path
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_imshow, increment_path, set_logging, check_img_size, non_max_suppression, scale_coords
from utils.plots import draw_boxes
from utils.torch_utils import select_device, time_synchronized, TracedModel

from pose.utils.datasets import LoadImages as PoseLoadImages
from pose.detect import detect
from sort import Sort


def model_load(weights, device, imgsz):
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    return model, stride, imgsz


def view_image(p, im0):
    cv2.imshow(str(p), im0)
    if cv2.waitKey(1) == ord('q'): 
        cv2.destroyAllWindows()


def img_prep(img, device, half):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img


def track(det, img, im0, sort_tracker):
    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
    dets_to_sort = np.empty((0, 6))

    for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
        dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))
    tracked_dets = sort_tracker.update(dets_to_sort)

    return tracked_dets
                

def visualize(tracked_dets, im0, model, imgsz, stride, device, half, names, results):
    bbox_xyxy = tracked_dets[:, :4]
    identities = tracked_dets[:, -1]
    categories = tracked_dets[:, 4]

    for idx, box in enumerate(bbox_xyxy):
        cat = int(categories[idx]) if categories is not None else 0
        if cat != 0:
            continue
        id = int(identities[idx]) if identities is not None else 0
        x1, y1, x2, y2 = [int(x) for x in box]
        obj = im0[y1:y2, x1:x2]
        if not obj.shape[0] or not obj.shape[1]:
            continue
        d = PoseLoadImages(obj, imgsz, stride)
        kpts, obj = detect(d, model, device, half, xy=[x1, y1])
        if kpts == None:
            continue
        if id in results.keys():
            results[id].append(kpts)
        else:
            results[id] = [kpts]
        im0[y1:y2, x1:x2] = obj
    draw_boxes(im0, bbox_xyxy, identities, categories, names)
    
    return results
    

def main(source):
    device = '0'
    img_size, conf_thres, iou_thres = 640, 0.25, 0.45
    view_img, save_json = False, True
    save_img = not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))

    sort_tracker = Sort(max_age=5, min_hits=2, iou_threshold=0.2)

    save_dir = Path(increment_path(Path('output') / 'obj', exist_ok=False))
    save_dir.mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'

    model, stride, imgsz = model_load('yolov7.pt', device, img_size)
    model_, stride_, imgsz_ = model_load('yolov7-w6-pose.pt', device, img_size)
    model = TracedModel(model, device, img_size)

    if half:
        model.half()
        model_.half()

    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmarrk = True
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    names = model.module.names if hasattr(model, 'module') else model.names
    
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters()))) 
        model_(torch.zeros(1, 3, imgsz_, imgsz_).to(device).type_as(next(model_.parameters())))

    t0 = time.time()
    nf = 0

    results = {}
    for path, img, im0s, vid_cap in dataset:
        nf += 1
        img = img_prep(img, device, half)

        t1 = time_synchronized()
        pred = model(img, augment=False)[0]
        t2 = time_synchronized()
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=0)
        t3 = time_synchronized()

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)
            save_path = str(save_dir / p.name)

            if len(det):
                tracked_dets = track(det, img, im0, sort_tracker)
                if len(tracked_dets) > 0:
                    results = visualize(tracked_dets, im0, model_, imgsz_, stride_, device, half, names, results)
            else:
                tracked_dets = sort_tracker.update()
        print(f'Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

        if view_img:
            view_image(p, im0)
        
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
                print(f" The image with the result is saved in: {save_path}")
            else:  # 'video' or 'stream'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer.write(im0)

    if save_json:
        t4 = time.time()
        results['tag'] = {'time': t4-t0,
                       'num of frame': nf,
                       'total detected': len(results.keys()),
                       'frame/time': nf/(t4-t0)
                       }
        with open(save_path.split('.')[0]+'.json', 'w') as f:
                json.dump(results, f, indent=4)


if __name__ == '__main__':
    source = 'test.mp4'
    with torch.no_grad():
        main(source)
