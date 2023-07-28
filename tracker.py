import os
import time
import json
import cv2
import torch
import argparse
from pathlib import Path
from numpy import random
from random import randint
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, apply_classifier, scale_coords, strip_optimizer, set_logging, increment_path
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.download_weights import download
from utils.plots import draw_boxes

from pose.utils.datasets import LoadImages as PoseLoadImages
from pose.models.experimental import attempt_load as pose_attempt_load
from pose.detect import detect
from sort import *



def tracker():
    source, weights, view_img, save_txt, imgsz, trace, save_bbox_dim, save_with_object_id, save_kpts = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace, opt.save_bbox_dim, opt.save_with_object_id, opt.save_kpts
    save_img = not opt.nosave and not source.endswith('.txt')  # Save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    
    if save_kpts:
        results = {}

    # Initialize SORT
    sort_max_age = 5 
    sort_min_hits = 2
    sort_iou_thresh = 0.2
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh)
    
    # Rand Color for every trk
    rand_color_list = []
    amount_rand_color_prime = 5003  # prime number
    for i in range(0,amount_rand_color_prime):
        r = randint(0, 255)
        g = randint(0, 255)
        b = randint(0, 255)
        rand_color = (r, g, b)
        rand_color_list.append(rand_color)

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt or save_with_object_id else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load models (*_: for pose estimation)
    model_ = pose_attempt_load('yolov7-w6-pose.pt', map_location=device)
    stride_ = int(model_.stride.max()) 
    imgsz_ = imgsz
    if isinstance(imgsz_, (list,tuple)):
        assert len(imgsz_) ==2; "height and width of image has to be specified"
        imgsz_[0] = check_img_size(imgsz_[0], s=stride_)
        imgsz_[1] = check_img_size(imgsz_[1], s=stride_)
    else:
        imgsz_ = check_img_size(imgsz_, s=stride_)

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model_.half()
        model.half()  # to FP161

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        model_(torch.zeros(1, 3, imgsz_, imgsz_).to(device).type_as(next(model_.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    nf = 0  # number of frames

    print(f'🤔 Object Tracking & Pose Estimation 시작')
    for path, img, im0s, vid_cap in dataset:
        nf += 1
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]
        
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # Detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # path/name
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # path/labels/name_*.txt

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                dets_to_sort = np.empty((0, 6))

                # NOTE: We send in detected object class too
                for x1, y1, x2, y2, conf, detclass in det.cpu().detach().numpy():
                    dets_to_sort = np.vstack((dets_to_sort, 
                                np.array([x1, y1, x2, y2, conf, detclass])))
                
                # Run SORT
                tracked_dets = sort_tracker.update(dets_to_sort)
                tracks =sort_tracker.getTrackers()

                txt_str = ""

                if save_txt and not save_with_object_id:
                    with open(txt_path + '.txt', 'a') as f:
                        f.write(txt_str)
            
                if len(tracked_dets) > 0:
                    bbox_xyxy = tracked_dets[:,:4]
                    identities = tracked_dets[:, -1]
                    categories = tracked_dets[:, 4]

                    if save_kpts:
                        for idx, box in enumerate(bbox_xyxy):
                            cat = int(categories[idx]) if categories is not None else 0
                            if cat != 0:
                                continue
                            id = int(identities[idx]) if identities is not None else 0
                            x1, y1, x2, y2 = [int(x) for x in box]
                            obj = im0[y1:y2, x1:x2]
                            if not obj.shape[0] or not obj.shape[1]:
                                continue
                            d = PoseLoadImages(obj, img_size=imgsz_, stride=stride_)
                            kpts, obj = detect(d, model=model_, device=device, half=half, xy=[x1, y1])
                            if kpts == None:
                                continue
                            if id in results.keys():
                                results[id] += kpts
                            else:
                                results[id] = kpts
                            im0[y1:y2, x1:x2] = obj
                    draw_boxes(im0, bbox_xyxy, identities, categories, names, save_with_object_id, txt_path)
           
            else:  # SORT should be updated even with no detections
                tracked_dets = sort_tracker.update()

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            
            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    cv2.destroyAllWindows()
                    # raise StopIteration
            
            # Save results (image with detections)
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

    if save_txt or save_img or save_with_object_id:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # print(f"Results saved to {save_dir}{s}")

    if save_kpts:
        t4 = time.time()
        results['tag'] = {'time': t4-t0,
                       'num of frame': nf,
                       'total detected': len(results.keys()),
                       'frame/time': nf/(t4-t0)
                       }
        with open(save_path.split('.')[0]+'.json', 'w') as f:
            json.dump(results, f, indent=4)
        print(f'time: ({t4-t0:.3f}s)')
        print(f'num of frame: {nf}')
        print(f'total detected: {len(results.keys())-1}')
        print(f'frame/time: {nf/(t4-t0)}')

    print('============= 종 료 =============')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--download', action='store_true', help='download model weights automatically')
    parser.add_argument('--no-download', dest='download', action='store_false',help='not download model weights if already exist')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='output', help='save results to project/name')
    parser.add_argument('--name', default='obj', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--save-bbox-dim', action='store_true', help='save bounding box dimensions with --save-txt tracks')
    parser.add_argument('--save-with-object-id', action='store_true', help='save results with object id to *.txt')
    parser.add_argument('--save-kpts', action='store_true', help='save keypoints data with object id to *.json')
    parser.set_defaults(download=True)
    opt = parser.parse_args()
    print(opt)

    # check_requirements(exclude=('pycocotools', 'thop'))
    if opt.download and not os.path.exists(''.join(opt.weights)):
        print('Model weights not found. Attempting to download now...')
        download('./')

    with torch.no_grad():
        if opt.update:  # Update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                tracker()
                strip_optimizer(opt.weights)
        else:
            tracker()