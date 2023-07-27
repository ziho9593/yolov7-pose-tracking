import torch
from pose.utils.general import non_max_suppression, scale_coords


def detect(dataset, model, device, half, kpt_label=True):
    kpts = None
    for img, im0s in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]
        # print(pred[...,4].max())
        # Apply NMS
        pred = non_max_suppression(pred, kpt_label=kpt_label)

        # Process detections
        for det in pred:  # detections per image
            s, im0 = '', im0s.copy()
            s += '%gx%g ' % img.shape[2:]  # print string
            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                kpts = det[:, 6:]
                kpts = kpts.tolist()

    return kpts