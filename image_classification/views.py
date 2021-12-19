from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext, loader
from django.http.response import StreamingHttpResponse

import cv2 as cv
import numpy as np

import torchvision.utils as utils
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
import torchvision.transforms.functional as f
from PIL import Image

# Home page ----------------------------------------------------------------------------------

# index view
def index(request):
    #img = stream()
    #context = {'image': img}
    try:
        return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')
    except:
        pass
    return render(request, 'index.html')

# Display camera


def stream():

    cam_id = 0
    vid = cv.VideoCapture(cam_id)

    if not vid.isOpened():
        print("Cannot open camera")
        exit()
    else:
        while True:
            # frame-by-frame prediction
            frame = get_prediction(vid)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

            #if cv.waitKey(1) == ord('q'):
                #break

        #vid.release()



COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()


def transform_image(frame):
    my_transforms = transforms.Compose([transforms.ToTensor()])
    my_transforms2 = transforms.Compose([transforms.PILToTensor()])

    pil_image = Image.fromarray(frame)
    return my_transforms(pil_image).unsqueeze(0), my_transforms2(pil_image)


def get_prediction(vid):
    # Capture frame-by-frame
    ret, frame = vid.read()
    if ret:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_tensor, frame_int = transform_image(frame)
        outputs = model.forward(frame_tensor)
        b_boxes = outputs[0]['boxes'][outputs[0]['scores'] > 0.75]
        labels = [COCO_INSTANCE_CATEGORY_NAMES[label] for label in outputs[0]['labels'][outputs[0]['scores'] > 0.75]]
        frame_int = utils.draw_bounding_boxes(frame_int, boxes=b_boxes, colors='red', width=4, labels=labels)
        frame_int = cv.cvtColor(np.asarray(f.to_pil_image(frame_int)), cv.COLOR_RGB2BGR)
        _, jpeg = cv.imencode('.jpg', frame_int)
        return jpeg.tobytes()

    else:
        raise ValueError("Can't receive image")

