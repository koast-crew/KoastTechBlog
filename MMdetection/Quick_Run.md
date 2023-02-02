# 1. 기존에 정의된 모델 및 표준 데이터 세트를 이용하여 학습 및 추론


MMDetection은 [Model zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html) 에서 수많은 객체 탐지 모델을 제공하고, Pascal VOC, COCO, CityScapes, LVIS 등을 포함한 여러가지 데이터 세트를 지원한다.

---

## 목차



1. [기존 모델을 사용하여 주어진 이미지 추론](#기존-모델을-사용하여-주어진-이미지-추론)

2. [표준 데이터 세트에서 기존 모델을 테스트]()

3. [표준 데이터 세트에서 사전 정의된 모델을 학습]()

---

## 기존 모델을 사용하여 주어진 이미지 추론
추론이란 훈련된 모델을 사용해 이미지에서 객체를 탐색하는 것을 의미한다.
MMDetection에서 모델은 구성 파일에 의해 정의되고 기존 모델의 매개변수는 체크포인트파일에 저장된다.

우선 이 구성파일(configure file)과 체크포인트(ckp) 파일을 사용하여 Faster RCNN을 사용 해보는 것을 권장한다.
<br>  
<br>  

**<U>추론을 위한 고급 API 사용 예시</U>**

``` python
## git mmdetection 이 설치되어있어야 한다.

from mmdet.apis import init_detector, inference_detector
import mmcv

# Specify the path to model config and checkpoint file
config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

## 
checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

# build the model from a config file and a checkpoint file
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 쿠다 오류가 있는 경우 해당설정을 통해 에러가 발생하지 않음.
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single image and show the results
img = 'demo.jpg'  # or img = mmcv.imread(img), which will only load it once
result = inference_detector(model, img)
# visualize the results in a new window
model.show_result(img, result)
# or save the visualization results to image files
model.show_result(img, result, out_file='result.jpg')

# test a video and show the results
video = mmcv.VideoReader('demo.mp4')
for frame in video:
    result = inference_detector(model, frame)
    # show option을 추가해서 프레임단위로 결과를 볼 수 있지만 프레임 수가 많으므로 주의!
    model.show_result(frame, result, wait_time=1) 
```

***입력 이미지***
![input_img](./img/demo.jpg)

***출력 이미지***
![output_img](./img/output_img.png)






