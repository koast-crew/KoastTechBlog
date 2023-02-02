- [[#사전준비|사전준비]]
- [[#설치|설치]]
	- [[#표준|표준]]
	- [[#설치 검증|설치 검증]]
	- [[#사용자 정의 설치|사용자 정의 설치]]
		- [[#사용자 정의 설치#CUDA 버전|CUDA 버전]]
		- [[#사용자 정의 설치#MIM 없이 MMCV 설치|MIM 없이 MMCV 설치]]
		- [[#사용자 정의 설치#CPU 전용 플랫폼에 설치|CPU 전용 플랫폼에 설치]]
		- [[#사용자 정의 설치#Google Colab에서 설치|Google Colab에서 설치]]
		- [[#사용자 정의 설치#Docker를 이용한 MMDetection 사용|Docker를 이용한 MMDetection 사용]]
- [[#벤치마크와 MODEL ZOO|벤치마크와 MODEL ZOO]]
	- [[#미러 사이트|미러 사이트]]
	- [[#일반 설정|일반 설정]]
	- [[#ImageNet 사전학습된 모델|ImageNet 사전학습된 모델]]
	- [[#베이스라인|베이스라인]]
			- [[#Docker를 이용한 MMDetection 사용#Other datasets|Other datasets]]
			- [[#Docker를 이용한 MMDetection 사용#사전 학습된 모델들|사전 학습된 모델들]]
	- [[#벤치마크 속도|벤치마크 속도]]
		- [[#벤치마크 속도#학습 속도 벤치마크|학습 속도 벤치마크]]
		- [[#벤치마크 속도#추론 속도 벤치마크|추론 속도 벤치마크]]
	- [[#Detectron2와 비교|Detectron2와 비교]]
		- [[#Detectron2와 비교#Hardware|Hardware]]
		- [[#Detectron2와 비교#Software environment|Software environment]]
		- [[#Detectron2와 비교#Performance|Performance]]
		- [[#Detectron2와 비교#Training Speed|Training Speed]]
		- [[#Detectron2와 비교#추론 속도|추론 속도]]
		- [[#Detectron2와 비교#학습 메모리|학습 메모리]]


# 사전준비
- PyTorch 환경을 구축하기 위한 방법을 소개합니다.
- MMDetection은 Linux, Windows, 그리고 macOS에서 동작이 가능하며, Python 3.7+, CUDA 9.2+, PyTorch 1.5+ 등이 요구됩니다.
>[!NOTE]
>만약 PyTorch를 설치해본적 있거나 설치되어 있다면, 해당 부분은 건너뛰고 [다음 단계](#설치)로 이동하세요, 그렇지 않으면 아래 단계에 따라 준비를 할 수 있습니다.

**Step 0.** Miniconda ([다운로드](https://docs.conda.io/en/latest/miniconda.html))를 다운로드 받고 설치합니다.

**Step 1.** conda 가상환경을 만들고 활성화합니다.

Step 2. PyTorch를 설치합니다. [설치 참고](https://pytorch.org/get-started/locally/)    
- 예시      
```
# on GPU platforms:
conda install pytorch torchvision -c pytorch   
```

```
# on CPU Platforms:
conda install pytorch torchvision cpuonly -c pytorch
``` 

# 설치
- 사용자분들이 표준설치 방법에 따라 MMDetection을 설치하기를 추천합니다. 그러나 전체 프로세스는 사용자가 세부적으로 정의할 수 있습니다. 자세한 내용은 [[#사용자 정의 설치|사용자 정의 설치]]를 참고하세요. 

## 표준     
**Step 0.** [MIM](https://github.com/open-mmlab/mim)을 이용하여 [MMCV](https://github.com/open-mmlab/mmcv)를 설치합니다.
> MIM : OpenMMLab 프로젝트를 설치하고 시작하기 위한 인터페이스 제공
> MMCV : 컴퓨터 비전 연구를 위한 라이브러리
```
pip install -U openmim    
mim install mmcv-full
``` 

**Step 1.** MMDetection을 설치합니다.
- Case a : mmdet를 직접 개발하고 실행하려면, 소스코드를 이용하여 설치하십시오. 
```
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```
- Case b : mmdet를 종속 패키지 등으로 사용하시려면, pip를 이용하여 설치하십시오.
```
pip install mmdet
```

## 설치 검증
- MMDetection이 올바르게 설치되었는지 검증하기 위해, 추론 데모를 실행할 수 있는 몇가지 샘플 코드를 제공합니다.

Step 1. config 파일과 checkpoint 파일을 다운로드합니다.
```
mim download mmdet --config yolov3_mobilenetv2_320_300e_coco --dest .
```

- 다운로드는 네트환경에 따라 수초 또는 그 이상 걸릴 수 있습니다. 다운로드가 완료되면, 현재 경로에 두 개의 파일을 찾을 수 있습니다.  
	- *yolov3_mobilenetv2_320_300e_coco.py*
	- *yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth*

**Step 2.** 추론 데모를 검증하십시오.
- Option (a). 소스코드로부터 mmdetection을 설치하였다면, 아래 명령어를 따라 실행하십시오.
```
python demo/image_demo.py demo/demo.jpg yolov3_mobilenetv2_320_300e_coco.py yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth --device cpu --out-file result.jpg
```
실행이 끝나면 현재 경로에 자동차 등에 bounding box가 그려진 *result.jpg* 이미지가 생성됩니다. 

- Option (b). pip 명령어를 통해 mmdetection을 설치하였다면, python 인터프리터를 열고 아래 코드를 복사&붙여넣기 하십시오.
```
from mmdet.apis import init_detector, inference_detector

config_file = 'yolov3_mobilenetv2_320_300e_coco.py'
checkpoint_file = 'yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/demo.jpg')
```
해당코드를 실행하면 탐지된 bounding box를 나타내는 배열 값들이 출력됩니다.

## 사용자 정의 설치
### CUDA 버전
PyTorch 설치시, CUDA 버전을 지정해야합니다. 어떤 것을 선택해야할지 명확하지 않다면, 아래 추천에 따르십시오.
- Ampere 기반의 NVIDIA GPU (GeForce 30 시리즈, A100 등)는 CUDA 11 버전이 반드시 필요합니다.
- 이전 NVIDIA GPU는 CUDA 11 이전 버전과 호환되지만, CUDA 10.2 버전을 추천합니다.
GPU 드라이버가 최소 버전 요구 사항을 충족하는지 확인하십시오. 자세한 내용은 [링크](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions__table-cuda-toolkit-driver-versions)를 확인하십시오.
>[!NOTE]
>위의 표준에 따라 설치를 했다면, CUDA 코드가 로컬에서 컴파일되지 않기 때문에 CUDA 런타임 라이브러리만 설치해도 됩니다. 그러나 소스코드를 이용하여 MMCV를 컴파일하거나 다른 CUDA 연산자를 개발하는 경우 NVIDIA 웹사이트에서 완전한 CUDA toolkit 설치해야되며, 해당 버전은 PyTorch의 CUDA 버전(즉, *conda install* 명령어에서 지정된 cudatoolkit 버전)과 일치해야합니다. 

### MIM 없이 MMCV 설치
MMCV에는 C++ 및 CUDA 확장이 포함되어 있으므로 복잡한 방식으로 PyTorch에 의존합니다. MIM은 이러한 종속성을 자동으로 해결하고 설치를 더 쉽게 만듭니다. 그러나 필수는 아닙니다.
MIM 대신 pip를 이용하여 MMCV를 설치하기 위해서는 [MMCV 설치 가이드](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)를 참고하십시오. 이를 위해 PyTorch 버전 및 해당 CUDA 버전을 기반으로 find-url을 수동으로 지정해야합니다.
예를 들어 다음 명령은 PyTorch 1.10.x 및 CUDA 11.3용으로 빌드된 mmcv-full을 설치합니다.
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
```

### CPU 전용 플랫폼에 설치
MMDetection은 CPU 전용 환경에서 설치가 가능합니다. CPU 모드에서 모델 학습, 테스트, 추론 모두 가능합니다. (MMCV 1.4.4 이상의 버전이 필요)
그러나 CPU 모드에서 아래의 기능은 사용할 수 없습니다 :
- Deformable Convolution
- Modulated Deformable Convoluiton
- ROI pooling
- Deformable ROI pooling
- CARAFE
- SyncBatchNorm
- CirssCrossAttention
- MaskedConv2d
- Temporal Interlace Shift
- nms_cuda
- sigmoid_focal_loss_cuda
- bbox_overlaps
만약 위 기능을 포함하여 모델 학습/테스트/추론을 시도한다면, 오류가 발생할 것입니다. 다음 표에는 영향을 받는 알고리즘 목록이 적혀 있습니다.
|Operator|Model|
|:--------------------:|:--------------------:|
|Deformable Convolution/Modulated Deformable Convolution|DCN, Guided Anchoring, RepPoints, CentripeetalNet, VFNet, CascadeRPN, NAS-FCOS, DetectoRS|
|MaskedConv2d|Guided Anchoring|
|CARAFE|CARAFE|
|SyncBacthNorm|ResNeSt|

### Google Colab에서 설치
[Google Colab](https://research.google/) 에는 일반적으로 PyTorch가 설치되어 있기 때문에 다음과 같이 MMCV와 MMDetection 설치만 필요합니다.

**Step 1.** [MIM](https://github.com/open-mmlab/mim)을 이용하여 [MMCV](https://github.com/open-mmlab/mmcv)를 설치합니다.
```
!pip3 install openmim
!mim install mmcv-full
```

Step 2. 소스를 이용하여 MMDetection을 설치합니다.
```
!git clone https://github.com/open-mmlab/mmdetection.git
%cd mmdetection
!pip install -e .
```

Step 3. 검증
```
import mmdet
print(mmdet.__version__)
# Example output: 2.23.0
```
>[!NOTE]
>Jupyter에서 느낌표`!`는 외부 실행 명령을 위해 사용되며, `%cd` 는 파이썬에서 현재 작업 디렉토리를 변경하기 위한 [magic command](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-cd)입니다.

### Docker를 이용한 MMDetection 사용
이미지 빌드를 위한 [Dockerfile](https://github.com/open-mmlab/mmdetection/blob/master/docker/Dockerfile)을 제공합니다. [Docker version](https://docs.docker.com/engine/install/) 이 19.03 이상인지 확인하세요.
```
# build an image with PyTorch 1.6, CUDA 10.1
# If you prefer other versions, just modified the Dockerfile
docker build -t mmdetection docker/
```
다음과 같이 실행합니다.
```
docker run --gpus all --shm-size=8g -it -v {DATA_DIR}:/mmdetection/data mmdetection
```

# 벤치마크와 MODEL ZOO
## 미러 사이트
MMDetection V2.0 이후로 [model zoo](https://modelzoo.co/)를 유지하기 위해 aliyun(알리바바 클라우드)만 사용합니다. V1.x 의 Zoo 모델은 사용되지 않습니다.
## 일반 설정
- 모든 모델은 `coco_2017_train` 으로 학습되었으며, `coco_2017_val` 으로 테스트되었습니다.
- 분산 학습을 사용합니다.
- ImageNet의 모든 pytorch 스타일의 사전 학습된 backbone은 PyTorch model zoo로부터 가져온 것이고, caffe 스타일의 사전 학습된 backbone은 새로 출시된 detectron2 모델에서 변환된 것입니다.
- 다른 코드베이스와의 공정한 비교를 위해 GPU 메모리를 총 8개 GPU에 대해 `torch.cuda.max_memory_allocated()` 의 최대값으로 사용합니다. 이 값은 일반적으로 `nvidia-smi` 에 표시된 것보다 작습니다.
- 데이터 로딩 시간읠 제외하고, 전체 네트워크 처리 시간으로 추론 시간을 사용합니다. 2000개의 이미지에 대한 평균 시간을 계산하는 [benchmark.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/benchmark.py) 스크립트로 결과를 생성합니다.
## ImageNet 사전학습된 모델
ImageNet 분류 문제에서 사전학습된 backbone으로 초기화를 수행하는 것이 일반적입니다.  [open_mmlab](https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json) 에서 사전 학습된 모델 링크들을 찾아 볼 수 있습니다. `img_norm_cfg` 과 가중치 소스에 따르면, 모든 ImageNet 사전학습된 모델 가중치를 몇 가지 경우들로 분류할 수 있습니다.
- TorchVison : ResNet50, NerNet101등이 torchvision 가중치에 해당합니다. `img_norm_cfg` 는 `dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)` 입니다.
- Pycls : RegNetX 등이 [pycls](https://github.com/facebookresearch/pycls) 가중치에 해당됩니다.
- MSRA 스타일 : ResNet50_Caffe와 ResNet101_Caffe 등이 [MSRA](https://github.com/KaimingHe/deep-residual-networks) 가중치에 해당됩니다.
- Caffe2 스타일 : 현재는 ResNext101_32x8d 만 해당됩니다. `img_norm_cfg` 는 `dict(mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)` 입니다.
- 다른 스타일 : E.g SSD (`img_norm_cfg` 는 `dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)`)와 YOLOv3 (`img_norm_cfg` 는 `dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)`)
MMDetection에서 일반적으로 사용되는 backbone 모델은 아래와 같습니다 : 
|model|source|link|description|
|:------------------:|:------------------:|:------------------:|
|ResNet50|TorchVision|[torchvision's ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth)|From torchvision's ResNet-50.|
|ResNet101|TorchVision|[torchvision's ResNet-101](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth)|From torchvision's ResNet-101.|
|ResNetX|Pycls|[RegNetX_3.2gf](https://download.openmmlab.com/pretrain/third_party/regnetx_3.2gf-c2599b0f.pth),[RegNetX_800mf](https://download.openmmlab.com/pretrain/third_party/regnetx_800mf-1f4be4c7.pth)|From pycls.|
|ResNet50_Caffe|MSRA|[MSRA's ResNet-50](https://download.openmmlab.com/pretrain/third_party/resnet50_caffe-788b5fa3.pth)|[Detectron2's R-50.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl) 의 변환된 사본입니다. 원본 가중치는 [MSRA's original ResNet-50](https://github.com/KaimingHe/deep-residual-networks) 에서 가져옵니다.|
|ResNet101_Caffe|MSRA|[MSRA's ResNet-101](https://download.openmmlab.com/pretrain/third_party/resnet101_caffe-3ad79236.pth)|[Detectron2's R-101.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-101.pkl) 의 변환된 사본입니다. 원본 가중치는 [MSRA's original ResNet-101](https://github.com/KaimingHe/deep-residual-networks) 에서 가져옵니다.|
|ResNet101_32x8d|Caffe2|[Caffe2 ResNet101_32x8d](https://download.openmmlab.com/pretrain/third_party/resnext101_32x8d-1516f1aa.pth)|[Detectron2's X-101-32x8d.pkl](https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/FAIR/X-101-32x8d.pkl) 의 변환된 사본입니다. ResNeXt-101-32x8d은 페이스북(FB)에서 Caffe2를 이용하여 학습한 모델입니다.|

## 베이스라인
[RPN]()
[Faster R-CNN]()
[Mask R-CNN]()
[Fast R-CNN(with pre-computed proposals)]()
[RetinaNet]()
[Cascade R-CNN and Cascade Mask R-CNN]()
[Hybrid Task Cascade (HTC)]()
[SSD]()
[Group Noemalization(GN)]()
[Weight Sandardization]()
[Deformable Convolution v2]()
[CARAFE: Content-Aware ReAssembly of FEatures]()
[Instaboost]()
[Libra R-CNN]()
[Guided Anchoring]()
[FCOS]()
[FoveaBox]()
[RepPoints]()
[FreeAnchor]()
[Grid R-CNN (plus)]()
[GHM]()
[GCNet]()
[HRNet]()
[Mask Scoring R-CNN]()
[Train from Scratch]()
[NAS-FPN]()
[ATSS]()
[FSAF]()
[RegNetX]()
[Res2Net]()
[GRoIE]()
[Dynamic R-CNN]()
[PointRend]()
[DetectoRS]()
[Generalized Focal Loss]()
[CornerNet]()
[YOLOv3]()
[PAA]()
[SABL]()
[CetripetalNet]()
[ResNeSt]()
[DETR]()
[Deformable DETR]()
[AutoAssign]()
[YOLOF]()
[Seesaw Loss]()
[CenterNet]()
[YOLOX]()
[PVT]()
[SOLO]()
[QueryInst]()
[PanopticFPN]()
[MaskFormer]()
[DyHead]()
[Mask2Former]()
[Efficientnet]()
[RF-Next]()
#### Other datasets
[PASCAL VOC](), [Cityscapes](), [OpenImages](), [WIDER FACE]() 에 대해서도 벤치마크를 수행하였습니다.
#### 사전 학습된 모델들
또한 ResNet-50과 [RegNetX-3.2G]()를 사용하여 다중 규모 및 긴 스케쥴로  [Faster R-CNN](), [ResNet-50]()을 학습합니다. 이러한 모델은 편의를 위해 다운스트림 작업에 대한 강력한 사전학습된 모델 역할을 합니다.

## 벤치마크 속도
### 학습 속도 벤치마크
반복학습의 평균시간을 계산하기 위해 [analyze_logs.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/analyze_logs.py) 를 제공합니다. [Log Analysis](https://mmdetection.readthedocs.io/en/latest/useful_tools.html#log-analysis) 에서 예제를 확인할 수 있습니다.

Mask R-CNN와 다른 유명한 프레임워크(detectron2에서 데이터를 복사함)의 학습속도를 비교하였습니다. mmdetection에서는 detectron2의 [mask_rcnn_R_50_FPN_noaug_1x.yaml](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_poly_1x_coco_v1.py) 과 동일한 설정값으로[mask_rcnn_r50_caffe_fpn_poly_1x_coco_v1.py](https://github.com/open-mmlab/mmdetection/blob/master/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_poly_1x_coco_v1.py) 를 벤치마크합니다. 또한 참조용 [checkpoint](https://download.openmmlab.com/mmdetection/v2.0/benchmark/mask_rcnn_r50_caffe_fpn_poly_1x_coco_no_aug/mask_rcnn_r50_caffe_fpn_poly_1x_coco_no_aug_compare_20200518-10127928.pth) 와 [training log](https://download.openmmlab.com/mmdetection/v2.0/benchmark/mask_rcnn_r50_caffe_fpn_poly_1x_coco_no_aug/mask_rcnn_r50_caffe_fpn_poly_1x_coco_no_aug_20200518_105755.log.json) 를 제공합니다. 처리량은 GPU 워밍업 시간을 제외한 100-500회 반복의 평균 처리량으로 계산됩니다.
|Implementation|Throughput(img/s)|
|:------------------:|:------------------:|
|Detectron2|62|
|MMDetection|61|
|tensorpack|50|
|simpledet|39|
|Detectron|19|
|matterport/Mask_RCNN|14|

### 추론 속도 벤치마크
추론 시간을 벤치마크하기 위한 [benchmark.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/analysis_tools/benchmark.py) 를 제공합니다. 해당 스크립트는 2000장의 이미지로 모델을 벤치마크하며, 첫 5회를 제외한 평균 시간을 계산합니다. `LOG-INTERVAL` 값 설정을 통해 log가 출력되는 간격 (기본값 : 50)을 변경할 수 있습니다.
```
python tools/benchmark.py ${CONFIG} ${CHECKPOINT} [--log-interval $[LOG-INTERVAL]] [--fuse-conv-bn]
```

model zoo에 있는 모든 모델의 지연시간은 `fuse-conv-bn` 설정 없이 벤치마크되었으며, 이 값을 설정하여 지연시간을 더 낮출 수 있습니다.

## Detectron2와 비교
속도와 성능에 대해 mmdetection을 detectron2와 비교합니다. detectron  commit id [185c27e](https://github.com/facebookresearch/detectron2/tree/185c27e4b4d2d4c68b5627b3765420c6d7f5a659) (30/4/2020)를 사용합니다. 공정한 비교를 위해 두 프레임워크를 동일한 장비에 설치하고 수행합니다.
### Hardware
- 8 NVIDIA Tesla V100 (32G) GPUs
- Intel(R) Xeon(R) Gold 6148 CPU @ 2.40aGHz
### Software environment
- Python 3.7
- PyTorch 1.4
- CUDA 10.1
- CUDNN 7.6.03
- NCCL 2.4.08
### Performance
|Type|Lr schd|Detectron2|mmdetection|Download|
|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
|Faster R-CNN|1x|[37.9](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml)|38.0|[model](https://download.openmmlab.com/mmdetection/v2.0/benchmark/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-5324cff8.pth)/[log](https://download.openmmlab.com/mmdetection/v2.0/benchmark/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco_20200429_234554.log.json)|
|Mask R-CNN|1x|[38.6 & 35.2](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml)|38.8 & 35.4|[model](https://download.openmmlab.com/mmdetection/v2.0/benchmark/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco-dbecf295.pth)/[log](https://download.openmmlab.com/mmdetection/v2.0/benchmark/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco_20200430_054239.log.json)
|Retinanet|1x|[36.5](https://github.com/facebookresearch/detectron2/blob/master/configs/COCO-Detection/retinanet_R_50_FPN_1x.yaml)|38.0|[model](https://download.openmmlab.com/mmdetection/v2.0/benchmark/retinanet_r50_caffe_fpn_mstrain_1x_coco/retinanet_r50_caffe_fpn_mstrain_1x_coco-586977a0.pth)/[log](https://download.openmmlab.com/mmdetection/v2.0/benchmark/retinanet_r50_caffe_fpn_mstrain_1x_coco/retinanet_r50_caffe_fpn_mstrain_1x_coco_20200430_014748.log.json)
### Training Speed
학습 속도 단위는 s/iter 입니다. 더 낮을수록 좋습니다.
|Type|Detectron2|mmdetection|
|:--------------------:|:--------------------:|:--------------------:|
|Faster R-CNN|0.210|0.216|
|Mask R-CNN|0.261|0.265|
|Retinanet|0.200|0.205|
### 추론 속도
추론 속도는 싱글 GPU에서 fps(img/s)로 측정되며, 더 높을수록 좋습니다. Detectron2와 일관성을 유지하기 위해, 데이터 로드 시간을 제외하고 순수 추론 속도만을 측정합니다. Mask R-CNN의 경우 후처리에서 RLE 인코딩 시간을 배제합니다. 또한 괄호 안의 값은 공식적으로 보고된 속도이며, 하드웨어의 차이로 인해 서버에서 테스트한 결과보다는 성능이 더 좋게 나타납니다. 
|Type|Detectron2|mmdetection|
|:--------------------:|:--------------------:|:--------------------:|
|Faster R-CNN|25.6(26.3)|22.2|
|Mask R-CNN|22.5(23.3)|19.6|
|Retinanet|17.8(18.2)|20.6|
### 학습 메모리
|Type|Detectron2|mmdetection|
|:--------------------:|:--------------------:|:--------------------:|
|Faster R-CNN|3.0|3.8|
|Mask R-CNN|3.4|3.9|
|Retinanet|3.9|3.4|

