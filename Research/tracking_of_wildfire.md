# Real-time tracking of wildfire boundaries using satellite imagery  
2023.02.03 - google ai blog (https://ai.googleblog.com/2023/02/real-time-tracking-of-wildfire.html)  

- Introduction
위성 이미지와 ML 기법을 사용하여 실시간으로 산불을 추적하는 시스템
10~15분마다 정보 업데이트  
사전 연구 (https://medium.com/google-earth/how-to-generate-wildfire-boundary-maps-with-earth-engine-b38eadc97a38) 보다 성능 개선 
미국, 멕시코, 캐나다, 호주 등에서 발생하는 대형 화재를 구글 검색 및 지도에 표시  

- Inputs
정지 위성 사용  
북미(GOES-16, GOES-18), 호주 (Himawari-9, GK2A) 
위성 바로 아래지점부터 공간해상도 2km 이며, 위성에서 멀어질수록 해상도가 낮아짐  
화재시 발생하는 연기로 인해 산불범위를 결정하기 어려움  
이 문제를 극복하기 위해 3~4μm 파장 범위에서 적외선(IR) 주파수에 의존하는 것이 일반적임    
IR 대역에서 추가된 정보가 있어도 화재의 방출 강도가 다양하고 여러 가지 다른 현상으로 인해 IR 방사선이 간섭받기 때문에 화재의 범위를 정확하게 결정하는 것은 어려움 

- Model  
화재 감지 연구는 물리 알고리즘을 기반으로 하는 것이 일반적이었음   
산불 추적 시스템에서 모델은 모든 위성 입력에 대해 서로 다른 주파수 대역의 상대적 중요성을 학습  
모델은 구름으로 인한 장애를 보완하기 위해 각 밴드에서 가장 최근의 3개 이미지 시퀀스를 입력받음  
또한, 두 개의 정지위성으로부터 입력을 수신하여 두 위성의 픽셀 크기에 따라 감지 정확도가 향상되는 초해상도 효과를 달성   
북미 모델에 대해서는 NOAA 화재 product를 추가적으로 입력      
마지막으로 태양과 위성의 상대 각도를 계산하고 이를 모델에 추가 입력으로 제공   

모든 입력자료는 균일한 1km-square grid로 리샘플링되어 CNN에 제공   
CNN + 1x1 컨볼루션 레이어를 사용하여 화재 및 구름 픽셀에 대한 별도의 분류 헤드 생성   
픽셀이 구름으로 식별되면 대량의 구름이 근본적인 화재를 가리기 때문에 모든 화재 감지를 무시함   
(구름 분류 작업을 통해 엣지 케이스를 더 잘 식별하도록 시스템에 인센티브를 제공하므로 화재 감지 성능이 향상)   

![image](https://user-images.githubusercontent.com/76670294/217757470-06f30fb6-b4dd-480d-bd0e-bc3476793c4c.png)

MODIS 및 VIIRS 극궤도 위성의 열 이상(thermal anomalies) 데이터를 레이블로 사용    
MODIS 및 VIIRS는 정지 위성보다 공간 해상도(750~1000m)가 더 높지만, 몇 시간에 한 번만 지정된 위치를 커버하므로 빠르게 진행되는 화재는 놓칠 수 있음   
따라서 MODIS 및 VIIRS로 데이터 셋을 구성하지만 추론 시점에는 정지 위성의 고주파 이미지에 의존    
불타지 않는 픽셀에 대한 모델의 편향을 줄이기 위해 학습 데이터 셋의 화재 픽셀을 업샘플링하고 focal loss를 적용하여 드물게 잘못 분류된 화재 픽셀을 개선하려고 함   

- Evaluation  
데이터 셋을 만드는 과정에서 라벨 지정 오류(예: 구름 관련 오탐지) 등이 발생할 수 있으나, 이와 상관없이 평가를 수행   
평가는 지방 당국에서 측정한 화재 흔적(전체 불에 탄 면적의 모양)을 기준으로 함   

![image](https://user-images.githubusercontent.com/76670294/217770376-36854fab-72b7-4816-9f9f-5902fffae222.png)

분류 오류의 공간적인 심각성을 정량화하기 위해, false positive 또는 false negative 픽셀과 true positive fire 픽셀 사이의 최대 거리를 계산
그런 다음, 모든 화재에 대해 각 메트릭의 평균을 구함
|Model|Number of fires|Precision|Recall|False positive max distance mean|False negative max distance mean|
|:-------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
|Canada|13|66.1%|63.1%|1.6km|2.7km|
|USA|55|80.1%|78.7%|1.8km|2.7km|
|Mexico|27|67.3%|70.2%|2.3km|2.9km|
|Australia|11|76.7%|95.9%|2.8km|1.6km|

NOAA의 GOES-16 및 GOES-17 화재 product만 의존하는 초기모델을 평가하고 2022년 데이터를 추가로 수집하여 새로운 데이터 셋으로 평가하는 등 두 가지 추가 실험을 수행
|Model|Number of fires|Precision|Recall|False positive max distance mean|False negative max distance mean|
|:-------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|:-------------------:|
|NOAA fire product on original test set|55|76.7%|67.8%|2.3km|3.3km|
|Our model on original test set|55|80.1%|78.7%|1.8km|2.7km|
|Our model on 2022 wildfires|121|76.1%|80.8%|2.0km|2.5km|
