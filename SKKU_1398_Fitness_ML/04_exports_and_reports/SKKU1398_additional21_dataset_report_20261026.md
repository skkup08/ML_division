# 1398 피트니스 추가 21종 데이터 조사
2026-10-26. 기존 학습 5종 제외. 7개 공개 YOLO 데이터셋을 Colab에서 실제 다운로드하고 라벨 파일로 기구별 이미지 수를 집계하고 기본 좌표 검사를 통과한 박스가 있는 이미지 수를 계산했다.

수량은 기본 좌표 검사 통과 이미지의 train/valid/test 합계이며 증강본 포함이다. 데이터셋끼리 중복될 수 있어 단순 합계는 제시하지 않는다. 괄호는 .rf 이후 및 mirrored 문자열을 제거한 파일명 그룹 수다. 그룹은 원본 사진 수나 독립 촬영 수가 아니며 파일명 충돌·유사사진이 남는다. 판단은 초기 전이학습 가능성에 대한 잠정 판단이며 센터 실물 성능을 검증한 것이 아니다.

|센터 번호|기구|출처와 대응 클래스|이미지 수(파일명 그룹)|학습 충분성 판단|
|---|---|---|---|---|
|2|Recumbent Bike|Q recumbent bike; T Recumbent Bike|Q 10(10); T 1(1)|매우 부족. 별도 수집 필요|
|3|Upright Bike|T Stationary Bike; D stationary bike|T 120(40); D 144(18)|부족. 업라이트형 선별 필요|
|4|Elliptical|Q elliptical; F Elliptical; T Elliptical|Q 63(63); F 39(17); T 33(18)|부족. 증강 전 다양성 확보 필요|
|6|Pectoral Fly|G chest-fly; F Chest Fly Machine|G 942(276); F 243(100)|초기 학습 후보. 센터 펙덱 구조에 맞게 검수|
|7|High Fly|G cable-lat-pulldown; T Lat Pull Down Machine|G 500(155); T 362(103)|초기 실험 가능. 제공 사진 기준 랫풀다운 대응|
|9|Long Pull|G seated-cable-row; T Seated Row Machine|G 374(121); T 343(104)|조건부 초기 실험. 케이블 로우와 레버 로우 선별|
|10|Arm Curl|G arm-curl; T Arm Curl Machine; H armcurl|G 744(237); T 172(85); H 100(87)|초기 학습 후보. 프리처 벤치와 머신 구분|
|11|Abdominal|G abdominal-machine; F Ab Crunch Machine|G 274(85); F 31(14)|부족. 초기 실험은 가능하나 다양성 보완|
|12|Chin Up|G dip-chin-assist; T Pull Up Bar|G 244(102); T 48(48)|기구 형태 확정 필요. 어시스트형과 단순 철봉을 합치면 안 됨|
|13|Rotary Torso|G torso-rotation-machine|G 502(140)|초기 실험 가능. 같은 기구 반복 촬영·분할 중복 주의|
|14|Leg Extension|G leg-extension; T Leg Extension Machine; H legextension|G 332(105); T 354(102); H 118(118)|초기 실험 가능. 출처 간 중복 확인 후 결합|
|15|Leg Curl|G lying-down-leg-curl / seated-leg-curl; H legcurl|G 누운형 246(57), 앉은형 298(90); H 107(107)|센터 형태별 선별 필요. 각 하위형은 부족|
|17|Cable Cross Over|Z Cable Crossover Machine; G pulley-machine; F Functional Trainer|Z 14(6); G 458(145); F 117(50)|정확 대응 데이터 부족. 일반 케이블 기구에서 재선별 필요|
|19|Mega Max|확정 대응 클래스 미발견|확정 확보 0|학습 불가. 제공 사진과 제품명 대응 미확정|
|20|Chinning Dipping Hip Flexor|T Leg Raise Tower; Z Power Tower|T 367(100); Z 11(5)|조건부 초기 실험. 세 기능이 있는 타워형 선별 필요|
|21|Power Leg Press|H powerlegpress|H 101(101)|초기 소규모 실험 가능, 충분하지 않음. 일반 leg-press와 자동 병합 금지|
|22|Power Rack|G squat-rack; D squat rack|G 372(125); D 333(45)|조건부 초기 실험. 풀랙·하프랙·스쿼트 스탠드 선별|
|23|Sit Up|F Abdominal Bench|F 35(15)|매우 부족|
|24|Double Twist|기구 Bounding Box 데이터셋 미발견|확정 확보 0|학습 불가. 제품 사진 출처는 있으나 학습셋 아님|
|25|Hyper Extension|F Back Extension Machine; T Roman Chair|F 75(35); T 1(1)|부족. 중량식 백익스텐션과 로만체어 구분 필요|
|26|Super Bench Press (Incline/Decline)|G incline-bench; D adjustable bench; Q declined bench press|G 360(121); D 271(37); Q 디클라인 2(2)|인클라인 초기 후보, 디클라인 매우 부족. 벤치 단독/프레스 결합형 선별|

## 데이터 출처
- F FitFuel v2, CC BY 4.0: https://universe.roboflow.com/fitfuel/all-gym-equipment/dataset/2
- G GymBro v2, CC BY 4.0: https://universe.roboflow.com/rizzlabzz/gymbro/dataset/2
- T Trainers Workspace v1, CC BY 4.0: https://universe.roboflow.com/trainers-workspace/gym-equipment-t6kck-fkuoc
- Z Gymze v1, CC BY 4.0: https://universe.roboflow.com/gymze/gymze/dataset/1
- Q Gym v5, CC BY 4.0: https://universe.roboflow.com/gym-qknei/gym-shkud
- D Home Gym Equipment Detection v6, MIT: https://universe.roboflow.com/dans-new-workspace/home-gym-equipment-detection
- H healnion v1, CC BY 4.0: https://universe.roboflow.com/hn-2kixi/healnion-f4l32

## 판단 근거와 한계
대략 수백 장과 100~200개 이상 파일명 그룹이 있으면 사전학습 모델의 초기 실험 후보로 평가했다. 이 수치는 본 프로젝트의 작업 우선순위이며 보장된 최소 데이터 요구량이 아니다. 정확한 클래스 형태, 독립 촬영 다양성, 배경·가림·각도, 라벨 정확도, 센터 홀드아웃 성능이 충분성 판단의 최종 기준이다.
Ultralytics YOLOv5 학습 지침은 좋은 결과를 위한 참고 권장량으로 클래스당 1,500장 이상 및 실제 배포 환경 다양성을 제시한다. YOLO11의 필수 최소량으로 해석하지 않는다: https://docs.ultralytics.com/yolov5/tutorials/tips-for-best-training-results/
이번 작업은 데이터 조사·집계이며 추가 21종을 학습한 것이 아니다. 클래스별 전수 시각 검수와 데이터셋 간 이미지 중복 제거는 아직 수행하지 않았다. 검색에서 발견한 Gym_item(RJIT)은 원문 fetch가 404여서 실제 확보량에 포함하지 않았다.

실행 코드와 결과: https://colab.research.google.com/drive/1lVGnJa2CzjvjodBBgJisaP-DE5dzXQT8 (셀 15~18)

## 기본 라벨 검사
대응 이미지 누락은 7개 데이터셋 모두 0건. 좌표 오류 박스 행은 F 58, G 91, T 48, Z 3, Q 0, D 1393, H 0건. 표는 해당 오류 박스를 제외한 후의 이미지 수다. 한 이미지에 남은 유효 박스가 있으면 해당 클래스의 이미지로 집계했다. 이는 학습셋 정제 완료를 의미하지 않는다. 검사 기준: 5개 수치, 유한값, x/y/w/h 각각 [0,1], 양수 w/h. 박스 모서리 경계 및 시각적 적합성은 전수 검사하지 않았다.
