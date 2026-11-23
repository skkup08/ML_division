from pathlib import Path
import json,shutil,zipfile,os,sys,hashlib
import numpy as np
from PIL import Image,ImageDraw
from ultralytics import YOLO
ROOT=Path('/content/drive/MyDrive/SKKU_1398_Fitness_ML');RUN=Path((ROOT/'04_exports_and_reports/latest_run.txt').read_text().strip());DEST=ROOT/'04_exports_and_reports'/RUN.name;DEST.mkdir(exist_ok=True)
DATA=Path('/content/skku1398_workspace/prepared/grouped_v1');model=YOLO(str(RUN/'baseline/weights/best.pt'))
failures=[];stats={c:{'tp':0,'fp':0,'fn':0} for c in range(5)}
def iou(a,b):
    lt=np.maximum(a[:2],b[:2]);rb=np.minimum(a[2:],b[2:]);wh=np.maximum(rb-lt,0);inter=wh.prod()
    return float(inter/(np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter+1e-9))
for p in sorted((DATA/'test/images').glob('*')):
    pred=model.predict(str(p),imgsz=640,conf=.25,iou=.7,device=0,verbose=False)[0];h,w=pred.orig_shape
    gt=[]
    for line in (DATA/'test/labels'/(p.stem+'.txt')).read_text().splitlines():
        c,x,y,bw,bh=map(float,line.split());gt.append((int(c),np.array([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])))
    used=set();fp=0
    for b in pred.boxes:
        c=int(b.cls.item());box=b.xyxy[0].cpu().numpy();matches=[(iou(box,v),j) for j,(gc,v) in enumerate(gt) if gc==c and j not in used]
        best=max(matches,default=(0,-1))
        if best[0]>=.5:used.add(best[1]);stats[c]['tp']+=1
        else:stats[c]['fp']+=1;fp+=1
    missing=[j for j in range(len(gt)) if j not in used]
    for j in missing:stats[gt[j][0]]['fn']+=1
    if missing or fp:
        failures.append({'image':p.name,'false_negatives':[model.names[gt[j][0]] for j in missing],'false_positives':fp,'predicted_classes':[model.names[int(b.cls.item())] for b in pred.boxes]})
        if len(failures)<=8:
            fd=DEST/'failure_examples';fd.mkdir(exist_ok=True)
            im=Image.open(p).convert('RGB');d=ImageDraw.Draw(im)
            for c,b in gt:d.rectangle(tuple(b),outline='lime',width=4);d.text((b[0],b[1]),model.names[c],fill='lime')
            im.save(fd/(p.stem+'_ground_truth.jpg'));pred.save(filename=str(fd/(p.stem+'_prediction.jpg')))
(DEST/'failure_cases.json').write_text(json.dumps({'confidence':.25,'nms_iou':.7,'match_iou':.5,'per_class_counts':{model.names[c]:v for c,v in stats.items()},'cases':failures},indent=2))
# Reuse the provided development API and Flutter HTTP example; correct ndarray color order.
with zipfile.ZipFile(ROOT/'01_notebooks/SKKU_1398_Colab_Project.zip') as z:
    server=z.read('SKKU_1398_Colab_Project/src/server.py').decode().replace('source=np.asarray(rgb)','source=np.asarray(rgb)[..., ::-1].copy()')
    (DEST/'server.py').write_text(server)
    (DEST/'flutter_http_example.dart').write_bytes(z.read('SKKU_1398_Colab_Project/docs/flutter_http_example.dart'))
shutil.copy2('/content/skku1398_ml.py',DEST/'skku1398_ml.py')
os.environ['SKKU_MODEL_PATH']=str(RUN/'baseline/weights/best.pt');sys.path.insert(0,str(DEST))
from fastapi.testclient import TestClient
import server
client=TestClient(server.app);photo=next((DATA/'test/images').glob('*'))
with photo.open('rb') as f:response=client.post('/predict',files={'file':(photo.name,f,'image/jpeg')})
assert response.status_code==200,response.text
(DEST/'http_api_validation.json').write_text(json.dumps({'status_code':response.status_code,'response':response.json(),'tested':'FastAPI TestClient in Colab; no Flutter runtime or phone tested'},indent=2))
metrics=json.loads((RUN/'public_grouped_test_metrics.json').read_text());audit=json.loads((ROOT/'02_dataset/fitfuel_v2_grouped_v1/audit.json').read_text());bench=json.loads((RUN/'inference_benchmark.json').read_text())
contract=json.loads((DEST/'tensor_contract.json').read_text()) if (DEST/'tensor_contract.json').exists() else None
lines=['# SKKU 1398 운동기구 인식 baseline 실행 보고서','', '범위: FitFuel 공개 데이터에서 재분할한 proxy 모델. 센터 현장 성능 및 Flutter 휴대폰 실행은 검증하지 않았습니다.','', '## 실행 결과',f"- mAP@0.5: {metrics['map50']:.4f}",f"- mAP@0.5:0.95: {metrics['map50_95']:.4f}",f"- Precision: {metrics['precision']:.4f}",f"- Recall: {metrics['recall']:.4f}",f"- 가중치: {metrics['weights_bytes']} bytes",f"- T4 단일 사진 전체 처리 중앙값: {bench['end_to_end_ms_median']:.2f} ms (휴대폰 속도 아님)",'','| 클래스 | AP50 | AP50:95 | Precision | Recall |','|---|---:|---:|---:|---:|']
for c,v in metrics['class_metrics'].items():lines.append(f"| {c} | {v['ap50']:.4f} | {v['ap50_95']:.4f} | {v['precision']:.4f} | {v['recall']:.4f} |")
lines+=['','## 데이터 및 한계','- 원본 다운로드의 data.yaml은 28종. 공개 페이지의 29종 표기와 차이가 있어 실제 파일 기준으로 처리했습니다.','- 공식 test에는 목표 5종이 없고 valid에는 Chest Press가 없어 공식 분할을 사용할 수 없었습니다.','- 원본 파일명 및 0/90/180/270도 pHash 유사도 그룹으로 묶어 seed 42로 새 분할을 학습 전에 고정했습니다.','- 최종 학습 753장, 검증 147장, 테스트 156장. 증강본이 포함되어 이미지 수가 독립 촬영 수를 뜻하지 않습니다.','- 잘못된 라벨 이미지 48장 제외, 정확 중복 195장 제외, 추가 유사 그룹 이미지 6장 격리.','- 5종이 없는 사진 3412장은 이번 학습 및 평가에서 제외했습니다. 실제 헬스장의 미지원 기구/배경 오탐률은 검증되지 않았습니다.','- 제품·광고 사진과 회전 이미지가 많습니다. 세션/촬영자/웹 출처 독립성 및 모든 라벨의 완전성은 보장하지 않습니다.','- 원본 5개 ID: Treadmill 27 → 0, Chest Press Machine 7 → 1, Shoulder Press Machine 23 → 2, Leg Press Machine 18 → 3, Smith Machine 24 → 4.','- Chest/Leg Press는 시티드 세부 형태를 보증하지 않습니다.','- 학습 중 모델 선택은 검증 데이터 기준. 테스트 결과에 따른 재학습/임계값 조정은 하지 않았습니다.','- Precision/Recall은 Ultralytics 평가 지표이며 앱 confidence 0.25에서의 동작점과 다릅니다. 별도 failure_cases.json이 confidence 0.25, matching IoU 0.5의 TP/FP/FN을 담습니다.','', '## 출처','FitFuel, All Gym Equipment v2, Roboflow Universe (2024), CC BY 4.0. 클래스 필터·라벨 매핑·그룹 분할 변경.', 'https://universe.roboflow.com/fitfuel/all-gym-equipment/dataset/2','https://docs.ultralytics.com/integrations/litert','', '## 앱 연결','- Python: skku_inference.py의 predict_photo(image_path, best_pt_path).','- HTTP: server.py의 /predict. FastAPI TestClient로 사진 전송과 JSON 응답 확인.','- flutter_http_example.dart는 기존 HTTP 연결 예제이며 온디바이스 구현 또는 실행 완료를 뜻하지 않습니다.','- 출력: class_id, class_name, confidence, bbox_xyxy_pixels = [x1,y1,x2,y2]; 원본 이미지 좌표.','- 모바일 내보내기 및 실제 텐서 계약: tensor_contract.json, conversion_verification.json 확인.','- EXIF 방향 보정 후 RGB, 종횡비 유지 letterbox 640, 패딩 114, float32/255. 실제 텐서 layout은 tensor_contract.json 기준.','- YOLO11 출력은 4개 좌표 + 5개 클래스 점수. 별도 objectness는 없습니다. 클래스별 NMS 후 letterbox 패딩을 빼고 resize 비율로 나눕니다.','- 카메라 YUV→RGB, 회전, 미러링, 화면 미리보기 좌표 변환은 앱에서 별도로 검증해야 합니다.']
if contract:lines+=['', '실제 내보내기 계약:', '```json',json.dumps(contract,ensure_ascii=False,indent=2),'```']
(DEST/'REPORT_KO.md').write_text('\n'.join(lines))
for f in ['public_grouped_test_metrics.json','inference_benchmark.json','dataset_audit.json','status.json']:shutil.copy2(RUN/f,DEST/f)
print('FINAL_REPORT_SAVED',str(DEST),'failure_images',len(failures),'HTTP_TEST',response.status_code,flush=True)
shutil.copy2(RUN/'baseline/weights/best.pt',DEST/'skku1398_yolo11n_best.pt')
shutil.copy2(ROOT/'01_notebooks/flutter_litert_example.dart',DEST/'flutter_litert_example.dart')
with (DEST/'REPORT_KO.md').open('a') as f:
    f.write('\n\nFlutter 온디바이스 예제: flutter_litert_example.dart. 입력/출력 형태 검사와 클래스별 NMS를 포함합니다. Dart 컴파일 및 휴대폰 실행은 수행하지 않았습니다. tflite_flutter 네이티브 런타임과 모델 연산자 호환성은 앱팀 검증이 필요합니다.\n참고: https://github.com/tensorflow/flutter-tflite\n')
(DEST/'artifact_integrity.json').write_text(json.dumps({str(p.relative_to(DEST)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in DEST.rglob('*') if p.is_file() and p.name!='artifact_integrity.json'},indent=2))
