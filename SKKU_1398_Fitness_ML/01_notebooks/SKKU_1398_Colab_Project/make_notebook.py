from pathlib import Path
import nbformat as nbf

root = Path(__file__).parent
library_code = (root / 'src' / 'skku1398_ml.py').read_text(encoding='utf-8')
nb=nbf.v4.new_notebook()
nb.metadata.update({
  'colab': {'provenance': [], 'name':'SKKU_1398_5class_Colab_Training.ipynb'},
  'kernelspec': {'name':'python3','display_name':'Python 3'},
  'language_info':{'name':'python'},
  'accelerator':'GPU',
})
md=lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))
code=lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip()))

md('''
# 성균관대 자연과학캠퍼스 1398피트니스센터 · 운동기구 인식 첫 모델

**실행 목표:** 공개 운동기구 데이터를 이용해 **5종 객체탐지 기준 모델**을 만들어 학습·평가하고, 현장 사진으로 별도 검증할 준비를 합니다.

> **반드시 구분:** 이것은 일반 헬스장 사진으로 학습한 **공개 데이터 기반 proxy 모델**입니다. 1398피트니스센터의 실제 기구와 외형이 일치한다고 검증되지 않았습니다. 센터 촬영 사진을 추가하고, 독립 테스트를 해야 현장 인식 성능을 말할 수 있습니다.

첫 5종: `treadmill / seated_chest_press / shoulder_press / seated_leg_press / smith_machine`

데이터 출처: [FitFuel / All Gym Equipment v2](https://universe.roboflow.com/fitfuel/all-gym-equipment/dataset/2), Roboflow Universe, **CC BY 4.0**. 이 데이터의 원래 범주인 `Chest Press Machine`, `Leg Press Machine`은 센터의 **SEATED** 세부 형태와 정확히 대응하지 않을 수 있습니다.

데이터는 Roboflow 계정의 **Private API Key**로 다운로드하거나, 본인이 준비한 YOLO ZIP 파일로 시작합니다. 키는 **Colab 비밀 키 메뉴**의 `ROBOFLOW_API_KEY`에 넣으며, 코드 셀에 직접 적지 않습니다. 사진과 라벨은 서버/Drive에 저장하지만 사용자 사진을 Roboflow에 자동 업로드하지 않습니다.

실행 순서: ① 환경/Drive → ② 데이터 다운로드 → ③ 5종 정리·정답 검수 → ④ 학습 → ⑤ 평가·추론 → ⑥ 모바일 변환(옵션) → ⑦ Drive 결과 보관.
''')

md('''## 1. 설정: 첫 실행은 반드시 `smoke`부터

- `smoke`: 소량의 공개 데이터를 학습하여 연결만 검증. 성능 수치를 보고하지 않습니다.
- `baseline`: 데이터 검수 후 전체 학습·평가. 이것도 **센터 실환경 성능이 아닙니다**.
- `DATASET_ZIP_NAME`: Roboflow 대신 이미 갖고 있는 YOLO 데이터 ZIP을 Drive의 `02_dataset` 폴더에 넣었다면 파일명으로 설정합니다.
- `RUN_MOBILE_EXPORT`: 실제 학습 완료 후 `True`로 바꿔 LiteRT 변환을 진행합니다.
''')
code('''
RUN_MODE = "smoke"             # "smoke" 또는 "baseline"
DATASET_ZIP_NAME = None       # 예: "my_center_yolo_dataset.zip" (Drive/02_dataset)
RUN_MOBILE_EXPORT = False     # 모델 학습 후 True로 바꿀 수 있음
RUN_PHOTO_UPLOAD = False      # 본인 사진을 직접 넣어 예측하려면 True
DISPLAY_CONFIDENCE = 0.25
IMAGE_SIZE = 320 if RUN_MODE == "smoke" else 640
EPOCHS = 2 if RUN_MODE == "smoke" else 30
BATCH = 8 if RUN_MODE == "smoke" else 16
SEED = 42
GOOGLE_DRIVE_ROOT_FOLDER = "SKKU_1398_Fitness_ML"
assert RUN_MODE in ("smoke", "baseline")
print("설정:", RUN_MODE, "입력 크기:", IMAGE_SIZE, "epochs:", EPOCHS)
''')
md('''## 2. 패키지 설치 및 GPU 확인

유료 Colab이어도 GPU가 자동으로 붙지는 않습니다. **런타임 → 런타임 유형 변경**에서 GPU를 선택한 후 실행하세요. 첫 실험은 CPU에서도 동작할 수 있으나 느릴 수 있습니다.
''')
code('''
import os, sys, subprocess
subprocess.check_call([sys.executable,"-m","pip","install","-q", "ultralytics>=8.3,<9", "roboflow", "PyYAML", "Pillow"])
import torch
from ultralytics import YOLO
print("Python",sys.version.split()[0],"| torch",torch.__version__,"| CUDA:",torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:",torch.cuda.get_device_name(0))
else:
    print("주의: GPU가 확인되지 않았습니다. smoke 모드만 권장합니다.")
''')
md('''## 3. Google Drive 연결

여기서 나타나는 Google 인증창에서는 **본인이 계정과 접근 권한을 확인**해야 합니다. 인증 코드나 비밀번호를 채팅에 입력하지 마세요.
''')
code('''
from pathlib import Path
from datetime import datetime, timezone
from google.colab import drive

drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive")/GOOGLE_DRIVE_ROOT_FOLDER
if not DRIVE_ROOT.exists():
    raise FileNotFoundError(f"프로젝트 Drive 폴더가 보이지 않습니다: {DRIVE_ROOT}. 구글 계정 및 Drive 마운트를 확인하세요.")
for folder in ("01_notebooks", "02_dataset", "03_training_runs", "04_exports_and_reports"):
    (DRIVE_ROOT/folder).mkdir(parents=True,exist_ok=True)
WORK=Path("/content/skku1398_workspace")
WORK.mkdir(parents=True,exist_ok=True)
RUN_NAME=f"skku1398_public_proxy_{RUN_MODE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
DRIVE_RUN=DRIVE_ROOT/"03_training_runs"/RUN_NAME
DRIVE_RUN.mkdir(parents=True,exist_ok=True)
print("Drive 작업 위치:", DRIVE_ROOT)
print("이번 실험 보관 위치:", DRIVE_RUN)
''')
md('''## 4. 내장된 데이터 가공 코드 설치

별도 Python 파일 없이 **이 노트북 하나만** 올려도 실행할 수 있도록 코드를 내장했습니다. ZIP 패키지의 `src/skku1398_ml.py`와 동일한 모듈입니다.
''')
code('''
LIB_CODE = ''' + repr(library_code) + '''
from pathlib import Path
LIB_PATH=Path('/content/skku1398_ml.py')
LIB_PATH.write_text(LIB_CODE,encoding='utf-8')
import sys
sys.path.insert(0,'/content')
from skku1398_ml import CLASS_NAMES, prepare_yolo_dataset, make_annotation_preview, predictions_to_json
print("모델 5개 클래스:",CLASS_NAMES)
''')
md('''## 5. 공개 데이터 또는 Drive ZIP 다운로드

Roboflow Universe는 데이터를 제공하지만 **계정별 다운로드 인증이 필요할 수 있습니다**. Colab 열쇠(비밀 키) 메뉴에 `ROBOFLOW_API_KEY`가 없으면 다운로드를 중단하고 명확한 오류를 보여줍니다. 또는 데이터셋을 YOLO 포맷 ZIP으로 내려받아 Drive `02_dataset`에 넣고 `DATASET_ZIP_NAME`을 지정하면 됩니다.
''')
code('''
from pathlib import Path
import shutil, zipfile
RAW_ROOT=WORK/'raw_source'
RAW_ROOT.mkdir(parents=True,exist_ok=True)

def safe_unzip(zip_path: Path, dest: Path) -> None:
    dest=dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target=(dest/info.filename).resolve()
            if target!=dest and dest not in target.parents:
                raise ValueError('ZIP contains an unsafe path')
        zf.extractall(dest)

if DATASET_ZIP_NAME:
    zip_input=DRIVE_ROOT/'02_dataset'/DATASET_ZIP_NAME
    if not zip_input.is_file():
        raise FileNotFoundError(f"Drive에 다음 ZIP이 보이지 않습니다: {zip_input}")
    local_zip=WORK/zip_input.name
    shutil.copy2(zip_input,local_zip)
    safe_unzip(local_zip,RAW_ROOT)
    SOURCE_ROOT=RAW_ROOT
    print('사용자 YOLO ZIP 불러오기 완료:',local_zip)
else:
    from google.colab import userdata
    try:
        ROBOFLOW_KEY=userdata.get('ROBOFLOW_API_KEY')
    except Exception as exc:
        ROBOFLOW_KEY=None
    if not ROBOFLOW_KEY:
        raise RuntimeError('Colab 비밀 키 메뉴에 ROBOFLOW_API_KEY를 등록하거나 DATASET_ZIP_NAME을 지정하세요.')
    from roboflow import Roboflow
    rf=Roboflow(api_key=ROBOFLOW_KEY)
    dataset=rf.workspace('fitfuel').project('all-gym-equipment').version(2).download('yolov8',location=str(RAW_ROOT/'fitfuel_v2'))
    SOURCE_ROOT=Path(dataset.location)
    del ROBOFLOW_KEY  # 로그에 키를 출력하지 않음
    print('공개 데이터 다운로드 완료:',SOURCE_ROOT)
''')
md('''## 6. 클래스 정리, 중복 검사, 정답 데이터 생성

- 원본 라벨에서 지정한 5종만 추출합니다.
- 드물게 있는 정확히 동일한 파일은 **학습/검증/테스트에 중복 사용하지 않게 제거**합니다.
- 유사 이미지는 별도 CSV로 남기므로 반드시 육안 검토하세요. 자동 중복 검사만으로 독립성이 보장되는 것은 아닙니다.
- 라벨에 문제가 있으면 `label_issues.csv`에 기록합니다.
- 이 공개 데이터는 `Chest Press Machine` 등의 일반 클래스이므로 **센터 기구와 일치하지 않을 수 있습니다**.
''')
code('''
import json
PREPARED=WORK/'prepared'/RUN_NAME
report=prepare_yolo_dataset(
    SOURCE_ROOT, PREPARED,
    mode=RUN_MODE, max_train=200, max_val=60, max_test=40, seed=SEED
)
print(json.dumps(report['source_split_counts'],ensure_ascii=False,indent=2))
print('클래스별 박스:',json.dumps(report['box_counts_by_split'],ensure_ascii=False,indent=2))
print('오류 행 수:',report['label_issue_count'])
print('유사 중복 경고:',report['near_duplicate_warning_count'])
print('원본 → 5종 매핑:',json.dumps(report['id_mapping'],ensure_ascii=False,indent=2))
for f in (PREPARED/'reports').iterdir():
    if f.is_file(): shutil.copy2(f, DRIVE_RUN/f.name)
shutil.copy2(PREPARED/'data.yaml',DRIVE_RUN/'data_reference.yaml')
''')
md('''## 7. 실제 정답 박스 검수

아래 이미지에서 **물체와 정답 상자가 일치하는지** 확인하세요. 심각한 오라벨, 너무 작은 박스, 누락이 많으면 학습 전에 데이터를 수정해야 합니다. 이 단계의 출력은 모델 예측이 아니라 **원본 라벨 검수 그림**입니다.
''')
code('''
from IPython.display import display
from PIL import Image
PREVIEW=make_annotation_preview(PREPARED,WORK/'annotation_review.jpg',max_images=9)
display(Image.open(PREVIEW))
shutil.copy2(PREVIEW,DRIVE_RUN/'annotation_review.jpg')
''')
md('''## 8. YOLO11n 학습

`smoke`는 2회만 학습하며 **품질 지표로 사용하지 마세요**. `baseline`은 전체 공개 데이터를 이용하므로 시간이 더 걸립니다. 학습 중간 가중치는 5 epochs마다 Drive에 백업합니다.
''')
code('''
import shutil, torch, json
from pathlib import Path
from ultralytics import YOLO
MODEL=YOLO('yolo11n.pt')
DEVICE=0 if torch.cuda.is_available() else 'cpu'
if DEVICE=='cpu' and RUN_MODE=='baseline':
    print('경고: baseline을 CPU에서 학습하면 매우 느릴 수 있습니다.')

def backup_to_drive(trainer):
    if (trainer.epoch+1)%5==0 or (trainer.epoch+1)==EPOCHS:
        path=Path(trainer.last)
        if path.exists():
            shutil.copy2(path, DRIVE_RUN/'last_checkpoint.pt')
MODEL.add_callback('on_model_save',backup_to_drive)
training=MODEL.train(
    data=str(PREPARED/'data.yaml'),
    imgsz=IMAGE_SIZE, epochs=EPOCHS,
    batch=BATCH if DEVICE != 'cpu' else 4,
    seed=SEED, device=DEVICE,
    workers=2, cache=False, patience=12 if RUN_MODE=='baseline' else 2,
    project=str(WORK/'runs'),name=RUN_NAME,exist_ok=False,
    plots=True, verbose=True,
)
TRAIN_DIR=Path(MODEL.trainer.save_dir)
BEST=TRAIN_DIR/'weights'/'best.pt'
if not BEST.exists():
    raise FileNotFoundError('학습 종료 후 best.pt 파일이 생성되지 않았습니다')
shutil.copy2(BEST,DRIVE_RUN/'best.pt')
for filename in ('results.csv','results.png','args.yaml','confusion_matrix.png','confusion_matrix_normalized.png'):
    f=TRAIN_DIR/filename
    if f.exists(): shutil.copy2(f,DRIVE_RUN/f.name)
print('학습된 모델 저장:',DRIVE_RUN/'best.pt')
''')
md('''## 9. 별도 분할 평가

여기서 나오는 수치는 **FitFuel 공개 데이터 내 평가 결과**입니다. 1398피트니스센터 현장 성능이 아닙니다. `smoke` 결과는 연결 시험용입니다. 테스트는 모델 선정 이후 한 번만 수행하고, 이 값을 보고 가중치나 임계값을 재선정하지 마세요.
''')
code('''
from ultralytics import YOLO
import json
MODEL_FINAL=YOLO(str(DRIVE_RUN/'best.pt'))
val_metrics=MODEL_FINAL.val(data=str(PREPARED/'data.yaml'),split='test',
                            imgsz=IMAGE_SIZE,batch=8,device=DEVICE,plots=True,
                            project=str(WORK/'eval'), name=RUN_NAME,exist_ok=True)
result_metrics={
 'mode':RUN_MODE,'scope':'public_proxy_not_center','classes':CLASS_NAMES,
 'map50':float(val_metrics.box.map50),
 'map50_95':float(val_metrics.box.map),
 'precision_mean':float(val_metrics.box.mp),
 'recall_mean':float(val_metrics.box.mr),
 'per_class_map50_95':{CLASS_NAMES[i]:float(s) for i,s in enumerate(val_metrics.box.maps)},
 'warning':'This is NOT validated on real SKKU 1398 Fitness Center photos.'
}
(DRIVE_RUN/'public_test_metrics.json').write_text(json.dumps(result_metrics,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(result_metrics,indent=2,ensure_ascii=False))
''')
md('''## 10. 실제 추론: 사진 → 기구명·박스·점수 JSON

`RUN_PHOTO_UPLOAD=True`로 바꾸면 본인 사진을 직접 업로드할 수 있습니다. 기본값은 공개 평가 분할의 샘플 한 장을 사용합니다. 업로드한 사진은 학습·평가 자료에 자동으로 합쳐지지 않습니다.
''')
code('''
import json
from pathlib import Path
from PIL import Image
from IPython.display import display
if RUN_PHOTO_UPLOAD:
    from google.colab import files
    uploaded=files.upload()
    if not uploaded:
        raise RuntimeError('사진이 업로드되지 않았습니다')
    photo_name=next(iter(uploaded))
    PHOTO=Path('/content')/photo_name
    PHOTO.write_bytes(uploaded[photo_name])
else:
    PHOTO=next((PREPARED/'test'/'images').iterdir())
output=MODEL_FINAL.predict(source=str(PHOTO),imgsz=IMAGE_SIZE,
                           conf=DISPLAY_CONFIDENCE,device=DEVICE,verbose=False)[0]
response=predictions_to_json(output,threshold=DISPLAY_CONFIDENCE)
print(json.dumps(response,indent=2,ensure_ascii=False))
annotated=Image.fromarray(output.plot()[...,::-1])
annotated.save(DRIVE_RUN/'single_photo_prediction.jpg')
(DRIVE_RUN/'single_photo_result.json').write_text(json.dumps(response,ensure_ascii=False,indent=2),encoding='utf-8')
display(annotated)
''')
md('''## 11. 모바일 파일 변환 (선택)

아래 단계는 `RUN_MOBILE_EXPORT=True`일 때만 실행합니다. 현재 Ultralytics는 `format="litert"`로 `.tflite` 형식의 모델을 내보낼 수 있습니다. 하지만 **변환 성공은 Flutter 실기기 성공과 다르며**, 앱팀의 런타임에서 전처리·출력 해석을 별도 비교해야 합니다.
''')
code('''
from pathlib import Path
import json
if RUN_MOBILE_EXPORT:
    tflite_path=Path(MODEL_FINAL.export(format='litert',imgsz=IMAGE_SIZE))
    shutil.copy2(tflite_path, DRIVE_ROOT/'04_exports_and_reports'/tflite_path.name)
    print('모바일 변환 파일:', DRIVE_ROOT/'04_exports_and_reports'/tflite_path.name)
else:
    print('모바일 변환 건너뜀. 학습 확인 후 RUN_MOBILE_EXPORT=True로 전환하세요.')
''')
md('''## 12. 실행 보고서 저장

최종 보고서에는 **실제 실행된 범위**가 기록됩니다. 센터 사진 추가 수집과 독립 테스트를 수행하기 전에는 **정식 현장용 모델로 납품하면 안 됩니다**.
''')
code('''
import json
from datetime import datetime
summary={
 'run_name':RUN_NAME,'created_local_time':datetime.now().isoformat(timespec='seconds'),
 'source':'FitFuel / All Gym Equipment v2 (CC BY 4.0)' if not DATASET_ZIP_NAME else str(DATASET_ZIP_NAME),
 'classes':CLASS_NAMES, 'image_size':IMAGE_SIZE, 'epochs_requested':EPOCHS,
 'run_mode':RUN_MODE, 'trained_model':str(DRIVE_RUN/'best.pt'),
 'data_audit':str(DRIVE_RUN/'dataset_audit.json'),
 'test_metrics':str(DRIVE_RUN/'public_test_metrics.json'),
 'next_step':'Real SKKU 1398 equipment photos + independent session-wise held-out evaluation; Flutter on-device validation',
 'not_claimed':'No accuracy claim for SKKU gym without site photos'
}
(DRIVE_RUN/'run_report.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('실행 결과가 Google Drive에 저장됐습니다:',DRIVE_RUN)
''')

nbf.validate(nb)
out=root/'SKKU_1398_5class_Colab_Training.ipynb'
nbf.write(nb,out)
print(out,len(nb.cells),'cells',out.stat().st_size,'bytes')
