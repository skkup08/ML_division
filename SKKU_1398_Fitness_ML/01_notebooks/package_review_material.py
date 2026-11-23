from pathlib import Path
import json, zipfile, time, shutil
ROOT=Path('/content/drive/MyDrive/SKKU_1398_Fitness_ML/02_dataset/SKKU1398_equipment_v1')
WORK=Path('/content/SKKU1398_equipment_v1_work')
def main():
 summary=json.loads((ROOT/'reports/automated_build_summary.json').read_text())
 rows=json.loads((ROOT/'reports/class_counts.json').read_text())
 sources=json.loads((ROOT/'manifests/sources.json').read_text())
 out=WORK/'inspection_material.zip'
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted((WORK/'review_pack/contact_sheets').glob('*.jpg')):
   if int(p.name[:2]) in {2,4,17,25}:z.write(p,'sheets/'+p.name)
  z.write(WORK/'review_pack/sheet_index.json','sheet_index.json')
  for p in (ROOT/'reports').glob('*.json'):z.write(p,'reports/'+p.name)
  z.write(ROOT/'manifests/images.jsonl','images.jsonl')
 text='SKKU1398 equipment v1 — 구축 상태\n\n원본 보존 및 자동 정제 완료. 시각 검수 대기 후보와 승인된 학습셋은 별개입니다.\n'
 text+='원본 이미지: '+str(summary['raw_images'])+'\n검수 대표 이미지: '+str(summary['review_representative_images'])+'\n'
 text+='\n구성\nraw_sources: 원본 이미지·라벨·README 압축파일. 수정하지 않음.\nmanifests: 이미지별 출처·원본 파일·좌표·중복 그룹·검수 상태.\nreview: 검수 대기 대표 이미지·YOLO 제안 라벨·24장씩 검수 시트.\nyolo: 명시적으로 검수 승인된 사진만 별도 배포.\nreports: 실제 처리 수량 및 품질 검사 결과.\n'
 text+='\n자동 검사 기준\n이미지 디코딩, 유한 좌표, 정수 클래스 ID, 박스 모서리 경계, 양수 폭/높이. 오류가 있는 사진은 일부 박스만 버리고 승인하지 않고 사진 전체를 보류. EXIF 회전 사진은 라벨 동반 변환 검수 전 보류. 원본 크기 유지.\n'
 text+='\n중복 처리\nSHA256 동일 파일, 같은 출처의 증강 파일명 그룹, 원본/좌우 반전 pHash 거리 4 이내 후보를 연결 그룹으로 묶음. 그룹·대응 클래스 조합별 대표 사진 선정. 지각 해시 연결은 동일 사진 확정이나 독립 촬영 수의 보장이 아님. 큰 그룹은 과도한 병합 가능성도 있음. 임베딩 유사도/모든 회전·크롭/연속 촬영의 완전한 검사는 미완료.\n'
 text+='\n분할\n그룹 ID와 seed 1398의 결정적 해시로 80:10:10 확률 할당. 실제 장수는 승인 결과에 따라 다름. 평가 사진은 증강하지 않음. 기존 5종 모델의 holdout을 재사용한 성능으로 비교하지 않음.\n'
 text+='\n출처\n'+'\n'.join(s['url']+' | '+s['license_reported']+' | '+str(s['archive_bytes'])+' bytes' for s in sources)
 text+='\n\n기구별 후보 / 기본 검사 통과 / 대표 검수 대상\n'+'\n'.join(f"{r['center_id']:02d} {r['name_ko']}: {r['candidate_files']} / {r['technical_valid_files']} / {r['review_representatives']}" for r in rows)
 text+='\n\n0장인 메가맥스와 더블 트위스트는 학습 클래스에 자동 편입하지 않음. 공개 출처 라이선스 표시는 기록했으나 개별 사진 촬영자의 원출처는 미확인. 현장 성능 검증 및 모델 학습은 수행하지 않음.\n'
 (ROOT/'README_KO.txt').write_text(text,encoding='utf-8')
 shutil.copyfile(out,ROOT/'review/inspection_material.zip')
 shutil.copyfile(ROOT/'README_KO.txt',ROOT.parent.parent/'04_exports_and_reports/SKKU1398_equipment_v1_build_status.txt')
 print('INSPECTION_MATERIAL',str(out),out.stat().st_size,flush=True)
if __name__=='__main__':main()
