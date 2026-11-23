"""Bounded public ZIP directory reads, never video/image/archive download.
Resolves the previously frozen159-new-worker pool before any feature extraction.
It does NOT train, process videos, or report accuracy.
"""
from pathlib import Path
import requests,struct,json,gzip,hashlib,time
from datetime import datetime,timezone
ROOT=Path('/content/drive/MyDrive/Allinone/Exercise_Quality_95_Target_20261028')
OUT=ROOT/'qevd_round2_metadata';OUT.mkdir(exist_ok=True)
BASE='https://softwarecenter.qualcomm.com/api/download/software/dataset/AIDataset/Qualcomm_Exercise_Video_Dataset/'

def bounded(url,start,end=None,suffix=False):
    for attempt in range(4):
        try:
            with requests.get(url,headers={'Range':f'bytes=-{start}' if suffix else f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,timeout=(20,120)) as r:
                if r.status_code!=206:raise RuntimeError('Expected public bounded206; got'+str(r.status_code))
                data=bytearray()
                for b in r.iter_content(1024*1024):
                    data.extend(b)
                    if len(data)>32*1024**2:raise ValueError('Metadata budget exceeded32MiB')
                return bytes(data),r.headers.get('Content-Range')
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)

def directory(part):
    url=BASE+f'QEVD-FIT-300k-Part-{part}/QEVD-FIT-300k-Part-{part}.zip'
    tail,content_range=bounded(url,65536,suffix=True);i=tail.rfind(b'PK\x05\x06')
    if i<0:raise ValueError('No end record')
    h=struct.unpack_from('<4s4H2IH',tail,i);disk,cd_disk,size,offset=h[1],h[2],h[5],h[6]
    if size==0xffffffff or offset==0xffffffff:
        j=tail.rfind(b'PK\x06\x06')
        if j<0:raise ValueError('ZIP64 footer outside bounded tail')
        q=struct.unpack_from('<4sQ2H2I4Q',tail,j);disk,cd_disk,size,offset=q[4],q[5],q[8],q[9]
    if disk!=cd_disk or size>32*1024**2:raise ValueError('Unsupported central directory layout')
    cd,_=bounded(url,offset,offset+size-1);pos=0;result=[]
    while cd[pos:pos+4]==b'PK\x01\x02':
        h=struct.unpack_from('<4s6H3I5H2I',cd,pos);fn,el,cl=h[10:13]
        name=cd[pos+46:pos+46+fn].decode('utf-8','replace')
        entry={'name':name,'archive_part':part,'method':h[4],'crc32':h[7],'compressed':h[8],'uncompressed':h[9],'disk':h[13],'offset':h[16]}
        extra=cd[pos+46+fn:pos+46+fn+el];k=0
        while k+4<=len(extra):
            tag,sz=struct.unpack_from('<HH',extra,k);v=extra[k+4:k+4+sz];k+=4+sz
            if tag==1:
                q=0
                for field,width in [('uncompressed',8),('compressed',8),('offset',8),('disk',4)]:
                    if entry[field]==(0xffffffff if width==8 else 0xffff):entry[field]=int.from_bytes(v[q:q+width],'little');q+=width
        if name.endswith('.mp4'):result.append(entry)
        pos+=46+fn+el+cl
    return result,{'url':url,'content_range':content_range,'member_videos':len(result),'directory_sha256':hashlib.sha256(cd).hexdigest()}

plan=json.loads((ROOT/'qevd_round2_frozen_new_workers.json').read_text());indexes={};sources=[]
try:
    for part in (1,2,3):
        path=OUT/f'part{part}_members.json.gz'
        if path.exists():rows=json.load(gzip.open(path,'rt'));meta={'part':part,'cached':True}
        else:
            rows,meta=directory(part)
            with gzip.open(path,'wt') as f:json.dump(rows,f)
        for r in rows:
            basename=Path(r['name']).name
            if basename in indexes:raise ValueError('Duplicate source video member across archives')
            indexes[basename]=r
        sources.append(meta);print('METADATA_PART_RESOLVED',part,len(rows),flush=True)
    resolved=[];unresolved=[]
    for r in plan['records']:
        if r['video_path'] in indexes:resolved.append({**r,'zip_member':indexes[r['video_path']]})
        else:unresolved.append(r)
    # Do not silently shrink the frozen holdout denominator for unavailable files.
    result={k:v for k,v in plan.items() if k!='records'}
    result.update(status='metadata_resolution_complete',raw_videos_processed=0,resolved=len(resolved),unresolved=len(unresolved),records=resolved,unresolved_records=unresolved,original_frozen_denominator=len(plan['records']),source_metadata=sources)
    with gzip.open(OUT/'resolved_extraction_plan.json.gz','wt') as f:json.dump(result,f,ensure_ascii=False)
    report={k:v for k,v in result.items() if k not in ['records','unresolved_records','old_workers_excluded']}
    (OUT/'source_status.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2));print('ROUND2_METADATA_RESOLVED_NO_VIDEOS_PROCESSED')
except Exception as exc:
    (OUT/'failure.json').write_text(json.dumps({'error':repr(exc),'time_utc':datetime.now(timezone.utc).isoformat(),'raw_videos_processed':0},indent=2));raise
