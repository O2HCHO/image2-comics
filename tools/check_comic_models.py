"""Offline real-page QA: no generation API, writes only build/comic-qa."""
from pathlib import Path
import json
import sys
import time
import ctypes
from ctypes import wintypes

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
from comic_color.text_protection import detection, fuse, rgb, png, align, regions
from comic_color.comic_models import engine
from comic_color.config import ORIGINAL


class Memory(ctypes.Structure):
    _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]


def main():
    out = ROOT/'build/comic-qa'
    out.mkdir(parents=True,exist_ok=True)
    report = []
    model = engine()
    manifests = list((ROOT/'outputs').rglob('tasks.json'))
    entries = []
    for manifest in manifests:
        output = manifest.parent.parent
        for record in json.loads(manifest.read_text(encoding='utf-8')).get('entries',{}).values():
            source = Path(record['root'])/record['source']
            result = output/record.get('raw_output',record.get('output',''))
            if source.is_file() and result.is_file() and record.get('status')=='success' and not record.get('text_mask'):
                entries.append((source,result,output,record))
    # Include one page from the previous comparison, then a spread of available pages.
    selected = entries[::max(1,len(entries)//4)][:4]
    seen = set()
    for index,(source,result,output,record) in enumerate(selected):
        if source in seen: continue
        seen.add(source)
        folder = out/f'page-{index+1}'
        folder.mkdir(exist_ok=True)
        started = time.perf_counter()
        data = detection(source,folder)
        item = {'source':str(source),'generated':str(result),'regions':len(regions(data)),'bubbles':len(data['bubbles']),'skipped':len(data['skipped']),'issues':data['issues'],'detection_seconds':time.perf_counter()-started}
        original = rgb(source)
        try:
            started = time.perf_counter()
            content,mask,info = fuse(source,result.read_bytes(),data,ORIGINAL)
            cleanup = info.pop('cleanup_png')
            item.update(success=True,fusion_seconds=time.perf_counter()-started,**info)
            (folder/'fused.png').write_bytes(content)
            (folder/'restore.png').write_bytes(mask)
            (folder/'cleanup.png').write_bytes(cleanup)
            aligned,_ = align(original,rgb(result),regions(data))
            merged = rgb(content)
            changed = (rgb(mask)[:,:,0]>0)|(rgb(cleanup)[:,:,0]>0)
            assert np.array_equal(merged[~changed],aligned[~changed]), 'Pixels outside both masks changed'
            overlay = original.copy()
            amount = rgb(mask)[:,:,0,None]/255*.65
            overlay = np.rint(overlay*(1-amount)+np.array([255,30,30])*amount).astype(np.uint8)
            for bubble in data['bubbles']:
                cv2.polylines(overlay,[np.asarray(bubble['contour'],np.int32)],True,(30,100,255),1)
            tiles = [original,aligned,merged,overlay,np.repeat(rgb(cleanup)[:,:,:1],3,axis=2)]
            canvas = Image.new('RGB',(500*len(tiles),740),'#e8edf4')
            draw = ImageDraw.Draw(canvas)
            for col,(tile,label) in enumerate(zip(tiles,['Original','Generated aligned','CTD fused','Restore mask / bubbles','Cleanup mask'])):
                image = Image.fromarray(tile); image.thumbnail((480,700))
                canvas.paste(image,(col*500+10,30));draw.text((col*500+10,8),label,fill='black')
            canvas.save(folder/'comparison.jpg',quality=94)
        except Exception as error:
            item.update(success=False,error=str(error))
        report.append(item)
        print(json.dumps(item,ensure_ascii=False),flush=True)
    memory = Memory();memory.cb=ctypes.sizeof(memory)
    kernel = ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Memory),wintypes.DWORD]
    psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(memory),memory.cb)
    summary = {'model_load_seconds':model.load_seconds,'peak_working_set_mb':memory.PeakWorkingSetSize/1024**2,'pages':report,'all_fusions_passed':bool(report) and all(x['success'] for x in report)}
    (out/'report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
