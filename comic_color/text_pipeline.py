"""Versioned comic segmentation cache and foreground-only compositing."""
from pathlib import Path
import hashlib
import json
import logging
from io import BytesIO

import cv2
import numpy as np

from . import comic_models
from .config import atomic_bytes, atomic_json, read_json
from .imaging import fingerprint, png_bytes
from .text_geometry import closed_bubbles, safe_interior, strokes, envelope, rectangle_mask, polygon_mask, inpaint_inside
from .stage_cache import StageCache, Timing

VERSION = 'ctd-balloon-strokes-3'


def detect_page(image):
    probability, lines, blocks, candidates = comic_models.infer(image)
    bubbles, skipped = closed_bubbles(image,candidates)
    automatic, details, issues = [], [], []
    refined = np.zeros(image.shape[:2],np.uint8)
    raw = np.rint(np.clip(probability,0,1)*255).astype(np.uint8)
    for index,bubble in enumerate(bubbles):
        safe = safe_interior(image.shape,bubble)
        contained_lines = []
        for polygon in lines:
            line_mask = polygon_mask(image.shape,[polygon])>0
            if np.count_nonzero(line_mask & (safe>0))/max(1,np.count_nonzero(line_mask)) >= .95:
                contained_lines.append(polygon)
        # Reject a balloon candidate around only a fragment of a larger line
        # (a common failure on artwork and oversized sound effects).
        crossing = any(.05 < np.count_nonzero((polygon_mask(image.shape,[p])>0) & (safe>0))/max(1,np.count_nonzero(polygon_mask(image.shape,[p]))) < .95 for p in lines)
        if crossing:
            issues.append({'bubble':index,'box':bubble['box'],'reason':'文字跨越候选气泡边界，可能是背景或拟声词，自动跳过'})
            skipped.append({**bubble,'reason':'文字未完整包含于气泡内部'})
            continue
        ink,_ = strokes(image,probability,safe,contained_lines)
        evidence = (probability > .3) & (safe > 0)
        boundary = (safe>0) & (cv2.erode(safe,np.ones((5,5),np.uint8))==0)
        if np.count_nonzero((probability>.5)&boundary)>3:
            issues.append({'bubble':index,'box':bubble['box'],'reason':'原稿文字接近气泡边线，需核对归属'})
        possible,_ = strokes(image,np.ones(image.shape[:2],np.float32),safe,manual=True)
        if np.count_nonzero(possible)>10 and not np.any(evidence):
            issues.append({'bubble':index,'box':bubble['box'],'reason':'气泡内部有墨迹但模型未确认文字'})
        if np.count_nonzero(evidence) >= 3 and not ink.any():
            issues.append({'bubble':index,'box':bubble['box'],'reason':'气泡内有疑似文字但无法提取可靠字形'})
        if not ink.any():
            continue
        # Multiple lines belong to one balloon; the stored box is only a UI hit target.
        x,y,w,h = cv2.boundingRect(cv2.findNonZero(ink))
        automatic.append([x,y,x+w,y+h])
        details.append({'bubble':index,'lines':contained_lines})
        refined |= ink
    for block in blocks:
        region = rectangle_mask(image.shape,[block['box']]) > 0
        if not any(np.count_nonzero(region & (safe_interior(image.shape,b)>0))/max(1,np.count_nonzero(region)) > .9 for b in bubbles):
            skipped.append({**block,'reason':'文字不属于已确认的封闭气泡'})
    return {'automatic':automatic,'details':details,'bubbles':bubbles,'skipped':skipped,'issues':issues,'lines':lines,'blocks':blocks}, raw, refined


def detect_regions(image):
    return detect_page(image)[0]['automatic']


def _asset(data,name):
    from .text_protection import NeedsReview, rgb
    item = data.get('assets',{}).get(name)
    if item is None:
        raise NeedsReview('文字掩码缓存缺失，请重新检测')
    root = Path(data['cache_root']).resolve()
    path = (root/item['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or fingerprint(path) != item['sha256']:
        raise NeedsReview('文字掩码缓存校验失败，请重新检测')
    array = rgb(path)[:,:,0]
    if list(array.shape[::-1]) != data['size']:
        raise NeedsReview('文字掩码尺寸不匹配，请重新检测')
    return array


def _intersects(a,b):
    return max(a[0],b[0]) < min(a[2],b[2]) and max(a[1],b[1]) < min(a[3],b[3])


def detection(source,output):
    from .text_protection import cache_path, rgb, png, NeedsReview
    source,output = Path(source),Path(output).resolve()
    digest = fingerprint(source)
    path = cache_path(output,source)
    previous = read_json(path)
    if previous:
        previous['cache_root'] = str(output)
    manifest = comic_models.model_manifest()
    model_id = {name:{'sha256':manifest[name]['sha256'],'revision':manifest[name]['revision']} for name in ('ctd','bubble')}
    if previous.get('source_hash') == digest and previous.get('algorithm') == VERSION and previous.get('models') == model_id:
        try:
            _asset(previous,'strokes'); _asset(previous,'raw')
            logging.getLogger(__name__).info('original detection cache hit %s',source.name)
            return previous
        except NeedsReview:
            pass
    image = rgb(source)
    with Timing('original detection'):
        data,raw,refined = detect_page(image)
    same = previous.get('source_hash') == digest
    excluded = list(previous.get('excluded',[])) if same else []
    if same:
        excluded += [b for i,b in enumerate(previous.get('automatic',[])) if i in previous.get('removed',[])]
    data.update(source_hash=digest,algorithm=VERSION,schema=2,size=list(image.shape[1::-1]),models=model_id,
                removed=[i for i,b in enumerate(data['automatic']) if any(_intersects(b,e) for e in excluded)],
                excluded=excluded,added=previous.get('added',[]) if same else [],
                expand=previous.get('expand',1) if same else 1,feather=previous.get('feather',1) if same else 1,
                confirmed_empty=previous.get('confirmed_empty',False) if same else False,cache_root=str(output),assets={})
    token = hashlib.sha256((digest+VERSION+json.dumps(model_id,sort_keys=True)).encode()).hexdigest()[:24]
    for name,array in [('raw',raw),('strokes',refined)]:
        relative = Path('.comic-color/text')/(path.stem+'-'+token+'-'+name+'.png')
        atomic_bytes(output/relative,png(array))
        data['assets'][name] = {'path':relative.as_posix(),'sha256':fingerprint(output/relative)}
    if fingerprint(source) != digest:
        raise NeedsReview('检测期间原稿改变，请重新检测')
    atomic_json(path,data)
    return data


def regions(data):
    return [b for i,b in enumerate(data.get('automatic',[])) if i not in data.get('removed',[])] + data.get('added',[])


def selection(image,data):
    from .text_protection import NeedsReview
    shape = image.shape[:2]
    groups = []
    cached = _asset(data,'strokes') if data.get('automatic') else np.zeros(shape,np.uint8)
    for i,box in enumerate(data.get('automatic',[])):
        if i in data.get('removed',[]):
            continue
        try:
            bubble = data['bubbles'][data['details'][i]['bubble']]
        except (KeyError,IndexError) as error:
            raise NeedsReview('文字缓存格式已过期，请重新检测') from error
        allowed = safe_interior(shape,bubble)
        core = cv2.bitwise_and(cached,allowed)
        groups.append((allowed,core,False))
    for box in data.get('added',[]):
        allowed = rectangle_mask(shape,[box])
        core,_ = strokes(image,np.ones(shape,np.float32),allowed,manual=True)
        if not core.any():
            raise NeedsReview('人工补选区域无法提取可靠字形，请调整范围')
        groups.append((allowed,core,True))
    return groups


def restoration_mask(image,data):
    result = np.zeros(image.shape[:2],np.uint8)
    for allowed,core,_ in selection(image,data):
        support = envelope(core,allowed,data.get('expand',1),data.get('feather',1))
        opacity,_ = foreground_layer(image,core,support)
        result = np.maximum(result,np.rint(opacity*255).astype(np.uint8))
    return result


def foreground_layer(original,core,support):
    gray = cv2.cvtColor(original,cv2.COLOR_RGB2GRAY).astype(np.float32)
    paper = cv2.morphologyEx(original,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))).astype(np.float32)
    bg = cv2.cvtColor(paper,cv2.COLOR_RGB2GRAY)
    fg = float(np.percentile(gray[core>0],5))
    opacity = np.clip((bg-gray)/np.maximum(bg-fg,1),0,1)
    opacity[(bg-gray)<4] = 0
    opacity *= support/255.
    opacity = np.rint(opacity*255)/255.
    foreground = np.clip((original.astype(np.float32)-(1-opacity[...,None])*paper)/np.maximum(opacity[...,None],1e-5),0,255)
    return opacity,foreground


def make_mask(size,boxes,expand,feather,background=None):
    if background is None:
        raise ValueError('字形掩码需要原稿图像，不能以矩形代替')
    allowed = rectangle_mask(background.shape,boxes)
    core,_ = strokes(background,np.ones(background.shape[:2],np.float32),allowed,manual=True)
    return envelope(core,allowed,expand,feather)


def fuse(source,colored,data,ratio,progress=None):
    from .text_protection import NeedsReview
    if fingerprint(source) != data.get('source_hash'):
        raise NeedsReview('原稿改变，文字区域已失效，请重新检测')
    if data.get('algorithm') != VERSION:
        raise NeedsReview('文字缓存版本已过期，请重新检测')
    if data.get('automatic'):
        _asset(data,'strokes')
    content = colored if isinstance(colored,bytes) else Path(colored).read_bytes()
    cache = StageCache(data.get('cache_root'))
    key = cache.key('final-v2',[VERSION,data,hashlib.sha256(content).hexdigest(),ratio])
    cached = cache.load(key)
    if cached:
        blobs,info = cached
        if all(k in blobs for k in ('result.png','restore.png','cleanup.png')):
            if progress:
                progress('复用已校验的融合预览')
            return blobs['result.png'],blobs['restore.png'],{**info,'cleanup_png':blobs['cleanup.png'],'cache_hit':True}
    with Timing('fusion total'):
        result,mask,info = _fuse(source,content,data,ratio,progress)
    cache.save(key,{'result.png':result,'restore.png':mask,'cleanup.png':info['cleanup_png']},
               {k:v for k,v in info.items() if k != 'cleanup_png'})
    return result,mask,info


def _fuse(source,colored,data,ratio,progress=None):
    from .text_protection import NeedsReview, rgb, png, align
    original = rgb(source)
    if fingerprint(source) != data.get('source_hash'):
        raise NeedsReview('原稿改变，文字区域已失效，请重新检测')
    if data.get('algorithm') != VERSION:
        raise NeedsReview('文字缓存版本已过期，请重新检测')
    boxes = regions(data)
    active_bubbles = {data['details'][i]['bubble'] for i in range(len(data.get('automatic',[]))) if i not in data.get('removed',[])} if data.get('details') else set()
    excluded = [b for i,b in enumerate(data.get('automatic',[])) if i in data.get('removed',[])]
    unresolved = [issue for issue in data.get('issues',[]) if not any(_intersects(issue.get('box',[0,0,0,0]),b) for b in excluded+data.get('added',[]))]
    if unresolved and not data.get('confirmed_empty'):
        raise NeedsReview('气泡内疑似文字无法可靠提取，请人工核对或补选')
    if not boxes:
        if data.get('issues') and not data.get('confirmed_empty'):
            raise NeedsReview('气泡内疑似文字需核对')
        zero = np.zeros(original.shape[:2],np.uint8)
        cleanup = png(zero)
        generated = rgb(colored)
        if generated.shape != original.shape:
            generated, alignment = align(original,generated,[])
        else:
            alignment = {'coverage':1.,'edge_score':1.,'local_scores':[]}
        return png_bytes(png(generated),ratio),png(zero),{'regions':0,'algorithm':VERSION,'models':data.get('models',{}),'cleanup_png':cleanup,**alignment,'zero_mask':True}
    groups = selection(original,data)
    cache = StageCache(data.get('cache_root'))
    align_key = cache.key('alignment-v2',[VERSION,data['source_hash'],hashlib.sha256(colored).hexdigest(),boxes])
    aligned_cache = cache.load(align_key)
    if progress:
        progress('读取对齐缓存' if aligned_cache else '原稿与生成图对齐')
    if aligned_cache and 'aligned.png' in aligned_cache[0]:
        aligned,info = rgb(aligned_cache[0]['aligned.png']),aligned_cache[1]
    else:
        with Timing('alignment'):
            aligned,info = align(original,rgb(colored),boxes)
        cache.save(align_key,{'aligned.png':png(aligned)},info)
    try:
        detect_key = cache.key('generated-text-v1',[VERSION,data.get('models'),hashlib.sha256(aligned.tobytes()).hexdigest()])
        detected = cache.load(detect_key)
        if progress:
            progress('读取生成图文字缓存' if detected else '检测生成图文字')
        if detected and 'probability.npy' in detected[0]:
            try:
                probability = np.load(BytesIO(detected[0]['probability.npy']),allow_pickle=False)
                if probability.shape != aligned.shape[:2] or not np.isfinite(probability).all():
                    raise ValueError('invalid probability')
                lines = detected[1]['lines']
            except (ValueError,KeyError):
                detected = None
        else:
            detected = None
        if detected is None:
            with Timing('generated detection'):
                probability,lines,_,_ = comic_models.infer(aligned,include_bubbles=False)
            buffer = BytesIO()
            np.save(buffer,probability,allow_pickle=False)
            cache.save(detect_key,{'probability.npy':buffer.getvalue()},{'lines':lines})
    except Exception as error:
        raise NeedsReview('生成图文字检测失败：'+str(error)) from error
    merged = aligned.copy()
    cleanup = np.zeros(original.shape[:2],np.uint8)
    restore = np.zeros_like(cleanup)
    if progress:
        progress('精修字形、清字与合成')
    for allowed,core,manual in groups:
        generated,_ = strokes(aligned,probability,allowed,lines)
        for image,ink in ((original,core),(aligned,generated)):
            background_pixels = (allowed>0) & (cv2.dilate(ink,np.ones((5,5),np.uint8))==0)
            samples = image[background_pixels].astype(np.float32)
            if len(samples)<20 or np.max(np.percentile(samples,95,axis=0)-np.percentile(samples,5,axis=0))>45:
                raise NeedsReview('气泡内部底纹复杂或存在未确认线条，文字需核对')
        # Evidence against the safe boundary is unsafe to erase or silently clip.
        boundary = (allowed>0) & (cv2.erode(allowed,np.ones((5,5),np.uint8))==0)
        if np.count_nonzero((probability>.5)&boundary) > 3:
            raise NeedsReview('生成文字靠近气泡边线或超出安全内部，请核对')
        clean_mask = envelope(generated,allowed,1,0)
        try:
            merged = inpaint_inside(merged,clean_mask,allowed)
        except ValueError as error:
            raise NeedsReview(str(error)) from error
        cleanup |= clean_mask
        support = envelope(core,allowed,data.get('expand',1),data.get('feather',1))
        opacity,foreground = foreground_layer(original,core,support)
        candidate = np.rint(foreground*opacity[...,None]+merged*(1-opacity[...,None])).clip(0,255).astype(np.uint8)
        changed = opacity>0
        merged[changed] = candidate[changed]
        restore = np.maximum(restore,np.rint(opacity*255).astype(np.uint8))
    return png_bytes(png(merged),ratio),png(restore),{**info,'regions':len(groups),'algorithm':VERSION,'models':data.get('models',{}),'cleanup_png':png(cleanup),'cleanup_pixels':int(np.count_nonzero(cleanup)),'restored_pixels':int(np.count_nonzero(restore))}
