"""Closed balloon validation and stroke masks, independent of model inference."""
import cv2
import numpy as np


def rectangle_mask(shape, boxes):
    mask = np.zeros(shape[:2], np.uint8)
    h, w = mask.shape
    for box in boxes:
        x1,y1,x2,y2 = map(int, box)
        x1,y1,x2,y2 = max(0,x1),max(0,y1),min(w,x2),min(h,y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2,x1:x2] = 255
    return mask


def polygon_mask(shape, polygons):
    mask = np.zeros(shape[:2], np.uint8)
    for polygon in polygons:
        points = np.asarray(polygon, np.int32)
        if points.size >= 6:
            cv2.fillPoly(mask, [points.reshape(-1,2)], 255)
    return mask


def closed_bubbles(image, candidates):
    """Never close gaps: a component reaching the padded ROI is rejected."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    results, rejected = [], []
    for candidate in candidates:
        x1,y1,x2,y2 = candidate['box']
        pad = max(8, round(min(x2-x1,y2-y1)*.15))
        rx1,ry1,rx2,ry2 = max(0,x1-pad),max(0,y1-pad),min(w,x2+pad),min(h,y2+pad)
        patch = gray[ry1:ry2,rx1:rx2]
        best = None
        for threshold in (225, 205, 185):
            contours, _ = cv2.findContours((patch > threshold).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                bx,by,bw,bh = cv2.boundingRect(contour)
                area = cv2.contourArea(contour)
                if bx <= 1 or by <= 1 or bx+bw >= patch.shape[1]-1 or by+bh >= patch.shape[0]-1:
                    continue
                if area < 60 or area > h*w*.4 or area < bw*bh*.4:
                    continue
                gx1,gy1,gx2,gy2 = bx+rx1,by+ry1,bx+bw+rx1,by+bh+ry1
                intersection = max(0,min(gx2,x2)-max(gx1,x1))*max(0,min(gy2,y2)-max(gy1,y1))
                union = bw*bh+(x2-x1)*(y2-y1)-intersection
                iou = intersection/max(1,union)
                if iou < .4:
                    continue
                filled = np.zeros_like(patch)
                cv2.drawContours(filled,[contour],-1,255,cv2.FILLED)
                rim = cv2.dilate(filled,np.ones((5,5),np.uint8))-filled
                # A true ink boundary must support nearly the whole perimeter.
                support = np.mean(patch[rim > 0] < 100) if np.any(rim) else 0
                paper = patch[(filled > 0) & (patch > threshold)]
                if support < .45 or paper.size < area*.55 or np.std(paper) > 15:
                    continue
                score = iou + support*.1
                if best is None or score > best[0]:
                    points = cv2.approxPolyDP(contour, .6, True).reshape(-1,2)+[rx1,ry1]
                    best = (score, {'box':[gx1,gy1,gx2,gy2], 'contour':points.tolist(), 'category':candidate.get('category',0), 'score':candidate.get('score',1.), 'boundary_support':float(support)})
        if best is None:
            rejected.append({**candidate,'reason':'未确认封闭且平坦的气泡内部'})
        elif not any(abs(best[1]['box'][0]-b['box'][0]) < 3 and abs(best[1]['box'][1]-b['box'][1]) < 3 for b in results):
            results.append(best[1])
    return results, rejected


def safe_interior(shape, bubble):
    filled = polygon_mask(shape, [bubble['contour']])
    return cv2.erode(filled, np.ones((5,5),np.uint8))


def strokes(image, probability, allowed, polygons=(), manual=False):
    """Select full dark components supported by model evidence, retaining dots."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
    paper = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,kernel).astype(np.float32)
    if np.any(allowed):
        paper = np.maximum(paper, np.percentile(gray[allowed>0],90))
    contrast = paper-gray.astype(np.float32)
    ink = ((contrast > 16) & (gray < paper*.93) & (allowed > 0)).astype(np.uint8)
    # A line polygon locates text, but cannot turn every ink stroke in its
    # rectangle into text. Require independent pixel-level model evidence.
    evidence = (probability > .4).astype(np.uint8)
    line_support = polygon_mask(gray.shape,polygons)>0 if polygons else None
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink,8)
    selected = np.zeros_like(gray)
    boundary = (allowed > 0) & (cv2.erode(allowed,np.ones((3,3),np.uint8)) == 0)
    for label in range(1,count):
        x,y,bw,bh,area = stats[label]
        local = labels[y:y+bh,x:x+bw] == label
        if area < 2 or area > np.count_nonzero(allowed)*.3:
            continue
        if np.any(boundary[y:y+bh,x:x+bw][local]):
            continue
        supported = np.mean(evidence[y:y+bh,x:x+bw][local]>0)
        confident = np.mean(probability[y:y+bh,x:x+bw][local])
        in_line = line_support is not None and np.mean(line_support[y:y+bh,x:x+bw][local]) >= .9
        # Long thin components are usually speed lines / panel decoration.
        thin_line = max(bw,bh) > max(24,min(bw,bh)*12) and area/(bw*bh) > .5
        if manual or (not thin_line and supported >= .65 and confident >= .45 and (in_line or confident >= .75)):
            selected[y:y+bh,x:x+bw][local] = 255
    return selected, paper


def envelope(core, allowed, expand=1, feather=1):
    expand, feather = max(0,min(30,int(expand))),max(0,min(12,int(feather)))
    if expand:
        core = cv2.dilate(core,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(expand*2+1,expand*2+1)))
    if not np.any(core):
        return core
    if feather:
        distance = cv2.distanceTransform(255-core,cv2.DIST_L2,3)
        core = np.rint(np.clip(1-distance/(feather+1),0,1)*255).astype(np.uint8)
    return cv2.bitwise_and(core,allowed)


def inpaint_inside(image, mask, allowed):
    if not np.any(mask):
        return image.copy()
    points = cv2.findNonZero(allowed)
    if points is None:
        raise ValueError('清字区域为空')
    x,y,w,h = cv2.boundingRect(points)
    local = image[y:y+h,x:x+w].copy()
    safe = allowed[y:y+h,x:x+w] > 0
    clean = safe & (mask[y:y+h,x:x+w] == 0)
    if np.count_nonzero(clean) < 20:
        raise ValueError('气泡内部没有足够的干净背景，文字需核对')
    samples = local[clean].astype(np.float32)
    if np.max(np.percentile(samples,95,axis=0)-np.percentile(samples,5,axis=0)) > 45:
        raise ValueError('气泡内部底纹复杂或存在未确认线条，文字需核对')
    # Extend the nearest *interior* pixels outwards so Telea never samples artwork.
    outside = (~safe).astype(np.uint8)
    _, nearest = cv2.distanceTransformWithLabels(outside,cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
    samples = local[safe]
    local[~safe] = samples[np.clip(nearest[~safe]-1,0,len(samples)-1)]
    repaired = cv2.inpaint(local,mask[y:y+h,x:x+w],3,cv2.INPAINT_TELEA)
    result = image.copy()
    target = result[y:y+h,x:x+w]
    change = mask[y:y+h,x:x+w] > 0
    target[change] = repaired[change]
    return result
