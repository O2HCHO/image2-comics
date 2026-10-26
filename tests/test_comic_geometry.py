import cv2
import numpy as np
import pytest

from comic_color.text_geometry import closed_bubbles, safe_interior, strokes, envelope, inpaint_inside


def balloon(opened=False, rectangle=False):
    image = np.full((260,300,3),230,np.uint8)
    if rectangle:
        cv2.rectangle(image,(50,40),(230,200),(0,0,0),3)
        cv2.rectangle(image,(53,43),(227,197),(255,255,255),-1)
    else:
        cv2.ellipse(image,(140,120),(85,75),0,0,360,(255,255,255),-1)
        cv2.ellipse(image,(140,120),(85,75),0,0,360,(0,0,0),3)
    if opened:
        cv2.rectangle(image,(130,30),(150,60),(255,255,255),-1)
    cv2.putText(image,'Hi.',(105,125),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,0),2)
    return image


@pytest.mark.parametrize('rectangle',[True,False])
def test_closed_geometry_and_bounded_envelope(rectangle):
    image = balloon(rectangle=rectangle)
    box = [48,38,233,203] if rectangle else [53,43,228,198]
    bubbles, skipped = closed_bubbles(image,[{'box':box,'score':.99,'category':0}])
    assert len(bubbles) == 1 and not skipped
    allowed = safe_interior(image.shape,bubbles[0])
    probability = np.zeros(image.shape[:2],np.float32)
    probability[100:132,102:150] = 1
    core,_ = strokes(image,probability,allowed)
    mask = envelope(core,allowed,1,1)
    assert np.count_nonzero(mask)>50
    assert not np.any(mask[allowed==0])
    assert mask[160,140] == 0
    assert np.any(core[122:127,131:136])  # independent punctuation survives


def test_open_balloon_and_page_frame_rejected():
    image = balloon(opened=True)
    bubbles,_ = closed_bubbles(image,[{'box':[53,43,228,198],'score':.99}])
    assert bubbles == []
    page = np.full((300,300,3),255,np.uint8)
    cv2.rectangle(page,(3,3),(296,296),(0,0,0),3)
    assert closed_bubbles(page,[{'box':[2,2,298,298]}])[0] == []


def test_white_object_without_ink_boundary_rejected():
    image = np.full((300,300,3),160,np.uint8)
    cv2.ellipse(image,(150,150),(60,70),0,0,360,(255,255,255),-1)
    assert closed_bubbles(image,[{'box':[89,79,212,222]}])[0] == []


def test_cleanup_only_changes_mask_and_preserves_color():
    image = balloon()
    bubbles,_ = closed_bubbles(image,[{'box':[53,43,228,198]}])
    allowed = safe_interior(image.shape,bubbles[0])
    image[allowed>0] = (220,235,250)
    cv2.putText(image,'BAD',(95,125),cv2.FONT_HERSHEY_SIMPLEX,.7,(10,10,10),2)
    core,_ = strokes(image,np.ones(image.shape[:2],np.float32),allowed,manual=True)
    mask = envelope(core,allowed,1,0)
    result = inpaint_inside(image,mask,allowed)
    assert np.array_equal(result[mask==0],image[mask==0])
    assert np.min(result[core>0]) > 200
    assert np.mean(result[core>0,2]) > np.mean(result[core>0,0])
