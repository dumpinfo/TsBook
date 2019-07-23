from PIL import Image as image

def get_scaled_dims(org_w, org_h, dest_w, dest_h):
    scale = dest_w / org_w
    dh = scale * org_h
    new_w = dest_w
    new_h = dest_h
    if dh < dest_h:
        new_h = dh
    else:
        scale = dest_h / org_h
        new_w = scale * org_w
    return int(new_w), int(new_h)

def get_resized_dim(ori_w, ori_h, dest_w, dest_h):
    widthRatio = heightRatio = None
    ratio = 1
    if (ori_w and ori_w > dest_w) or (ori_h and ori_h > dest_h):
        if dest_w and ori_w > dest_w:
            widthRatio = float(dest_w) / ori_w #正确获取小数的方式
        if dest_h and ori_h > dest_h:
            heightRatio = float(dest_h) / ori_h

        if widthRatio and heightRatio:
            if widthRatio < heightRatio:
                ratio = widthRatio
            else:
                ratio = heightRatio

        if widthRatio and not heightRatio:
            ratio = widthRatio
        if heightRatio and not widthRatio:
            ratio = heightRatio
            
        newWidth = int(ori_w * ratio)
        newHeight = int(ori_h * ratio)
    else:
        newWidth = ori_w
        newHeight = ori_h
    return newWidth, newHeight

def resize_img_file(org_img, dest_img, dest_w, dest_h, save_quality=35):
    print('resize the image')
    im = image.open(org_img)
    print('im={0}'.format(im))
    ori_w, ori_h = im.size
    newWidth, newHeight = get_resized_dim(ori_w, ori_h, dest_w, dest_h)
    im.resize((newWidth,newHeight),image.ANTIALIAS).save(dest_img,quality=save_quality)
    
def resize_img(im, dest_w, dest_h, save_quality=35):    
    ori_w, ori_h = im.size
    newWidth, newHeight = get_resized_dim(ori_w, ori_h, dest_w, dest_h)    
    return im.resize((newWidth,newHeight),image.ANTIALIAS)
    
