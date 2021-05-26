"""
Mote Marine Laboratory Collaboration

Manatee Matching Program

Written by Nate Wagner, Rosa Gradilla 
"""

from PIL import Image
import json
import os
import cv2
import numpy as np
import random
import time

annotations = {}
for an in os.listdir('/Users/natewagner/Downloads/easy_annotations/'):
    if an[-1] == "n":
        f = open('/Users/natewagner/Downloads/easy_annotations/' + an)
        data = json.load(f)
        if data['objects'] != []:
            annotations[str(an)] = data
            
path_to_blank = '/Users/natewagner/Documents/Mote_Manatee_Project/data/BLANK_SKETCH_updated.jpg'
blank = cv2.imread(path_to_blank,cv2.IMREAD_GRAYSCALE)
blank.shape
blank_m = blank.copy()
Image.fromarray(blank_m).resize((256, 561))

f = open('/Users/natewagner/Documents/Mote_Manatee_Project/Build outline mask/ds0/ann/BLANK_SKETCH_updated.jpg.json')
data = json.load(f)
pnts = data['objects'][0]['points']['exterior']
mask = np.zeros(blank.shape[:2], np.uint8)
cv2.drawContours(mask, [np.array(pnts)], -1, (255), -1, cv2.LINE_8)

Image.fromarray(mask)

# get all scar annotations and corresponding image name
image_annots = {}
image_names = list(annotations.keys())
# iterate through images
for nm in image_names:
    scar_points = []
    # iterate through scar annotations
    for annot in annotations[nm]['objects']:        
        points = annot['points']['exterior']
        class_name = annot['classTitle']
        # add scar annots to list 
        scar_points.append([np.array(points), class_name])
    # add all annots to dict
    image_annots[nm[0:-5]] = scar_points

#image_annots['DU707.jpg']
#annotations['DU707.jpg.json']

path_to_ims = '/Users/natewagner/Documents/Mote_Manatee_Project/AI/current_train_data/train_images/'
t = None
def extractScars(path_to_ims, image_annots):
    global t
    labeled_ims = {}
    for im in os.listdir(path_to_ims):
        annot_im = cv2.imread(path_to_ims + im, cv2.IMREAD_GRAYSCALE)
        labeled_ims[im[0:-4]] = annot_im
    t = labeled_ims
    
    scar_num = 0
    scar_annots_and_image = {}
    my_ims = list(image_annots.keys())
    
    im_dta = []
    
    for i in my_ims:
        im = labeled_ims[i[0:-4]]
        im_areas = []
        if im.shape == (559, 256):
            for scar_ann in image_annots[i]:            
                mask = np.zeros(im.shape[:2], np.uint8)
                cv2.drawContours(mask, [scar_ann[0]], -1, (255), -1, cv2.LINE_8)
                area = cv2.contourArea(scar_ann[0])            
                if area <= 50:
                    cat="small"                    
                if area > 50 and area <= 130:
                    cat="medium"
                if area > 130 and area <= 700:
                    cat ="large"
                if area > 700:
                    cat="outlier"
                dst = cv2.bitwise_and(im, im, mask=mask)
    
                bg = np.ones_like(im, np.uint8)*255
                cv2.bitwise_not(bg,bg, mask=mask)
                dst2 = bg + dst
                
                im_areas.append(cat)
                
                scar_annots_and_image[scar_num] = [scar_ann[0], bg, dst, dst2, cat, scar_ann[1]]
                scar_num += 1
            im_dta.append(im_areas)
   
    return scar_annots_and_image, im_dta

scar_annots_and_image, im_dta = extractScars(path_to_ims, image_annots)
scar_annots_and_image[1]

my_scars = list(scar_annots_and_image.keys())
smalls = {}
mediums = {}
larges = {}
outliers = {}
mutes = {}
for i in my_scars:
    if scar_annots_and_image[i][5] == "Mute":
        mutes[i] = scar_annots_and_image[i]
    else:
        if scar_annots_and_image[i][4] == "small":
            smalls[i] = scar_annots_and_image[i]
        if scar_annots_and_image[i][4] == "medium":
            mediums[i] = scar_annots_and_image[i]
        if scar_annots_and_image[i][4] == "large":
            larges[i] = scar_annots_and_image[i]
        if scar_annots_and_image[i][4] == "outlier":
            outliers[i] = scar_annots_and_image[i]        

len(smalls)
len(mediums)
len(larges)
len(outliers)
len(mutes)

scar_annot_list = [smalls, mediums, larges, outliers, mutes]
def checkPoints(scar_points, pnts, amount_to_move_horiz, amount_to_move_vert):
    new_points = []
    for i in list(scar_points): 
        # Change one pixel       
        if cv2.pointPolygonTest(np.array(pnts), (i[0] + amount_to_move_horiz, i[1] + amount_to_move_vert), measureDist=False) == 1.0:
            new_points.append([i[0] + amount_to_move_horiz, i[1] + amount_to_move_vert])     
        else:
            dist = cv2.pointPolygonTest(np.array(pnts), (i[0] + amount_to_move_horiz, i[1] + amount_to_move_vert), measureDist=True)
            return [amount_to_move_horiz, amount_to_move_vert], dist, None
    return [amount_to_move_horiz, amount_to_move_vert], None, new_points

def shiftScar(annot, bg, dst, dst2, mask_pnts):
    amount_to_move_horiz = random.randint(-20,20)
    amount_to_move_vert = random.randint(-40,40)

    moves, off_by, shifted_points = checkPoints(annot, mask_pnts, amount_to_move_horiz, amount_to_move_vert)
    if off_by is not None:
        while off_by is not None:
            amount_to_move_horiz = random.randint(-20,20)       
            amount_to_move_vert = random.randint(-40,40)
            moves = [amount_to_move_horiz, amount_to_move_vert]
            moves, off_by, shifted_points = checkPoints(annot, mask_pnts, moves[0], moves[1])
    
    white_image = np.zeros(bg.shape, np.uint8)
    white_image[:] = 255
    M = np.float32([[1,0,amount_to_move_horiz],[0,1,amount_to_move_vert]]) 
    
    dst_translation = cv2.warpAffine(dst, M, (dst.shape[1], dst.shape[0]))
    bg_translation = cv2.warpAffine(bg, M, (bg.shape[1], bg.shape[0]), white_image,  borderMode=cv2.BORDER_TRANSPARENT)
    dst2_translation = bg_translation + dst_translation

    return np.array(shifted_points), bg_translation, dst_translation, dst2_translation       

def buildManatee(num_small, num_medium, num_large, num_outliers, num_mute, scar_annot_list, scar_annots_and_image, pnts, form):
    wanted, mute_wanted = list(), list()
    wanted.extend(random.choices(list(scar_annot_list[0].keys()), k=num_small))
    wanted.extend(random.choices(list(scar_annot_list[1].keys()), k=num_medium))
    wanted.extend(random.choices(list(scar_annot_list[2].keys()), k=num_large))
    wanted.extend(random.choices(list(scar_annot_list[3].keys()), k=num_outliers))
    mute_wanted.extend(random.choices(list(scar_annot_list[4].keys()), k=num_mute))
    
    new_annots, new_bg_translation, new_dst_translation, new_dst2_translation = [],[],[],[]
    for i in wanted:        
        annot, bg, dst, dst2, size, typ = scar_annots_and_image[i]
        annot2, bg_translation, dst_translation, dst2_translation = shiftScar(annot, bg, dst, dst2, pnts)
        new_annots.append(annot2)
        new_bg_translation.append(bg_translation)
        new_dst_translation.append(dst_translation)
        new_dst2_translation.append(dst2_translation)
    
    blank_m2 = np.array(Image.fromarray(blank_m).resize((new_dst2_translation[0].shape[1], new_dst2_translation[0].shape[0])))
    h = new_dst2_translation[0]
    for i in range(1,len(new_annots)):
        h2 = new_dst2_translation[i]
        h[h==0] = 2
        h2[h2==0] = 2
        h = np.add(h, h2)
        h = h + 1
    blank_m2[blank_m2==0] = 2
    res = h + blank_m2
    res = res + 1

    # build JSON
    if form == "VGG":
        annots_json = {}
        annots_json['filename'] = random.randint(1000, 100000)
    else:
        annots_json = {}
        annots_json['description'] = ''
        annots_json['tags'] = []
        annots_json['size'] = {'height': new_dst2_translation[0].shape[0], 'width': new_dst2_translation[0].shape[1]}
        objects = []
        for i in range(0,len(new_annots)):
            f = {}
            f['id'] = random.randint(1000, 100000)
            f['classId'] = 2876852
            f['description'] = ''
            f['geometryType'] = 'polygon'
            f['labelerLogin'] = 'natewag10'
            f['createdAt'] = time.asctime()
            f['updatedAt'] = time.asctime()
            f['tags'] = []
            f['classTitle'] = 'Scar'
            pt = []
            for p in new_annots[i]:
                pt.append(list(map(int, p)))
            f['points'] = {'exterior': pt, 'interior': []}
            objects.append(f)
        annots_json['objects'] = objects

    return res, new_annots, annots_json

res, a, annots_json = buildManatee(4, 4, 6, 1, 0, scar_annot_list, scar_annots_and_image, pnts, "supervisly")
Image.fromarray(res)

