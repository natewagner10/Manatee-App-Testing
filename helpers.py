"""
Mote Marine Laboratory Collaboration

Manatee Matching Program

Written by Nate Wagner, Rosa Gradilla 
"""

import numpy as np
import os
import pandas as pd
import dash_html_components as html
import dash_table
import random
import cv2
import plotly.graph_objects as go
from PIL import Image
import plotly.express as px
from svgpath2mpl import parse_path

############################################################
#  Needed functions & tools
############################################################

def readjust(weights):
    """
    Reparameterize max values for weight range.

    Parameters
    ----------
    weights : list
        list of scar weights.

    Returns
    -------
    list
        reparameterized weights

    """
    new_weights = weights.copy()
    for i, weight in enumerate(weights):
        amt_unused = 1 - weight
        excess = amt_unused/(len(weights)-1)
        before = new_weights[:i]
        after = new_weights[i+1:]
        before_arr = np.array(before)
        after_arr = np.array(after)
        new_weights[:i] = before_arr + excess
        new_weights[i+1:] = after_arr + excess
    return(new_weights)

def getFilePaths(root_dir):
    """
    Finds all images inside the root directory. Also checks to see if it's in 
    the tail mute info CSV

    Parameters
    ----------
    root_dir : str
        path to all dataset images
    tail_mute_info : str
        path to tail mute CSV

    Returns
    -------
    file_paths : list
        paths to dataset images
    itemDict : dict
        dictionary of image name and tail mute information

    """
    file_paths = []              
    for root, _, fnames in sorted(os.walk(root_dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            if path[-1] == 'g' or path[-1] == 'G':
                file_paths.append(path)
    return file_paths

def buildDataTables(matches, find_matches_func):  
    images_url = []
    names_df = []
    mean_df = []
    sum_df = []
    min_df = []
    orien_df = []
    MA_df = []
    ma_df = []
    aspect_df = []      
    for entry in matches:
        names_df.append(entry[1][0:-4])
        needed_im = '![myImage-' + str(entry[0]) + '](assets/' + str(find_matches_func.getImagePath(entry[1])) + ')'            
        images_url.append(needed_im)
        mean_df.append(round(entry[2],2))
        sum_df.append(round(entry[3],2))
        min_df.append(round(entry[4],2))
        orien_df.append(round(entry[5],2))
        MA_df.append(round(entry[6],2))
        ma_df.append(round(entry[7],2))
        aspect_df.append(round(entry[8],2))            

    orientation_i = find_matches_func.scar_i[0]
    length_i = find_matches_func.scar_i[1]
    width_i = find_matches_func.scar_i[2]
    aspect_i = find_matches_func.scar_i[3]
    scar_num = find_matches_func.scar_i[4]    
       
    scar_i_tups = list(zip(scar_num,orientation_i,length_i,width_i,aspect_i))
    scar_i_data_table_df = pd.DataFrame(scar_i_tups, columns=['Scar Number','Orientation','Length','Width','Aspect Ratio']) 

    data_tups = list(zip(names_df,images_url,mean_df,orien_df,MA_df,ma_df,aspect_df))
    data_table_df = pd.DataFrame(data_tups, columns=['Name','Image','Mean','O','L','W','A'])  

    browse_matches_table = html.Div(
               [                           
                   dash_table.DataTable(
                       id='table',
                       columns=[{'id': 'Name', 'name': 'Name'}, {'id': 'Image', 'name': 'Image', 'presentation': 'markdown'}, {'id': 'Mean', 'name': 'Mean'}, {'id': 'O', 'name': 'O'}, {'id': 'L', 'name': 'L'}, {'id': 'W', 'name': 'W'}, {'id': 'A', 'name': 'A'}],
                       page_size=50,
                       style_header={'height':'auto'},
                       data=data_table_df.to_dict('records'),
                       style_table={'overflowX': 'auto', 'minHeight': '625px', 'height': '625px', 'maxHeight': '625px', 'overflowY': 'auto', 'minWidth': '100%', 'Width': '100%', 'maxWidth': '100%'},
                       style_cell={'textAlign': 'center', 'font_size': '20px','overflowX': 'auto', 'height': 'auto'},
                       sort_action="native",
                       fixed_rows={'headers': True}, 
                       filter_action="native",                       
                       style_cell_conditional=[
                            {
                                'if': {'column_id': 'Image'},
                                'width': '270px'
                            },                           
                        ],                          
                   ),  
               ],
               style = {'textAlign': 'center',
                        'width': '100%',
                        'align-items': 'center'})
    
    scar_statistics_table = html.Div(
               [                           
                   dash_table.DataTable(
                       id='table1',
                       columns=[{'id': 'Scar Number', 'name': 'Type'}, {'id': 'Orientation', 'name': 'Orien'}, {'id': 'Length', 'name': 'Len'}, {'id': 'Width', 'name': 'Wid'}, {'id': 'Aspect Ratio', 'name': 'Aspect'}],                           
                       data=scar_i_data_table_df.to_dict('records'),
                       style_header={'overflowY': 'auto', 'minWidth': '10px', 'width': '10px', 'maxWidth': '10px', 'backgroundColor': 'white'},
                       style_cell={'textAlign': 'center', 'font_size': '15px', 'height':'auto', 'minWidth': '10px', 'width': '10px', 'maxWidth': '10px'},
                       fixed_rows={'headers': True},
                       style_as_list_view=True,
                   ),  
               ],
               style = {'textAlign': 'center',
                        'width': '100%',
                        'align-items': 'center',
                        'margin-bottom': '15px'})
    return browse_matches_table, scar_statistics_table, names_df

def extractModelOutput(r):
    polys = []
    classes = []    
    for i in range(0, r['masks'].shape[2]):
        img_gray = r['masks'][:,:,i]      
        mask = img_gray[0:512,139:373]       
        mask = mask.astype(np.uint8)  #convert to an unsigned byte
        mask*=255
        mask = cv2.resize(mask, (256,559), interpolation = cv2.INTER_LINEAR)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        polys.append(contours[0])
        classes.append(r['class_ids'][i])
    if len(polys) != r['masks'].shape[2]:
        print("Warning, more polygons then mask")        
    annots_json = {}
    objects = []        
    for i in range(0,len(polys)):
        f = {}
        f['id'] = random.randint(1000, 100000)
        f['classId'] = 2876852
        f['description'] = ''
        f['geometryType'] = 'polygon'
        f['labelerLogin'] = 'natewag10'
        if classes[i] == 1:
            f['classTitle'] = 'Scar'
        elif classes[i] == 2:
            f['classTitle'] = 'Mute'
        pt = []
        for p in polys[i][0]:
            pt.append(list(map(int, p[0])))
        f['points'] = {'exterior': pt, 'interior': []}        
        contour = np.array(pt)
        x, y, w, h = cv2.boundingRect(contour)                          
        area = cv2.contourArea(contour)  
        if area > 5:
            (x,y), (ma,MA), angle = cv2.fitEllipseDirect(contour)   
            if MA == 0:
                MA = 1
            aspect_ratio = float(ma)/MA                                               
        f['orientation'] = round(angle,2)
        f['length'] = round(MA,2)
        f['width'] = round(ma,2)
        f['aspect'] = round(aspect_ratio,2)        

        objects.append(f)
    annots_json['num_scars'] = len(polys)
    annots_json['objects'] = objects    
    if 2 in classes:
        annots_json['has_mute'] = 1
    else:
        annots_json['has_mute'] = 0    
    return annots_json

def tranformImageAndRois(im, rois):
    # crop image and resize
    new_arr = np.asarray(im)
    cropped = new_arr[0:512, 128:362]
    cropped = Image.fromarray(cropped).resize((256, 559))    
    # crop bb dims
    new_rois = []
    for r in rois:                        
        r = [r[0], r[1]-128, r[2], r[3]-128]
        # resize bb dims
        x_ = 234
        y_ = 512
        x_scale = 256 / x_
        y_scale = 559 / y_
        
        r[1] = int(np.round(r[1] * x_scale))
        r[0] = int(np.round(r[0] * y_scale))
        r[3] = int(np.round(r[3] * x_scale))
        r[2] = int(np.round(r[2] * y_scale))   
    
        new_rois.append(r)
    return cropped, np.array(new_rois)

def cropImageAndROI(im, rois):    
    new_im, new_rois = tranformImageAndRois(im, rois)
    new_im_arr = np.asarray(new_im)
    
    top1 = np.min([[r[0], r[2]] for r in new_rois])
    top2 = np.max([[r[0], r[2]] for r in new_rois])
    left1 = np.min([[r[1], r[3]] for r in new_rois])
    left2 = np.max([[r[1], r[3]] for r in new_rois])
    r = np.array([top1-10, left1-10, top2+10, left2+10])
    
    crop_img = new_im_arr[r[0]:r[2], r[1]:r[3]]
    
    r2 = r.copy()
    new_rois2 = []
    for ri in new_rois:                        
        r2 = [ri[0]-r[0], ri[1]-r[1], ri[2]-r[0], ri[3]-r[1]]
        new_rois2.append(r2)
    
    return Image.fromarray(crop_img), np.array(new_rois2)

def buildFigure(r, im):
    new_im, new_rois = cropImageAndROI(im, r['rois'])
    fig = go.Figure()
    img_width, img_height = new_im.size    
    fig.add_trace(
        go.Scatter(
            x=[0, img_width],
            y=[0, img_height],
            mode="markers",
            marker_opacity=0
        )
    )
    fig.update_xaxes(
        visible=False,
        range=[0, img_width]
    )    
    fig.update_yaxes(
        visible=False,
        range=[0, img_height],
        scaleanchor="x"
    )    
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width,
            y=img_height,
            sizey=img_height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=new_im)
    )        
    for bb in new_rois:    
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))    
        fig.add_shape(type="rect",
            x0=bb[1], y0=img_height - bb[0], x1=bb[3], y1=img_height - bb[2],
            line=dict(
                color=color,
                width=2,
                dash="dot",
            ),                
        )
    fig.update_layout(
        width=img_width,
        height=img_height,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},        
    )
    fig.update_shapes(dict(xref='x', yref='y'))        
    fig.update_traces(
       hoverinfo='skip'
    )        
    return fig
    
def buildFigure2(im, objs, color):
    im_arr = np.array(im)
    for i in range(0,len(objs)):
        pts_arr_h2 = np.array(objs[i])
        if color is None:
            cv2.fillPoly(im_arr,pts=[pts_arr_h2],color=(random.randint(0,255),random.randint(0,255),random.randint(0,255)))
        else:
            cv2.fillPoly(im_arr,pts=[pts_arr_h2],color=(color))
    new_im = Image.fromarray(im_arr)    
    layout = go.Layout(hovermode=False)
    fig = go.Figure(layout=layout)    
    img_width, img_height = new_im.size    
    fig.add_trace(
        go.Scatter(
            x=[0, img_width],
            y=[0, img_height],
            mode="markers",
            marker_opacity=0,
        )
    )    
    fig.update_xaxes(
        visible=False,
        range=[0, img_width]
    )    
    fig.update_yaxes(
        visible=False,
        range=[0, img_height],
        scaleanchor="x"
    )    
    fig.add_layout_image(
        dict(
            x=0,
            sizex=img_width,
            y=img_height,
            sizey=img_height,
            xref="x",
            yref="y",
            opacity=1.0,
            layer="below",
            sizing="stretch",
            source=new_im)
    )        
    fig.update_layout(
        width=img_width,
        height=img_height,
        margin={"l": 0, "r": 0, "t": 0, "b": 0},        
    )
    fig.update_shapes(dict(xref='x', yref='y'))        
    return fig
    
def buildDataExplorer(images, find_matches_func):
    names_df = []
    images_url = []
    number_scars = []
    has_mute = []
    for entry in images:
        try:
            names_df.append(entry[0:-4])
            needed_im = '![myImage-' + str(entry) + '](assets/' + str(find_matches_func.getImagePath(entry)) + ')'            
            images_url.append(needed_im)
            number_scars.append(len(find_matches_func.processed_images[entry]['objects']))
            has_mute.append(find_matches_func.processed_images[entry]['has_mute'])
        except:
            continue
        
    data_tups = list(zip(names_df,number_scars,has_mute,images_url))
    data_table_df = pd.DataFrame(data_tups, columns=['Name','# of Scars','Has Mute','Image'])   
    data_table_df = data_table_df.sample(frac=1).reset_index(drop=True)

    browse_matches_table = html.Div(
               [                           
                   dash_table.DataTable(
                       id='datatable-interactivity',
                       columns=[{'id': 'Name', 'name': 'Name', "selectable": False}, {'id': '# of Scars', 'name': '# of Scars', "selectable": False}, {'id': 'Has Mute', 'name': 'Has Mute', "selectable": False}, {'id': 'Image', 'name': 'Image', 'presentation': 'markdown', "selectable": False}],
                       page_action='none',
                       virtualization=True,
                       style_header={'height':'auto'},
                       data=data_table_df.to_dict('records'),
                       style_table={'minHeight': '625px', 'height': '625px', 'maxHeight': '625px', 'overflowY': 'auto', 'textAlign': 'center'},
                       style_cell={'font_size': '20px', 'height': 'auto', 'textAlign': 'center'},                      
                       filter_action="native",
                       sort_action="native",
                       sort_mode="multi",
                       column_selectable="single",
                       selected_columns=[],
                       style_as_list_view=True,
                       fixed_rows={'headers': True},    
                       style_data_conditional=[
                            {'if': {'column_id': 'Image'},
                             'width': '270px'},
                            {'if': {'column_id': 'Name'},
                             'width': '225px'},
                            {'if': {'column_id': '# of Scars'},
                             'width': '180px'},
                            {'if': {'column_id': 'Has Mute'},
                             'width': '180px'},
                       ],                       
                   ),  
               ])   
    return browse_matches_table, data_table_df    
    
def buildAnnotate(img):
    fig = px.imshow(img)
    fig.update_layout(
        dragmode="drawclosedpath",
        coloraxis_showscale=False,
        height=559,
        width=256,
        margin=dict(l=1, r=1, b=1, t=1),
        hovermode=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)    
    config = {
        "modeBarButtonsToAdd": [
            "drawclosedpath",
            "eraseshape",
        ]
    }
    fig.update_traces(
       hoverinfo='skip'
    )        
    
    return fig, config        
    
def extractNewAnnots(new_annots):    
    if len(new_annots) == 1:
        annots = new_annots[0]['shapes'][0]['path']
        mpl_path = parse_path(annots)
        coords = mpl_path.to_polygons()[0].astype(int)
        return coords
    else:
        for i in range(1, len(new_annots)):
            if 'shapes' in list(new_annots[-i].keys()):
                annots = new_annots[-i]
                break
        my_coords = []
        for a in range(len(annots['shapes'])):
            svg = annots['shapes'][a]['path']
            mpl_path = parse_path(svg)
            coords = mpl_path.to_polygons()[0].astype(int)
            my_coords.append(coords)
        return my_coords

def buildAnnotateFromApp(an):
    polys = an[0]
    if an[1] == True:
        classes = [2] * len(polys)
    else:
        classes = [1] * len(polys)       
    annots_json = {}
    objects = []        
    for i in range(0,len(polys)):
        f = {}
        f['id'] = random.randint(1000, 100000)
        f['classId'] = 2876852
        f['description'] = ''
        f['geometryType'] = 'polygon'
        f['labelerLogin'] = 'natewag10'
        if classes[i] == 1:
            f['classTitle'] = 'Scar'
        elif classes[i] == 2:
            f['classTitle'] = 'Mute'
        pt = []
        for p in polys[i]:
            pt.append(list(map(int, p)))
        f['points'] = {'exterior': pt, 'interior': []}        
        contour = np.array(pt)
        x, y, w, h = cv2.boundingRect(contour)                          
        area = cv2.contourArea(contour)  
        if area > 5:
            (x,y), (ma,MA), angle = cv2.fitEllipseDirect(contour)   
            if MA == 0:
                MA = 1
            aspect_ratio = float(ma)/MA                                               
        f['orientation'] = round(angle,2)
        f['length'] = round(MA,2)
        f['width'] = round(ma,2)
        f['aspect'] = round(aspect_ratio,2)        

        objects.append(f)
    annots_json['num_scars'] = len(polys)
    annots_json['objects'] = objects    
    if 2 in classes:
        annots_json['has_mute'] = 1
    else:
        annots_json['has_mute'] = 0    
    return annots_json        
    
def extendAnnotations(an, class_type, existing_json):
    polys = an
    if class_type == True:
        classes = [2] * len(polys)
    else:
        classes = [1] * len(polys)       
    annots_json = existing_json      
    for i in range(0,len(polys)):
        f = {}
        f['id'] = random.randint(1000, 100000)
        f['classId'] = 2876852
        f['description'] = ''
        f['geometryType'] = 'polygon'
        f['labelerLogin'] = 'natewag10'
        if classes[i] == 1:
            f['classTitle'] = 'Scar'
        elif classes[i] == 2:
            f['classTitle'] = 'Mute'
        pt = []
        for p in polys[i]:
            pt.append(list(map(int, p)))
        f['points'] = {'exterior': pt, 'interior': []}        
        contour = np.array(pt)
        x, y, w, h = cv2.boundingRect(contour)                          
        area = cv2.contourArea(contour)  
        if area > 5:
            (x,y), (ma,MA), angle = cv2.fitEllipseDirect(contour)   
            if MA == 0:
                MA = 1
            aspect_ratio = float(ma)/MA                                               
        f['orientation'] = round(angle,2)
        f['length'] = round(MA,2)
        f['width'] = round(ma,2)
        f['aspect'] = round(aspect_ratio,2)        
        annots_json['objects'].append(f)
    annots_json['num_scars'] = len(polys) + annots_json['num_scars']
    if 2 in classes:
        annots_json['has_mute'] = 1
    else:
        annots_json['has_mute'] = 0    
    return annots_json
        
def extractScarDetailsFromJSON(an):
    an['num_scars'] = len(an['objects'])        
    classes = []
    remove = []
    for i in range(0 , len(an['objects'])):
        pts = an['objects'][i]['points']['exterior']
        contour = np.array(pts)
        x, y, w, h = cv2.boundingRect(contour)                          
        area = cv2.contourArea(contour)  
        try:
            if area > 5:
                (x,y), (ma,MA), angle = cv2.fitEllipseDirect(contour)   
                if MA == 0:
                    MA = 1
                aspect_ratio = float(ma)/MA  
        except:
            remove.append(an['objects'][i]['id'])
        an['objects'][i]['orientation'] = round(angle,2)
        an['objects'][i]['length'] = round(MA,2)
        an['objects'][i]['width'] = round(ma,2)
        an['objects'][i]['aspect']    = round(aspect_ratio,2) 
        classes.append(an['objects'][i]['classTitle'])
    if remove != []:
        for bad_an in remove:
            an['objects'] = [x for x in an['objects'] if not (bad_an == x.get('id'))]   
        an['num_scars'] = len(an['objects'])        
    if 'Mute' in classes:
        an['has_mute'] = 1
    return an
