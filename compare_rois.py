"""
Mote Marine Laboratory Collaboration

Manatee Matching Program

Written by Nate Wagner, Rosa Gradilla 
"""

import pandas as pd
import numpy as np
import os
import ntpath
import json
from shapely.geometry import Polygon
import cv2
from math import atan2,degrees

############################################################
#  Compare region of interest class
############################################################

class Compare_ROIS(object):    
    def __init__(self, paths, path_to_jsons, input_sketch, input_annots, roi, low_end, high_end, tail, model, scar_i, check, orientation_bool, roi_low, roi_high, in_roi, roi_types, new_annots, new_annots_coords, new_json):
        self.paths = paths
        self.input_sketch = input_sketch
        self.input_annots = input_annots
        self.roi = roi
        self.processed_images = None
        self.low_end = 1
        self.high_end = 5
        self.tail = 'unknown'
        self.all_tail_info = None
        self.path_to_jsons = path_to_jsons
        self.model = model
        self.scar_i = None
        self.check = None
        self.orientation_bool = True
        self.roi_low = 1
        self.roi_high = 3
        self.in_roi = False
        self.roi_types = None
        self.new_annots = []
        self.new_annots_coords = []
        self.new_json = None
    def compare_rois(self):
        scores = []    
        n_scars = []
        orientation = []
        length = []
        width = []
        aspect = []
        cnt = 0
        for ob in self.input_annots['objects']:
            orientation.append(ob['orientation'])
            length.append(ob['length'])
            width.append(ob['width'])
            aspect.append(ob['aspect'])
            cnt+=1
            n_scars.append(ob['classTitle'])                       
        self.scar_i = [orientation, length, width, aspect, n_scars]
        # iterate through each box
        for b in range(len(self.roi)):
            box = self.roi[b]
            x,y,w,h = box
            poly_box = Polygon(np.array([[x,y], [x+w, y], [x+w, y+h], [x, y+h]]))
            # iterate through each image JSON
            for im in self.processed_images:
                sketch_info = self.processed_images[im]  

                mute_index = [[0], [1], [0,1]]
                if self.tail == 'unknown':
                    holder = 2
                if self.tail == 'mute':
                    holder = 1
                if self.tail == 'no_mute':
                    holder = 0                

                if sketch_info['num_scars'] in range(self.low_end, self.high_end+1) and sketch_info['has_mute'] in mute_index[holder]:
                    in_box = []
                    for i in range(0,len(sketch_info['objects'])):                        
                        poly = sketch_info['objects'][i]['points']['exterior']  
                        poly_obj = Polygon(poly)
                        if self.in_roi == False:
                            if poly_obj.intersects(poly_box):
                                in_box.append(sketch_info['objects'][i]['id'])
                        if self.in_roi == True:
                            if poly_box.contains(poly_obj):
                                in_box.append(sketch_info['objects'][i]['id'])
                    if len(in_box) in range(self.roi_low, self.roi_high+1):
                        if in_box != []:
                            if self.roi_types[b] == 'normal':                            
                                dist = self.getDistance(sketch_info['objects'], in_box, self.input_annots['objects'], im, False, None) 
                            if self.roi_types[b] == 'series': 
                                 # get series stats
                                our_series_stats = self.computeSeriesStats(sketch_info['objects'], in_box, False)
                                input_series_stats = self.computeSeriesStats(self.input_annots['objects'], in_box, True)
                                dist = self.getDistance(sketch_info['objects'], in_box, self.input_annots['objects'], im, True, [our_series_stats, input_series_stats]) 
                            if dist != 'NA':
                                scores.append(dist) 
        our_scar_df = self.buildDataFrame(scores)                           
        return our_scar_df                                                    
    def getDistance(self, our_object, in_box, input_objects, name, series, series_info):
        final_scores = []
        for input_obj in input_objects:
            
            input_orientation = input_obj['orientation']
            input_length = input_obj['length']
            input_width = input_obj['width']
            input_aspect = input_obj['aspect']  
            
            if series == True:
                
                input_series_orientation_left = series_info[1][1]
                input_series_length_left = series_info[1][0]
                input_series_orientation_right = series_info[1][3]
                input_series_length_right = series_info[1][2]                
                our_series_orientation_left = series_info[0][1]
                our_series_length_left = series_info[0][0]
                our_series_orientation_right = series_info[0][3]
                our_series_length_right = series_info[0][2]                
            
                diff_in_orientation_left = abs(input_series_orientation_left - our_series_orientation_left)
                diff_in_length_left = abs(input_series_length_left - our_series_length_left)
                diff_in_orientation_right = abs(input_series_orientation_right - our_series_orientation_right)
                diff_in_length_right = abs(input_series_length_right - our_series_length_right)                
            
                
                percentage_orientation_left = 100 * (diff_in_orientation_left / input_series_orientation_left)
                percentage_length_left = 100 * (diff_in_length_left / input_series_length_left)
                percentage_orientation_right = 100 * (diff_in_orientation_right / input_series_orientation_right)
                percentage_length_right = 100 * (diff_in_length_right / input_series_length_right)                
                
            
            for num, obj in enumerate(our_object):
                
                if obj['id'] in in_box:
                
                    scar_type = obj['classTitle']            
                    
                    if scar_type == input_obj['classTitle']:               
                        
                        our_orientation = obj['orientation']
                        our_length = obj['length']
                        our_width = obj['width']
                        our_aspect = obj['aspect']                            
                        
                        diff_in_orientation = abs(input_orientation - our_orientation)
                        diff_in_length = abs(input_length - our_length)
                        diff_in_width = abs(input_width - our_width)
                        diff_in_aspect = abs(input_aspect - our_aspect)            
                        
                        percentage_orientation = 100 * (diff_in_orientation / input_orientation)
                        percentage_length = 100 * (diff_in_length / input_length)
                        percentage_width = 100 * (diff_in_width / input_width)
                        percentage_aspect = 100 * (diff_in_aspect / input_aspect)            
                        
                        if self.orientation_bool == True:
                            score = 1/4 * (0.20 * percentage_orientation + 0.30 * percentage_length + 0.30 * percentage_width + 0.20 * percentage_aspect)
                            individual_scores = [0.20 * percentage_orientation, 0.30 * percentage_length, 0.30 * percentage_width, 0.20 * percentage_aspect]
                        else:
                            score = 1/3 * (0.35 * percentage_length + 0.35 * percentage_width + 0.30 * percentage_aspect)
                            individual_scores = [0.00, 0.35 * percentage_length, 0.35 * percentage_width, 0.30 * percentage_aspect]
                        
                        if series == True:
                            score += 0.10*np.mean([percentage_orientation_left, percentage_length_left, percentage_orientation_right, percentage_length_right])
                        
                        final_scores.append([num, round(score, 2), [round(i, 2) for i in individual_scores]])
        if final_scores != []:
            distances, distances_info = self.computeScore(final_scores, len(input_objects))                
            return [name, np.mean(distances), np.sum(distances), np.min(distances), distances_info[0], distances_info[1], distances_info[2], distances_info[3]]
        else:
            return 'NA'    
    def buildDataFrame(self, df):
        distance_dict_df = pd.DataFrame(df)        
        distance_dict_df.columns = ['Name', 'Mean', 'Sum', 'Min', 'Orien', 'MA', 'ma', 'aspect']
        unique_names_cnts = distance_dict_df.groupby('Name').agg({'Mean': np.nanmean, 'Sum': np.nansum, 'Min': np.nanmin, 'Name': 'count', 'Orien': np.nanmin, 'MA': np.nanmin, 'ma': np.nanmin, 'aspect': np.nanmin})
        unique_names_cnts['names'] = unique_names_cnts.index
        has_all_scars = unique_names_cnts[unique_names_cnts['Name'] >= len(self.roi)]
        has_all_scars.columns = ['mean', 'sum', 'min', 'count', 'Orien', 'MA', 'ma', 'aspect', 'name']
        returned = has_all_scars[['name', 'mean', 'sum', 'min', 'Orien', 'MA', 'ma', 'aspect']]        
        returned_list = returned.values.tolist()     
        returned_list = sorted(returned_list, key = lambda x: x[1])         
        returned_list2 = []
        for idx, img in enumerate(returned_list):
            returned_list2.append([idx+1, img[0], img[1], img[2], img[3], img[4], img[5], img[6], img[7]])                    
        return returned_list2
    def preLoadData(self):
        all_jsons = {}
        for file in os.listdir(self.path_to_jsons):
            if file[-4:] == 'json':
                filename = self.path_to_jsons + file
                f = open(filename,)
                json_data = json.load(f)            
                json_data['sketch_name'] = file[:-5]
                all_jsons[file[:-5]] = json_data
        sketch_names = self.paths    
        for i in range(len(sketch_names)):
            # get sketch path
            sketch_path = sketch_names[i]        
            # get sketch name
            sketch_name = ntpath.basename(sketch_names[i])
            try:
                all_jsons[sketch_name]['path_to_image'] = sketch_path
            except:
                continue
        self.processed_images = all_jsons                        
    def computeScore(self, dist, num_input_scars):   
        scores = []    
        num_lookup_scars = len(list(set([el[0] for el in dist]))) 
        while len(scores) <= num_input_scars - 1:
            if len(scores) >= num_lookup_scars:
                break
            current_lowest = dist[np.argmin([el[1] for el in dist])]        
            if len(dist) != 0:
                scores.append(current_lowest)
            dist = [item for item in dist if item[0] != current_lowest[0]]            
        best_scores = [el[1] for el in scores]
        diff_in_scar_stats = [el[2] for el in scores]
        diff_in_oriten = [] 
        diff_in_MA = []
        diff_in_ma = []
        diff_in_aspect = []                    
        for i in diff_in_scar_stats:
            diff_in_oriten.append(i[0])
            diff_in_MA.append(i[1])
            diff_in_ma.append(i[2])
            diff_in_aspect.append(i[3])                    
        return best_scores, [min(diff_in_oriten), min(diff_in_MA), min(diff_in_ma), min(diff_in_aspect)]
    def getImagePath(self, image_name):
        for name in self.paths:
            if name.endswith(str(image_name)):
                my_path = os.path.normpath(name)
                components = my_path.split(os.sep)
                return os.path.join(components[-2], components[-1])   
    def computeSeriesStats(self, objects, in_box, input_s):
        all_left = []
        all_right = []
        for obj in objects:
            if input_s == False:
                if obj['id'] in in_box:
                    cnt = obj['points']['exterior']
                    arr = np.array(cnt)
                    leftmost = tuple(arr[arr[:,0].argmin()])
                    rightmost = tuple(arr[arr[:,0].argmax()]) 
                    all_left.append(leftmost)
                    all_right.append(rightmost)
            if input_s == True:
                cnt = obj['points']['exterior']
                arr = np.array(cnt)
                leftmost = tuple(arr[arr[:,0].argmin()])
                rightmost = tuple(arr[arr[:,0].argmax()]) 
                all_left.append(leftmost)
                all_right.append(rightmost)                
                
        cnt_left = np.array(all_left)
        cnt_right = np.array(all_right)                
        
        [vx_left,vy_left,x_left,y_left] = cv2.fitLine(cnt_left, cv2.DIST_L2,0,0.01,0.01)
        [vx_right,vy_right,x_right,y_right] = cv2.fitLine(cnt_right, cv2.DIST_L2,0,0.01,0.01)
        
        topmost_left = tuple(cnt_left[cnt_left[:,1].argmin()])
        bottommost_left = tuple(cnt_left[cnt_left[:,1].argmax()]) 
        
        topmost_right = tuple(cnt_right[cnt_right[:,1].argmin()])
        bottommost_right = tuple(cnt_right[cnt_right[:,1].argmax()])       
        
        x1_left, y1_left = topmost_left
        x2_left, y2_left = bottommost_left
        dist_left = np.sqrt((y2_left-y1_left)**2+(x2_left-x1_left)**2)
        
        x1_right, y1_right = topmost_right
        x2_right, y2_right = bottommost_right
        dist_right = np.sqrt((y2_right-y1_right)**2+(x2_right-x1_right)**2)        

        xDiff_left = x2_left - x1_left
        yDiff_left = y2_left - y1_left
        angle_left = degrees(atan2(yDiff_left, xDiff_left))
        
        xDiff_right = x2_right - x1_right
        yDiff_right = y2_right - y1_right
        angle_right = degrees(atan2(yDiff_right, xDiff_right))
        
        
        return ([round(dist_left,2), round(angle_left,2), round(dist_right,2), round(angle_right,2)])
             