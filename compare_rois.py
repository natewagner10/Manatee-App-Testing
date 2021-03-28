"""
Mote Marine Laboratory Collaboration

Manatee Matching Program

Written by Nate Wagner, Rosa Gradilla 
"""


import cv2
import pandas as pd
import numpy as np
import os
import ntpath
import torch
from scipy.stats import wasserstein_distance


############################################################
#  Compare region of interest class
############################################################

class Compare_ROIS(object):    
    def __init__(self, paths, path_to_template, path_to_tail_mute_info, input_sketch, roi, mask, orien_perc, MA_perc, ma_perc, aspect_perc, low_end, high_end, tail, model, scar_i, search_mode, check):
        """
        Class to store image information and compute scar similarity metric

        Parameters
        ----------
        paths : list
            file paths to images
        path_to_template : str
            file path to template image
        path_to_tail_mute_info : str
            file path to tail mute CSV file             
        input_sketch : np.array
            the input sketch the user draws
        roi : list
            contains the user defined bounding box dimensions 
        mask : str
            path to manatee mask
        orien_perc : int
            orientation weight
        MA_perc : int
            length of scar weight
        ma_perc : int
            width of scar weight
        aspect_perc : int
            aspect ratio scar weight
        low_end : int
            user defined number of scars returned low end 
        high_end : int
            user defined number of scars returned high end 
        tail : str
            indicates whether user applies tail mute filter
        model : CNN
            tail mute classifier 

        Returns
        -------
        None

        """
        self.paths = paths
        self.mask = mask
        self.input_sketch = input_sketch
        self.roi = roi #[x1,y1,x2,y2]
        self.processed_images = None
        self.orien_perc = 0.20
        self.MA_perc = 0.30
        self.ma_perc = 0.30
        self.aspect_perc = 0.20
        self.low_end = 1
        self.high_end = 3
        self.tail = 'unknown'
        self.all_tail_info = None
        self.path_to_template = path_to_template
        self.path_to_tail_mute_info = path_to_tail_mute_info
        self.model = model
        self.scar_i = None
        self.search_mode = "shape"
        self.check = None
    def compare_rois(self):
        """
        Compares the input roi with the roi of dataset

        Returns
        -------
        List containing the matched images along with scar similarity score

        """     
        if self.search_mode == "shape":
            # get ROI array
            input_contour_info = []
            for input_bb in self.roi:  
                input_sketch_roi = self.input_sketch[int(input_bb[0]): int(input_bb[1]), int(input_bb[2]): int(input_bb[3])]
                # preprocess input sketch roi
                input_roi = self.preprocess(input_sketch_roi)        
                # find contours in input sketch roi
                input_contours = cv2.findContours(input_roi , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                # find contour rois in input sketch roi       
                input_shapes, input_area, input_num_contour, input_bb_dim = self.find_contours(input_contours[0], input_roi)
                input_contour_info.append([input_shapes, input_area, input_num_contour, input_bb_dim, self.tail])
            self.scar_i = input_contour_info
            distance_dict = []
        # First get all file names in list file_names    
            for i in range(len(self.processed_images)):
                # get ROI array and preprocess               
                for x in range(len(self.roi)):                
                    if self.tail == 'mute' or self.tail == 'no_mute':
                        if self.tail == 'mute':
                            mute_switch = 0
                        else:
                            mute_switch = 1
                        if mute_switch == self.processed_images[i][2]:                 
                            sketch_roi = self.processed_images[i][1][int(self.roi[x][0]): int(self.roi[x][1]), int(self.roi[x][2]): int(self.roi[x][3])]
                            # find contours in ROI
                            contours = cv2.findContours(sketch_roi , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                            # get contours rois in sketch roi                
                            contours_shapes, contour_area, num_contours, bb_dims  = self.find_contours(contours[0], sketch_roi)  
                            distances = self.compute_distances(input_contour_info[x][3], bb_dims, str(self.processed_images[i][0]))  
                            if distances != "NA":
                                distance_dict.append((str(self.processed_images[i][0]), distances[0], distances[1], distances[2], distances[3], distances[4], distances[5], distances[6])) 
                    else:
                        sketch_roi = self.processed_images[i][1][int(self.roi[x][0]): int(self.roi[x][1]), int(self.roi[x][2]): int(self.roi[x][3])]
                        # find contours in ROI
                        contours = cv2.findContours(sketch_roi , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                        # get contours rois in sketch roi
                        contours_shapes, contour_area, num_contours, bb_dims  = self.find_contours(contours[0], sketch_roi)  
                        distances = self.compute_distances(input_contour_info[x][3], bb_dims, str(self.processed_images[i][0]))  
                        if distances != "NA":
                            distance_dict.append((str(self.processed_images[i][0]), distances[0], distances[1], distances[2], distances[3], distances[4], distances[5], distances[6]))              
            our_scar_df = self.buildDataFrame(distance_dict)
            return our_scar_df
        else:
            # get ROI array
            input_roi_info = []
            for input_bb in self.roi:  
                input_sketch_roi = self.input_sketch[int(input_bb[0]): int(input_bb[1]), int(input_bb[2]): int(input_bb[3])]
                # preprocess input sketch roi                
                input_roi = self.preprocess(input_sketch_roi)                 
                input_roi_info.append(input_roi)
            distance_dict = []
            for i in range(len(self.processed_images)):
                # get ROI array and preprocess               
                for x in range(len(self.roi)):                
                    if self.tail == 'mute' or self.tail == 'no_mute':
                        if self.tail == 'mute':
                            mute_switch = 0
                        else:
                            mute_switch = 1
                        if mute_switch == self.processed_images[i][2]:                 
                            sketch_roi = self.processed_images[i][1][int(self.roi[x][0]): int(self.roi[x][1]), int(self.roi[x][2]): int(self.roi[x][3])]
                            EMD_1 = self.computeEMD(input_roi_info[x], sketch_roi)
                            if EMD_1 != "NA":
                                distance_dict.append((str(self.processed_images[i][0]), EMD_1[0], EMD_1[1], EMD_1[2], EMD_1[3], EMD_1[4], EMD_1[5], EMD_1[6])) 
                    else:
                        sketch_roi = self.processed_images[i][1][int(self.roi[x][0]): int(self.roi[x][1]), int(self.roi[x][2]): int(self.roi[x][3])]                        
                        EMD_1 = self.computeEMD(input_roi_info[x], sketch_roi)
                        if EMD_1 != "NA":
                            distance_dict.append((str(self.processed_images[i][0]), EMD_1[0], EMD_1[1], EMD_1[2], EMD_1[3], EMD_1[4], EMD_1[5], EMD_1[6]))             
            our_scar_df = self.buildDataFrame(distance_dict)
            return our_scar_df
    def preprocess(self, img):
        """
        Preprocesses images by blurring, bitwise-not and converting to binary. 

        Parameters
        ----------
        img : np.array
            image to preprocess

        Returns
        -------
        img : np.array
            Preprocessed image

        """    
        # blur
        img[img < 235] = 0
        img = cv2.blur(img, (2,2))
        # black background
        img = cv2.bitwise_not(img)
        # threshold
        _,img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)        
        return img
    def find_contours(self, contours_list, sketch_roi):
        """
        Computes some scar contour information

        Parameters
        ----------
        contours_list : list
            list of all contours found
        sketch_roi : np.array
            region of interest from dataset

        Returns
        -------
        contours_rois : list
            list of np.arrays containing each scar roi
        contour_area : list
            list containing area of each scar contour
        num_contours : int
            the number of scars found in the roi
        bb_dims : list
            list containing some more metrics: bb height, bb width, area, angle, scar height, scar width, contour coordinates, aspect ratio and extent

        """
        contours_rois = []
        contour_area = []
        num_contours = 0
        bb_info = []
        for contour in contours_list:
            #filled = cv2.fillPoly(sketch_roi.copy(), contours_list, 255)           
            x, y, w, h = cv2.boundingRect(contour)
            roi = sketch_roi[y:y + h, x:x + w]            
            # contour center coordinates
            contour_x = round(x + (w/2)) 
            contour_y = round(y + (h/2))               
            area = cv2.contourArea(contour)  
            # extent
            rect_area = w*h
            extent = float(area)/rect_area
            if area > 5:
                (x,y), (MA,ma), angle = cv2.fitEllipse(contour)   
                if MA == 0:
                    MA = 1
                aspect_ratio = float(ma)/MA                           
                contours_rois.append(roi)
                contour_area.append(area)
                bb_info.append([np.array([w,h]), area, angle, np.array([ma,MA]), np.array([contour_x, contour_y]), aspect_ratio, extent]) 
                num_contours += 1
        return contours_rois, contour_area, num_contours, bb_info    
    def compute_distances(self, input_contours_shape, contours_shape, name):
        """
        Computes similarity between input scars and dataset scars

        Parameters
        ----------
        input_contours_shape : list
            input scar information
        contours_shape : list
            dataset scar information
        name : str
            name of dataset manatee

        Returns
        -------
        float
            scar similarity score

        """
        num_input_scars = len(input_contours_shape)
        num_scars = len(contours_shape)    
        if num_input_scars == 0 and num_scars > 0:
            return 'NA'
        if num_input_scars == 0 and num_scars == 0:
            return [np.nan,np.nan,np.nan]
        #if num_input_scars != 0 and num_scars != 0:
        if num_scars <= self.high_end and num_scars >= self.low_end:
        #if num_input_scars != 0 and num_scars != 0:
            comparisons = []
            for shape in input_contours_shape:                
                for num, shape2 in enumerate(contours_shape):
                    # Separate h,w and MA,ma and x,y
                    input_oriten = shape[2]
                    oriten = shape2[2]                    
                    input_MA, input_ma = shape[3]  
                    MA, ma = shape2[3]
                    input_aspect = shape[5]
                    aspect = shape2[5]                    
                    diff_in_MA = abs(input_MA - MA)
                    percentage_MA = 100*(diff_in_MA/input_MA)
                    diff_in_ma = abs(input_ma - ma)/ input_ma
                    percentage_ma = 100*(diff_in_ma/input_ma)
                    diff_in_aspect = abs(input_aspect - aspect)/ input_aspect
                    percentage_aspect = 100*(diff_in_aspect/input_aspect)
                    diff_in_angle = abs(input_oriten - oriten)
                    percentage_angle = 100*(diff_in_angle/input_oriten)
                    comparisons.append([num, 1/4*(self.orien_perc * percentage_angle + self.MA_perc * percentage_MA + self.ma_perc * percentage_ma + self.aspect_perc * percentage_aspect),
                                        [round(percentage_angle,2), round(percentage_MA,2), round(percentage_ma,2), round(percentage_aspect,2)], [oriten, MA, ma, aspect]])
            if len(comparisons) != 0:
                distances, distances_info = self.computeScore(comparisons, num_input_scars)               
            return [np.mean(distances), np.sum(distances), np.min(distances), distances_info[0], distances_info[1], distances_info[2], distances_info[3]]
        else:
            return 'NA'
    def computeEMD(self, input_roi, sketch_roi):              
        input_roi[input_roi==255] = 1
        sketch_roi[sketch_roi==255] = 1                
        input_roi_rowsums = input_roi.sum(axis=1).ravel()
        sketch_roi_rowsums = sketch_roi.sum(axis=1).ravel() 
        input_roi_colsums = input_roi.sum(axis=0).ravel()
        sketch_roi_colsums = sketch_roi.sum(axis=0).ravel()              
        #sketch_roi_hist = cv2.calcHist([sketch_roi],[0],None,[256],[0,256])
        #input_roi_hist = cv2.calcHist([input_roi],[0],None,[256],[0,256])        
        emd_1_row = wasserstein_distance(input_roi_rowsums, sketch_roi_rowsums)
        emd_1_col = wasserstein_distance(input_roi_colsums, sketch_roi_colsums) 
        min_dist = emd_1_row + emd_1_col  
        return [min_dist, min_dist, min_dist, min_dist, min_dist, min_dist, min_dist]
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
    def removeOutline(self, img, mask):
        """
        Removes manatee outline and extracts features
 
        Parameters
        ----------
        img : np.array
            image we want to remove manatee outline from 
        mask : str
            path to manatee mask

        Returns
        -------
        img : np.array
            manatee image with outline removed

        """
        img2 = img.copy()
        mask = cv2.imread(self.mask, cv2.IMREAD_GRAYSCALE)
        template = cv2.imread(self.path_to_template, cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img2,template,4)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        crop_img = img2[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        crop_img[mask == 0] = 255        
        img2 = cv2.resize(crop_img, (259, 559), interpolation= cv2.INTER_NEAREST)                                     
        return img2
    def preLoadData(self):
        """
        Preloads the images after launching application

        Returns
        -------
        None

        """
        processed_images = []
        sketch_names = self.paths    
        for i in range(len(sketch_names)):
            # get sketch path
            sketch_path = sketch_names[i]
            # read sketch in grayscale format
            sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)                                                                             
            sk_name = sketch_path.split(os.sep)[-1]
            if sk_name[-1] == "g" or sk_name[-1] == "G":
                if self.all_tail_info[sk_name] == 'NA':
                    sketch = cv2.resize(sketch, (259, 559), interpolation= cv2.INTER_NEAREST)                                        
                    tail = sketch[363:559,30:226] / 255                
                    im_tensor = torch.tensor(tail.astype(np.float32))    
                    probs = self.model(im_tensor.unsqueeze(0).unsqueeze(0))                
                    _, predicted = torch.max(probs.data, 1)      
                    self.all_tail_info[sk_name] = predicted.item()
                    new_df = pd.DataFrame(self.all_tail_info.items())
                    new_df.to_csv(self.path_to_tail_mute_info, index=False)   
                sketch_no_outline = self.removeOutline(sketch, self.mask)
                preprocessed_img = self.preprocess(sketch_no_outline)
                sketch_name = ntpath.basename(sketch_names[i])
                processed_images.append([sketch_name, preprocessed_img, int(self.all_tail_info[sk_name]), sketch])
        self.processed_images = processed_images
    def computeScore(self, dist, num_input_scars):
        """
        Ensures that best match is taken for each input scar.

        Parameters
        ----------
        dist : list
            list of similarity scores for all scars
        num_input_scars : int
            number of input scars

        Returns
        -------
        float
            sum of similarity scores
            
        """         
        scores = []    
        num_lookup_scars = len(list(set([el[0] for el in dist]))) 
        while len(scores) <= num_input_scars - 1:
            if len(scores) >= num_lookup_scars:
                break
            current_lowest = dist[np.argmin([el[1] for el in dist])]        
            if len(dist) != 0:
                # dta = current_lowest[2]
                # info += 'diff in angle: ' + str(dta[0]) + '\n diff in length: ' + str(dta[1]) + '\n diff in width: ' + str(dta[2]) + '\n diff in area: ' + str(dta[3]) +  '\n diff in aspect: ' + str(dta[4]) + '\n diff in x: ' + str(dta[5]) + '\n diff in y: ' + str(dta[6]) + ' ' 
                # info += '\n'
                scores.append(current_lowest)
            dist = [item for item in dist if item[0] != current_lowest[0]]            
        best_scores = [el[1] for el in scores]
        best_scores_info = [el[3] for el in scores]
        diff_in_scar_stats = [el[2] for el in scores]
        diff_in_oriten = [] 
        diff_in_MA = []
        diff_in_ma = []
        diff_in_aspect = []         
        self.check = best_scores_info            
        for i in diff_in_scar_stats:
            diff_in_oriten.append(i[0])
            diff_in_MA.append(i[1])
            diff_in_ma.append(i[2])
            diff_in_aspect.append(i[3])                    
        return best_scores, [min(diff_in_oriten), min(diff_in_MA), min(diff_in_ma), min(diff_in_aspect)]
    def getImagePath(self, image_name):
        """
        Returns the path to the folder the image is in.

        Parameters
        ----------
        image_name : str
            image name

        Returns
        -------
        str
            image path

        """
        for name in self.paths:
            if name.endswith(str(image_name)):
                my_path = os.path.normpath(name)
                components = my_path.split(os.sep)
                return os.path.join(components[-2], components[-1])

