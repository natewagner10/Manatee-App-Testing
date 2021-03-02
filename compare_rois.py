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



############################################################
#  Compare region of interest class
############################################################

class Compare_ROIS(object):    
    def __init__(self, paths, path_to_template, path_to_tail_mute_info, input_sketch, roi, mask, orien_perc, MA_perc, ma_perc, area_perc, aspect_perc, locX_perc, locY_perc, low_end, high_end, tail, model):
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
        area_perc : int
            size of scar weight
        aspect_perc : int
            aspect ratio scar weight
        locX_perc : int
            location x-direction scar weight
        locY_perc : int
            location y-direction scar weight
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
        self.orien_perc = 0.08
        self.MA_perc = 0.23
        self.ma_perc = 0.23
        self.area_perc = 0.10
        self.aspect_perc = 0.36
        self.locX_perc = 0.10
        self.locY_perc = 0.10
        self.low_end = 1
        self.high_end = 3
        self.tail = 'unknown'
        self.all_tail_info = None
        self.path_to_template = path_to_template
        self.path_to_tail_mute_info = path_to_tail_mute_info
        self.model = model
    def compare_rois(self):
        """
        Compares the input roi with the roi of dataset

        Returns
        -------
        List containing the matched images along with scar similarity score

        """     
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
                            distance_dict.append((str(self.processed_images[i][0]), distances))
                else:                  
                    sketch_roi = self.processed_images[i][1][int(self.roi[x][0]): int(self.roi[x][1]), int(self.roi[x][2]): int(self.roi[x][3])]
                    # find contours in ROI
                    contours = cv2.findContours(sketch_roi , cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    # get contours rois in sketch roi                
                    contours_shapes, contour_area, num_contours, bb_dims  = self.find_contours(contours[0], sketch_roi)  
                    distances = self.compute_distances(input_contour_info[x][3], bb_dims, str(self.processed_images[i][0]))  
                    if distances != "NA":
                        distance_dict.append((str(self.processed_images[i][0]), distances))                    
        distance_dict_df = pd.DataFrame(distance_dict)
        unique_names_cnts = distance_dict_df.groupby(0)[1].agg(['count', 'mean'])
        unique_names_cnts['names'] = unique_names_cnts.index
        has_all_scars = unique_names_cnts[unique_names_cnts['count'] >= len(self.roi)]
        returned = has_all_scars[['names', 'mean']]        
        returned_list = returned.values.tolist()     
        returned_list = sorted(returned_list, key = lambda x: x[1])         
        returned_list2 = []
        for idx, img in enumerate(returned_list):
            returned_list2.append([idx+1, img[0], img[1]])
        return returned_list2
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
        img[img < 215] = 0
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
        bb_dims = []
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
            if area > 20 and len(contour) >= 5:
                (x,y), (MA,ma), angle = cv2.fitEllipse(contour)   
                if MA == 0:
                    MA = 1
                aspect_ratio = float(ma)/MA                           
                contours_rois.append(roi)
                contour_area.append(area)
                bb_dims.append([np.array([w,h]), area, angle, np.array([MA,ma]), np.array([contour_x, contour_y]), aspect_ratio, extent]) 
                num_contours += 1
        return contours_rois, contour_area, num_contours, bb_dims    
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
            return 0
        #if num_input_scars != 0 and num_scars != 0:
        if num_scars <= self.high_end and num_scars >= self.low_end:
        #if num_input_scars != 0 and num_scars != 0:
            comparisons = []
            for shape in input_contours_shape:                
                for num, shape2 in enumerate(contours_shape):
                    # Separate h,w and MA,ma and x,y
                    input_h, input_w = shape[0]
                    h,w = shape2[0]
                    input_MA, input_ma = shape[3]  
                    MA, ma = shape2[3]
                    input_x, input_y = shape[4]
                    x,y = shape2[4]    
                    input_area = shape[1]
                    area = shape2[1]
                    input_aspect = shape[5]
                    aspect = shape2[5]                    
                    # Compute percentage differences for each feature
                    diff_in_x = abs(input_x - x)
                    percentage_in_x = (100*diff_in_x)/input_x
                    diff_in_y = abs(input_y - y)
                    percentage_in_y = (100*diff_in_y)/input_y
                    diff_in_MA = abs(input_MA - MA)
                    percentage_MA = (100*diff_in_MA)/input_MA
                    diff_in_ma = abs(input_ma - ma)/ input_ma
                    percentage_ma = (100*diff_in_ma)/input_ma
                    diff_in_area = abs(input_area - area)/ input_area
                    percentage_area = (100*diff_in_area)/input_area
                    diff_in_aspect = abs(input_aspect - aspect)/ input_aspect
                    percentage_aspect = (100*diff_in_aspect)/input_aspect                                        
                    diff_in_angle = abs(shape[2] - shape2[2])
                    percentage_angle = (100*(diff_in_angle))/shape[2]
                    comparisons.append([num, 1/8*(self.orien_perc * percentage_angle + self.MA_perc * percentage_MA + self.ma_perc * percentage_ma + self.area_perc * percentage_area + self.aspect_perc * percentage_aspect + self.locX_perc * percentage_in_x + self.locY_perc * percentage_in_y)])
            if len(comparisons) != 0:
                distances = self.computeScore(comparisons, num_input_scars)                                    
            return np.mean(distances)
        else:
            return 'NA'
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
        mask = cv2.imread(self.mask, cv2.IMREAD_GRAYSCALE)
        template = cv2.imread(self.path_to_template, cv2.IMREAD_GRAYSCALE)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img,template,4)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        crop_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        crop_img[mask == 0] = 255        
        img = cv2.resize(crop_img, (259, 559), interpolation= cv2.INTER_NEAREST)                                     
        return img
    def preLoadData(self):
        """
        Preloads the images after launching application

        Returns
        -------
        None

        """
        #sketch_names = [] 
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
                processed_images.append([sketch_name, preprocessed_img, int(self.all_tail_info[sk_name])])
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
                scores.append(current_lowest)
            dist = [item for item in dist if item[0] != current_lowest[0]]    
        return np.sum([el[1] for el in scores])
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


                      
