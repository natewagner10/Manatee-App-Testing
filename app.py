
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import cv2
import base64
import os
import json
import numpy as np
from dash_canvas import DashCanvas
from dash_canvas.utils import parse_jsonstring
import pandas as pd
from PIL import Image
import ntpath
import dash_table


######################################################################################################

### Needed Paths ###
path_to_images = '/Users/natewagner/Documents/Mote_Manatee_Project/data/data_folders/'
path_to_mask = '/Users/natewagner/Documents/Mote_Manatee_Project/canny_filled2.png'
path_to_blank = '/Users/natewagner/Documents/Mote_Manatee_Project/data/BLANK_SKETCH_updated.jpg'

# slider 
orientation_perc = 1
MA_perc = 1
ma_perc = 1
area_perc = 1
aspect_perc = 1
locX_perc = 1
locY_perc = 1
pixs_perc = 1

def Navbar():
    navbar = dbc.NavbarSimple(
        brand="Manatee Identification",
        color="primary",
        expand = 'md',
        dark=True,
    )
    return navbar

class Compare_ROIS(object):
    def __init__(self, paths, input_sketch, roi, mask, orien_perc, MA_perc, ma_perc, area_perc, aspect_perc, locX_perc, locY_perc, pixs_perc, low_end, high_end):
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
        self.pixs_perc = 0.10
        self.low_end = 1
        self.high_end = 3
    def compare_rois(self):
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
            input_contour_info.append([input_shapes, input_area, input_num_contour, input_bb_dim])
        distance_dict = []
    # First get all file names in list file_names
        for i in range(len(self.processed_images)):
            # get ROI array and preprocess     
            for x in range(len(self.roi)):
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
        # blur
        img[img < 215] = 0
        img = cv2.blur(img, (2,2))
        # black background
        img = cv2.bitwise_not(img)
        # threshold
        _,img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)        
        return img
    def find_contours(self, contours_list, sketch_roi):
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
            avg_pixel = np.sum(roi)
            if area > 20 and len(contour) >= 5:
                (x,y), (MA,ma), angle = cv2.fitEllipse(contour)   
                if MA == 0:
                    MA = 1
                aspect_ratio = float(ma)/MA                           
                contours_rois.append(roi)
                contour_area.append(area)
                bb_dims.append([np.array([w,h]), area, angle, np.array([MA,ma]), np.array([contour_x, contour_y]), avg_pixel, aspect_ratio, extent]) 
                num_contours += 1
        return contours_rois, contour_area, num_contours, bb_dims    
    def compute_distances(self, input_contours_shape, contours_shape, name):
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
                    input_aspect = shape[6]
                    aspect = shape2[6]                    
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
                    diff_in_pixs = abs(shape[5] - shape2[5])
                    percentage_pixs = (100*(diff_in_pixs))/shape[5]
                    diff_in_angle = abs(shape[2] - shape2[2])
                    percentage_angle = (100*(diff_in_angle))/shape[2]
                    comparisons.append([num, 1/8*(self.orien_perc * percentage_angle + self.MA_perc * percentage_MA + self.ma_perc * percentage_ma + self.area_perc * percentage_area + self.aspect_perc * percentage_aspect + self.locX_perc * percentage_in_x + self.locY_perc * percentage_in_y + self.pixs_perc * percentage_pixs)])
            if len(comparisons) != 0:
                distances = self.computeScore(comparisons, num_input_scars)                                    
            return np.mean(distances)
        else:
            return 'NA'
    def removeOutline(self, img, mask):
        mask = cv2.imread(path_to_mask, cv2.IMREAD_GRAYSCALE)
        blur = cv2.GaussianBlur(mask,(5,5),0)
        mask = cv2.addWeighted(blur,1.5,mask,-0.5,0)
        mask[mask != 0] = 1          
        img = cv2.resize(img, (259, 559), interpolation= cv2.INTER_NEAREST)
        img[mask == 1] = 255
        return img
    def preLoadData(self):
        #sketch_names = [] 
        processed_images = []
        sketch_names = self.paths
        for i in range(len(sketch_names)):
            # get sketch path
            sketch_path = sketch_names[i]
            # read sketch in grayscale format
            try:
                sketch = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
                sketch_no_outline = self.removeOutline(sketch, self.mask)
                preprocessed_img = self.preprocess(sketch_no_outline)
                sketch_name = ntpath.basename(sketch_names[i])
                processed_images.append([sketch_name, preprocessed_img])
            except:
                continue
        self.processed_images = processed_images
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
        return np.sum([el[1] for el in scores])
    def getImagePath(self, image_name):
        for name in self.paths:
            if name.endswith(str(image_name)):
                my_path = name.split('/')
                return my_path[-2] + '/' + my_path[-1]

def readjust(weights):
    """
    Reparameterize max values for weight range.
    """
    new_weights = weights.copy()
    for i, weight in enumerate(weights):
        # We need all other weights except for the one we are currently
        # looking at because we need to adjust them accordingly
        amt_unused = 1 - weight
        # add unused amount to other values proportionally
        excess = amt_unused/(len(weights)-1)
        before = new_weights[:i]
        after = new_weights[i+1:]
        before_arr = np.array(before)
        after_arr = np.array(after)
        new_weights[:i] = before_arr + excess
        new_weights[i+1:] = after_arr + excess
    return(new_weights)


def getFilePaths(root_dir):
    file_paths = []
    for root, _, fnames in sorted(os.walk(root_dir, followlinks=True)):
        for fname in sorted(fnames):
            path = os.path.join(root, fname)
            #if path.lower().endswith('.jpg'):
            file_paths.append(path)
    return file_paths
    
if 'find_matches_func' not in globals():
    find_matches_func = Compare_ROIS(None, None, None, path_to_mask, orientation_perc, MA_perc, ma_perc, area_perc, aspect_perc, locX_perc, locY_perc, pixs_perc, 1, 3)

paths_to_images = getFilePaths(path_to_images)
find_matches_func.paths = paths_to_images
find_matches_func.preLoadData()
find_matches_func.getImagePath('U4757.jpg')

######################################################################################################

# LITERA
app = dash.Dash(__name__, meta_tags=[{"content": "width=device-width"}], external_stylesheets=[dbc.themes.LITERA], assets_folder=path_to_images)
server = app.server

# dash canvas info
filename = Image.open(path_to_blank)
canvas_width = 259   # this has to be set to 259 because we use the canvas as input to the model

score_html = " "
n_html = " "
num_matches_html = " "
name_html = " "
name_info = None

app.layout = html.Div(
    [
     Navbar(),
     html.Br(),
     dbc.Row(
         [
             dbc.Col(
                 dbc.Card(
                    [
                       html.H6(id="orientation", children=['Orientation'], style={'textAlign': 'center', 'font-weight': 'normal'}),                    
                       dcc.Slider(
                           id='my-slider',
                           min=0,
                           max=2,
                           step=0.10,
                           value=1,
                           marks={
                               0: {'label': '0'},
                               1: {'label': '1'},                                                                
                               2: {'label': '2'}
                           }
                       ),
                dbc.Tooltip("Useful for tilted scars or scars with some sense of orientation.",target="orientation"),                       
                       html.Div(id='hidden-div', style={'display':'none'}),
                       html.H6(id="length", children=['Length'], style={'textAlign': 'center', 'font-weight': 'normal'}),                                                                        
                       dcc.Slider(
                           id='my-slider1',
                           min=0,
                           max=2,
                           step=0.10,
                           value=1,
                           marks={
                               0: {'label': '0'},
                               1: {'label': '1'},                                                                
                               2: {'label': '2'}
                           }
                       ),
                dbc.Tooltip("Length of the scar (with respect to the ellipse fitted to the scar).",target="length"),                              
                       html.Div(id='hidden-div2', style={'display':'none'}),                                                    
                       html.H6(id="width", children=['Width'], style={'textAlign': 'center', 'font-weight': 'normal'}),                                                                        
                       dcc.Slider(
                           id='my-slider2',
                           min=0,
                           max=2,
                           step=0.10,
                           value=1,
                           marks={
                               0: {'label': '0'},
                               1: {'label': '1'},                                                                
                               2: {'label': '2'}
                           }
                       ),  
                dbc.Tooltip("Width of the scar (with respect to the ellipse fitted to the scar).",target="width"),                       
                       html.Div(id='hidden-div3', style={'display':'none'}),                                                    
                       html.H6(id="area", children=['Area'], style={'textAlign': 'center', 'font-weight': 'normal'}),                                                                        
                       dcc.Slider(
                           id='my-slider3',
                           min=0,
                           max=2,
                           step=0.10,
                           value=1,
                           marks={
                               0: {'label': '0'},
                               1: {'label': '1'},                                                                
                               2: {'label': '2'}
                           }
                       ), 
                dbc.Tooltip("Area of the scar (size of scar). Increasing this will add additional penalties to scars having a greater area than the drawn scar. ",target="area"),                                              
                       html.Div(id='hidden-div4', style={'display':'none'}),                                                    
                       html.H6(id="aspect", children=['Aspect Ratio'], style={'textAlign': 'center', 'font-weight': 'normal'}),                                                                        
                       dcc.Slider(
                           id='my-slider4',
                           min=0,
                           max=2,
                           step=0.10,
                           value=1,
                           marks={
                               0: {'label': '0'},
                               1: {'label': '1'},                                                                
                               2: {'label': '2'}
                           }
                       ),   
                dbc.Tooltip("Ratio of width to height of rectangle fitted to scar. Increasing this will penalize scars that don’t have similar width-to-height ratios. ",target="aspect"),                       
                       html.Div(id='hidden-div5', style={'display':'none'}), 
                       html.H6(id="locationx", children=['Location X'], style={'textAlign': 'center', 'font-weight': 'normal'}),                                                                        
                       dcc.Slider(
                           id='my-slider5',
                           min=0,
                           max=2,
                           step=0.10,
                           value=1,
                           marks={
                               0: {'label': '0'},
                               1: {'label': '1'},                                                                
                               2: {'label': '2'}
                           }
                       ),   
                dbc.Tooltip("Scar Location (x-direction): horizontal axis location with respect to the whole sketch. Increasing this will add additional penalties to scars being further away from the drawn scar.",target="locationx"),                                              
                       html.Div(id='hidden-div6', style={'display':'none'}), 
                       html.H6(id="locationy", children=['Location Y'], style={'textAlign': 'center', 'font-weight': 'normal'}),                                                                        
                       dcc.Slider(
                           id='my-slider6',
                           min=0,
                           max=2,
                           step=0.10,
                           value=1,
                           marks={
                               0: {'label': '0'},
                               1: {'label': '1'},                                                                
                               2: {'label': '2'}
                           }
                       ),   
                dbc.Tooltip("Scar Location (y-direction): vertical axis location with respect to the whole sketch. Increasing this will add additional penalties to scars being further away from the drawn scar. ",target="locationy"),                       
                       html.Div(id='hidden-div7', style={'display':'none'}),
                       html.H6(id="sumpixs", children=['Sum of Pixels'], style={'textAlign': 'center', 'font-weight': 'normal'}),                                                                        
                       dcc.Slider(
                           id='my-slider7',
                           min=0,
                           max=2,
                           step=0.10,
                           value=1,
                           marks={
                               0: {'label': '0'},
                               1: {'label': '1'},                                                                
                               2: {'label': '2'}
                           }
                       ),   
                dbc.Tooltip("Sum of Scar Pixel Values: Good for tail mutilations.",target="sumpixs"),                       
                       html.Div(id='hidden-div8', style={'display':'none'}),                       
                       #html.Hr(),
                       dbc.Button(id="clear", children="Update Weights", color="primary", style = {'width': '76%', 'margin-left': '12.5%'}),
                       html.Span(id="slider-sum", style={"vertical-align": "middle"}),
                       html.Div(
                           [
                            html.H6(children=['Number of Scars in Match:'], style={'textAlign': 'center', 'font-weight': 'normal', 'margin-bottom': '10px'}),
                            dcc.RangeSlider(
                                    id='my-range-slider',
                                    min=0,
                                    max=10,
                                    step=1,
                                    marks={
                                        0: '0',
                                        1: '1',
                                        2: '2',
                                        3: '3',
                                        4: '4',
                                        5: '5',
                                        6: '6',
                                        7: '7',
                                        8: '8',
                                        9: '9',
                                        10: '10',                                        
                                    },
                                    value=[1, 3]
                                ),
                           html.Div(id='hidden-div12', style={'display':'none'}),                       
                           html.H6(children=['Brush Width'], style={'textAlign': 'center', 'font-weight': 'normal'}),
                               dcc.Slider(
                                   id='bg-width-slider',
                                   min=1,
                                   max=40,
                                   step=1,
                                   value=1,
                               ),                                         
                               ],
                           style = {"margin-top": "10px"}),                                                   
                       ],
                    style = {'width': '100%',
                             'margin-top': '5px',
                             'margin-right': '100px',
                             'display': 'inline-block',                                      
                             'box-shadow': '0 0 10px lightgrey'}
                    ),md = 2),
             dbc.Col(
                 dbc.Card(
                     [
                     dbc.CardHeader(
                     html.H4("Sketch", className="card-text"), style={"width": "100%", 'textAlign': 'center'}
                     ),
                        html.Div(
                            [                                       
                                DashCanvas(
                                    id='canvas',
                                    width=canvas_width,
                                    filename=filename,
                                    lineColor='black',
                                    hide_buttons=['line', 'select', 'zoom', 'pan'],
                                    goButtonTitle="Search"                                    
                                    ),
                            ], style = {'margin-left': '25%',
                                        'margin-top': '7px'}
                            ),
                    ],
                    style = {#'textAlign': 'center',
                             'width': '100%',
                             #'height': '110%',
                             'display': 'inline-block',
                             'box-shadow': '0 0 10px lightgrey'}
                    ),
                    md = 4),               
             dbc.Col(
                 dbc.Card(
                     [
                         dbc.CardHeader(
                         html.H4("Browse Matches", className="card-text", style = {'textAlign': 'center'}), style={"width": "100%"}
                         ),
                          html.Span(id="sketch_output", style={"vertical-align": "middle"}),
                          dbc.CardFooter(
                              dbc.Row(
                                  [
                                      dbc.Col(html.Div(id = 'sketch_output_info1'), md = 12),
                                  ]
                                  )
                              )                                 
                     ],
                     style={'width': '100%',
                            #'height': '',                            
                            'align-items': 'center',
                            'display': 'inline-block',
                            'box-shadow': '0 0 10px lightgrey'}
                     ),
                     md = 6),
        ]
        ),
    ],
    style={'padding': '0px 40px 0px 40px', 'height': '100%'}
)

@app.callback([Output('sketch_output', 'children'),
               Output('sketch_output_info1', 'children')],
                [Input('canvas', 'json_data'),
                 Input('canvas', 'n_clicks')],
                [State('canvas', 'image_content')])
def update_data(string,image,n):    
    global name_info, names, switch, count, find_matches_func, num_returned, path_to_mask
    blank = base64.b64encode(open(path_to_blank, 'rb').read())    
    switch = True
    is_rect = False
    if string:
        data = json.loads(string)
        bb_info = data['objects'][1:]         
        bounding_box_list = []        
        for i in bb_info:
            if i['type'] == 'rect':  
                is_rect = True
                top = i['top']
                left = i['left']
                wd = i['width']
                ht = i['height']
                bounding_box_list.append((top, top+ht, left, left+wd))
            else:
                continue        
        if is_rect == False:
            bounding_box_list.append((0, 559, 0, 259))        
        mask = parse_jsonstring(string, shape=(559, 259))
        mask = (~mask.astype(bool)).astype(int)
        mask[mask == 1] = 255
        mask = mask.astype(np.uint8)
        if 'find_matches_func' in globals():
            find_matches_func.input_sketch = mask
            find_matches_func.roi = bounding_box_list
        matches = find_matches_func.compare_rois()
        name_info = matches
        is_rect = False 
        images_url = []
        names_df = []
        for entry in matches:
            names_df.append(entry[1][0:-4])
            needed_im = '![myImage-' + str(entry[0]) + '](assets/' + str(find_matches_func.getImagePath(entry[1])) + ')'            
            images_url.append(needed_im)
            
        data_tups = list(zip(names_df,images_url))
        data_table_df = pd.DataFrame(data_tups, columns=['Name','Image'])
        #if data_table_df is not None:
        return html.Div(
                       [                           
                           dash_table.DataTable(
                              id='table',
                              columns=[{"name": 'Name', "id": 'Name'},
                                  {
                                    'id': 'Image',
                                    'name': 'Image',
                                    'presentation': 'markdown',
                                  },
                              ],
                              page_size=50,
                              data=data_table_df.to_dict('records'),
                              style_table={'height': '575px', 'overflowY': 'auto'},
                              style_cell={'textAlign': 'center', 'font_size': '26px'},
                              style_header = {'display': 'none'},
                              style_as_list_view=True,
                              css=[
                                      {
                                         'selector': 'tr:first-child',
                                         'rule': 'display: none',
                                      },
                                    ],
                          ),  
                       ],
                       style = {'textAlign': 'center',
                                'width': '100%',
                                'align-items': 'center',
                                }), html.H5("Number of Matches: " + str(len(names_df)))            
    return html.Div([
    html.Img(src='data:image/png;base64,{}'.format(blank.decode()))
    ], style = {'align-items': 'center','margin-left':'35%'}), html.H5("Number of Matches: ")

@app.callback(Output("canvas", "json_objects"), [Input("clear", "n_clicks")])
def clear_canvas(n):
    if n is None:
        return dash.no_update
    strings = ['{"objects":[ ]}', '{"objects":[]}']
    return strings[n % 2]

@app.callback(Output('canvas', 'lineColor'),
            [Input('color-picker', 'value')])
def update_canvas_linecolor(value):
    if isinstance(value, dict):
        return value['hex']
    else:
        return value

@app.callback(Output('canvas', 'lineWidth'),
            [Input('bg-width-slider', 'value')])
def update_canvas_linewidth(value):
    return value

@app.callback(Output('hidden-div', 'children'),
            [Input('my-slider', 'value')])
def update_orientation(value):
    global orientation_perc
    orientation_perc = value
    find_matches_func.orien_perc = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})

@app.callback(Output('hidden-div2', 'children'),
            [Input('my-slider1', 'value')])
def update_MA(value):
    global MA_perc
    MA_perc = value
    find_matches_func.MA_perc = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})

@app.callback(Output('hidden-div3', 'children'),
            [Input('my-slider2', 'value')])
def update_ma(value):
    global ma_perc
    ma_perc = value
    find_matches_func.ma_perc = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})

@app.callback(Output('hidden-div4', 'children'),
            [Input('my-slider3', 'value')])
def update_area(value):
    global area_perc
    area_perc = value
    find_matches_func.area_perc = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})

@app.callback(Output('hidden-div5', 'children'),
            [Input('my-slider4', 'value')])
def update_aspect(value):
    global aspect_perc
    aspect_perc = value
    find_matches_func.aspect_perc = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})

@app.callback(Output('hidden-div6', 'children'),
            [Input('my-slider5', 'value')])
def update_locationX(value):
    global locX_perc
    locX_perc = value
    find_matches_func.locX_perc = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})

@app.callback(Output('hidden-div7', 'children'),
            [Input('my-slider6', 'value')])
def update_locationY(value):
    global locY_perc
    locY_perc = value
    find_matches_func.locY_perc = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})

@app.callback(Output('hidden-div8', 'children'),
            [Input('my-slider7', 'value')])
def update_sum_pixs(value):
    global pixs_perc
    pixs_perc = value
    find_matches_func.pixs_perc = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})

@app.callback(Output('slider-sum', 'children'),
            [Input('my-slider', 'value'),
             Input('my-slider1', 'value'),
             Input('my-slider2', 'value'),
             Input('my-slider3', 'value'),
             Input('my-slider4', 'value'),
             Input('my-slider5', 'value'),
             Input('my-slider6', 'value'),
             Input('my-slider7', 'value')])
def sum_slider(value,value1,value2,value3,value4,value5,value6,value7):
    summ = round((value+value1+value2+value3+value4+value5+value6+value7),2) 
    return html.Div([
                #html.Hr(),
                html.H6(children=['Weights Sum: ' + str(summ)], style={'textAlign': 'center', 'font-weight': 'normal', 'margin-top':'10px'}),                                                                        
                html.Hr(),                
                ], style = {'align-items': 'center'})

@app.callback([Output('my-slider', 'value'),
             Output('my-slider1', 'value'),
             Output('my-slider2', 'value'),
             Output('my-slider3', 'value'),
             Output('my-slider4', 'value'),
             Output('my-slider5', 'value'),
             Output('my-slider6', 'value'),
             Output('my-slider7', 'value')],
             [Input('clear', 'n_clicks')])
def updateSliders(value):
    new,new1,new2,new3,new4,new5,new6,new7 = readjust([orientation_perc,MA_perc,ma_perc,area_perc,aspect_perc,locX_perc,locY_perc,pixs_perc])
    return new,new1,new2,new3,new4,new5,new6,new7

@app.callback(
    Output('hidden-div12', 'children'),
    [Input('my-range-slider', 'value')])
def update_range_slider(value):
    find_matches_func.low_end = value[0]
    find_matches_func.high_end = value[1]    
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'}) 

if __name__ == '__main__':
    app.run_server()
   
