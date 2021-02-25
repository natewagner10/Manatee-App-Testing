"""
Mote Marine Laboratory Collaboration

Manatee Matching Program

Written by Nate Wagner, Rosa Gradilla 
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import base64
import json
import numpy as np
from dash_canvas import DashCanvas
from dash_canvas.utils import parse_jsonstring
import pandas as pd
from PIL import Image
import dash_table
import torch
from Documents.Mote_Marine_Manatees.app3.compare_rois import Compare_ROIS
from Documents.Mote_Marine_Manatees.app3.cnn import CNN
from Documents.Mote_Marine_Manatees.app3.helpers import readjust, getFilePaths

############################################################
#  Needed paths
############################################################

path_to_images = '/Users/natewagner/Documents/Mote_Manatee_Project/data/data_folders/'
path_to_mask = '/Users/natewagner/Documents/Mote_Manatee_Project/data/new_mask_temp4.png'
path_to_blank = '/Users/natewagner/Documents/Mote_Manatee_Project/data/BLANK_SKETCH_updated.jpg'
path_to_cnn = '/Users/natewagner/Documents/Mote_Manatee_Project/data/assets/tail_mute_cnn_15_epochs_99.pt'
path_to_tail_mute_info = '/Users/natewagner/Documents/Mote_Manatee_Project/data/tail_mute_info.csv'
path_to_template = '/Users/natewagner/Documents/Mote_Manatee_Project/data/BLANK_SKETCH_updated_cropped.jpg'

############################################################
#  Initiate model and 
#  Compare_ROIS class, initial slider weights
############################################################

orientation_perc = 1
MA_perc = 1
ma_perc = 1
area_perc = 1
aspect_perc = 1
locX_perc = 1
locY_perc = 1
pixs_perc = 1

model = CNN()
model.load_state_dict(torch.load(path_to_cnn, map_location=torch.device('cpu')))
    
if 'find_matches_func' not in globals():
    find_matches_func = Compare_ROIS(None, None, None, path_to_mask, orientation_perc, MA_perc, ma_perc, area_perc, aspect_perc, locX_perc, locY_perc, 1, 3, "unkown", path_to_template, path_to_tail_mute_info, model)

paths_to_images, find_matches_func.all_tail_info = getFilePaths(path_to_images, path_to_tail_mute_info)
find_matches_func.paths = paths_to_images
find_matches_func.preLoadData()

############################################################
#  User interface
############################################################

app = dash.Dash(__name__, meta_tags=[{"content": "width=device-width"}], external_stylesheets=[dbc.themes.LITERA], assets_folder=path_to_images)
server = app.server

# dash canvas info
filename = Image.open(path_to_blank)
canvas_width = 259   # this has to be set to 259 because we use the canvas as input to the model
canvas_height = 559

app.layout = html.Div(
    [
     dbc.NavbarSimple(
        brand="Manatee Matching Application",
        color="primary",
        expand = 'md',
        dark=True,
    ),
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
                dbc.Tooltip("Scar Location (y-direction): horizontal axis location with respect to the whole sketch. Increasing this will add additional penalties to scars being further away from the drawn scar.",target="locationy"),                                                                     
                html.Div(id='hidden-div8', style={'display':'none'}),                       
                    #html.Hr(),
                    html.Div([
                        dcc.RadioItems(
                            id="tail_filter",
                             options=[
                                 {'label': 'Tail Mute', 'value': 'mute'},
                                 {'label': 'No Mute', 'value': 'no_mute'},
                                 {'label': 'Unkown', 'value': 'unkown'},
                             ],
                             value='unkown', 
                             inputStyle={"margin-right": "10px", "margin-left": "10px"},
                         )
                                       ],style = {'align-items': 'center', 'margin-left': '10%'}),                        
                       html.Div(id='hidden-div13', style={'display':'none'}), 
                       dbc.Button(id="update_the_weights", children="Update Weights", color="primary", style = {'width': '76%', 'margin-left': '12.5%'}),
                       html.Span(id="slider-sum", style={"vertical-align": "middle"}),
                       html.Div(
                           [
                            html.H6(children=['Number of Scars in Match:'], style={'textAlign': 'center', 'font-weight': 'normal'}),
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
                           html.A(dbc.Button('Refresh',color="primary", style = {'width': '76%', 'margin-left': '12.5%', 'margin-bottom': '5px'}),href='/')                                       
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
                                    height=canvas_height,
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


############################################################
#  Callbacks
############################################################                                                                                
                                        
@app.callback([Output('sketch_output', 'children'),
               Output('sketch_output_info1', 'children')],
                [Input('canvas', 'json_data'),
                 Input('canvas', 'n_clicks')],
                )
def update_data(string,image):    
    global count, find_matches_func, num_returned, path_to_mask
    blank = base64.b64encode(open(path_to_blank, 'rb').read())    
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


@app.callback(Output('slider-sum', 'children'),
            [Input('my-slider', 'value'),
             Input('my-slider1', 'value'),
             Input('my-slider2', 'value'),
             Input('my-slider3', 'value'),
             Input('my-slider4', 'value'),
             Input('my-slider5', 'value'),
             Input('my-slider6', 'value')])
def sum_slider(value,value1,value2,value3,value4,value5,value6):
    summ = round((value+value1+value2+value3+value4+value5+value6),2) 
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
             Output('my-slider6', 'value')],
             [Input('update_the_weights', 'n_clicks')])
def updateSliders(value):
    new,new1,new2,new3,new4,new5,new6 = readjust([orientation_perc,MA_perc,ma_perc,area_perc,aspect_perc,locX_perc,locY_perc])
    return new,new1,new2,new3,new4,new5,new6

@app.callback(
    Output('hidden-div12', 'children'),
    [Input('my-range-slider', 'value'),])
def update_range_slider(value):
    find_matches_func.low_end = value[0]
    find_matches_func.high_end = value[1]    
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'}) 

@app.callback(
    Output('hidden-div13', 'children'),
    [Input('tail_filter', 'value')])
def tail_mute_filter(value):
    find_matches_func.tail = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'}) 

if __name__ == '__main__':
    app.run_server()
    
    
