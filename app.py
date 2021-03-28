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
import dash_daq as daq
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
path_to_mask = '/Users/natewagner/Documents/Mote_Manatee_Project/data/MANATEE_MASK.png'
path_to_blank = '/Users/natewagner/Documents/Mote_Manatee_Project/data/BLANK_SKETCH_updated.jpg'
path_to_cnn = '/Users/natewagner/Documents/Mote_Manatee_Project/data/assets/tail_mute_cnn_15_epochs_99.pt'
path_to_tail_mute_info = '/Users/natewagner/Documents/Mote_Manatee_Project/data/tail_mute_info.csv'
path_to_template = '/Users/natewagner/Documents/Mote_Manatee_Project/data/BLANK_SKETCH_updated_cropped.jpg'

############################################################
#  Initiate model and Compare_ROIS class
############################################################

model = CNN()
model.load_state_dict(torch.load(path_to_cnn, map_location=torch.device('cpu')))
    
find_matches_func = Compare_ROIS(None, path_to_template, path_to_tail_mute_info, None, None, path_to_mask, 0.20, 0.30, 0.30, 0.20, 1, 3, "unknown", model, None, "shape", None)

find_matches_func.paths, find_matches_func.all_tail_info = getFilePaths(path_to_images, path_to_tail_mute_info)
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
    html.Br(),
    dbc.Row(
        [
            dbc.Col(
                html.H2(className="title", children="Manatee Matching Application")
            ),
            dbc.Col(
                dbc.Button(id="how-to-use", children="How To Use", color="primary", outline=True, style = {'margin-left': '80%'}),
            ),
        ]
    ),
    html.Br(),
    dbc.Row(
        [    
            dbc.Col(
                [
                    html.Div(
                        [   
                            html.Div(
                                [
                                    html.Div(id='search-mode-out'),
                                        daq.ToggleSwitch(
                                            id='search-mode',
                                            value=False,
                                            size=50
                                        ),
                                ],
                                style = {'width': '100%',
                                         'margin-top': '5px',
                                         'margin-right': '100px',
#                                         'box-shadow': '0 0 10px lightgrey',
                                         'display': 'inline-block'}                                
                            ), 
                            html.Span(id="s-mode", style={"vertical-align": "middle"}),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Hr(),
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
                                        ],
                                        style = {'align-items': 'center', 'margin-left': '10%'}
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
                                 'display': 'inline-block'}
                    )
                ], 
                md = 2
            ),
            dbc.Col(
                [                
                    html.Div(
                        [
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
                                ],
                                style = {'margin-left': '25%',
                                         'margin-top': '7px'}
                            ),
                            html.A(dbc.Button('Refresh',color="primary", outline=True, style = {'width': '50%', 'margin-left': '27%', 'margin-top': '3px'}),href='/')                            
                                            
                        ],
                        style = {'width': '100%',
                                 'height': '675px',
                                 'display': 'inline-block',
                                 'box-shadow': '0 0 10px lightgrey'}
                    ),
                    html.H5("Scar Statistics: ", style={'margin-left': '10px',
                                                        'margin-top': '10px'}),
                    html.Div(id="scar_statistic_df"),
                ],
                md = 4
            ),               
            dbc.Col(
                [
                    html.Div(
                        [
                            html.Span(id="sketch_output", style={"vertical-align": "middle"}),                              
                        ],
                        style={'width': '100%',                       
                               'align-items': 'center',
                               'display': 'inline-block',
                               'box-shadow': '0 0 10px lightgrey',
                               'height': '675px'}
                    ),
                    html.Div(id = 'sketch_output_info1', style={'margin-left': '10px',
                                                                'margin-top': '10px'})
                ],
                md = 6
            ),
        ]    
    ),
    ],
    style={'padding': '0px 40px 0px 40px', 'height': '100%'}
)


############################################################
#  Callbacks
############################################################                                                                                
                                        
@app.callback([Output('sketch_output', 'children'),
               Output('sketch_output_info1', 'children'),
               Output('scar_statistic_df', 'children')],
                [Input('canvas', 'json_data'),
                 Input('canvas', 'n_clicks')],
                )
def update_data(string,image):  
    """
    

    Parameters
    ----------
    string : TYPE
        DESCRIPTION.
    image : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    global find_matches_func
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

        scar_num = []
        orientation_i = []
        length_i = []
        width_i = []
        aspect_i = []        
        if find_matches_func.search_mode == 'shape':
            if len(find_matches_func.scar_i) > 1: 
                for num, info in enumerate(find_matches_func.scar_i):
                    scar_num.append(num+1)
                    orientation_i.append(round(info[3][0][2], 2))
                    length_i.append(round(info[3][0][3][0], 2))
                    width_i.append(round(info[3][0][3][1], 2))    
                    aspect_i.append(round(info[3][0][5], 2))
            else:
                for num, info in enumerate(find_matches_func.scar_i[0][3]):
                    scar_num.append(num+1)
                    orientation_i.append(round(info[2], 2))
                    length_i.append(round(info[3][0], 2))
                    width_i.append(round(info[3][1], 2))    
                    aspect_i.append(round(info[5], 2))   
        scar_i_tups = list(zip(scar_num,orientation_i,length_i,width_i,aspect_i))
        scar_i_data_table_df = pd.DataFrame(scar_i_tups, columns=['Scar Number','Orientation','Length','Width','Aspect Ratio']) 

        data_tups = list(zip(names_df,images_url,mean_df,sum_df,min_df,orien_df,MA_df,ma_df,aspect_df))
        data_table_df = pd.DataFrame(data_tups, columns=['Name','Image','Mean','Sum','Min','O','L','W','A'])      
        #if data_table_df is not None:
        return html.Div(
                   [                           
                       dash_table.DataTable(
                           id='table',
                           columns=[{'id': 'Name', 'name': 'Name'}, {'id': 'Image', 'name': 'Image', 'presentation': 'markdown'}, {'id': 'Mean', 'name': 'Mean'}, {'id': 'Sum', 'name': 'Sum'}, {'id': 'Min', 'name': 'Min'}, {'id': 'O', 'name': 'O'}, {'id': 'L', 'name': 'L'}, {'id': 'W', 'name': 'W'}, {'id': 'A', 'name': 'A'}],
                           page_size=50,
                           #style_as_list_view=True,
                           style_header={'height':'auto'},
                           data=data_table_df.to_dict('records'),
                           style_table={'overflowX': 'auto', 'minHeight': '625px', 'height': '625px', 'maxHeight': '625px', 'overflowY': 'auto'},
                           style_cell={'textAlign': 'center', 'font_size': '20px','overflowX': 'auto', 'height': 'auto'},
                           sort_action="native",
                           fixed_rows={'headers': True},                           
                       ),  
                   ],
                   style = {'textAlign': 'center',
                            'width': '100%',
                            'align-items': 'center'}), html.H5("Number of Matches: " + str(len(names_df))), html.Div(
                   [                           
                       dash_table.DataTable(
                           id='table1',
                           columns=[{'id': 'Scar Number', 'name': 'Scar'}, {'id': 'Orientation', 'name': 'Orien'}, {'id': 'Length', 'name': 'Len'}, {'id': 'Width', 'name': 'Wid'}, {'id': 'Aspect Ratio', 'name': 'Aspect'}],                           
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
    return html.Div([
    html.Img(src='data:image/png;base64,{}'.format(blank.decode()))
    ], style = {'align-items': 'center','margin-left':'35%', 'margin-top': '5%'}), html.H5("Number of Matches: "), html.Div(id='hidden-div200', style={'display':'none'})

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
    find_matches_func.orien_perc = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})

@app.callback(Output('hidden-div2', 'children'),
            [Input('my-slider1', 'value')])
def update_MA(value):
    find_matches_func.MA_perc = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})

@app.callback(Output('hidden-div3', 'children'),
            [Input('my-slider2', 'value')])
def update_ma(value):
    find_matches_func.ma_perc = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})

@app.callback(Output('hidden-div5', 'children'),
            [Input('my-slider3', 'value')])
def update_aspect(value):
    find_matches_func.aspect_perc = value
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})

@app.callback([Output('my-slider', 'value'),
               Output('my-slider1', 'value'),
               Output('my-slider2', 'value'),
               Output('my-slider3', 'value')],
              [Input('update_the_weights', 'n_clicks')])
def updateSliders(value):
    if value is not None and value > 0:
        new,new1,new2,new3 = readjust([find_matches_func.orien_perc,find_matches_func.MA_perc,find_matches_func.ma_perc,find_matches_func.aspect_perc])
        return new,new1,new2,new3
    else:
        new,new1,new2,new3 = [find_matches_func.orien_perc,find_matches_func.MA_perc,find_matches_func.ma_perc,find_matches_func.aspect_perc]
        return new,new1,new2,new3 

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

@app.callback([Output('search-mode-out', 'children'),
               Output('s-mode', 'children')],
              [Input('search-mode', 'value')])
def update_output(value):
    if value is False:
        mode = 'Contour Mode'
        find_matches_func.search_mode = 'shape'
        return 'Search Mode: {}'.format(mode), html.Div(
                                [
                                    html.Hr(),
                                    html.H6(id="orientation", children=['Orientation'], style={'textAlign': 'center', 'font-weight': 'normal'}),                    
                                    dcc.Slider(
                                        id='my-slider',
                                        min=0,
                                        max=2,
                                        step=0.10,
                                        value=0.20,
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
                                        value=0.30,
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
                                        value=0.30,
                                        marks={
                                            0: {'label': '0'},
                                            1: {'label': '1'},                                                                
                                            2: {'label': '2'}
                                        }
                                    ),  
                                    dbc.Tooltip("Width of the scar (with respect to the ellipse fitted to the scar).",target="width"),                                                           
                                    html.Div(id='hidden-div4', style={'display':'none'}),                                                    
                                    html.H6(id="aspect", children=['Aspect Ratio'], style={'textAlign': 'center', 'font-weight': 'normal'}),                                                                        
                                    dcc.Slider(
                                        id='my-slider3',
                                        min=0,
                                        max=2,
                                        step=0.10,
                                        value=0.20,
                                        marks={
                                            0: {'label': '0'},
                                            1: {'label': '1'},                                                                
                                            2: {'label': '2'}
                                        }
                                    ),   
                                    dbc.Tooltip("Ratio of width to height of rectangle fitted to scar. Increasing this will penalize scars that don’t have similar width-to-height ratios. ",target="aspect"),                       
                                    html.Div(id='hidden-div5', style={'display':'none'}),       
                                    html.Div(id='hidden-div8', style={'display':'none'}),                                                             
                                    html.Div(id='hidden-div13', style={'display':'none'}), 
                                    dbc.Button(id="update_the_weights", children="Update Weights", color="primary", outline=True, style = {'width': '76%', 'margin-left': '12.5%', 'margin-bottom': '5px'}),
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
                                ],
                                style = {'width': '100%',
                                         'margin-top': '5px',
                                         'margin-right': '100px',
                                         'display': 'inline-block'}                                 
                            ),
    else:
        mode = 'Pixel Mode'
        find_matches_func.search_mode = 'dist'
        return 'Search Mode: {}'.format(mode), html.Div(id='hidden-div20', style={'display':'none'})

if __name__ == '__main__':
    app.run_server()
    
        
