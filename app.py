"""
Mote Marine Laboratory Collaboration

Manatee Matching Program

Written by Nate Wagner, Rosa Gradilla 
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_daq as daq
import base64
import json
import numpy as np
from dash_canvas import DashCanvas
from dash_canvas.utils import parse_jsonstring
from PIL import Image
import cv2
from tensorflow import keras
import tensorflow as tf
from Documents.Mote_Marine_Manatees.app8.compare_rois import Compare_ROIS
from Documents.Mote_Marine_Manatees.app8.model_configs import manateeAIConfig
from Documents.Mote_Marine_Manatees.app8.helpers import (
    getFilePaths,
    buildDataTables,
    extractModelOutput,
    buildFigure,
    buildFigure2,
    buildAnnotate,
    extractNewAnnots,
    buildAnnotateFromApp,
    extendAnnotations,
    extractScarDetailsFromJSON
)
from Documents.Mote_Marine_Manatees.app8.mrcnn import model as modellib
import os
import pandas as pd
import dash_table

############################################################
#  Needed paths
############################################################
path_to_images = '/Users/natewagner/Documents/Mote_Manatee_Project/data/data_folders/'
path_to_blank = '/Users/natewagner/Documents/Mote_Manatee_Project/data/BLANK_SKETCH_updated.jpg'
path_to_jsons = '/Users/natewagner/Documents/Mote_Manatee_Project/AI/predictions/predicted_annotations2/'
path_to_ManateeAI = '/Users/natewagner/Documents/Mote_Marine_Manatees/app8/manateeAI_Scar_Finder.h5'

############################################################
#  Initiate model and Compare_ROIS class
############################################################

config = tf.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

session = tf.Session(config=config)

keras.backend.set_session(session)

model_config = manateeAIConfig()
model = modellib.MaskRCNN(mode="inference", config=model_config, model_dir=path_to_ManateeAI)
model.load_weights(path_to_ManateeAI, by_name=True)   
model.keras_model._make_predict_function()    

find_matches_func = Compare_ROIS(None, path_to_jsons, None, None, None, 1, 3, "unknown", model, None, None, True, 1, 3, False, None, [], [], None)
find_matches_func.paths = getFilePaths(path_to_images)
find_matches_func.preLoadData()

############################################################
#  Page 2 Graph
############################################################

names = []
for im in find_matches_func.paths:
    my_path = os.path.normpath(im)
    name = my_path.split(os.sep)[-1]
    names.append(name)

names_df = []
images_url = []
number_scars = []
has_mute = []
for entry in names:
    try:
        number_scars.append(len(find_matches_func.processed_images[entry]['objects']))            
        needed_im = '![myImage-' + str(entry) + '](assets/' + str(find_matches_func.getImagePath(entry)) + ')'        
        names_df.append(entry[0:-4])                    
        images_url.append(needed_im)        
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
                   columns=[{'id': 'Name', 'name': 'Name', "selectable": False},
                            {'id': '# of Scars', 'name': '# of Scars', "selectable": False},
                            {'id': 'Has Mute', 'name': 'Has Mute', "selectable": False},
                            {'id': 'Image', 'name': 'Image', 'presentation': 'markdown', "selectable": False}],
                   style_header={'height':'auto'},
                   style_table={'minHeight': '625px', 'height': '625px', 'maxHeight': '625px', 'overflowY': 'auto', 'textAlign': 'center'},
                   style_cell={'font_size': '20px', 'height': 'auto', 'textAlign': 'center'},                      
                   page_current=0,
                   page_size=100,
                   page_action='custom',            
                   sort_action='custom',
                   sort_mode='multi',
                   sort_by=[],
                   style_as_list_view=True,
                   fixed_rows={'headers': True}, 
                   filter_action='custom',
                   filter_query='',                   
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


############################################################
#  User interface
############################################################

app = dash.Dash(__name__, meta_tags=[{"content": "width=device-width"}], external_stylesheets=[dbc.themes.LITERA], assets_folder=path_to_images)
server = app.server

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

index_page = html.Div([
    dcc.Link('Go to Page 1', href='/page-1'),
    html.Br(),
    dcc.Link('Go to Page 2', href='/page-2'),
])


############################################################
#  Layout 1
############################################################

# dash canvas info
filename = Image.open(path_to_blank)
canvas_width = 256   # this has to be set to 256 because we use the canvas as input to the model
canvas_height = 559

layout1 = html.Div(
    [
    html.Br(),
    dbc.Row(
        [
            dbc.Col(
                html.H2(className="title", children="ManateeAI")
            ),
            dbc.Col(                
                dbc.Button(id="how-to-use", children=dcc.Link(
                                                "Explore Data",
                                                href="/page-2",
                                                className="tab first"),
                color="primary", outline=True, style = {'margin-left': '80%'}),
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
                                            html.H6(id = 'num-scar',children=['# Scars in Sketch:'], style={'textAlign': 'center', 'font-weight': 'normal', 'margin-top': '15px'}),
                                            html.Div(
                                                [
                                                    daq.NumericInput(
                                                        id='image-low',
                                                        min=1,
                                                        max=150,
                                                        value=1
                                                    ),
                                                    daq.NumericInput(
                                                        id='image-high',
                                                        min=1,
                                                        max=150,                                                        
                                                        value=15
                                                    ),                                                                                                        
                                                ], style = {'display': 'flex',
                                                            'justify-content': 'space-between',
                                                            'width': '50%',
                                                            'margin': '10px auto'}
                                                ),
                                            html.H6(id = 'num-scar-roi', children=['# Scars in ROI:'], style={'textAlign': 'center', 'font-weight': 'normal', 'margin-top': '15px'}),
                                            html.Div(
                                                [
                                                    daq.NumericInput(
                                                        id='roi-low',
                                                        min=1,
                                                        max=150,                                                        
                                                        value=1
                                                    ),
                                                    daq.NumericInput(
                                                        id='roi-high',
                                                        min=1,
                                                        max=150,                                                        
                                                        value=3
                                                    ),                                                                                                        
                                                ], style = {'display': 'flex',
                                                            'justify-content': 'space-between',
                                                            'width': '50%',
                                                            'margin': '10px auto'}
                                                ),
                                            html.Hr(),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.H6(id='orien',children=['Orientation:'], style={'font-weight': 'normal'}),
                                                            html.H6(id = 'in-roi', children=['In-ROI:'], style={'font-weight': 'normal'}),
                                                            html.H6(id = 'series', children=['Series ROI:'], style={'font-weight': 'normal'}),                                                            
                                                        ], style={'display': 'inline-block', 'vertical-align': 'middle', 'margin-top': '5px'}),
                                                    html.Div(
                                                        [
                                                            daq.BooleanSwitch(
                                                                id='orientation-switch',
                                                                on=True,
                                                                ),
                                                            daq.BooleanSwitch(
                                                                id='in-box-switch',
                                                                on=False,   
                                                                style = {'margin-top': '5px'},
                                                                ),
                                                            daq.BooleanSwitch(
                                                                id='series-switch',
                                                                on=False,
                                                                style = {'margin-top': '5px'}
                                                                ),                                                            
                                                        ], style={'display': 'inline-block', 'vertical-align': 'middle'}),
                                                ], style = {'display': 'flex',
                                                            'justify-content': 'space-between',
                                                            'width': '80%',
                                                            'margin': '10px auto'}),                                                       
                                    html.Hr(),                        
                                    html.Div(
                                        [                                            
                                            dcc.RadioItems(
                                                id="tail_filter",
                                                options=[
                                                    {'label': 'Mute', 'value': 'mute'},
                                                    {'label': 'No Mute', 'value': 'no_mute'},
                                                    {'label': 'Unknown', 'value': 'unknown'},
                                                ],
                                                value='unknown', 
                                                inputStyle={"margin-right": "10px", "margin-left": "10px"},
                                                )
                                        ], style = {'display': 'flex',
                                                    'justify-content': 'space-between',
                                                    'width': '45%',
                                                    'margin': '5px auto'}
                                    ),   
                                    html.Hr(),
                                    html.Div(id='hidden-div12', style={'display':'none'}),
                                    html.Div(id='hidden-div16', style={'display':'none'}),
                                    html.Div(id='hidden-div14', style={'display':'none'}),
                                    html.Div(id='hidden-div15', style={'display':'none'}),                                    
                                    html.Div(id='hidden-div10', style={'display':'none'}),
                                    html.Div(id='hidden-div13', style={'display':'none'}),
                                    html.Div(id='hidden-div44', style={'display':'none'}),
                                    html.Div(id='hidden-div144', style={'display':'none'}),                              
                                    html.H6(children=['Brush Width'], style={'textAlign': 'center', 'font-weight': 'normal'}),
                                    dcc.Slider(
                                        id='bg-width-slider',
                                        min=1,
                                        max=40,
                                        step=1,
                                        value=1,
                                    ),                                                                       
                                    html.Div([html.A(dbc.Button('Refresh',color="primary", outline=True),href='/')], style = {'display': 'flex', 'justify-content': 'center', 'margin-bottom': '15px'}),                                 
                                ],
                                style = {"margin-top": "5px"}
                            ),                                                  
                        ],
                        style = {'width': '100%',                                 
                                 'display': 'flex',
                                 'justify-content': 'center',
                                 'box-shadow': '0 0 10px lightgrey',
                                 'backgroundColor': '#FFFFFF'}
                    ),
                    html.H5("Extracted Scars: ", style={'margin-left': '10px',
                                                        'margin-top': '20px'}),                    
                    html.Div(
                        [
                            html.Div(id = 'image-results')
                        ],
                        style = {'width': '100%',                                 
                                 'display': 'flex',
                                 'margin-top': '20px',
                                 'justify-content': 'center',
                                 'box-shadow': '0 0 10px lightgrey',
                                 'backgroundColor': '#FFFFFF'})                    
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
                                        hide_buttons=['line', 'select', 'zoom', 'pan'],
                                        goButtonTitle="Search"                                    
                                    ),                                                            
                                ],
                                style = {'top': '50%',
                                         'display': 'flex',
                                         'justify-content': 'center',
                                         'margin-top': '7px',                                     
                                         'align-items': 'center'}
                            ),                                                                        
                        ],
                        style = {'width': '100%',
                                 'display': 'inline-block',
                                 'box-shadow': '0 0 10px lightgrey',
                                 'backgroundColor': '#FFFFFF'}
                    ),
                    html.H5("Scar Statistics: ", style={'margin-left': '10px',
                                                        'margin-top': '10px'}),
                    html.Div(id="scar_statistic_df")

                ],
                md = 4
            ),               
            dbc.Col(
                [
                    html.Div(
                        [
                            html.Div(id="sketch_output", style={"vertical-align": "middle"}),                              
                        ],
                        style={'width': '100%',                       
                               'align-items': 'center',
                               'display': 'inline-block',
                               'box-shadow': '0 0 10px lightgrey',
                               'height': '675px',
                               'backgroundColor': '#FFFFFF'}
                    ),
                    html.Div(id = 'sketch_output_info1', style={'margin-left': '10px',
                                                                'margin-top': '10px'})
                ],
                md = 6
            ),
        ]    
    ),
    dbc.Tooltip("Useful for tilted scars or scars with some sense of orientation.",target="orien"),
    dbc.Tooltip("Sets the minimum and maximum number of scars that can be in a sketch.",target="num-scar"),
    dbc.Tooltip("Sets the minimum and maximum number of scars that can be in a ROI.",target="num-scar-roi"),
    dbc.Tooltip("Turn on if the entire scar should be in the ROI. Else the scar just has to be touching the ROI.",target="in-roi"),
    dbc.Tooltip("This enables a series ROI. Use this when searching for series.",target="series"),    
    ],
    style={'padding': '0px 40px 0px 40px',
           'min-height': '100vh',
           'backgroundColor': '#EBFFFD'}
)


                                                    
############################################################
#  Layout 2
############################################################                                                    
                                                    

layout2 = html.Div([
    html.Br(),
    dbc.Row(
        [
            dbc.Col(
                html.H2(className="title", children="ManateeAI - Data Explorer")
            ),
            dbc.Col(                
                dbc.Button(id="how-to-use", children=dcc.Link(
                                                "Main",
                                                href="/page-1",
                                                className="tab first"),
                color="primary", outline=True, style = {'margin-left': '80%'}),
            ),
        ]
    ),
    html.Br(),
    html.Div(id='hidden-div1', style={'display':'none'}),
    html.Div(id='hidden-div2', style={'display':'none'}),
    html.Div(id='hidden-div3', style={'display':'none'}),    
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(      
                        [
                            browse_matches_table
                        ],  style={'align-items': 'center',
                                   'display': 'inline-block',
                                   'box-shadow': '0 0 10px lightgrey',
                                   'height': '675px',
                                   'backgroundColor': '#FFFFFF',
                                   'width': '100%'}),
                    html.Div(
                        [
                            dcc.Upload(
                                id='upload-data',
                                children=html.Div([
                                    html.A('Upload an Annotation')
                                ]), style = {
                                            'border-style': 'dotted',
                                            'border-radius': '20px',
                                            'display': 'flex',
                                            'justify-content': 'center',
                                            'align-items': 'center',
                                            'margin-top': '20px'}),
                        ]
                    )
                ], md = 6
            ),
            dbc.Col(
                [
                    html.Div(
                        [
                            html.H2(className="title", children="Model Output:"),
                            html.Div(id='datatable-interactivity-container'), 
                            dbc.Button(id='re-annotate', children = ["Re-Annotate Image"], color="primary", outline=True, style = {'margin-top': '20px', 'width': '256px'})
                        ]
                    ),  
                ], md = 2,
            ),
            dbc.Col(
                [
                    html.Div(
                        [
                            html.Div(id="annotate")
                            
                        ]
                    )
                ], md = 2,
            ),
            dbc.Col(
                [
                    html.Div(
                        [
                            html.Div(id="annotate2")
                            
                        ]
                    )
                ], md = 2,
            )            
        ])        
    ],
    style={'padding': '0px 40px 0px 40px',
           'min-height': '100vh',
           'backgroundColor': '#EBFFFD'}
    )
            
                                   
jumbotron = html.Div(
            [

                html.P(
                    "Click on a sketch name \n to see model output.",
                    className="lead",  
                    style = {'width': '50%'}                      
                ),                   
            ], style = {'height': '559px',
                        'width': '256px',
                        'display': 'flex',
                        'justify-content': 'center',
                        'align-items': 'center',
                        'margin-top': '25px'}
        )
      

                                                    
                                                    
############################################################
#  Callbacks - Page 1
############################################################                                                                                
            
                                     
@app.callback([Output('sketch_output', 'children'),
               Output('sketch_output_info1', 'children'),
               Output('scar_statistic_df', 'children'),
               Output('image-results', 'children')],
                [Input('canvas', 'json_data'),
                 Input('canvas', 'n_clicks')],
                )
def update_data(string,image):  
    global find_matches_func
    blank = base64.b64encode(open(path_to_blank, 'rb').read())    
    is_rect = False
    if string:
        data = json.loads(string)      
        bb_info = data['objects'][1:]         
        bounding_box_list = []      
        bounding_box_list_types = []
        for i in bb_info:
            if i['type'] == 'rect':  
                is_rect = True
                top = i['top']
                left = i['left']
                wd = i['width']
                ht = i['height']
                bounding_box_list.append((left, top, wd, ht))
                if i['stroke'] == '#B00000':
                    bounding_box_list_types.append('series')
                else:
                    bounding_box_list_types.append('normal')
            else:
                continue        
        if is_rect == False:
            bounding_box_list.append((0, 559, 0, 256))        
        mask = parse_jsonstring(string, shape=(559, 256))
        mask = (~mask.astype(bool)).astype(int)
        mask[mask == 1] = 255
        mask = mask.astype(np.uint8)        
        
        new = Image.new('RGB', (512, 512), 0)            
        mask = cv2.blur(mask, (2,2))
        mask=Image.fromarray(mask).resize((234, 512))
        new.paste(mask, (128,0))
        new_arr = np.asarray(new)
        with session.as_default():
            with session.graph.as_default():        
                results = model.detect([new_arr], verbose=0)
        r = results[0]
        input_annots = extractModelOutput(r)             
        if 'find_matches_func' in globals():
            find_matches_func.input_annots = input_annots
            find_matches_func.input_sketch = mask
            find_matches_func.roi = bounding_box_list
            find_matches_func.roi_types = bounding_box_list_types            
        matches = find_matches_func.compare_rois()        
        is_rect = False 

        browse_matches_table, scar_stats_table, names_df = buildDataTables(matches, find_matches_func)
        
        fig = buildFigure(r, new)
        
        return browse_matches_table, html.H5("Number of Matches: " + str(len(names_df))), scar_stats_table, dcc.Graph(figure=fig)
    
    return html.Div([
    html.Img(src='data:image/png;base64,{}'.format(blank.decode()))
    ], style = {'align-items': 'center','margin-left':'35%', 'margin-top': '5%'}), html.H5("Number of Matches: "), html.Div(id='hidden-div200', style={'display':'none'}), html.Div(id='hidden-div200', style={'display':'none'}) 

@app.callback(Output('canvas', 'lineWidth'),
            [Input('bg-width-slider', 'value')])
def update_canvas_linewidth(value):
    return value
@app.callback(
    Output('hidden-div12', 'children'), 
    [Input('image-low', 'value'),])
def update_num_scars_low(value):
    find_matches_func.low_end = value  
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'}) 
@app.callback(
    Output('hidden-div16', 'children'), 
    [Input('image-high', 'value'),])
def update_num_scars_high(value):
    find_matches_func.high_end = value  
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'}) 
@app.callback(
    Output('hidden-div14', 'children'), 
    [Input('roi-low', 'value'),])
def update_num_scars_low_roi(value):
    find_matches_func.roi_low = value  
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'}) 
@app.callback(
    Output('hidden-div15', 'children'), 
    [Input('roi-high', 'value'),])
def update_num_scars_high_roi(value):
    find_matches_func.roi_high = value  
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
@app.callback(
    dash.dependencies.Output('hidden-div10', 'children'),
    [dash.dependencies.Input('orientation-switch', 'on')])
def orientation_switch(on):    
    if on == True:
        find_matches_func.orientation_bool = True
    if on == False:
        find_matches_func.orientation_bool = False
    return 'The switch is {}.'.format(on)
@app.callback(
    dash.dependencies.Output('hidden-div44', 'children'),
    [dash.dependencies.Input('in-box-switch', 'on')])
def in_roi_switch(on): 
    if on == True:
        find_matches_func.in_roi = True
    if on == False:
        find_matches_func.in_roi = False
    return 'The switch is {}.'.format(on)
@app.callback(    
    dash.dependencies.Output('canvas', 'lineColor'),
    [dash.dependencies.Input('series-switch', 'on')])    
def series_switch(on): 
    color = '#000000'
    if on == True:
        color = '#B00000'
    if on == False:
        color = '#000000'
    return color


############################################################
#  Callbacks - Page 2
############################################################

@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/page-1':
        return layout1
    elif pathname == '/page-2':
        return layout2
    else:
        return layout1  
@app.callback(
    Output('datatable-interactivity-container', "children"),
    [Input('datatable-interactivity', 'data'),
     Input('datatable-interactivity', 'selected_row_ids'),
     Input('datatable-interactivity', 'active_cell')])
def update_graphs(rows, selected_row_ids, active_cell):
    global data_table_df
    if active_cell is not None:
        dff = pd.DataFrame(rows)
        selected_image = dff.loc[active_cell['row']]['Name']    
        path = find_matches_func.processed_images[selected_image + '.jpg']['path_to_image']
        im = Image.open(path)
        im = im.convert('RGB')
        rois = []
        for pts in find_matches_func.processed_images[selected_image + '.jpg']['objects']:
            rois.append(pts['points']['exterior'])
        try:
            fig = buildFigure2(im, rois, None)
        
            return dcc.Graph(figure=fig, config={'displayModeBar': False})
        except:
            fig = None
    else:
        return jumbotron
operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]
def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]
                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part
                return name, operator_type[0].strip(), value
    return [None] * 3
@app.callback(
    Output('datatable-interactivity', 'data'),
    [Input('datatable-interactivity', "page_current"),
    Input('datatable-interactivity', "page_size"),
    Input('datatable-interactivity', 'sort_by'),
    Input('datatable-interactivity', 'filter_query')])
def update_table(page_current, page_size, sort_by, filter2):
    filtering_expressions = filter2.split(' && ')
    dff = data_table_df
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    if len(sort_by):
        dff = dff.sort_values(
            [col['column_id'] for col in sort_by],
            ascending=[
                col['direction'] == 'asc'
                for col in sort_by
            ],
            inplace=False
        )

    page = page_current
    size = page_size
    return dff.iloc[page * size: (page + 1) * size].to_dict('records')
@app.callback(Output('annotate', 'children'),
              [Input('re-annotate', 'n_clicks'),
               Input('datatable-interactivity', 'active_cell'),
               Input('datatable-interactivity', 'data')])
def buildAnnotGraph(nclicks, active_cell, rows): 
    if nclicks:
        if active_cell is not None:
            dff = pd.DataFrame(rows)
            selected_image = dff.loc[active_cell['row']]['Name']    
            path = find_matches_func.processed_images[selected_image + '.jpg']['path_to_image']
            im = Image.open(path)
            im = im.convert('RGB')
            im = im.resize((256,559))
            fig, config = buildAnnotate(np.asarray(im))            
            nclicks = None
            return html.Div([html.H2(className="title", children="Annotations:"),
                             dcc.Graph(id="fig-annots", figure=fig, config=config),
                             html.Pre(id="annotations-data-pre"),                             
                             dbc.Button(id='confirm-annotate', children = ["Confirm Annotations"], color="primary", outline=True, style = {'width': '256px'}),
                             html.Div([
                             html.H6(id='mute-s',children=['Mute:'], style={'font-weight': 'normal'}),
                             daq.BooleanSwitch(id='mute-switch', on=False)], style = {'display': 'flex',
                                                                                      'justify-content': 'space-between',
                                                                                      'width': '80%',
                                                                                      'margin': '10px auto'})])            
@app.callback(Output("annotations-data-pre", "children"),
             [Input("fig-annots", "relayoutData")])
def annotation(relayout_data):    
    if relayout_data is not None:
        find_matches_func.new_annots.append(relayout_data)
        return dash.no_update        
@app.callback(Output("annotate2", "children"),
             [Input("confirm-annotate", "n_clicks"),
              Input('datatable-interactivity', 'active_cell'),
              Input('datatable-interactivity', 'data'),
              Input("fig-annots", "relayoutData"),
              Input('mute-switch', 'on')])
def confirm(n_clicks, active_cell, rows, relayout_data, on):  
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]    
    if 'confirm-annotate.n_clicks' in changed_id:    
        if active_cell is not None:
            new_coords = extractNewAnnots(find_matches_func.new_annots)
            find_matches_func.new_annots_coords = [new_coords, on]
            dff = pd.DataFrame(rows)
            selected_image = dff.loc[active_cell['row']]['Name']    
            path = find_matches_func.processed_images[selected_image + '.jpg']['path_to_image']
            im = Image.open(path)
            im = im.convert('RGB')
            im = im.resize((256,559)) 
            if on == True:
                color = (200,50,50)
            if on == False:
                color = (0,220,200)             
            if type(new_coords) != list:
                fig = buildFigure2(im, [new_coords], color)   
            else:
                fig = buildFigure2(im, new_coords, color)   
            return html.Div([html.H2(className="title", children="Results:"),
                             dcc.Graph(figure=fig, config={'displayModeBar': False}),
                             dbc.Button(id='process-annot', children = ["Add Annotations"], color="primary", outline=True, style = {'width': '256px', 'margin-top': '15px'}),
                             dbc.Button(id="open", children = ["Save Annotation"], color="primary", style = {'width': '256px', 'margin-top': '15px'}),
                             dbc.Modal([                                 
                                 dbc.ModalBody('This will permanently delete the existing file. Are you sure you want to continue?'),          
                                 dbc.ModalFooter(html.Div([dbc.Button("Save", id="save", className="ml-auto", color="danger"),
                                                           dbc.Button("Cancel", id="close", className="ml-auto")], style = {'display': 'flex','justify-content': 'space-between','width': '40%','margin': '10px auto'})),
                                 ], id='modal',
                             )])
    if n_clicks:
        return dash.no_update            
@app.callback(Output('hidden-div1', "children"),
             [Input('process-annot', "n_clicks")])
def buildAnnotation(n_clicks):  
    if n_clicks:
        if find_matches_func.new_json is None:
            if type(find_matches_func.new_annots_coords[0]) != list:
                json_dta = buildAnnotateFromApp([[find_matches_func.new_annots_coords[0]], find_matches_func.new_annots_coords[1]])    
            else:
                json_dta = buildAnnotateFromApp(find_matches_func.new_annots_coords)    
            find_matches_func.new_json = json_dta
            return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})
        if find_matches_func.new_json is not None:
            new = len(find_matches_func.new_annots_coords[0])
            have = len(find_matches_func.new_json['objects'])    
            get = new - have    
            new_coords = find_matches_func.new_annots_coords[0][-get:]            
            extended_json = extendAnnotations(new_coords, find_matches_func.new_annots_coords[1], find_matches_func.new_json)
            find_matches_func.new_json = extended_json            
    return html.Div([
                html.H5("test"),                
                ], style = {'align-items': 'center'})     
@app.callback(Output("modal", "is_open"),
             [Input("open", "n_clicks"), 
              Input("close", "n_clicks"),
              Input("save", "n_clicks"),
              Input('datatable-interactivity', 'active_cell'),
              Input('datatable-interactivity', 'data')],
             [State("modal", "is_open")])
def saveAnnotation(openn, close, save, active_cell, rows, is_open):
    if save:
        dff = pd.DataFrame(rows)
        selected_image = dff.loc[active_cell['row']]['Name']  
        data = find_matches_func.new_json
        try:
            os.remove(path_to_jsons + selected_image + '.jpg.json') 
        except:
            "no file there"
        with open(path_to_jsons + selected_image + '.jpg.json', 'w') as json_file:
            json.dump(data, json_file)  
    if openn or close:
        return not is_open    
    return is_open      
@app.callback(Output('hidden-div3', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename'),
              State('upload-data', 'last_modified')])
def uploadData(list_of_contents, list_of_names, list_of_dates):    
    if list_of_contents is not None:
        content_type, content_string = list_of_contents.split(',')
        decoded = base64.b64decode(content_string)    
        res_dict = json.loads(decoded)
        new_annots = extractScarDetailsFromJSON(res_dict)
        with open(path_to_jsons + 'updated_' + list_of_names + '.json', 'w') as json_file:
            json.dump(new_annots, json_file)        

if __name__ == '__main__':
    app.run_server()
        
