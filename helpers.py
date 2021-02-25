"""
Mote Marine Laboratory Collaboration

Manatee Matching Program

Written by Nate Wagner, Rosa Gradilla 
"""

import numpy as np
import os
import csv


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


def getFilePaths(root_dir, tail_mute_info):
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
    if not os.path.exists(tail_mute_info):
        with open(tail_mute_info, 'w') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Sketch_ID', 'Has_Tail_Mute'])
    with open(tail_mute_info) as f:
        reader = csv.reader(f)
        current_sketch_names = [i[0] for i in reader]                
    for root, _, fnames in sorted(os.walk(root_dir, followlinks=True)):
        for fname in sorted(fnames):
            if fname not in current_sketch_names:
                with open(tail_mute_info, 'a', newline='') as csvfile:
                    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    filewriter.writerow([str(fname), 'NA'])
            path = os.path.join(root, fname)
            file_paths.append(path)
    with open(tail_mute_info) as f:
        reader = csv.reader(f)
        tail_mute_data = [i for i in reader]    
    itemDict = {item[0]: item[1] for item in tail_mute_data}
    return file_paths, itemDict
