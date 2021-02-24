# Manatee Program

Code repository for the manatee matching program.

![alt text](https://github.com/natewagner10/Manatee-App-Testing/blob/main/assets/dash_example2.png)

## Setup

1. Open the terminal or command line and clone the repository: <br />
```sh
git clone https://github.com/natewagner10/Manatee-App-Testing.git
```
2. Then install the requirements: <br />
```sh
pip install -r requirements.txt
```
3. Finally, to launch the program: <br />
```sh
python app.py
```
## Usage

The program takes a scar and bounding box as input, and then it cross references the image with the existing database and returns similar matches (similar scars in similar locations).  <br />
<br />
The application also supports multiple bounding box regions, as well as empty bounding boxes, in which the program will not return images that contain scars in that region. <br />
<br />
Changing the weights can have a big effect of the quality of matches depending on the input scars, so we include the option to adjust the weights (i.e. put more emphasis on orientation). <br />
<br />
The program also supports filters such as the number of scars that should be present in the bounding box search regions. <br />
<br />
When running the program the first time, it runs the manatee database through a CNN classifier and creates a csv file (stored in the assets folder) specifiying whether a manatee has a tail mutilation or not. Then the user can specify a filter to include or exclude matches with tail mutilations. The program will recognize when new images are added to the database and will run these through the classifier and storing the tail mutilation information in the csv file accordingly. <br />
<br />

Weight options: <br />
* Scar Orientation <br />
* Scar Length <br />
* Scar Width <br />
* Scar Area <br />
* Scar Aspect Ratio <br />
* Scar Location (x-direction) <br />
* Scar Location (y-direction) <br />

## How it works

The user draws a scar, then puts a bounding box around the scar.  We then extract the scar contours, and compute some metrics such as height, width, scar orientation, and some others.   We then take that bounding box, and extract all scar contours that are within the given region from the database.  Finally, we compute a similarity measure based on how similar the scar contours are. <br />
<br />

## Contributors
Rosa Gradilla <br />
Nate Wagner <br />





