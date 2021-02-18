# Manatee Program

Code repository for New College of Florida and Mote Marine Research Laboratory collaboration.

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

Enter the path to the images and select **Go**. The import status will change if the images are successfully imported. <br />
<br />
The program takes a scar and bounding box as input, and then it cross references the image with the existing database and returns similar matches.  <br />
<br />
The application also supports multiple bounding box regions, as well as empty bounding boxes, in which the program will not return images that contain scars in that region. <br />
<br />
Certain scars return better matches with different weights, so we include the option to adjust the weights. <br />
<br />
The program also supports filters such as the number of scars that should be present in the bounding box search regions. <br />
<br />
When running the program the first time, it runs the manatee database through a CNN classifier and creates a csv file (stored in the assets folder) specifiying whether a manatee has a tail mutilation or not. Then the user can specify a filter to include matches with tail mutilations or not. The program will recognize new sketches in the database and run these through the classifier and storing the tail mutilation information in the csv accordingly. <br />
<br />

Weight options: <br />
* Scar Orientation <br />
* Scar Length <br />
* Scar Width <br />
* Scar Area <br />
* Scar Aspect Ratio <br />
* Scar Location (x-direction) <br />
* Scar Location (y-direction) <br />

## Contributors
Rosa Gradilla <br />
Nate Wagner <br />





