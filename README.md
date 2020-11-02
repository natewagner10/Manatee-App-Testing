# Manatee Program

Code repository for New College of Florida and Mote Marine Research Laboratory collaboration.

![alt text](https://github.com/natewagner10/Manatee-App-Testing/blob/main/assets/dash_example.png)

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
## How to Use

Enter the path to the images and select **Go**. The import status will change if the images are successfully imported. <br />
<br />
The program takes a scar and bounding box as input, and then it cross references the image with the existing database and returns similar matches.  <br />
<br />
The application also supports multiple bounding box regions, as well as empty bounding boxes, in which the program will not return images that contain scars in that region. <br />
<br />
Certain scars return better matches with different weights, so we include the option to adjust the weights. <br />
<br />
Weight options: <br />
* Scar Orientation <br />
* Scar Length <br />
* Scar Width <br />
* Scar Area <br />
* Scar Aspect Ratio <br />
* Scar Location (x-direction) <br />
* Scar Location (y-direction) <br />
* Sum of Scar Pixel Values <br />

## Contributors
Rosa Gradilla 
Nate Wagner





