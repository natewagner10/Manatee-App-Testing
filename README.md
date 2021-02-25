# Manatee Program

Code repository for the manatee matching program.

![alt text](https://github.com/natewagner10/Manatee-App-Testing/blob/main/assets/app_example.png)

## 1. Setup

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
## 2. Usage

### 2.1 Assets folder

The assets folder contains items needed to run the program. Specifically, it contains the blank-manatee-sketch, manatee-outline-removal-mask, tail-mutilation-model-weights and the CSV-file containing the tail mutilation information.  <br />
<br />

### 2.2 First time launching application

The first time running the application, the program takes all images in the dataset and runs them through a convolutional neural network classifier.  The classifier inputs 196x196 cropped images of the tail and outputs a label indicating whether the manatee has a tail mutilation.  Due to this process being time consuming, we only run the images through the classifier on the initial program launch, and we store the labels in a file tail_mute_info.csv in the assets folder.  The file contains two fields, Sketch_ID and Has_Tail_Mute, where Sketch_ID contains the name of the image and Has_Tail_Mute takes value 0 if a tail mute is present and 1 if not.  Subsequent program launches, the application reads in the CSV file and cross references the data with the image database and checks if new images were added.  If new images were added to the dataset, the program runs the new image through the classifier and stores the tail mutilation information in the CSV accordingly.  

Important: The first time running the application it’s critical not to interrupt the process because the program is filling out all the tail mutilation information and problems can occur.  <br />
<br />

### 2.3	Sketching

To start finding similar sketches, select the pen tool from the toolbox at the bottom of the Sketch box and draw the desired scar(s). <br />
<br />

<p align="center">
<img src="https://github.com/natewagner10/Manatee-App-Testing/blob/main/assets/figure1.png" width="70%" height="70%">
</p>

Afterwards, select the bounding box tool and place a box around either both scars (a.) or each scar individually (b.). <br />
<br />

![alt text](https://github.com/natewagner10/Manatee-App-Testing/blob/main/assets/figure2.png)

Given we want to find similar scars in similar locations, the bounding box takes care of scar location. <br />
<br />




## How it works

The user draws a scar, then puts a bounding box around the scar.  We then extract the scar contours, and compute some metrics such as height, width, scar orientation, and some others.   We then take that bounding box, and extract all scar contours that are within the given region from the database.  Finally, we compute a similarity measure based on how similar the scar contours are. <br />
<br />

## Contributors
Rosa Gradilla <br />
Nate Wagner <br />





