#!/usr/local/bin/python 

#*** DO NOT EDIT - GENERATED FROM services.ipynb ****

import os
import pandas as pd
from  mangorest.mango import webapi

#*** new stuff for running inference ***
import cv2
import tensorflow as tf
import os # don't need this
import numpy as np
from django.http import HttpResponse
from PIL import Image
import io
import base64
from django.http import JsonResponse
import concurrent.futures
import re
import json



#--------------------------------------------------------------------------------------------------------
# Checks queue for images
@webapi("/app1/getPotholeQueue")
def view_pothole_queue(request,  **kwargs):
    # Directory containing the files to parse
    directory = "PotholeQueue"

    # List to hold parsed data for each file
    data_list = []
    
    # Loop over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has a .jpg or .png extension
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Extract latitude, longitude, and datetime from the filename using a regular expression
            match = re.match(r"seattle:(?P<latitude>-?\d+\.\d+)_(?P<longitude>-?\d+\.\d+)_(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2})\.(jpg|png)", filename)
            if match:
                # Create a dictionary with the extracted values
                data = {
                    "latitude": float(match.group("latitude")),
                    "longitude": float(match.group("longitude")),
                    "datetime": match.group("datetime"),
                    "image_type": match.group(4) # Add the image file type to the dictionary
                }
                
                # Read the image file and encode it in base64 format
                with open(os.path.join(directory, filename), "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                
                # Add the base64-encoded image to the dictionary
                data["image"] = encoded_image
                
                # Add the dictionary to the list of parsed data
                data_list.append(data)
            else:
                print(f"Invalid filename format for file {filename}.")
        else:
            print(f"File {filename} is not a supported image file type.")
    
    # Return the list of parsed data as a JSON string
    return json.dumps(data_list)

# Inferene Handeler
def run_inference(image, model):
    image_exists = os.path.isfile(image)
    model_exists = os.path.isfile(model)
    print(f"\nrun_inference(image={image} ({image_exists}), model={model} ({model_exists}))\n")
    
    # Set output file path
    output_file = os.path.join("..", "dynamic", "output.png")
    # Check if output file exists and delete it if it does
    if os.path.isfile(output_file):
        os.remove(output_file)
    
    # Load model and image, perform inference, and add labels
    model_final = tf.keras.models.load_model(model)
    img = cv2.imread(image)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(img)
    ss.switchToSelectiveSearchFast()
    ssresults = ss.process()
    imout = img.copy()
    for e,result in enumerate(ssresults):
        if e < 500:
            x,y,w,h = result
            timage = imout[y:y+h,x:x+w]
            resized = cv2.resize(timage, (224,224), interpolation = cv2.INTER_AREA)
            img = np.expand_dims(resized, axis=0)
            out= model_final.predict(img)
            if out[0][0] > 0.60:
                # Add Labels & Design To Rectangles 
                label = "Pothole: " + str(round(out[0][0], 2))
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 2
                color = (255, 0, 0)
                text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = x + w // 2 - text_size[0] // 2
                text_y = y - text_size[1] - 5
                cv2.putText(imout, label, (text_x, text_y), font, font_scale, color, thickness)
                cv2.rectangle(imout, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)

    # Save output image to output_file
    cv2.imwrite(output_file, imout)
    print(f"Saving output image to {output_file}")
    
    # Return the file location
    return output_file



#--------------------------------------------------------------------------------------------------------    
@webapi("/app1/test")
def test( request,  **kwargs):
    return "APP 1 TEST version 1.0"
#--------------------------------------------------------------------------------------------------------    
@webapi("/app1/uploadfile")
def uploadfile( request,  **kwargs):
    par = dict(request.GET)
    par.update(request.POST)
   # DESTDIR ="/tmp/MYAPP/" 
    DESTDIR = os.path.join("..", "dynamic")

    os.makedirs(DESTDIR, exist_ok=True)  # Create the destination directory if it does not exist

    ret = ""
    for file_ in request.FILES.getlist('file'):
        filename = str(file_)
        content = file_.read()
        file_path = os.path.join(DESTDIR, str(file_))
        with open(file_path, "wb") as f:
            f.write(content)
        ret += filename
        if os.path.isfile(file_path):  # Check if the file exists in the destination directory
            print(" saved successfully.")
        else:
            print("error uploading file")
    return ret
#--------------------------------------------------------------------------------------------------------
@webapi("/app1/processall")
def processfiles(request, **kwargs):
    # Directory containing the files to parse
    directory = "PotholeQueue"

    # List to hold parsed data for each file
    data_list = []

    # Loop over all files in the directory
    for filename in os.listdir(directory):
        # Check if the file has a .jpg or .png extension
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Extract latitude, longitude, and datetime from the filename using a regular expression
            match = re.match(r"seattle:(?P<latitude>-?\d+\.\d+)_(?P<longitude>-?\d+\.\d+)_(?P<datetime>\d{4}-\d{2}-\d{2}-\d{2}-\d{2})\.(jpg|png)", filename)
            if match:
                # Create a dictionary with the extracted values
                data = {
                    "latitude": float(match.group("latitude")),
                    "longitude": float(match.group("longitude")),
                    "datetime": match.group("datetime"),
                    "image_type": match.group(4) # Add the image file type to the dictionary
                }
                
                # Get the full path of the image file
                image_path = os.path.join(directory, filename)
                
                # Call the run_inference function to generate a new image
                VGG = os.path.join('app1', 'Models/VGG.h5')
                new_image_path = run_inference(image_path, VGG)
                
                # Read the new image file and encode it in base64 format
                with open(new_image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
                
                # Add the base64-encoded image to the dictionary
                data["image"] = encoded_image
                
                # Remove the old image from the PotholeQueue directory
                #os.remove(image_path)
                
                # Add the dictionary to the list of parsed data
                data_list.append(data)

            else:
                print(f"Invalid filename format for file {filename}.")
        else:
            print(f"File {filename} is not a supported image file type.")
    
    # Return the list of parsed data as a JSON string
    return json.dumps(data_list)
        


#--------------------------------------------------------------------------------------------------------
@webapi("/app1/processfile")
def processfile(request, **kwargs):
    DESTDIR = os.path.join("..", "dynamic")
    filename = uploadfile(request, **kwargs)
    image_file = os.path.join(DESTDIR, filename)
    VGG = os.path.join('app1', 'Models/VGG.h5')
    output_path = run_inference(image_file, VGG)
    image = Image.open(output_path)
    # Apply a blue tint to the image
    blue_image = image.copy()

    # Save the modified image to a BytesIO object
    buffer = io.BytesIO()
    blue_image.save(buffer, format='JPEG')
    # Encode the image as a base64 string
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    # Return the modified image as a JSON response
    return JsonResponse({'image_data': img_str})
