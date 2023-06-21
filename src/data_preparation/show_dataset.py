"""
Display images associated with extreme traversal costs for visual analysis
"""

import os  # Provide OS functionalities
import pandas as pd  # To manage datasets
from PIL import Image, ImageDraw, ImageFont  # Image processing
from tqdm import tqdm  # Progress bar


DATASET = "datasets/dataset_multimodal_siamese2/"

# Read the csv file containing image names and
# associated pitch velocity variance measurements
csv_file = DATASET + "traversal_costs.csv"
dataframe = pd.read_csv(csv_file, converters={"image_id": str})

# Sort the dataframe by traversal cost
sorted_frame = dataframe.sort_values(by=["traversal_cost"])

# Create a new directory if it does not exist
try:
    os.makedirs("results/" + DATASET, exist_ok=True)
    print("results/" + DATASET + " folder created")
    
except OSError:
    print("Could not create results/" + DATASET + " folder")
    exit()

# Create a collage of images
collage = Image.new("RGB", (1700, 1100))
draw = ImageDraw.Draw(collage)
fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 30)
title_fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)

# Create a progress bar (total of 20 images)
pbar = tqdm(total=20)

# Display the 10 images with the lowest traversal costs
draw.text((680, 0), "Lowest values",
          font=title_fnt,
          fill=(255, 255, 255, 255))

for line in range(2):
    for col in range(5):
        pbar.update()
        
        # Display the image
        img_name = os.path.join(DATASET + "images",
                                sorted_frame.iloc[line*5 + col, 0])
        image = Image.open(img_name + ".png")
        image = image.resize((320, 180))
        collage.paste(image, (10 + 340*col, 40 + line*250))
        
        # Add the associated traversal cost
        draw.text((100 + 340*col, 220 + line*250),
                  f"{sorted_frame.iloc[line*5 + col, 1]:.5f}",
                  font=fnt, fill=(255, 255, 255, 255))
        
        draw.text((100 + 340*col, 250 + line*250),
                  f"{sorted_frame.iloc[line*5 + col, 3]:.3f} m/s",
                  font=fnt, fill=(255, 255, 255, 255))
        

# Display the 10 images with the highest traversal costs
draw.text((680, 540),
          "Highest values",
          font=title_fnt,
          fill=(255, 255, 255, 255))

# Get the total number of observations
nb_observations = len(dataframe)

for line in range(2):
    for col in range(5):
        pbar.update()
        
        # Display the image
        img_name = os.path.join(
            DATASET + "images",
            sorted_frame.iloc[nb_observations - 10 + line*5 + col, 0])
        image = Image.open(img_name + ".png")
        image = image.resize((320, 180))
        collage.paste(image, (10 + 340*col, 580 + line*250))
        
        # Add the associated traversal cost
        draw.text(
            (100 + 340*col, 760 + line*250),
            f"{sorted_frame.iloc[nb_observations - 10 + line*5 + col, 1]:.5f}",
            font=fnt, fill=(255, 255, 255, 255))
        
        draw.text(
            (100 + 340*col, 790 + line*250),
            f"{sorted_frame.iloc[nb_observations - 10 + line*5 + col, 3]:.3f} m/s",
            font=fnt, fill=(255, 255, 255, 255))


collage.save(f"results/{DATASET}collage.png", "PNG")
