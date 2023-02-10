"""
Display images associated with extreme traversal costs for visual analysis
"""

import os  # Provide OS functionalities
import pandas as pd  # To manage datasets
from PIL import Image, ImageDraw, ImageFont  # Image processing
from tqdm import tqdm  # Progress bar


DATASET = "datasets/dataset_path_grass_robust/"

# Read the csv file containing image names and
# associated pitch velocity variance measurements
csv_file = DATASET + "/traversal_costs.csv"
dataframe = pd.read_csv(csv_file)

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
collage = Image.new("RGB", (1700,960))
draw = ImageDraw.Draw(collage)
fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)

# Create a progress bar (total of 20 images)
pbar = tqdm(total=20)

# Display the 10 images with the lowest pitch velocity variances
draw.text((680, 0), "Lowest values", font=fnt, fill=(255, 255, 255, 255))

for line in range(2):
    for col in range(5):
        pbar.update()
        
        # Display the image
        img_name = os.path.join(DATASET + "images",
                                sorted_frame.iloc[line*5 + col, 0])
        image = Image.open(img_name)
        image = image.resize((320, 180))
        collage.paste(image, (10 + 340*col, 40 + line*220))
        
        # Add the associated pitch velocity variance measure
        draw.text((100 + 340*col, 220 + line*220),
                  f"{sorted_frame.iloc[line*5 + col, 1]:.5f}",
                  font=fnt, fill=(255, 255, 255, 255))

# Display the 10 images with the highest pitch velocity variances
draw.text((680,480), "Highest values", font=fnt, fill=(255, 255, 255, 255))

# Get the total number of observations
nb_observations = len(dataframe)

for line in range(2):
    for col in range(5):
        pbar.update()
        
        # Display the image
        img_name = os.path.join(DATASET + "images",
                                sorted_frame.iloc[nb_observations - 10 + line*5 + col, 0])
        image = Image.open(img_name)
        image = image.resize((320, 180))
        collage.paste(image, (10 + 340*col, 520 + line*220))
        
        # Add the associated pitch velocity variance measure
        draw.text((100 + 340*col, 440 + 260 + line*220),
                  f"{sorted_frame.iloc[nb_observations - 10 + line*5 + col, 1]:.5f}",
                  font=fnt, fill=(255, 255, 255, 255))


collage.save(f"results/{DATASET}collage.png", "PNG")
