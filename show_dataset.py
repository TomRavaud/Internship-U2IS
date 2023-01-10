"""
Rename images from a dataset and displays extreme values for visual analysis
"""

import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

DATASET = "datasets/dataset_sample_bag/"
# DATASET = "datasets/dataset_rania_2022-07-01-11-40-52/"

csv_file=DATASET+'/_imu_data.csv'
data_frame = pd.read_csv(csv_file)

sorted_frame = data_frame.sort_values(by=['y'])

try:
    os.makedirs("results/" + DATASET,exist_ok=True)
    print("results/" + DATASET + " folder created")
except OSError:
    print("couldn't create results/" + DATASET + " folder")
    exit()

collage = Image.new("RGB", (1700,960))
draw = ImageDraw.Draw(collage)
fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 40)

draw.text((680,0), "Lowest values", font=fnt, fill=(255, 255, 255, 255))
pbar = tqdm(total=20)
for line in range(2):
    for col in range(5):
        pbar.update()
        img_name = os.path.join(DATASET+'_zed_node_rgb_image_rect_color', sorted_frame.iloc[line*5+col,0])
        image = Image.open(img_name)
        image = image.resize((320,180))
        collage.paste(image, (10+340*col,40+line*220))
        draw.text((100+340*col,220+line*220), f"{sorted_frame.iloc[line*5+col,1]:.5f}", font=fnt, fill=(255, 255, 255, 255))

draw.text((680,480), "Highest values", font=fnt, fill=(255, 255, 255, 255))
sorted_frame = data_frame.sort_values(by=['y'],ascending=False )
for line in range(2):
    for col in range(5):
        pbar.update()
        img_name = os.path.join(DATASET+'_zed_node_rgb_image_rect_color', sorted_frame.iloc[line*5+col,0])
        image = Image.open(img_name)
        image = image.resize((320,180))
        collage.paste(image, (10+340*col,520+line*220))
        draw.text((100+340*col,440+260+line*220), f"{sorted_frame.iloc[line*5+col,1]:.5f}", font=fnt, fill=(255, 255, 255, 255))


collage.save(f"results/{DATASET}collage.png", "PNG")
