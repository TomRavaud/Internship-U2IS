"""
Script to convert a bagfile containing images, IMU and odometry into
a self-supervised dataset for traversability analysis
"""
import csv
import os
import sys
import time
from math import atan2, pi, sqrt

import numpy as np
import rosbag
import rospy
from cv_bridge import CvBridge
from PIL import Image
from tf.transformations import euler_from_quaternion
from tqdm import tqdm

debut = time.time()

# Topic names
IMAGE_TOPIC = '/zed_node/rgb/image_rect_color'
ODOM_TOPIC = "/odometry/filtered"
IMU_TOPIC = "/imu/data"

# Downsamplig ratio, par exemple s'il vaut 5, une donnee sur 5 sera enregistree
DOWNSAMPLING_RATIO = 2

# Les differents topics ne publient pas en meme temps.
# EPSILON_T correspond a l'ecart de temps qu'on s'autorise (seconde)
EPSILON_T = 0.1

# Distance between robot position and point seen at the center of the camera
DELTA_D = 2.5
# Threshold for matching distance
EPSILON_D = 0.1
# Threshold for direction (robot should follow a straight line)
DELTA_ALPHA = 10.0 / 180.0 * pi

# Time window for gathering IMU data to estimate traversability
DELTA_T_IMU = 0.2


# Verifie les arguments du script. On peut soit appeler le script
# en tapant "python read_bags.py" soit "python read_bags.py nom_du_rosbag.bag"

if len(sys.argv) > 2:
    print("invalid number of arguments:   " + str(len(sys.argv)))
    print("should be 2: 'read_bags.py' and 'bagName'")
    print("or just 1: 'read_bags.py'")
    sys.exit(1)
elif len(sys.argv) == 2:
    list_of_bag_files = [sys.argv[1]]
    number_of_files = len(list_of_bag_files)
    print(f"reading only 1 bagfile: {str(list_of_bag_files[0])}")
elif len(sys.argv) == 1:
    # get list of only bag files in current dir.
    list_of_bag_files = [f for f in os.listdir(".") if f[-4:] == ".bag"]
    number_of_files = len(list_of_bag_files)
    print(f"reading all {number_of_files} bagfiles in current directory: \n")
    for f in list_of_bag_files:
        print(f)
else:
    print("bad argument(s): " + str(sys.argv))  # shouldnt really come up
    sys.exit(1)


count = 0
# On fait les calculs/sauvegardes pour chaque rosbags si plusieurs rosbags
for bagFile in list_of_bag_files:
    count += 1
    print("reading file " + str(count) +
          " of  " + str(number_of_files) + ": " + bagFile)
    # acces au rosbag
    bag = rosbag.Bag(bagFile)
    bagContents = bag.read_messages()
    bagName = os.path.basename(bag.filename)
    # cree un nouveau dossier
    directory = os.path.abspath(os.getcwd())
    results_dir = directory + "/datasets/dataset_" + bagName[:-4]
    try:  # on cree le nouveau dossier seulement s'il n'existe pas deja
        os.mkdir(results_dir)
        print(results_dir + " folder created")
    except OSError:
        print("Existing directory " + results_dir)
        print("Aborting")
        exit()

    #########
    # Write images on disk and create a list of their timestamps
    #########

    full_image_list = []  # will contain, (timestamp of image, image name)

    # timestamp = nom de l'indicateur de temps pour ros. Utile car ne depend pas du topic,
    # reste le meme d'un topic a l'autre
    # on garde les timestamp pour ensuite pouvoir recuperer les donnees qui correspondent
    # aux instants ou les images ont ete prises.

    bridge = CvBridge()
    topic_name = IMAGE_TOPIC.replace('/', '_')
    topic_dir = results_dir + "/" + topic_name

    # creation d'un dossier s'il n'existe pas deja
    try:
        os.mkdir(topic_dir)
        print(topic_dir + " folder created")
    except OSError:
        pass

    print("Writing images")

    # creation d'un csv avec les timestamp
    csvname = topic_dir + "/" + topic_name + ".csv"
    with open(csvname, 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')

        # ecriture 1ere ligne du csv
        filewriter.writerow(["TimeStamp"])

        pbar = tqdm(total=bag.get_message_count(IMAGE_TOPIC))
        for image_number, (topic, msg, t) in enumerate(bag.read_messages(topics=IMAGE_TOPIC)):
            pbar.update()
            if image_number == 0:
                t0 = t  # save timestamp of first image for later
            if image_number % DOWNSAMPLING_RATIO == 0:  # on sauvegarde uniquement une donnee sur "downsamplingRatio"
                cv_img = bridge.imgmsg_to_cv2(
                    msg, desired_encoding="rgb8")
                im = Image.fromarray(cv_img)

                im = im.convert('RGB')
                image_name = f"{image_number:05d}.png"
                im.save(topic_dir + "/" +
                        image_name, "PNG")

                # on remplit la liste avec les instants qu'on veut sauvegarder
                full_image_list.append((rospy.Time.to_sec(t), image_name))
                filewriter.writerow([str(t)])
        pbar.close()
        print(
            f"Wrote {int(bag.get_message_count(IMAGE_TOPIC)/DOWNSAMPLING_RATIO + 1)} images")

    ########
    # For each image, find timestamp where the robot reaches the position
    # at the center of the image according to odometry, and remove image if nothing matches
    ########

    print("Filtering images that have corresponding positions")
    # will contain, (timestamp of image, image name, timestamp of position in front of robot)
    filtered_image_list = []
    last_t = t0  # start from the timestamp of first image

    pbar = tqdm(total=len(full_image_list))
    # on compare chaque instant du topic avec les instants
    for image_number, (t_image, image_name) in enumerate(full_image_list):
        # "t_image" ou nous avons sauvegarde une image
        pbar.update()
        for subtopic, msg, t in bag.read_messages(ODOM_TOPIC, start_time=last_t):
            if rospy.Time.to_sec(t) < t_image + EPSILON_T and rospy.Time.to_sec(t) > t_image - EPSILON_T:
                last_t = t

                # get position at the time of image
                ref_position = msg.pose.pose.position
                orientation_q = msg.pose.pose.orientation
                quat_list = [orientation_q.x, orientation_q.y,
                             orientation_q.z, orientation_q.w]
                _, _, ref_yaw = euler_from_quaternion(quat_list)

                # check for first future pose for correct displacement, and store its time
                for subtopic_2, msg_2, t_2 in bag.read_messages(ODOM_TOPIC, start_time=t):
                    new_position = msg_2.pose.pose.position

                    # motion direction since ref pose
                    alpha = atan2(new_position.y - ref_position.y,
                                  new_position.x - ref_position.x) - ref_yaw

                    # travelled distance
                    dist = sqrt((new_position.x - ref_position.x)*(new_position.x - ref_position.x)
                                + (new_position.y - ref_position.y)*(new_position.y - ref_position.y))

                    if dist > DELTA_D-EPSILON_D and abs(alpha) < DELTA_ALPHA:
                        filtered_image_list.append(
                            (t_image, image_name, rospy.Time.to_sec(t_2)))
                        break
                    if dist > DELTA_D+EPSILON_D:
                        break
                break
    pbar.close()
    print(
        f"Remaining {len(filtered_image_list)} images that have required data for computing traversability")

    #########
    # For each remaining image, compute the variance of pitch velocity around the pose in front of the robot
    #########

    print("Computing traversability")
    # Create a new CSV file for final data
    filename = results_dir + '/' + \
        str.replace(IMU_TOPIC, '/', '_') + '.csv'

    with open(filename, 'w+') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        headers = ["image_id", "y"]  # first column header
        filewriter.writerow(headers)

        last_t = t0
        nbr_valid_imgs = 0
        pbar = tqdm(total=len(filtered_image_list))
        for t_image, image_name, t_odom in filtered_image_list:
            pbar.update()
            # for each image, we look for the data in IMU_TOPIC
            pitch_vel_list = []
            for subtopic, msg, t in bag.read_messages(IMU_TOPIC, start_time=last_t):
                if rospy.Time.to_sec(t) < t_odom - DELTA_T_IMU:
                    last_t = t
                # keep IMU value around the time found with odometry
                if rospy.Time.to_sec(t) > t_odom - DELTA_T_IMU:
                    pitch_vel_list.append(msg.angular_velocity.y)
                if rospy.Time.to_sec(t) > t_odom + DELTA_T_IMU:
                    break

            if len(pitch_vel_list) > 2:
                values = [image_name, np.var(pitch_vel_list)]
                filewriter.writerow(values)
                nbr_valid_imgs += 1
            else:
                print("No values!")
    pbar.close()
    print(
        f"Remaining {nbr_valid_imgs} images that have required data for traversability")

    bag.close()

print(f"\nDone reading all {number_of_files} bag files.")

fin = time.time()
print(f"Processing time: {round(fin - debut, 1)} s.")
