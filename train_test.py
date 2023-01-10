import argparse

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

import loader as d
import model_simple as m
from generate_rand_params import get_params
from hyperband import Hyperband
from PIL import Image, ImageDraw, ImageFont



def save_loss_curves(num_epochs, train_loss_list_plot,validation_loss_list_plot):
    
    # Plotting loss for all epochs
    plt.figure(3)
    plt.xlim((0,num_epochs))
    plt.plot(train_loss_list_plot,
             'r', label="training loss")
    plt.xlabel("Number of epochs")

    plt.plot(validation_loss_list_plot,
             'b', label="validation loss")
    plt.xlabel("Number of epochs")
    plt.ylabel("MSE")

    plt.legend()
    plt.savefig('results/loss_curves.png')
    plt.close()

def main(arg, num_epochs=10):

    batch_size = args.batchsize  # 8 par défaut
    learning_rate = args.learning_rate
    modelnetwork = args.modelnetwork

    # 0.0001
    weight_decay = args.weight_decay

    print("Creating Dataloaders...")
    dataloaders = d.load(d.training_data1, batch_size)
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    test_loader = dataloaders['test']
    print(f"{len(train_loader)*batch_size} images loaded in {len(train_loader)} batches for training")
    print(f"{len(valid_loader)*batch_size} images loaded in {len(valid_loader)} batches for validation")
    print(f"{len(test_loader)*batch_size} images loaded in {len(test_loader)} batches for testing")

    if modelnetwork == "AlexNet":
        model = m.AlexNet().to(m.device)
    else:
        model = m.ResNet50(num_classes=1).to(m.device)

    model.apply(model.init_weights)

    print(f"Using {modelnetwork} model")


# Defining the model hyper parameters

    #learning_rate = 0.0001
    #weight_decay =  0.01

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


# Training process begins
    validation_loss_list = []
    train_loss_list = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}:', end=' ')
        train_loss = 0
        valid_loss = 0
     # Iterating over the training dataset in batches
        model.train()
        for i, samplebatch in enumerate(train_loader):
            images = samplebatch['image']
            labels = samplebatch['landmarks']
           # Extracting images and target labels for the batch being iterated
           # ici c le problème

            images = images.to(device=m.device, dtype=torch.float)
            labels = labels.type(torch.float)
            labels = labels.to(m.device)

           # Calculating the model output and the loss
            outputs = model(images)
            loss = criterion(outputs, labels)

           # Updating weights according to calculated loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

       # Printing loss for each epoch
        train_loss_list.append(train_loss/(len(train_loader)*batch_size))
        print(f"Training loss = {train_loss_list[-1]}")


        # validation part
        model.eval()

        with torch.no_grad():

           # Iterating over the training dataset in batches
            for i, samplebatch in enumerate(valid_loader):
                images = samplebatch['image']
                y_true = samplebatch['landmarks']

                images = images.to(m.device, dtype=torch.float)
                y_true = y_true.type(torch.float)
                y_true = y_true.to(m.device)

            # Calculating outputs for the batch being iterated
                outputs = model(images)

                _, y_pred = torch.max(outputs.data, 1)
                loss = criterion(outputs, y_true)
                valid_loss += loss.item()

            validation_loss_list.append(
                valid_loss/(len(valid_loader)*batch_size))
            print(f" validation loss = {validation_loss_list[-1]}")

        save_loss_curves(num_epochs, train_loss_list,validation_loss_list)


    best_training_loss = min(train_loss_list)
    best_validation_loss = min(validation_loss_list)
    print('best training loss', best_training_loss)
    print('best validation loss', best_validation_loss)


   # testing part our model with test loader created

    list_output = []
    test_loss_final = 0.0
    with torch.no_grad():

        collage = Image.new("RGB", (8*225,25+22*250))
        draw = ImageDraw.Draw(collage)
        fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 20)
        transform = transforms.ToPILImage()
        draw.text((225*4-300,0), "Test images, with prediction / groundtruth below", font=fnt, fill=(255, 255, 255, 255))

        # Iterating over the training dataset in batches
        for i, samplebatch in enumerate(test_loader):
            images = samplebatch['image']
            y_true = samplebatch['landmarks']

            images = images.to(m.device, dtype=torch.float)
            y_true = y_true.type(torch.float)
            y_true = y_true.to(m.device)

            # Calculating outputs for the batch being iterated
            outputs = model(images)

            loss = criterion(outputs, y_true)
            test_loss_final += loss.item()

            ###
            # un petit code pour voir le test data avec les outputs
            ###
            images_batch = samplebatch['image'] # used to have it in cpu


            grid = utils.make_grid(images_batch)
            image = transform(grid)
            collage.paste(image, (0,25+i*250))

            pred = [item for sublist in outputs.cpu().tolist()
                      for item in sublist] # just det a clean list of prediction
            gt = y_true.tolist()

            for id, (p,g) in enumerate(zip(pred,gt)):
                draw.text((30+225*id,25+225+i*250), f"{p:.5f}/{g:.5f}", font=fnt, fill=(255, 255, 255, 255))


        ### Finished tests
        collage.save("results/test_images.png", "PNG")
        test_error_final = test_loss_final/(len(test_loader)*batch_size)
        print(f" testing loss = {test_error_final}")

    return {"best train loss": best_training_loss, "best_val_loss": best_validation_loss}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('--batchsize', help='Batch size', default=8, type=int)
    parser.add_argument('-lr', '--learning_rate', help='Learning rate',
                        default=0.0001, type=float)  # same here
    # i need them just for training after finding the best paramaters by hyperband algorithm
    parser.add_argument('-wd', '--weight_decay',
                        help='Weight_decay', default=0.001, type=float)
    parser.add_argument(
        '--hyp', help='Test hyperparameters', default=0, type=int)
    parser.add_argument(
        '--num_epochs', help='Number of epoch', default=10, type=int)
    parser.add_argument('--modelnetwork', '--modelnetwork', help='type of neural network',
                        default="AlexNet", type=str, choices=['AlexNet', 'ResNet'])
    args = parser.parse_args()

    if (args.hyp == 0):
        hyperpar = {}

        hyperpar['batchsize'] = args.batchsize
        hyperpar['learning_rate'] = args.learning_rate

        hyperpar["weight_decay"] = args.weight_decay
        hyperpar["modelnetwork"] = args.modelnetwork

        main(hyperpar, args.num_epochs)

    elif (args.hyp == 1):
        hyp = Hyperband(args, get_params, main)
        hyp.run(dry_run=False, hb_result_file="hb_result.json",
                hb_best_result_file="hb_best_result.json")


# car on a argument test dont on n'a pas besoin
