import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import argparse
from tqdm import tqdm

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_parser():
    '''Parse user inputs from command line'''

    def check_c(c):
        if type(c)==str and c.endswith(".pth"):
            return c
        else:
            raise argparse.ArgumentTypeError("File path must end in '.pth' to save checkpoint.")

    parser = argparse.ArgumentParser()

    parser.add_argument("data_dir", type=str, default="flowers",
                        help="Path to image folder parent.")
    parser.add_argument("--check_dir", "-c", type=check_c,
                        help="Path to save checkpoint in. Must end in '.pth' file")
    parser.add_argument("--arch", "-a", type=str, default="densenet", choices=["densenet", "vgg"],
                        help="Architecture to use in the training model.")
    parser.add_argument("--learning_rate", "-lr", type=float, default=0.001,
                        help="Learning rate to be used by optimizer for backpropagation.")
    parser.add_argument("--epochs", "-e", type=int, default=15,
                        help="Number of epochs to train the model. Recommended range 5-20.")
    parser.add_argument("--hidden_units", "-hid", type=int, default=512,
                        help="Number of nodes in the hidden layer.")
    parser.add_argument("--gpu", "-gpu", action="store_true",
                        help="Use GPU.")

    return parser.parse_args()

def device_selection():
    '''Sets model to gpu or cpu'''

    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Device set to {torch.cuda.get_device_name(device=0)}")
        else:
            device = torch.device("cpu")
            print("No GPU detected. Device set to CPU")
    else:
        device = torch.device("cpu")
        print("Device set to CPU")

    return device

def data_loader(data_dir):
    '''Loads training, validation, and testing data and returns
       transforms, ImageFolders, and DataLoaders for each in correspondig dicts.
       Each sub-datset must be in folders called 'train', 'valid', and 'test'
       respectively inside the data_dir folder'''

    # Directories for training, validation, and testing sub-datasets
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Defines transforms for each sub-datasets
    data_transforms = {
        "train": transforms.Compose([
                 transforms.RandomRotation(45),
                 transforms.RandomResizedCrop(224),
                 transforms.RandomHorizontalFlip(),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        "valid": transforms.Compose([
                 transforms.Resize(255),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        "test": transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # Creates datasets to be passed into a DataLoader
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
        "test" : datasets.ImageFolder(test_dir, transform=data_transforms["test"])
    }


    # Creates dataloaders from ImageFolder datasets
    dataloaders = {
        "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
        "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=64, shuffle=True),
        "test" : torch.utils.data.DataLoader(image_datasets["test"], batch_size=64, shuffle=True)
    }

    print("Data loaded")

    return data_transforms, image_datasets, dataloaders

def save_checkpoint(filepath, architecture, classifier, epochs, lr,
                    model_state_dict, optimizer_state_dict, hidden_units,
                    class_to_idx):
    '''Saves model after training as well as hyperparameters used
       during training. Checkpoints can be loaded back for usage.'''

    # Model information dictionary
    checkpoint = {
        "architecture"          : architecture,
        "classifier"            : classifier,
        "epochs"                : epochs,
        "lr"                    : lr,
        "model_state_dict"      : model_state_dict,
        "optimizer_state_dict"  : optimizer_state_dict,
        "hidden_units"          : hidden_units,
        "class_to_idx"          : class_to_idx
    }

    torch.save(checkpoint, filepath)

    print(f"Checkpoint saved at {filepath}.")

def train_model(arch, epochs, learnrate, hidden_units):
    '''Training loop for model. Model can have 1 hidden layer with
       hidden_units nodes'''

    # Model selection
    if arch.lower()=="densenet":
        model = models.densenet201(pretrained=True)
        in_features = model.classifier.in_features
    if arch.lower()=="vgg":
        model = models.vgg19_bn(pretrained=True)
        in_features = model.classifier[0].in_features

    # Freezes model parameters in feature/convolutional layers
    for param in model.parameters():
        param.requires_grad = False

    # Redefines the classifier which whill be trained
    classes = len(image_datasets["train"].classes)
    model.classifier = nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(in_features, hidden_units)),
        ("relu1", nn.ReLU()),
        ("dropout", nn.Dropout(p=0.2)),
        ("fc2", nn.Linear(hidden_units, classes)),
        ("out", nn.LogSoftmax(dim=1))
        ]))

    # Network initial setup
    batch_loss = 0
    train_count = 0

    ## Defines optimizer and error function
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learnrate)

    # Moves model to selected device
    model.to(device)

    start = time.time()
    print(f"{arch.upper()} network initialized...")

    # Initialize training loop

    # Set model to training mode. (Dropout layer is ON)
    model.train()
    for e in range(epochs):
        print("")
        for images, labels in tqdm(dataloaders["train"], desc=f"Epoch {e+1:2}/{epochs}",
                                   unit="images", unit_scale=64):

            train_count += 1
            # Clears gradients and moves data to slected device
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)

            # Forwardpass
            logps = model(images)
            loss = criterion(logps, labels)
            # Backpropagation
            loss.backward()
            optimizer.step()
            # Loss accumulation
            batch_loss += loss.item()

        # Validation loop after each epoch
        else:
            accuracy = 0
            valid_count = 0
            valid_loss = 0

            # Sets model to evaluation mode (Dropout layer is OFF)
            model.eval()

            # Turns off gradients to speed up loop
            with torch.no_grad():
                for images, labels in tqdm(dataloaders["valid"], desc="Validation",
                                           unit="images", unit_scale=64, leave=False):
                    valid_count += 1

                    # Moves data to selected device
                    images, labels = images.to(device), labels.to(device)

                    # Forwardpass
                    logps = model(images)
                    # Error accumulation
                    loss = criterion(logps, labels)
                    valid_loss += loss.item()

                    # Accuracy measure
                    ps = torch.exp(logps)
                    mode_p, mode_class = ps.topk(1, dim=1)
                    equality = mode_class == labels.view(*mode_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()


            # Prints statistics
            print("")
            print(f"Train Loss:    : {batch_loss/train_count:.3f}")
            print(f"Validation Loss: {valid_loss/valid_count:.3f}")
            print(f"Accuracy       : {accuracy/valid_count*100:.1f}%")
            # Resets newtwork to train mode and error metrics
            train_count = 0
            batch_loss = 0
            model.train()

    #Network runtime
    running_time = time.time() - start
    print(f"\nRunning Time: {running_time//60:.0f}m {running_time % 60:.0f}s")

    if args.check_dir:
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        save_checkpoint(args.check_dir, arch.lower(), model.classifier, epochs,
                        learnrate, model_state_dict, optimizer_state_dict,
                        hidden_units, image_datasets['train'].class_to_idx)

if __name__ == '__main__':
    # Store user inputs from command line into args as attributes
    args = train_parser()
    # Device selection
    device = device_selection()
    # Loads datasets as training, vcalidation, and testing sub-sets
    data_transforms, image_datasets, dataloaders = data_loader(args.data_dir)
    # Trains model
    train_model(args.arch, args.epochs, args.learning_rate, args.hidden_units)
