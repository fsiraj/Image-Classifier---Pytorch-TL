import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import json
import argparse

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def predict_parser():
    '''Parses user inputs from the command line.'''

    def check_c(c):
        if type(c)==str and c.endswith(".pth"):
            return c
        else:
            raise argparse.ArgumentTypeError("File path must end in '.pth' to save checkpoint.")

    parser = argparse.ArgumentParser()

    parser.add_argument("image_path", type=str,
                        help="Path to image file.")
    parser.add_argument("checkpoint", type=check_c,
                        help="Path to model checkpoint.")
    parser.add_argument("category_names",
                        help="Path to json for category name mapping")
    parser.add_argument("--top_k", "-k", type=int, default=5,
                        help="Return top k classes from classifier.")
    parser.add_argument("--gpu", "-gpu", action="store_true",
                        help="Use GPU.")

    return parser.parse_args()

def device_selection():
    '''Selects either GPU or CPU to run model on'''
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

def reload_checkpoint(filepath):
    '''Reloads a model from a model checkpoint (.pth file.)'''

    checkpoint = torch.load(filepath)

    if checkpoint["architecture"] == "densenet":
        model = models.densenet201()
        print("Densenet201 loaded.")
    elif checkpoint["architecture"] == "vgg":
        model = models.vgg19_bn()
        print("VGG19_bn loaded.")
    else:
        print("Model not supported")

    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    model.idx_to_cat = {v: k for k, v in model.class_to_idx.items()}
    model.to(device)

    return model

def image_loader(image_path):
    '''Loads and transforms image into format that can
       be input into the classifier'''

    image = Image.open(image_path)
    transformations = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transformations(image)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    image_tensor = image_tensor.to(device)

    return image_tensor

def predict(model, image, top_k=5):
    '''Predicts the classification of the data and returns the top k
       classfication values and index.'''

    model.eval()
    with torch.no_grad():
        ps = torch.exp(model(image_tensor))
        mode_ps, mode_idx = ps.topk(top_k, dim=1)
        mode_ps = mode_ps.cpu().numpy().squeeze()
        mode_idx = mode_idx.cpu().numpy().squeeze()

    return mode_ps, mode_idx

def class_names(json_filepath):
    '''Returns the class name from the class index'''

    with open(json_filepath, 'r') as f:
        cat_to_name = json.load(f)

    class_names = []
    for i in mode_idx:
        i = model.idx_to_cat[i]
        class_name = cat_to_name[i]
        class_names.append(class_name)

    return class_names

def show_prediction(class_names, ps):
    '''Prints out category and probability to command line'''

    i = 1
    print("\n   Classification     :  Probability\n")
    for class_name, p in zip(class_names, ps):
        print(f"{i:2}. {class_name.title():18} : {p:5.2f}")
        i += 1

if __name__ == "__main__":
    # Parses user input
    args = predict_parser()
    # Selects device to run model on
    device = device_selection()
    # Loads in an image as a tensor and applies transforms
    image_tensor = image_loader(args.image_path)
    # Initializes model from .pth file
    model = reload_checkpoint(args.checkpoint)
    # Calculates top classifications
    mode_ps, mode_idx = predict(model, image_tensor, args.top_k)
    # Translates category index to category name
    class_names = class_names(args.category_names)
    # Displays classification data
    show_prediction(class_names, mode_ps)
