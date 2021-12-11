# sample execution (requires torchvision)

import torch
import argparse

from PIL import Image
from torchvision import transforms

import pytorch2timeloop

def gen_imagenet_workloads(model_name, batch_size, output_dir, convert_linear=True,  exception_modules=[]):

    model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)

    input_image = Image.open('dog.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    
    device = 'cuda:0' if torch.cuda.is_available() else "cpu"
    model.to(device)

    input_shape = (3, 224, 224)
    batch_size = 1

    exception_module_names = exception_modules

    # Now, convert!
    pytorch2timeloop.convert_model(model, input_shape, batch_size, model_name, output_dir, convert_linear, exception_module_names)

def main(args):
    
    model_name = args.model_name
    batch_size = args.batch_size
    output_dir = args.output_dir
    exception_modules = args.exception_modules

    layer_list = gen_imagenet_workloads(model_name, batch_size, output_dir, exception_modules)

    return layer_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser("description='Generate imagenet workloads")
    parser.add_argument('--model_name', type=str, help='name of the dnn model')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--output_dir', type=str, default='./workload', help='top level directory to store the yaml specifications')
    parser.add_argument('--exception_modules', type=str, nargs='+', default=[], help='layer names to exclude')

    layer_list = main(parser.parse_args())

