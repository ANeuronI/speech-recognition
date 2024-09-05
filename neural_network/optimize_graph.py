import argparse
import torch
from model import SpeechRecognition
from collections import OrderedDict

def main(args):
    print("loading model from", args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu'))
    h_params = SpeechRecognition.hyper_parameters
    model = SpeechRecognition(**h_params)

    model_state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k.replace("model.", "") # remove `model.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False

    print("saving checkpoint model to", args.save_path)
    torch.save(model.state_dict(), args.save_path)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freezes and optimizes the model. Use after training.")
    parser.add_argument('--model_checkpoint', type=str, default=None, required=True,
                        help='Checkpoint of model to optimize')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='path to save optimized model')

    args = parser.parse_args()
    main(args)
