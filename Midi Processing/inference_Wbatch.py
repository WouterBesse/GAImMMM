import os
import json
import pickle
from pathlib import Path
from argparse import ArgumentParser, Namespace
import math

import torch
import torchaudio
from omegaconf import DictConfig, OmegaConf
from audio_cls.src.model.net import ShortChunkCNN_Res
from midi_cls.src.model.net import SAN
from midi_cls.midi_helper.remi.midi2event import analyzer, corpus, event
from midi_cls.midi_helper.magenta.processor import encode_midi

path_data_root = "./midi_cls/midi_helper/remi/"
path_dictionary = os.path.join(path_data_root, 'dictionary.pkl')
midi_dictionary = pickle.load(open(path_dictionary, "rb"))
event_to_int = midi_dictionary[0]

# clearConsole script acquired from https://www.delftstack.com/howto/python/python-clear-console/
def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)

def torch_sox_effect_load(mp3_path, resample_rate):
    effects = [
        ['rate', str(resample_rate)]
    ]
    waveform, source_sr = torchaudio.load(mp3_path)
    if source_sr != 22050:
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, source_sr, effects, channels_first=True)
    return waveform

def remi_extractor(midi_path, event_to_int):
    midi_obj = analyzer(midi_path)
    song_data = corpus(midi_obj)
    event_sequence = event(song_data)
    
    quantize_midi = []
    for i in event_sequence:
        try:
            quantize_midi.append(event_to_int[str(i['name'])+"_"+str(i['value'])])
        except KeyError:
            
            if 'Velocity' in str(i['name']):
                quantize_midi.append(event_to_int[str(i['name'])+"_"+str(i['value']-2)])
            else:
                #skip the unknown event
                continue
    
    return quantize_midi

def magenta_extractor(midi_path):
    return encode_midi(midi_path)

def predict(args, filepath, task) -> None:
    device = args.cuda if args.cuda and torch.cuda.is_available() else 'cpu'
    # if args.cuda:
    #     print('GPU name: ', torch.cuda.get_device_name(device=args.cuda))
    config_path = Path("best_weight", args.types, task, "hparams.yaml")
    checkpoint_path = Path("best_weight", args.types, task, "best.ckpt")
    config = OmegaConf.load(config_path)
    label_list = list(config.task.labels)
    model = SAN(
        num_of_dim= config.task.num_of_dim,
        vocab_size= config.midi.pad_idx+1,
        lstm_hidden_dim= config.hparams.lstm_hidden_dim,
        embedding_size= config.hparams.embedding_size,
        r= config.hparams.r)
    state_dict = torch.load(checkpoint_path, map_location=torch.device(args.cuda))
    new_state_map = {model_key: model_key.split("model.")[1] for model_key in state_dict.get("state_dict").keys()}
    new_state_dict = {new_state_map[key]: value for (key, value) in state_dict.get("state_dict").items() if key in new_state_map.keys()}
    model.load_state_dict(new_state_dict)
    model.eval()
    model = model.to(args.cuda)

    if args.types == "midi_like":
        quantize_midi = magenta_extractor(filepath)
        model_input = torch.LongTensor(quantize_midi).unsqueeze(0)
        prediction = model(model_input.to(args.cuda))
    elif args.types == "remi":
        quantize_midi = remi_extractor(filepath, event_to_int)
        model_input = torch.LongTensor(quantize_midi).unsqueeze(0)
        prediction = model(model_input.to(args.cuda))
    
    pred_label = label_list[prediction.squeeze(0).max(0)[1].detach().cpu().numpy()]
    pred_value = prediction.squeeze(0).detach().cpu().numpy()
    #print(filepath, " is emotion", pred_label)
    #print("Inference values: ", pred_value)

    return pred_value

def batch(args):
    # Make a list of files
    PredictList = []
    filelist = os.listdir(args.input_path)
    print("Starting batch analysis")
    # Counting variable for the percentage
    z = 0
    furthestval = 0
    furthestar = 0
    vallist = []
    arlist = []
    # Go through each file and analyze it
    for file in filelist:
        # Print information
        z += 1
        percentage = z / len(filelist)
        # Analyze file and make region calculation
        full_path = os.path.join(args.input_path, file)
        # n_cls = [2.74, 1.33, -5.33, -1.1]
        if os.path.isfile(full_path):
            raw_val = predict(args, full_path, "valence")
            raw_ar = predict(args, full_path, "arousal")
            clearConsole()
            print(math.ceil(percentage * 100), '% progress')
            print("Current file:", file)
            print("valence = ", raw_val)
            print("arousal = ", raw_ar)
            for i in range(2):
                if abs(raw_val[i]) > furthestval:
                    furthestval = abs(raw_val[i])
                if abs(raw_ar[i]) > furthestar:
                    furthestar = abs(raw_ar[i])
            vallist.append(raw_val)
            arlist.append(raw_ar)
            # max_valence = abs(max(raw_valence,key=lambda x: abs(x)))
            # max_arousal = max(raw_arousal, key=lambda x: abs(x))
            # PredictList.append([file, norm_coord])
    normvallist = []
    normarlist = []
    for entry in vallist:
        normentry = []
        for value in entry:
            normentry.append(value / furthestval)
        normvallist.append(normentry)
    for entry in arlist:
        normentry = []
        for value in entry:
            normentry.append(value / furthestar)
        normarlist.append(normentry)
    print("Valence list: ")
    for valence in normvallist:
        print(valence)
    print("Arousal list: ")
    for arousal in normarlist:
        print(arousal)
    print("Done :)")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--types", default="midi_like", type=str, choices=["midi_like", "remi"])
    parser.add_argument("--task", default="ar_va", type=str, choices=["ar_va", "arousal", "valence"])
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument('--cuda', default='cuda:0', type=str)
    args = parser.parse_args()
    batch(args)
