import os
import json
import pickle
from pathlib import Path
from argparse import ArgumentParser, Namespace
import math
# from typing import ClassVar
from scipy import stats
import pandas as pd
import numpy as np

import torch
from omegaconf import DictConfig, OmegaConf
# from audio_cls.src.model.net import ShortChunkCNN_Res
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

# getListOfFiles script acquired from https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

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
    # label_list = list(config.task.labels)
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
    
    # pred_label = label_list[prediction.squeeze(0).max(0)[1].detach().cpu().numpy()]
    pred_value = prediction.squeeze(0).detach().cpu().numpy()
    # print(filepath, " is emotion", pred_label)
    # print("Inference values: ", pred_value)

    return pred_value

def batch(args):
    # Make a list of files
    filelist = getListOfFiles(args.input_path)
    print("Starting batch analysis")
    # Counting variable for the percentage
    z = 0
    vallist = []
    arlist = []
    valraw = []
    arraw = []
    namelist = []
    verbose = args.verbose
    # Go through each file and analyze it
    for file in filelist:
        
        
        # Counter to keep track of percentage done
        z += 1
        percentage = z / len(filelist)
        # Analyze file and make region calculation
        full_path = os.path.join(args.input_path, file)
        # n_cls = [2.74, 1.33, -5.33, -1.1]
        if os.path.isfile(full_path):
            try:
                # Predict valence and arousal
                raw_val = predict(args, full_path, "valence")
                raw_ar = predict(args, full_path, "arousal")
                # Print information
                clearConsole()
                print(round(percentage * 100, 2), '% progress')
                print("Current file:", file)
                print("valence = ", raw_val)
                print("arousal = ", raw_ar)
                # Add to total list
                namelist.append(file)
                valraw.append(raw_val[0])
                valraw.append(raw_val[1])
                arraw.append(raw_ar[0])
                arraw.append(raw_ar[1])
                vallist.append(raw_val)
                arlist.append(raw_ar)
                # max_valence = abs(max(raw_valence,key=lambda x: abs(x)))
                # max_arousal = max(raw_arousal, key=lambda x: abs(x))
            except Exception:
                clearConsole()
                print(round(percentage * 100, 2), '% progress')
                print("Current file:", file)
                print("valence = failed")
                print("arousal = failed")
                pass
    
    # Define normalised lists and vector lists
    furthestval = 0
    furthestar = 0

    # # Do z value analysis
    # valdf = pd.DataFrame({'data':valraw})
    # ardf = pd.DataFrame({'data':arraw})
    # valz = stats.zscore(valdf['data'])
    # arz = stats.zscore(ardf['data'])
    # valzcoupled = []
    # arzcoupled = []

    # # Couple z values back into groups after analysis
    # i = 0
    # while i < len(valz)/2 :
    #     valzcoupled.append([valz[i * 2], valz[i * 2 + 1]])
    #     i += 1

    # i = 0
    # while i < len(arz)/2 :
    #     arzcoupled.append([arz[i * 2], arz[i * 2 + 1]])
    #     i += 1

    # Remove outliers and find highest value for normalisation
    # i = 0
    # zthresh = 1.9
    # for arzed, valzed in zip(arzcoupled, valzcoupled):
    #     if valzed[0] >= zthresh or valzed[1] >= zthresh or arzed[0] >= zthresh or arzed[1] >= zthresh or valzed[0] <= -zthresh or valzed[1] <= -zthresh or arzed[0] <= -zthresh or arzed[1] <= -zthresh:
    #         del vallist[i]
    #         del arlist[i]
    #         del namelist[i]
    #         print("removed one")
    #         i -= 1
    #     else:
    #         if abs(vallist[i][0]) > furthestval:
    #             furthestval = abs(vallist[i][0])
    #         elif abs(vallist[i][1]) > furthestval:
    #             furthestval = abs(vallist[i][1])
    #         if abs(arlist[i][0]) > furthestar:
    #             furthestar = abs(arlist[i][0])
    #         elif abs(arlist[i][1]) > furthestar:
    #             furthestar = abs(arlist[i][1])
    #     i += 1
    # print("Furthest arousal", furthestar)
    # print("Furthes valence", furthestval)

    # print("Valence + Z list: ")
    # for zed in arzcoupled:
    #     print(zed)

    # print("Valence + Z list: ")
    # for valence, zed in zip(vallist, valzcoupled):
    #     print(valence, " -- Z: ",zed)
    # print("Arousal + Z list: ")
    # for valence, zed in zip(arlist, arzcoupled):
    #     print(valence, " -- Z: ", zed)

    # Do quartile value analysis
    valQ1 = np.percentile(sorted(valraw), 25)
    valQ3 = np.percentile(sorted(valraw), 75)
    valIQR = valQ3 - valQ1
    valUpper = valQ3 + (1.5 * valIQR)
    valLower = valQ1 - (1.5 * valIQR)

    arQ1 = np.percentile(sorted(arraw), 25)
    arQ3 = np.percentile(sorted(arraw), 75)
    arIQR = arQ3 - arQ1
    arUpper = arQ3 + (1.5 * arIQR)
    arLower = arQ1 - (1.5 * arIQR)

    # Remove outliers and find highest value for normalisation
    i = 0
    filteredval = []
    filteredar = []
    filterednames = []
    for val, ar, name in zip(vallist, arlist, namelist):
        if valLower < val[0] > valUpper or valLower < val[1] > valUpper or arLower < ar[0] > arUpper or arLower < ar[1] > arUpper:
            if verbose:
                print("removed one")
            i -= 1
        else:
            filteredval.append(val)
            filteredar.append(ar)
            filterednames.append(name)
            if abs(val[0]) > furthestval:
                furthestval = abs(val[0])
            elif abs(val[1]) > furthestval:
                furthestval = abs(val[1])
            if abs(ar[0]) > furthestar:
                furthestar = abs(ar[0])
            elif abs(ar[1]) > furthestar:
                furthestar = abs(ar[1])
        i += 1
    print("Furthest arousal", furthestar)
    print("Furthes valence", furthestval)

    normvallist = []
    normarlist = []

    vectarlist = []
    vectvallist = []
    catlist = []

    # Normalise valence/arousal and add both values for vector
    for valentry, arentry in zip(filteredval, filteredar):
        arnormentry = []
        valnormentry = []

        for valvalue, arvalue in (valentry, arentry):
            valnormentry.append(valvalue / furthestval)
            arnormentry.append(arvalue / furthestar)

        normvallist.append(valnormentry)
        normarlist.append(arnormentry)

        # Convert to vector and then category
        vectval = ((valnormentry[0] - valnormentry[1]) + 2) / 4
        vectar = ((arnormentry[0] - arnormentry[1]) + 2) / 4

        vectvallist.append(vectval)
        vectarlist.append(vectar)

        # For every vertical step (arousal) times 4 to count for whole row of 5 (it wil always stay below the actual one because of Math.floor)\
        # Then add the horizontal few 
        category = (math.floor(vectar * 5) * 5 + math.floor(vectval * 5))
        if category > 25:
            category = 25
        catlist.append(category)

    # Print lists
    # print("Valence list: ")
    # for valence in normvallist:
    #     print(valence)
    # print("Arousal list: ")
    # for arousal in normarlist:
    #     print(arousal)
    # print("Category list: ")
    # for category in catlist:
    #     print(category)

    print("==============")
    os.makedirs(args.output_path, exist_ok=True)
    if verbose:
        print("Analysis information:")
        print("Namelist length:", len(filterednames), " - Catlist length:", len(catlist), " - Vectvallist length:", len(vectvallist), " - Vectarlist length:", len(vectarlist))
        print("==============")
        HiVal = [item[0] for item in vallist]
        LoVal = [item[1] for item in vallist]
        HiAr = [item[0] for item in arlist]
        LoAr = [item[1] for item in arlist]
        print("Namelist length:", len(namelist), " - Catlist length:", len(catlist), " - Vallist length:", len(vallist), " - Arlist length:", len(arlist))
        rawoutputfile = os.path.join(args.output_path, "analysisraw.csv")
        rawdict = {'Filename': namelist, 'HiVal': HiVal, 'LoVal': LoVal, 'VQ1': valQ1, 'VQ3': valQ3, 'HiAr': HiAr, 'LoAr': LoAr, 'AQ1': arQ1, 'AQ3': arQ3}
        rawcsv = pd.DataFrame(rawdict)
        rawcsv.to_csv(rawoutputfile)
    outputfile = os.path.join(args.output_path, "analysis.csv")
    print("Saving to csv to: ", outputfile)
    dict = {'Filename': filterednames, 'Category': catlist, 'Valence vector': vectvallist, 'Arousal vector': vectarlist}
        
    csv = pd.DataFrame(dict)
    csv.to_csv(outputfile)

    print("Done :)")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--types", default="midi_like", type=str, choices=["midi_like", "remi"])
    parser.add_argument("--task", default="ar_va", type=str, choices=["ar_va", "arousal", "valence"])
    parser.add_argument('--cuda', default='cuda:0', type=str)
    parser.add_argument("--input_path", required=True, type=str)
    parser.add_argument('--output_path', default='./', type=str)
    parser.add_argument('--verbose', default=True, type=bool)
    args = parser.parse_args()
    batch(args)
