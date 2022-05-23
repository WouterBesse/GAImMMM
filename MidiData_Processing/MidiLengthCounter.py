# from pathlib import Path
from linecache import clearcache
from math import ceil
import os.path
import os
import time
from argparse import ArgumentError, ArgumentParser, Namespace
import pandas as pd
import numpy as np
import sys

from mido import MidiFile, MidiTrack, merge_tracks, MetaMessage

################################################################################
# config
################################################################################

parser = ArgumentParser()
parser.add_argument("--data_root", default='J:/Projects/2021-2022/GAImMMM Data/Analyzed', type=str)
parser.add_argument("--output_dir", default="./", type=str)
parser.add_argument("--namelist", default="J:/Projects/2021-2022/GAImMMM Data/corpus/fixed", type=str)
args = parser.parse_args()




# clearConsole script acquired from https://www.delftstack.com/howto/python/python-clear-console/
def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)

# getListOfFiles script acquired from https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
def getListOfFiles(dirName, title = 'None'):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    start_progress(title)
    i = 0
    for entry in listOfFile:
        progress(i / len(listOfFile))
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
        i += 1
    end_progress()
    return allFiles

def start_progress(title):
    global progress_x
    sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
    sys.stdout.flush()
    progress_x = 0

def progress(x):
    global progress_x
    x = int(x * 40 // 100)
    sys.stdout.write("#" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x

def end_progress():
    sys.stdout.write("#" * (40 - progress_x) + "]\n")
    sys.stdout.flush()

print("Getting list of files :)")
listOfFiles = getListOfFiles(args.data_root, "List of Files")
print("Getting list of names")
listOfNames = getListOfFiles(args.namelist, "List of Names")
basenamelist = []
for name in listOfNames:
    size = len(os.path.basename(name))
    basenamelist.append(os.path.basename(name)[:size - 4])

songlengths = [[],[]]
i = 1

start_progress("Voortgang files tellen")

for file in listOfFiles:
    if os.path.basename(file) in basenamelist:
        inputfile = MidiFile(file, clip=True)
        length = inputfile.length
        songlengths[0].append(length)
        songlengths[1].append(os.path.basename(file))
        i += 1
    # clearConsole()
    progress(i/len(listOfFiles))
    # print (i, "/", len(listOfFiles), " Done")

end_progress()
os.makedirs(args.output_dir, exist_ok=True)
lengthoutputfile = os.path.join(args.output_dir, "lengths.csv")
rawdict = {'Lengths': songlengths[0], 'Names': songlengths[1]}
rawcsv = pd.DataFrame(rawdict)
rawcsv.to_csv(lengthoutputfile)