# from pathlib import Path
import os.path
import os
import time

from mido import MidiFile, MidiTrack
from argparse import ArgumentParser, Namespace

# clearConsole script acquired from https://www.delftstack.com/howto/python/python-clear-console/
def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)

clearConsole()

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

def evaluate(args):
    # counting variable
    succeeded = 0
    failed = 0

    listOfFiles = getListOfFiles(args.input_path)
    # Quick information on what you are doing
    clearConsole()
    print("Amount of files to process =", len(listOfFiles))
    time.sleep(5)

    for file in listOfFiles:
        # Display, nice to see what's happening
        clearConsole()
        print("Files succeeded =", succeeded)
        print("Files failed =", failed)
        print("Current file =", file)

        full_path = os.path.join(args.input_path, file)
        if os.path.isfile(full_path):
            try:
                remove_drums(full_path, args.output_path)
                succeeded += 1
            except Exception:
                failed += 1
                pass

def remove_drums(inputpath, outputpath = "./newdataset/"):
    # Get the name of the file and make a file place for it
    filename = os.path.basename(inputpath)
    outputfile = os.path.join(outputpath, filename)

    # Import wanted midi file
    input_midi = MidiFile(inputpath, clip=True)
    # Create output midi file
    output_midi = MidiFile()
    # Copy time metrics between both files
    output_midi.ticks_per_beat = input_midi.ticks_per_beat

    for original_track in input_midi.tracks:
        new_track = MidiTrack()

        note_time = 0
        # Iterate through all messages in track
        for msg in original_track:
            # Add the time value of the current note
            # If previous note wasn't channel 9, this will be 0
            # If it was it will be added to the previous note
            note_time += msg.time

            # Only notes of this type have channel, so only check it for them
            if msg.type in ['note_on', 'note_off', 'control_change']:
                # Only add note if channel is not 9
                if msg.channel != 9:
                    new_track.append(msg.copy(type=msg.type, time=note_time, channel=2))
                    # Reset note time
                    note_time = 0
            else:
                # Always add all messages of other types
                new_track.append(msg.copy(type=msg.type, time=note_time))
                # Reset note time
                note_time=0

        # MIDI files are multitrack. Here we append
        # the new track with mapped notes to the output file
        output_midi.tracks.append(new_track)

    output_midi.save(outputfile)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", default="./newdataset/", type=str)
    args = parser.parse_args()
    evaluate(args)
