# from pathlib import Path
from math import ceil
import os.path
import os
import time

from mido import MidiFile, MidiTrack
from argparse import ArgumentError, ArgumentParser, Namespace
import multiprocessing as mp

succeeded = 0
failed = 0

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

def evaluate(args):
    # counting variable
    global succeeded
    global failed

    
    print("Getting list of files :)")
    listOfFiles = getListOfFiles(args.input_path)

    # Quick information on what you are doing
    clearConsole()
    print("Number of files to process =", len(listOfFiles))
    print("Action:", args.action)

    resultaten = []
    t = 0
    tt = 0
    ttt = 0
    for file in listOfFiles:
        ts = time.time()
        print("Current file =", file)
        print("Files succeeded =", succeeded)
        print("Files failed =", failed)
        result = tryremove(args.input_path, file, args.output_path, args.action, )
        failed += result[0]
        succeeded += result[1]
        resultaten.append(result)
        clearConsole()
        tt += time.time()-ts
        if t == 15:
            t = 0
            ttt = tt
            tt = 0
        t += 1
        print("Time in serial for ", t + 1,":", ttt)

    print("Done <3")

def evaluate_par(args):
    # counting variable
    global succeeded
    global failed

    
    print("Getting list of files :)")
    print("CPU count", mp.cpu_count())
    listOfFiles = getListOfFiles(args.input_path)

    # Quick information on what you are doing
    clearConsole()
    print("Amount of files to process =", len(listOfFiles))

    pool = mp.Pool(mp.cpu_count())

    succeeded = 123
    
    global c
    c = 0
    resultaten = []
    tt = 0
    resultaten = pool.starmap_async(tryremove, [(args.input_path, file, args.output_path, args.action) for file in listOfFiles]).get()
    print("appelflap")
    # while c < len(listOfFiles):
    #     ts = time.time()
    #     n = 0
    #     result = []
        
    #     print("Current file =", listOfFiles[c])
    #     print("Files succeeded =", succeeded)
    #     print("Files failed =", failed)
        
    #     n += 1
    #     c += 1
    #     succeeded += result[0]
    #     failed += result[1]
    #     resultaten.append(result)
    #     clearConsole()
    #     print("Time in parallel for 16: ", tt)
    #     tt = time.time()-ts
    pool.close
    pool.join
    print("Done <3")

    # for file in listOfFiles:
# inputpath, filename, outputpath = "./newdataset/"
def tryremove(inputpath, filename, outputpath = "./newdataset/", action = "rm-perc"):
    st = time.time()
    
    fail = 0
    succeed = 0
    filetosave = []

    full_path = os.path.join(inputpath, filename)
    if os.path.isfile(full_path):
        try:
            if action == "rm-perc":
                remove_drums(full_path, outputpath)
            elif action == "rm-silence":
                remove_silence(full_path, outputpath)
            else:
                raise ArgumentError("Invalid argument for --action:", action)
            print("Succeeded :D")
            succeed = 1
        except Exception as e:
            print("Failed :( error:", e.message)
            fail = 1
            pass
    clearConsole()
    print("Taken time: ", time.time() - st)
    succeed = 1
    return (fail, succeed)

def remove_drums(inputpath, outputpath):
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
                if msg.channel != 8:
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
    return [output_midi, outputfile]

def remove_silence(inputpath, outputpath):
    # Get the name of the file and make a file place for it
    filename = os.path.basename(inputpath)
    outputfile = os.path.join(outputpath, filename)

    # Import wanted midi file
    input_midi = MidiFile(inputpath, clip=True)
    # Create output midi file
    output_midi = MidiFile()
    # Copy time metrics between both files
    output_midi.ticks_per_beat = input_midi.ticks_per_beat

    first_time = 10000 # arbitrary for now; this will be the song's start time

    # Find the timestamp for the first note_on message in all tracks
    # and the timestamp for the last note
    for original_track in input_midi.tracks:
        # Store the first note's timestamp in first_time
        for msg in original_track:
            if msg.type == 'note_on':
                msg_time = msg.time
                if msg_time < first_time:
                    first_time = msg_time
                break
        
        # Mido can calculate end time; failsafe with final note_off message
        if original_track[-2].type == 'note_off':
            end_time = original_track[-2].time
            print('End time decided by note_off message:', end_time)
        else:
            end_time = ceil(input_midi.length)
            print('End time decided by mido length property:', end_time)

    print('First start time for this file:', first_time)

    for original_track in input_midi.tracks:
        # Subtract the start time from the first note
        # Since time is stored relatively,
        # the song should now begin start_time ticks earlier
        for msg in original_track:
            if msg.type == 'note_on':
                msg.time -= first_time
                if msg.time < 0:
                    msg.time = 0
                break

        # Make sure the song ends at end_time
        handled_msgs = 0
        for msg in reversed(original_track):
            # The last note_off should be at end_time
            if msg.type == 'note_off':
                if msg.time > end_time:
                    msg.time = end_time
                break

            # If this track seems to have no note_off messages,
            # just make sure the last message is directly at the end time
            elif msg.type == 'note_on' and handled_msgs > 3:
                original_track[-1].time = end_time
            
            handled_msgs += 1    

    input_midi.save(outputfile)
    return [output_midi, outputfile]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", default="./newdataset/", type=str)
    parser.add_argument("--parallel", default=1, type=int)
    parser.add_argument("--action", default="rm-perc", type=str,
        choices=['rm-perc', 'rm-silence'])
    args = parser.parse_args()
    if args.parallel == 0:
        evaluate(args)
    else:
        evaluate_par(args)
