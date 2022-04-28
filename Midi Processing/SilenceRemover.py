# from pathlib import Path
import os.path
import os
import time

from mido import MidiFile, MidiTrack
from argparse import ArgumentParser, Namespace
import multiprocessing as mp

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
    print("Amount of files to process =", len(listOfFiles))

    resultaten = []
    t = 0
    tt = 0
    ttt = 0
    for file in listOfFiles:
        ts = time.time()
        print("Current file =", file)
        print("Files succeeded =", succeeded)
        print("Files failed =", failed)
        result = tryremove(args.input_path, file, args.output_path)
        succeeded += result[0]
        failed += result[1]
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
    resultaten = pool.starmap_async(tryremove, [(args.input_path, file, len(resultaten), args.output_path) for file in listOfFiles]).get()
    print("appelflap")
    while c < len(listOfFiles):
        ts = time.time()
        n = 0
        result = []
        
        print("Current file =", listOfFiles[c])
        print("Files succeeded =", succeeded)
        print("Files failed =", failed)
        
        n += 1
        c += 1
        succeeded += result[0]
        failed += result[1]
        resultaten.append(result)
        clearConsole()
        print("Time in parallel for 16: ", tt)
        tt = time.time()-ts
    pool.close
    pool.join
    print("Done <3")

    # for file in listOfFiles:
# inputpath, filename, outputpath = "./newdataset/"
def tryremove(inputpath, filename, succeed, outputpath = "./newdataset/"):
    st = time.time()
    
    
    fail = 0
    filetosave = []

    full_path = os.path.join(inputpath, filename)
    if os.path.isfile(full_path):
        remove_silence(full_path, outputpath)
        try:
            remove_silence(full_path, outputpath)
            # print("Succeeded")
        except Exception:
            print("Failed :(")
            pass
    clearConsole()
    print("Taken time: ", time.time() - st)
    # print("Files done :", succeeded)
    return (fail, succeed)


def remove_silence(input_path, output_path):
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
        print('new track')
        found_first_note = False

        for msg in original_track:

            if (not found_first_note) and msg.type == 'note_on':
                msg_time = original_track[0].time
                if msg_time < first_time:
                    first_time = msg_time
                
                found_first_note = True

        # Mido can calculate end time; failsafe with final note_off message
        for msg in reversed(original_track):
            if original_track[-2].type == 'note_off':
                end_time = msg.time
            else:
                end_time = input_midi.length
    
    print('First start time for this file:', first_time)

    # Subtract the start time from all messages;
    # song should now begin start_time ticks earlier
    for original_track in input_midi.tracks:
        for msg in original_track:
            msg.dict()['time'] -= first_time
            if msg.dict()['time'] < 0:
                msg.dict()['time'] = 0

    # TODO: actually make end_time the end time

    return [output_midi, outputfile]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", default="./newdataset/", type=str)
    parser.add_argument("--parallel", default=1, type=int)
    args = parser.parse_args()
    if args.parallel == 0:
        evaluate(args)
    else:
        evaluate_par(args)