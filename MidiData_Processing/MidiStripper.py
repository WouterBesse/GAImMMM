# from pathlib import Path
from linecache import clearcache
from math import ceil
import os.path
import os
import time

from mido import MidiFile, MidiTrack, merge_tracks, MetaMessage
from argparse import ArgumentError, ArgumentParser, Namespace
import multiprocessing as mp

succeeded = 0
failed = 0

# clearConsole script acquired from https://www.delftstack.com/howto/python/python-clear-console/
def clearConsole():
    """
    Remove all text from the console.
    """
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)

# getListOfFiles script acquired from https://thispointer.com/python-how-to-get-list-of-files-in-directory-and-sub-directories/
def getListOfFiles(dirName):
    """
    Create a list of file and sub directories

    :param dirName: str: subdirectory to scan files of
    :return: list of full path for each file and directory within dirName
    """
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
    """
    Carry out the given action (remove percussion; remove silence; split into
    clips; change beat resolution; all of these) on every file in the given
    input path.
    Works on one CPU only; called when --parallel is set to 0

    :param args: The command-line arguments
    """

    # counting variable
    global succeeded
    global failed

    
    print("Getting list of files :)")
    listOfFiles = getListOfFiles(args.input_path)

    # Quick information on what you are doing
    if args.clear_csl == 1:
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
        resultaten = []
        result = trymodify(args.input_path, file, len(resultaten), 
            args.split_time, args.clear_csl, args.output_path, args.action)
        succeeded += result[0]
        failed += result[1]
        resultaten.append(result)
        if args.clear_csl == 1:
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
    """
    Carry out the given action (remove percussion; remove silence; split into
    clips; change beat resolution; all of these) on every file in the given
    input path.
    Works on any number of CPUs; called when --parallel is set to 1.

    :param args: The command-line arguments
    """

    # counting variable
    global succeeded
    global failed

    
    print("Getting list of files :)")
    print("CPU count", mp.cpu_count())
    listOfFiles = getListOfFiles(args.input_path)

    # Quick information on what you are doing
    if args.clear_csl == 1:
        clearConsole()
    print("Amount of files to process =", len(listOfFiles))

    pool = mp.Pool(mp.cpu_count())

    succeeded = 123
    
    global c
    c = 0
    resultaten = []
    tt = 0
    resultaten = pool.starmap_async(trymodify,
        [
            (args.input_path, file, len(resultaten), args.split_time, 
            args.clear_csl, args.output_path, args.action) 
            for file in listOfFiles
        ]
        ).get()

    pool.close
    pool.join
    print("Done <3")

def trymodify(inputpath, filename, succeed, split_time, clear_csl, 
              outputpath = "./newdataset/", action = "rm-perc"):
    """
    Carry out the given action on a file and save it.

    :param inputpath: str: full path to the directory holding filename
    :param filename: str: name of the file to process
    :param succeed: int: number of files successfully converted so far
    :param split_time: int: maximum playback time of a split clip in seconds
    :clear_csl: 0 or 1: whether of not to clear the console after each file
    :outputpath: str: where to save the processed files
    :action: str: given argument saying which action to perform (remove 
        percussion; remove silence; split into clips; change beat resolution; 
        all of these)

    :return: a list with length 2 containing the new number of succeeded files
        and the number of failed files, respectively
    """
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
            elif action == "split":
                split_midi(full_path, outputpath, split_time)
            elif action == "res":
                change_resolution(full_path, outputpath, 480)
            elif action == "all":
                nodrums = remove_drums(full_path, outputpath, 1)
                # Return a full track as a list of x second midi files
                splitup = split_midi(full_path, outputpath, split_time, nodrums, 1)

                curfile = 0

                # Remove silence from the list of midi files
                for split in splitup:
                    splitnosilence = remove_silence(full_path, outputpath, split, 1)
                    outname = os.path.basename(full_path)
                    outputfile = os.path.join(outputpath, str(curfile) + "_" + outname)
                    splitnosilence.save(outputfile)
                    curfile += 1
            else:
                raise ArgumentError("Invalid argument for --action:", action)
            print("Succeeded :D")
            succeed = 1
        except Exception as e:
            print("Failed :( error:", e)
            fail = 1
            pass
    if clear_csl == 1:
        clearConsole()
    print("Taken time: ", time.time() - st)
    return [succeed, fail]

def remove_drums(inputpath, outputpath, isall = 0):
    """
    Remove the percussion track and some other unnecessary tracks from this file
    and save it somewhere else.

    :param inputpath: str: full path to the file to process
    :param outputpath: str: relative path where to save the processed file
    :param isall: 0 or 1: whether the action that's being performed is to do
        all the actions; this fact influences the way this file should be saved

    :return: mido.MidiFile: the output midi file after processing
    """

    # Get the name of the file and make a file place for it
    filename = os.path.basename(inputpath)
    if isall == 0:       
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
                if msg.channel < 5:
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

    if isall == 0:
        output_midi.save(outputfile)
    return output_midi

def remove_silence(inputpath, outputpath, input_midi = MidiFile(), isall = 0):
    """
    Remove a few seconds of silence from the beginning of a midi file
    and save it somewhere else.

    :param inputpath: str: full path to the file to process
    :param outputpath: str: relative path where to save the processed file
    :param input_midi: mido.MidiFile: midi object for the file to process
    :param isall: 0 or 1: whether the action that's being performed is to do
        all the actions; this fact influences the way this file should be saved

    :return: mido.MidiFile: the output midi file after processing
    """
    # Get the name of the file and make a file place for it
    if isall == 0:
        filename = os.path.basename(inputpath)
        outputfile = os.path.join(outputpath, filename)

        # Import wanted midi file
        input_midi = MidiFile(inputpath, clip=True)
    # Create output midi file
    output_midi = MidiFile()
    # Copy time metrics between both files
    output_midi.ticks_per_beat = input_midi.ticks_per_beat

    # arbitrary for now; this will be the song's start time
    first_time = 10000 
    # These are the message types whose time property we might want to change
    timed_types = {'note_on', 'note_off'}

    for original_track in input_midi.tracks:
        # Check this track's messages until you find a note
        # Store the first note's timestamp in first_time
        for msg in original_track:
            if msg.type in timed_types:
                msg_time = msg.time
                if msg_time < first_time:
                    first_time = msg_time
                break
        print('First start time for this file:', first_time)
        
        # Subtract the start time from the first note
        # Also make time 0 for all other misc notes that come before that
        # Since time is stored relatively,
        # the song should now begin start_time ticks earlier
        for msg in original_track:
            if msg.type in timed_types:
                msg.time -= first_time
                if msg.time < 0:
                    msg.time = 0
                break
            else:
                msg.time = 0

        # Make sure the song ends at end_time
        for msg in reversed(original_track):
            # The last note_off should be at end_time
            if msg.type not in timed_types:
                msg.time == 0;

    if isall == 0:
        input_midi.save(outputfile)
    return input_midi

def get_tempos(input_track):
    """
    Create a map of all tempo changes that occur within a given MIDI track.

    :param input_track: mido.MidiTrack: the track to analyze

    :return: list containing 3 lists, whose indices connect to each other:
        * the time the tempo change occurs at
        * the new tempo
        * the message that this tempo change occurs at, with time set to 0
    """
    tempo_map = [[],[],[]]
    note_time = 0

    # Iterate through all messages in track
    for msg in input_track:
        # Add the time value of the current note, so we keep counting up
        note_time += msg.time

        if msg.type == 'set_tempo':
            # Making a vector with two other vectors, 
            # vector 1 = on which tick a new tempo is set, 
            # vector 2 = what that tempo is
            tempo_map[0].append(note_time)
            tempo_map[1].append(msg.tempo)
            tempo_map[2].append(msg.copy(time=0))
    
    return tempo_map

def save_midi(curtrack, destination, resolution, isall = 0):
    """
    Save a MidiTrack as a midi file.

    :param curtrack: mido.MidiTrack: the track to be saved
    :param destination: str: path to save the file to
    :param resolution: int: ticks per beat for the new midi file
    :param isall: 0 or 1: whether the action that's being performed is to do
        all the actions; this fact influences the way this file should be saved

    :return: if isall is set to 1, return a MidiFile object instead of saving it
    """
    output_midi = MidiFile()
    # Copy time metrics
    output_midi.ticks_per_beat = resolution
    # Save track
    output_midi.tracks.append(curtrack)
    if isall == 0:
        output_midi.save(destination)
    else:
        return output_midi

def split_midi(inputpath, outputpath, duration, input_midi = MidiFile(), isall = 0):
    """
    Split a midi file into clips with a max playback time
    and save them somewhere else.

    :param inputpath: str: full path to the file to process
        (used when isall == 0)
    :param outputpath: str: relative path where to save the processed file
    :param duration: int: maximum playback time of a split clip in seconds
    :param input_midi: mido.MidiFile: midi object for the file to process
        (used when isall == 1)
    :param isall: 0 or 1: whether the action that's being performed is to do
        all the actions; this fact influences the way this file should be saved

    :return: [mido.MidiFile]: list of midi files, one for each clip. This is
        necessary if we still need to remove silence.
    """
    curfile = 0

    if isall == 0:
        # Import midi file
        input_midi = MidiFile(inputpath, clip=True) 
    
    new_track = MidiTrack()
    returnablemidis = []

    mergedtracks = merge_tracks(input_midi.tracks)

    curtempo = 0
    tempo_map = get_tempos(mergedtracks)
    resolution = input_midi.ticks_per_beat

    time_per_tick = tempo_map[1][curtempo]/resolution

    note_time = 0
    time_elapsed = 0

    for i, msg in enumerate(mergedtracks):
        # Keep count of how many ticks have passed
        note_time += msg.time

        # If we passed a tempo change, add the time elapsed before the change 
        # and after the change to the time_elapsed and count that we are on a 
        # new tempo index
        try:
            if note_time >= tempo_map[0][curtempo + 1]:
                curtempo += 1
                ticks_after_change = note_time - tempo_map[0][curtempo]
                time_before_change = (msg.time - ticks_after_change) * time_per_tick
                time_per_tick = tempo_map[1][curtempo]/resolution
                time_after_change = ticks_after_change * time_per_tick
                
                time_elapsed += time_before_change + time_after_change
            else:
                time_elapsed += msg.time*time_per_tick
        except:
            time_elapsed += msg.time*time_per_tick
            pass
        # If the time counter is above duration in seconds and the current message is not at the same time as the previous one (that would be nice to keep)
        if time_elapsed > duration*(10**6) and msg.time != 0:
            print('Saving {0}\n   after {1} messages and {2} milliseconds'.format(inputpath, i, time_elapsed))
            # Create directory and save track
            filename = os.path.basename(inputpath)
            output_dir = os.path.join(outputpath, str(curfile) + "_" + filename)
            if isall == 0:
                save_midi(new_track, output_dir, resolution)
            else:
                returnablemidis.append(save_midi(new_track, output_dir, resolution, 1))
            # Reset time counter and create new midi track
            time_elapsed = 0
            curfile += 1
            new_track = MidiTrack()
            new_track.append(tempo_map[2][curtempo])
            new_track.append(msg.copy(time=0))
        # Also save the song at the end of the file if it's at least 5 seconds
        elif time_elapsed > 5000000 and i == len(mergedtracks) - 1:
            print('Saving {0}\n   at EOF after {1} messages and {2} milliseconds'.format(inputpath, i, time_elapsed))
            # Create directory and save track
            filename = os.path.basename(inputpath)
            output_dir = os.path.join(outputpath, str(curfile) + "_" + filename)
            if isall == 0:
                save_midi(new_track, output_dir, resolution)
            else:
                returnablemidis.append(save_midi(new_track, output_dir, resolution, 1))
        else:
            # If you haven't reached the max duration yet, add this note to the track
            new_track.append(msg.copy())
    
    # If we still need to remove silence, return all midi messages as a MidiFile()
    return returnablemidis

def change_resolution(inputpath, outputpath, new_resolution):
    """
    Change a midi file's "ticks per beat" resolution
    and save it somewhere else.

    :param inputpath: str: full path to the file to process
    :param outputpath: str: relative path where to save the processed file
    :param new_resolution: int: new resolution in ticks per beat

    :return: mido.MidiFile: the output midi file after processing
    """
    filename = os.path.basename(inputpath)
    # Import midi file
    input_midi = MidiFile(inputpath, clip=True) 
    
    
    new_track = MidiTrack()

    old_resolution = input_midi.ticks_per_beat
    time_multiplier = new_resolution / old_resolution

    outputfile = os.path.join(outputpath, filename)
    output_midi = MidiFile()
    output_midi.ticks_per_beat = new_resolution

    for track in input_midi.tracks:
        new_track = MidiTrack()
        for msg in track:
            new_track.append(msg.copy(time=int(msg.time * time_multiplier)))
        output_midi.tracks.append(new_track)

    output_midi.save(outputfile)
    print(output_midi)
    return output_midi

if __name__ == '__main__':
    args_description = """--input_path: folder of midi files to process
--output_path: where to save the new midi files
--parallel: either 0 or 1; whether to use multiple CPUs
--action: remove percussion; remove silence; split into clips; change beat resolution or all of these
--split_time: maximum playback time for the new midi clips after splitting in seconds; default is 40 seconds
--clear_csl: either 0 or 1; whether to clear the console after each processed file"""
    parser = ArgumentParser(description = args_description)

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", default="./newdataset/", type=str)
    parser.add_argument("--parallel", default=1, type=int)
    parser.add_argument("--action", default="rm-perc", type=str,
        choices=['rm-perc', 'rm-silence', 'split', 'res', 'all'])
    parser.add_argument("--split_time", default=40, type=int) # In seconds
    parser.add_argument("--clear_csl", default=1, type=int)

    args = parser.parse_args()
    if args.parallel == 0:
        evaluate(args)
    else:
        evaluate_par(args)
