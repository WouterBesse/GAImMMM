# from pathlib import Path
import mido
from mido import MidiFile, MidiTrack

# Import wanted midi file
input_midi = MidiFile("../../../EMOPIA_cls-main/empty dataset/00e6d5785a99002bc645765b8851e9ff.mid", clip=True)
# Create output midi file
output_midi = MidiFile()
# Copy time metrics between both files
output_midi.ticks_per_beat = input_midi.ticks_per_beat

for original_track in input_midi.tracks:
    new_track = MidiTrack()
    #print(original_track)
    note_time = 0
    for msg in original_track:

        note_time += msg.time

        if msg.type == 'note_on' or msg.type == 'note_off' or msg.type == 'control_change':
            if msg.channel != 9:
                new_track.append(msg.copy(type=msg.type, time=note_time))
                note_time = 0
        else:
            new_track.append(msg.copy(type=msg.type, time=note_time))
            note_time=0

    # MIDI files are multitrack. Here we append
    # the new track with mapped notes to the output file
    output_midi.tracks.append(new_track)

print(output_midi)
output_midi.save('./Murundu-remap.mid')

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
