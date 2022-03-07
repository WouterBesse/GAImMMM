#from pathlib import Path
import mido
from mido import MidiFile, MidiTrack

# Import wanted midi file
input_midi = MidiFile("H:/Projects/2021-2022/EMOPIA_cls-main/empty dataset/00e6d5785a99002bc645765b8851e9ff.mid", clip=True)
# Create output midi file
output_midi = MidiFile()
# Copy time metrics between both files
output_midi.ticks_per_beat = input_midi.ticks_per_beat

for original_track in input_midi.tracks:
    new_track = MidiTrack()

    for msg in original_track:
        if msg.type in ['note_on', 'note_off', 'control_change']:
            origin_note = msg.note

            if origin_note in note_map:
                new_track.append(
                    msg.copy(note=note_map[origin_note]['target_note']))
                print(note_map[origin_note]['description'])
            else:
                print("Origin note", origin_note, "not mapped")
        else:
            print(msg.type)
            new_track.append(msg)

    # MIDI files are multitrack. Here we append
    # the new track with mapped notes to the output file
    output_midi.tracks.append(new_track)

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
