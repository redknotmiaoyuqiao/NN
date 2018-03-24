# -*- coding: UTF-8 -*-
# Midi Function

import os
import subprocess

from music21 import converter, instrument, note, chord, stream
import pickle
import glob

def convertMidi2Mp3():
    input_file = "output.mid"
    output_file = "output.mp3"

    assert os.path.exists(input_file)

    print("Convert %s to mp3" % input_file)

    command = "timidity {} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 512k {}".format(input_file,output_file)

    subprocess.call(command,shell=True)

    print("Convert file is %s" % output_file)





def get_notes():
    notes = []
    i = 0
    for file in glob.glob('data/music_midi/*.mid'):
        print("Reader %s" % file)
        i += 1
        stream = converter.parse(file)

        parts = instrument.partitionByInstrument(stream)
        if parts:
            notes_to_parse = parts.parts[0].recurse()
        else:
            notes_to_parse = stream.flat.notes

        for element in notes_to_parse:
            #
            if isinstance(element,note.Note):
                notes.append(str(element.pitch))
            if isinstance(element,chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes','wb') as filepath:
        pickle.dump(notes, filepath)

    return notes





def create_music(prediction):
    offset = 0
    output_notes = []

    for data in prediction:
        if ('.' in data) or data.isdigit:
            notes_in_chord = data.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()

                notes.append(new_note)
            new_chord = chord.Chord(notes)

            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            new_note = note.Note(data)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5


    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi',fp='output.mid')

    convertMidi2Mp3()
