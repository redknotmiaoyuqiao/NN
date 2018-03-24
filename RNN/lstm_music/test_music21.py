# -*- coding: UTF-8 -*-

from music21 import converter, instrument

def print_notes():
    stream = converter.parse("output.mid")
    parts = instrument.partitionByInstrument(stream)
    if parts:
        notes = parts.parts[0].recurse()
    else:
        notes = stream.flat.notes
    
    for element in notes:
        print(str(element))


if __name__ == "__main__":
    print_notes()
