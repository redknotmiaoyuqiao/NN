# -*- coding: UTF-8 -*-
# Midi Function

import os
import subprocess

def convertMidi2Mp3():
    input_file = "output.mid"
    output_file = "output.mp3"

    assert os.path.exists(input_file)

    print("Convert %s to mp3" % input_file)

    command = "timidity {} -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 256k {}".format(input_file,output_file)

    subprocess.call(command,shell=True)

    print("Convert file is %s" % output_file)


if __name__ == "__main__":
    convertMidi2Mp3()
