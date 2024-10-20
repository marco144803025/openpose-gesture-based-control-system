
import sys
import cv2
import os
from sys import platform
import argparse

try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print('dir_path: {}'.format(dir_path))
    try:
        if platform == "win32":
            #
            sys.path.append(os.path.join(dir_path, '..','lib'))
            print(os.path.join(dir_path, '..','lib'))
            os.environ['PATH']  = os.environ['PATH']  + os.path.join(dir_path, '..','lib') +';' + os.path.join(dir_path, '..','lib')
            import pyopenpose as op
        else:
            sys.path.append('../../python')
           
            print('error occured')
         
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder? error:')
        raise e
    
    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default=os.path.join(dir_path, '..','media','COCO_val2014_000000000192.jpg'), help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = os.path.join(dir_path, '..','models')
    params["video"]= os.path.join(dir_path, '..','media','other','video (720p).mp4')
    params["hand"]=1
    params['net_resolution']='-1x256'    
    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item


    # op.init_argv(args[1])
    # oppython = op.OpenposePython()
    opWrapper = op.WrapperPython(op.ThreadManagerMode.Synchronous)
    opWrapper.configure(params)
    opWrapper.execute()
except Exception as e:
    print(e)
    sys.exit(-1)
