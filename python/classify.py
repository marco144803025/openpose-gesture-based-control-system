import sys
import cv2
import os
from sys import platform
import argparse
import os
from tensorflow import keras
import numpy as np
from pynput.keyboard import Controller
import mouse
import pyautogui
import time

#setting control system's input
controller = Controller()
prevLeftInput=None
prevRightInput=None
prevLeftGesture=None
prevRightGesture=None
prevFrame=None
user_input=None
gesture_counter=0
def mouseclick():
    global rightPredictedClassNum
    global prevRightInput
    global gesture_counter
    # if prevRightInput==rightPredictedClassNum:
    #     gesture_counter+=1
    # # if rightPredictedClassNum in rightinput and prevRightInput!=rightPredictedClassNum:  #mouse buttons    
    # if rightPredictedClassNum in rightinput:
    #     if gesture_counter==2:
    #         mouse.click(rightinput[rightPredictedClassNum])
    # prevRightInput=rightPredictedClassNum
    if rightPredictedClassNum in rightinput and prevRightInput!=rightPredictedClassNum:  #mouse buttons    
            mouse.click(rightinput[rightPredictedClassNum])
            print('clicked:',rightinput[rightPredictedClassNum])
    prevRightInput=rightPredictedClassNum
    
def mousehold():
    global rightPredictedClassNum
    global prevLeftGesture
    global leftPredictedClassNum
    global prevRightGesture
    if user_input=='m':
        predictedClassNum=leftPredictedClassNum
        prevGesture=prevLeftGesture

    else:
        predictedClassNum=rightPredictedClassNum
        prevGesture=prevRightGesture
    print("prevgesture, predictedclassnum is:",prevGesture,predictedClassNum)
    if prevGesture in rightinput and prevGesture!=predictedClassNum:
        mouse.release(button=rightinput[prevGesture])
    if prevGesture!=predictedClassNum:  #mouse buttons
        if predictedClassNum in rightinput:
            print('pressed gesture', predictedClassNum,rightinput[predictedClassNum])
            mouse.hold(button=rightinput[predictedClassNum])
        if user_input=='m':
            prevLeftGesture=predictedClassNum
        else:
            prevRightGesture= predictedClassNum
            
def singleKeyboardClick():
    global prevLeftInput
    global leftPredictedClassNum
    if prevLeftInput != leftPredictedClassNum and leftPredictedClassNum in leftinput:
        controller.press(leftinput[leftPredictedClassNum])  # Press new input
        controller.release(leftinput[leftPredictedClassNum])
    prevLeftInput=leftPredictedClassNum


def continousKeyboardClick():
    global gesture_counter
    global prevLeftGesture
    global leftPredictedClassNum
    global prevFrame
    global prevLeftInput
    #continous click (need balance between misinput and balance)
    if gesture_counter == 1 and leftPredictedClassNum != 6:
        # Press new key and release previous key
        if prevLeftGesture is not None and prevLeftGesture in leftinput:
            controller.release(leftinput[prevLeftGesture])
        if leftPredictedClassNum in leftinput:
            controller.press(leftinput[leftPredictedClassNum]) # Press new key  
        prevLeftGesture=leftPredictedClassNum          
    elif leftPredictedClassNum == 6:
        if prevLeftGesture in leftinput:
            controller.release(leftinput[prevLeftGesture])  # Release previous key

    if prevFrame == leftPredictedClassNum:
        gesture_counter += 1 
    prevFrame = leftPredictedClassNum
    
    if prevLeftInput != leftPredictedClassNum and leftPredictedClassNum != 6 and prevLeftInput is not None:
        if prevLeftInput in leftinput:
            controller.release(leftinput[prevLeftInput])  # Release previous input
        if leftPredictedClassNum in leftinput:
            controller.press(leftinput[leftPredictedClassNum])  # Press new input
        prevLeftInput = leftPredictedClassNum  # Update the previous input
    elif leftPredictedClassNum == 6:
        if prevLeftInput in leftinput:
            controller.release(leftinput[prevLeftInput])  # Release previous input
        prevLeftInput = leftPredictedClassNum  # Update the previous input

def normalize(handKeypoints,handId):
    handKeypoints = np.array(handKeypoints)
    handKeypoints = handKeypoints[handId,0]
    outputArray= None
    lengthFingers = [
        np.sqrt(
            (handKeypoints[0, 0] - handKeypoints[i, 0]) ** 2
            + (handKeypoints[0, 1] - handKeypoints[i, 1]) ** 2
        )
        for i in [1, 5, 9, 13, 17]
    ]
    for i in range(3):  # Add length of other segments of each fingers
        for j in range(len(lengthFingers)):
            x = (
                handKeypoints[1 + j * 4 + i + 1, 0]
                - handKeypoints[1 + j * 4 + i, 0]
            )
            y = (
                handKeypoints[1 + j * 4 + i + 1, 1]
                - handKeypoints[1 + j * 4 + i, 1]
            )
            lengthFingers[j] += np.sqrt(x ** 2 + y ** 2)
            normMax = max(lengthFingers)

            handCenterX = handKeypoints.T[0].sum() / handKeypoints.shape[0]
            handCenterY = handKeypoints.T[1].sum() / handKeypoints.shape[0]
            outputArray = np.array(
                [
                    (handKeypoints.T[0] - handCenterX) / normMax,
                    -(handKeypoints.T[1] - handCenterY) / normMax,
                ]
            )
    return outputArray

def flatten(array):
    x_coordinates = list(array[0,:])
    y_coordinates = list(array[1,:])
    # Reshape to match the dimensions
    x_coordinates_reshaped = np.array(x_coordinates).reshape(21, 1)
    y_coordinates_reshaped = np.array(y_coordinates).reshape(21, 1)

    # Concatenate along columns (axis 1)
    flattenedArray = np.hstack((x_coordinates_reshaped, y_coordinates_reshaped))
    flattenedArray = np.ravel(flattenedArray)
    flattenedArray = flattenedArray.reshape((1, 42))
    return flattenedArray

predictedClass = {
    0: 'fist',
    1: 'four',
    2: 'ok',
    3: 'palm',
    4: 'tick',
    5: 'v',
    6: 'other',
}
leftinput = {
    1: 'w',
    2: 'b',
    3: 's',
    4: 'd',
    5: 'a',
}
rightinput = {
    3: 'right',
    4: 'left',
} 
# ['fist', 'four', 'ok', 'palm', 'tick', 'v', 'other']
count=0
totalfps=0
try:
# setup openpose
    dir_path = os.path.dirname(os.path.realpath(sys.executable)) #for exe file
    dir_path = os.path.dirname(os.path.realpath(__file__))

    try:
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(os.path.join(dir_path, '..','lib'))
            print(os.path.join(dir_path, '..','lib'))
            os.environ['PATH']  = os.environ['PATH']  + os.path.join(dir_path, '..','lib') +';' + os.path.join(dir_path, '..','lib')
            import pyopenpose as op # type: ignore
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
    params["hand"] = 1
    params["model_pose"] = "BODY_25"
    params["model_folder"] = os.path.join(dir_path, '..','models')
    params["number_people_max"]=1
    params["net_resolution"] = "-1x256"
    params["camera_resolution"] = "640x360"
    params["hand_net_resolution"] = "256x256"
    params["render_pose"]= 1

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

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()
    
#Loading prediction model
    rightModelPath = os.path.join(dir_path, '..', 'models','gesture','aug_7_right.h5')
    leftModelPath = os.path.join(dir_path, '..', 'models','gesture','aug_7_left.h5')
    rightModel = keras.models.load_model(rightModelPath)
    leftModel  = keras.models.load_model(leftModelPath)    

    endFlag1 = False
    while not endFlag1:
        print("Which clicking mode do you prefer? (c for continuous, s for single click, m for mouse only): ")
        user_input = input()
        if user_input == 's' or user_input == 'c' or user_input == 'm':
            endFlag1 = True
        else:
            print("Invalid input. Please try again.")
    endFlag2 = False
    # while not endFlag2:
    #     print("Do you want to map your input (y/n) ")
    #     mapinputprompt = input()
    #     if mapinputprompt == 'n':
    #         endFlag2 = True
    #     elif mapinputprompt == 'y':
    #         left1=input("What keypress for gesture "+predictedClass[0]+":")
    #         left2=input("What keypress for gesture "+predictedClass[1]+":")
    #         left3=input("What keypress for gesture "+predictedClass[2]+":")
    #         left4=input("What keypress for gesture "+predictedClass[3]+":")
    #         left5=input("What keypress for gesture "+predictedClass[4]+":")
    #         left6=input("What keypress for gesture "+predictedClass[5]+":")
    #         left7=input("What keypress for gesture "+predictedClass[6]+":")
    #         endFlag2 = True
    #     else:
    #         print("Invalid input. Please try again.")

#   setting FPS calculationc
    frame_counter = 0
    start_time = time.time()
    
# Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    id=700
    cap=cv2.VideoCapture(id)
    windowWidth=640
    windowHeight=480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  windowWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, windowHeight)
#setting font 
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    color = (255, 255, 255)  # Blue color in BGR format
    thickness = 2
    lefttext='nothing detected'
    righttext='nothing detected'
# run looping events
    while True:
        datum=op.Datum()
        ret, frame = cap.read()

        # processing openpose
        datum.cvInputData = frame
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        outPutFrame=datum.cvOutputData
         
        # Create a named windowed
        outputFrame = cv2.flip(outPutFrame, 1)
        cv2.namedWindow("Control System", cv2.WINDOW_NORMAL)
        #add screen overlay
        screenshot = pyautogui.screenshot()
        screenshot_cv = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        # screenshot_cv = cv2.flip(cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR),1)
        screenshot_resized = cv2.resize(screenshot_cv, (640, 480))
        alpha = 0.4  # Adjust the transparency level as desired
        # overlay = cv2.flip(cv2.addWeighted(outPutFrame, 1 - alpha, screenshot_resized, alpha, 0), 1)
        overlay = cv2.addWeighted(outPutFrame, 1 - alpha, screenshot_resized, alpha, 0)
        #calculating if hands are detected
        if datum.handKeypoints[0] is not None:
            leftHandAccuaracyScore = datum.handKeypoints[0].T[2].sum()
        # Rest of your code using leftHandAccuracyScore
        else:
            leftHandAccuaracyScore = 0

        if datum.handKeypoints[1] is not None:
            rightHandAccuaracyScore = datum.handKeypoints[1].T[2].sum()
        # Rest of your code using leftHandAccuracyScore
        else:
            rightHandAccuaracyScore = 0

        leftPredictedClass=None
        rightPredictedClass=None
        if time.time() - start_time > 0.5:
            if leftHandAccuaracyScore > 1 and leftHandAccuaracyScore!=None:
                # normalizing keypoint
                norm_left = normalize(datum.handKeypoints,0)
                # changing array shape
                reshaped_keypoints = flatten(norm_left)

                leftPrediction=leftModel.predict(reshaped_keypoints,verbose=0)
                #predicting
                leftPredictedClassNum=np.argmax(leftPrediction)
                leftPredictedClass = predictedClass[leftPredictedClassNum]
                #pressing corrsponding input
                if prevFrame is not None and prevFrame != leftPredictedClassNum:
                    gesture_counter = 0  # Reset counter if gesture changes
                if user_input=='s':
                # single click mode 
                    singleKeyboardClick()
                if user_input=='c':
                # continous click
                    continousKeyboardClick()
                if user_input=='m':
                    mousehold()
                if leftPredictedClass!=None:
                    lefttext=leftPredictedClass
                else:
                    lefttext = 'nothing detected'
             


        if rightHandAccuaracyScore > 1 and rightHandAccuaracyScore!=None:
            norm_right = normalize(datum.handKeypoints,1)
            reshaped_keypoints = flatten(norm_right)
            rightPrediction=rightModel.predict(reshaped_keypoints,verbose=0)
            rightPredictedClassNum=np.argmax(rightPrediction)
            rightPredictedClass = predictedClass[rightPredictedClassNum]
            if user_input=='s':
                # single click mode
                mouseclick()
            if user_input=='c':
            # holding mouse click 
                mousehold()
            # Extract the x and y coordinates
            second_array = datum.handKeypoints[1]
            final_array = np.reshape(second_array, (-1, 3))
            xPos=int(final_array[8][0])
            yPos=int(final_array[8][1])
            screen_width, screen_height = pyautogui.size()
            # Get the size of the overlayed image
            overlay_width, overlay_height = overlay.shape[1], overlay.shape[0]
            # Calculate the equivalent position on the screen
            screen_xpos = int((xPos / overlay_width) * screen_width)
            screen_ypos = int((yPos / overlay_height) * screen_height)
            # Move the cursor to the calculated positdion
            pyautogui.moveTo(screen_xpos, screen_ypos,_pause=False)
        if rightPredictedClass!=None:
            righttext=rightPredictedClass
        else:
            righttext = 'nothing detected'
        cv2.putText(overlay, lefttext, (100, 50), font, font_scale, color, thickness)    
        cv2.putText(overlay, righttext, (350, 50), font, font_scale, color, thickness)
        
        frame_counter += 1
        
        if time.time() - start_time >= 1:
            # Calculate the new FPS
            fps = frame_counter / (time.time() - start_time)
            count+=1
            totalfps+=fps
            if count==10:
                print('average fps= ',totalfps/10)
                totalfps=0
                count=0
            # Print the new FPS
            print("FPS:", fps)
            # Reset the variables
            frame_counter = 0
            start_time = time.time()
        if not ret:
            print("Failed to capture frame from webcam.")
            break
        
        # Set the window to always stay on top
        cv2.setWindowProperty("Control System", cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow("Control System", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


# Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()
except Exception as e:
    print(e)
    sys.exit(-1)
