import numpy as np
import cv2 as cv

from yunet import Yunet

def Visualize(image, results, box_color=(0,255,0), text_color=(0,0,255), fps=None):
    output = image.copy()
    landmark_color = [
        (255, 0, 0), #right eye
        (0, 0, 255), #left eye
        (0, 255, 0), #nose
        (255, 0, 255), #right mount
        (255, 255, 0) #left mount
        ]
    
    if fps is not None:
        cv.putText(output, 'Fps: {:.2f}'.format(fps), (0,15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
    
    for det in (results if results is not None else []):
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, 2)

        conf = det[-1]
        cv.putText(output, '{:4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_COMPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)
    return output

if __name__ == "__main__":
    backendId = cv.dnn.DNN_BACKEND_OPENCV
    targetId = cv.dnn.DNN_TARGET_CPU

    model = Yunet()
    cap = cv.VideoCapture(0)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    model.setInputSize([w,h])
    frame_num = 0
    tm = cv.TickMeter()
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frame!')
            break
        frame_num+=1
        tm.start()
        results = model.infer(frame)
        tm.stop()
        if frame_num % 2 == 1: 
            frame = Visualize(frame, results, fps=tm.getFPS())
        cv.imshow("yunet", frame)

        tm.reset()