### imports 

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
import pandas as pd
from sort import *


class speedest:
    def __init__(self,args, line, line_speed_start, line_speed_end, top_left_x, top_left_y, bot_right_x, bot_right_y):
        self.cross_check = []
        self.tracker = Sort()
        self.memory = {}
        self.time_test = {}
        self.time_for_speed = []
        self.df = pd.DataFrame(columns= ["TrackingID","FrameID","LaneID"])
        self.df4 = pd.DataFrame(columns= ["TrackingID","Speed"])
        self.dict_id_speed = {}
        self.line = line
        self.lss = line_speed_start
        self.lse = line_speed_end
        self.top_left_x = top_left_x
        self.top_left_y = top_left_y
        self.bot_right_x = bot_right_x
        self.bot_right_y = bot_right_y
        self.counter1 = 0
        self.labelsPath = os.path.sep.join([args["yolo"] , "coco.names"])
        self.LABELS = open(self.labelsPath).read().strip().split("\n")

        self.weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
        self.configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        self.ln = self.net.getLayerNames()
        # print([i for i in net.getUnconnectedOutLayers()])
        self.ln = [self.ln[i - 1] for i in self.net.getUnconnectedOutLayers()]

        # initialize the video stream, pointer to output video file, and
        # frame dimensions
        self.vs = cv2.VideoCapture(args["input"])

        self.writer = None
        (self.W, self.H) = (None, None)

        self.frameIndex = 0
        try:
            prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
                else cv2.CAP_PROP_FRAME_COUNT
            self.total = int(self.vs.get(prop))
            print("[INFO] {} total frames in video".format(self.total))
        except:
            print("[INFO] could not determine # of frames in video")
            print("[INFO] no approx. completion time can be provided")
            self.total = -1
    def adjust_gamma(self,image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
    
        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)
    def intersect(self,A,B,C,D):
        return self.ccw(A,C,D) != self.ccw(B,C,D) and self.ccw(A,B,C) != self.ccw(A,B,D)

    def ccw(self,A,B,C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def calculation(self, args):
        while True:
        # read the next frame from the file
            (grabbed, frame) = self.vs.read()
            if not grabbed:
                break
            lane1 = np.array([[233, 36], [257, 37], [178, 316], [71, 317]], np.int32)
            lane2 = np.array([[235, 155],[186, 317], [311, 317],[303, 143], [287, 37],[260, 38]], np.int32)
            lane3 = np.array([[308, 156], [320, 320], [444, 316], [313, 36], [291, 38]], np.int32)
            lane4 = np.array([[316, 34], [336, 32],[427, 133], [547, 310], [449, 314],[365, 122]], np.int32)
            cv2.polylines(frame, [lane1], True, (0,255,0), thickness=1)
            cv2.polylines(frame, [lane2], True, (0,0,255), thickness=1)
            cv2.polylines(frame, [lane3], True, (0,255,0), thickness=1)
            cv2.polylines(frame, [lane4], True, (0,0,255), thickness=1)
            cv2.line(frame,self.lss[0],self.lss[1], (255, 255, 255), 1)
            cv2.line(frame,self.lse[0],self.lse[1], (255, 255, 255), 1)
            
            if self.W is None or self.H is None:
                (self.H, self.W) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (256, 256),
                swapRB=True, crop=False)
            self.net.setInput(blob)
            start = time.time()
            layerOutputs = self.net.forward(self.ln)
            end = time.time()
            boxes = []
            center = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > args["confidence"]:
                        box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        center.append(int(centerY))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])
            dets = []
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    dets.append([x, y, x+w, y+h, confidences[i]])
            np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
            dets = np.asarray(dets)
            tracks = self.tracker.update(dets)
            
            boxes = []
            indexIDs = []
            c = []
            
            previous = self.memory.copy()
            self.memory = {}

            for track in tracks:
                boxes.append([track[0], track[1], track[2], track[3]])
                indexIDs.append(int(track[4]))
                self.memory[indexIDs[-1]] = boxes[-1]
            
            if len(boxes) > 0:
                i = int(0)
                for box in boxes:
                    
                    if (int(self.top_left_x) <= int(box[0]) <= int(self.bot_right_x)):
                        if 30< (int(box[2])-int(box[0])) < 600 and (int(box[3])-int(box[1]))<600:
                            (x, y) = (int(box[0]), int(box[1]))
                            (w, h) = (int(box[2]), int(box[3]))
                            p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                            ct1 = cv2.pointPolygonTest(lane1, p0, False)
                            ct2 = cv2.pointPolygonTest(lane2, p0, False)
                            ct3 = cv2.pointPolygonTest(lane3, p0, False)
                            ct4 = cv2.pointPolygonTest(lane4, p0, False)
                                
                            color = (255,0,0) if ct1==1 else (0,255,0) if ct2==1 else (255,0,255) if ct3==1 else (0,255,255) if ct4==1 else (0,0,255)
                            cv2.rectangle(frame, (x, y), (w, h), color, 4)
                
                            if indexIDs[i] in previous:
                                previous_box = previous[indexIDs[i]]
                                (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                                (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                                p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                                p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                                cv2.line(frame, p0, p1, color, 3)
                                id = indexIDs[i]
                                #########################################################
                                if self.intersect(p0,p1,self.lss[0],self.lss[1]):
                                    time_start = np.round(time.time(),3)
        #                                print("ts",time_start)
                                    self.time_test.update({id:time_start})
                                elif self.intersect(p0,p1,self.lse[0],self.lse[1]):
                                    if id in self.time_test:
                                        time_taken = np.round(time.time(),3)-self.time_test.get(id)
                                        self.time_for_speed.append(time_taken)
                                        speed = (30*60*60)/(time_taken*1000)
                                        print({id: speed})
                                        self.df3 = pd.DataFrame([[id,speed]], columns= ["TrackingID","Speed"])

                                        self.df4=self.df4.append(self.df3,ignore_index=True)
                                        
                                        
                                if self.intersect(p0,p1,self.lse[0],self.lse[1]) and id in self.time_test:
                                    cv2.putText(frame, str({id:speed}), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 5)
                                    # del time_test[id]
                                else:
                                    cv2.putText(frame, str(id), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 5)
                                if self.intersect(p0, p1, self.line[0], self.line[1]) and indexIDs[i] not in self.cross_check:
                                    self.counter1 += 1
                                    self.cross_check.append(indexIDs[i])
                                    lane = 2 if ct1==1 else 3 if ct2==1 else 4 if ct3==1 else 5 if ct4==1 else 1
                                    self.df2 = pd.DataFrame([[id,self.frameIndex,lane]], columns= ["TrackingID","FrameID","LaneID"])
                                    self.df = self.df.append(self.df2,ignore_index=True)
                    i += 1

            cv2.line(frame, self.line[0], self.line[1], (0, 255, 255), 2)
            ##############################################
            counter_text = "counter:{}".format(self.counter1)
            cv2.putText(frame, counter_text, (100,250), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
            if self.writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                self.writer = cv2.VideoWriter(args["output"], fourcc, 30,
                    (1080, 720), True)

                # some information on processing single frame
                if self.total > 0:
                    elap = (end - start)
                    print("[INFO] single frame took {:.4f} seconds".format(elap))
                    print("[INFO] estimated total time to finish: {:.4f}".format(
                        elap * self.total))

            # write the output frame to disk
            new_dim = (1080,720)
            self.writer.write(cv2.resize(frame,new_dim, interpolation = cv2.INTER_AREA))

            # increase frame index
            self.frameIndex += 1
        return self.writer

    def save(self, writer):
        final_df1 = pd.merge_ordered(self.df, self.df4, how='outer', on='TrackingID')
        final_df1 = final_df1[final_df1.FrameID.notnull()]
        print("[INFO] cleaning up...")
        writer.release()
        self.vs.release()
        final_df1.to_csv("./final1.csv",index=False)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
        help="path to input video")
    ap.add_argument("-o", "--output", required=True,
        help="path to output video")
    ap.add_argument("-y", "--yolo", required=True,
        help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.40,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.40,
        help="threshold when applyong non-maxima suppression")
    args = vars(ap.parse_args())

    line = [(184, 118),(469, 117)]
    line_speed_start = [(93, 276),(635, 276)]
    line_speed_end = [(164, 148),(497, 148)]
    f1, g1 = 251, 0
    f2, g2 = 319, 0
    f3, g3 = 639, 276
    f4, g4 = 43, 359
    f5, g5 = 638, 355

    top_left_x = min([f1,f2,f3,f4,f5])
    top_left_y = min([g1,g2,g3,g4,g5])
    bot_right_x = max([f1,f2,f3,f4,f5])
    bot_right_y = max([g1,g2,g3,g4,g5])

    a = speedest(args, line, line_speed_start, line_speed_end, top_left_x, top_left_y, bot_right_x, bot_right_y)
    writer = a.calculation(args)
    a.save(writer)




