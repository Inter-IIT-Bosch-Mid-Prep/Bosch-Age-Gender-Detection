from utils.ringbuff import RingBuff

class CentroidTracker():

    import numpy as np

    def __init__(self, buffer_size=20):
        
        # centroid buffer
        self.CBuffer = RingBuff(buffer_size)
        self.ID = 0

    def euclidian_distance(self,p1,p2):
        """
        Rertun 2 2d points euclidian distance 

        Args:
            p1 (list): point 1 [x,y]
            p2 (list): point 2 [x,y]

        Returns:
            float : euclidian distance
        """

        return self.np.linalg.norm(self.np.array(p1)-self.np.array(p2))

    def RawTracker(self,detections,SEPARATION_DISTIANCE=25.0):
        """
        Process new detecton centroids, assign new ID, directly in the Circular Buffer.

        Args:
            detections (rect): rectangele coordinates [x,y,w,h]
            SEPARATION_DISTIANCE (float, optional): Maximum neighbor distance. Defaults to 25.0.
        """

        # loop over detection list,
        # assign to the circular buffer, format: cx,cy,ed,ID
        for d in detections:
            # centroide of the detected object
            cx = int(d[0] + d[2]/2)
            cy = int(d[1] + d[3]/2)
            
            # init tracked objects list with the first tracked object
            if len(self.CBuffer.get_buffer()) == 0:
                
                # fist element on the tracking list, use later as reference
                self.CBuffer.append((cx,cy, 0.0, self.ID))

            else:
                # serach tracking buffer for neighbors,
                # if found register the current elemet as a known one
                # else as a new element

                ordered = False

                for i in range (len(self.CBuffer.get_buffer())):
                    p = (self.CBuffer.get_buffer()[i][0],self.CBuffer.get_buffer()[i][1])
                    ed = self.euclidian_distance(p,(cx,cy))
                    
                    # check if there is a neigbhor
                    if ed < SEPARATION_DISTIANCE:
                        ordered = True
                        self.CBuffer.append((cx,cy,ed, self.CBuffer.get_buffer()[i][3]))
                        break
                
                # no neighbor found, register a new ID
                if ordered == False:
                    self.ID +=1
                    self.CBuffer.append((cx,cy,0.0,self.ID))
        

    def Update(self,detections,SEPARATION_DISTIANCE=25.0, MIN_DETECTIONS=5, REUSE_IDS=False):
        """
        Cyclic function called after object clasification, with a list of objects to be labeled frame after frame.

        Args:
            detections (list): List of rectagles detected [x,y,w,h]
            SEPARATION_DISTIANCE (float, optional): Euclidian distance between neighbors. Defaults to 25.0.
            MIN_DETECTIONS (int, optional): Minimum number of detected objects with same ID. Defaults to 5.
            REUSE_IDS (bool, optional): Optimize IDs, IDs with equal with number of objects detected. Defaults to False.

        Returns:
            [list]: List of ilteres rectanges, with corresponding IDs
        """
        
        # process detection, update Circular Buffer, 
        # give a rough output with every element ordered to an ID
        self.RawTracker(detections,SEPARATION_DISTIANCE=SEPARATION_DISTIANCE)
        
        # Post processing
        # Check miimum number of detections, put ob the return list only above threshold.
        # Select only 1 detection per ID to put on the return list

        ret = []
        if len(self.CBuffer.get_buffer())>0:

            # get the ID list
            tbuff = self.np.array(self.CBuffer.get_buffer(),dtype=self.np.int)[:,3]

            #count occurence of the IDs
            counts = [[x,list(tbuff).count(x)] for x in set(list(tbuff))]

            # reassign IDs based on the total count
            for i,c in enumerate(counts):
                
                # check if IDs are detected above threshold
                if c[1] >= MIN_DETECTIONS:
                    for j in range (len(self.CBuffer.get_buffer())):
                        if c[0]== self.CBuffer.get_buffer()[j][3]:

                            # get fist value of the detection,
                            # update ID with a new one by using the absolut
                            # numbers detected on the scene

                            val = self.CBuffer.get_buffer()[j]
                            if not REUSE_IDS:
                                ret.append(val)
                            else:
                                ret.append((val[0],val[1],val[2],i+1))

                            # we got what we need, exit
                            break

        # if no detectiions but circular buffer still full, start the clean-up
        if len(detections)== 0:
            # prevent to keep all old detections
            self.CBuffer.pop_element()

        return ret