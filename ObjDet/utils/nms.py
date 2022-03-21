"""
Simple implementation of the Non-Maximum-Surpression alorithm.
Uses only opencv cascade output, without probability rates (since opencv cascade does not proviede it)
Acticele: https://arxiv.org/pdf/1704.04503.pdf
"""

class NMS():
    
    import numpy as np

    def __init__(self):
        pass

    def intersection_over_union(self,val1,val2):
        """[calclate the intersection over union of two boxes]

        Args:
            val1 ([array]): [rectangle x,y,w,h]
            val2 ([array]): [rectangle x,y,w,h]

        Returns:
            [float]: [overlap ratio]
        """
        #calculeate union of boxes
        xu = self.np.min((val1[0],val2[0]))
        yu = self.np.min((val1[1],val2[1]))

        wu = self.np.max((val1[0]+val1[2], val2[0]+val2[2])) - xu
        hu = self.np.max((val1[1]+val1[3], val2[1]+val2[3])) - yu

        # calculate the intersection of boxes
        xi = self.np.max((val1[0],val2[0]))
        yi = self.np.max((val1[1],val2[1]))

        wi = self.np.min((val1[0]+val1[2], val2[0]+val2[2])) - xi
        hi = self.np.min((val1[1]+val1[3], val2[1]+val2[3])) - yi

        # the area of unification
        area_uni = wu*hu

        # the area of intersections
        area_int = wi*hi
        
        return area_int/area_uni


    def non_maximum_surpression(self,detections,THRESHOLD=0.1, USESORT=False,REVERSE_SORT=True):
        """[Non Maximum Surpression]

        Args:
            detections ([list]): [bounding boxex, format x,y,w,h]
            THRESHOLD (float, optional): [surpression threshold]. Defaults to 0.1.
            USESORT (bool, optional): [apply preliminarry sort area based]. Defaults to False.
            REVERSE_SORT (bool, optional): [ascending / descending]. Defaults to True.

        Returns:
            [list]: [detected bounding boxes]
        """
        det = []
        
        if len(detections)>0:
            tdet = list(detections.copy())

            # sort function    
            if USESORT == True:
                def sortCriteria(val):
                    return val[2]*val[3]

                tdet.sort(reverse=REVERSE_SORT, key=sortCriteria)
            
            # Loop till all elemets are preocessed. 
            # Calculate intersection over union of the overlaping areas, exclude the ones with higher overlap.
            while len(tdet)>0:
                tst = tdet[-1]
                det.append(tst)
                tdet.pop(-1)
            
                # check all elements overap rate
                for j in range (len(tdet)-1,-1,-1):
                    lv = tdet[j]
                    iou = self.intersection_over_union(tst,lv)

                    # if high overlap remove it
                    if iou > THRESHOLD:
                        tdet.pop(j)
        return det
