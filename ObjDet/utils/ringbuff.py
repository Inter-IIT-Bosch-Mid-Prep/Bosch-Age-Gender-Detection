class RingBuff():
    """
    Ring buffer implementations for lists
    """
    def __init__(self,bsize=10):
        self.bsize = bsize
        self.buff = []

    def append(self,val):
        """
        Add new element to the rign buffer

        Args:
            val (tuple): data to be stored in the Ring Buffer
        """
        # if list is full, make space and add element
        if (len(self.buff) + 1) % self.bsize != 0:
            self.buff.append(val)
        else:
            self.buff.pop(0)
            self.buff.append(val)

    def is_full(self):
        """
        Test if buffer is full

        Returns:
            boolean: [True/False]
        """
        return True if (len(self.buff)+1) == self.bsize else False

    def buff_size(self):
        """
        Returns the size of the buffer

        Returns:
            int: buffer size
        """
        return self.bsize
    
    def get_buffer(self):
        """
        Returns the buffer content

        Returns:
            tuple: data sored in buffer
        """
        return self.buff

    def pop_element(self):
        """
        Take the oldest element out from the list
        """
        if len(self.buff)>0:
            self.buff.pop(0)