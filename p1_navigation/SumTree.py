import numpy as np
class SumTree:
    """
    Leaf nodes hold experiences and intermediate nodes store experience priority sums.

    Adapted from: https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
    """

    def __init__(self, maxlen):
        """Initialize a SumTree object.
        Params
        ======
            maxlen (int): maximum size of replay buffer
        """
        self.sumList = np.zeros(maxlen * 2)
        self.experiences = np.zeros(maxlen * 2, dtype=object)
        self.maxlen = maxlen
        self.currentSize = 0
        # Set insertion marker for next item as first leaf
        self.tail = ((len(self.sumList) - 1) // 2) + 1

    def add(self, experience):
        """Add experience to array and experience priority to sumList."""
        if self.tail == len(self.sumList):
            self.tail = ((len(self.sumList) - 1) // 2) + 1
        self.experiences[self.tail] = experience
        old = self.sumList[self.tail]
        self.sumList[self.tail] = experience.priority
        if old == 0:
            change = experience.priority
            self.currentSize += 1
        else:
            change = experience.priority - old
        self.propagate(self.tail, change)
        self.tail += 1

    def propagate(self, index, change):
        """Updates sum tree to reflect change in priority of leaf."""
        parent = index // 2
        if parent == 0:
            return
        self.sumList[parent] += change
        self.propagate(parent, change)

    def get_sum(self):
        """Return total sum of priorities."""
        return self.sumList[1]

    def retrieve(self, start_index, num):
        """Return experience at index in which walking the array and summing the probabilities equals num."""
        # Return experience if we reach leaf node
        if self.left(start_index) > len(self.sumList) - 1:
            return self.experiences[start_index], start_index
        # If left sum is greater than num, we look in left subtree
        if self.sumList[self.left(start_index)] >= num:
            return self.retrieve(self.left(start_index), num)
        # If left sum is not greater than num, we subtract the left sum and look in right subtree
        return self.retrieve(self.right(start_index), num - self.sumList[self.left(start_index)])

    def update(self, index, experience):
        """Updates experience with new priority."""
        self.experiences[index] = experience
        old_e_priority = self.sumList[index]
        self.sumList[index] = experience.priority
        change = experience.priority - old_e_priority
        self.propagate(index, change)

    def left(self, index):
        return index * 2

    def right(self, index):
        return index * 2 + 1

    def __getitem__(self, index):
        return self.experiences[index]

    def __len__(self):
        return self.currentSize