# File: jetracer_plus.py
# Author: George Gorospe, george.gorospe@nmaia.net
# About: A collection of helpful classes for the machine learning process supporting the jetracer.

# Required Libraries
from threading import Timer

### CUSTOM CLASS: RepeatedTimer - we create a custom class so that we can genreate a RepeatedTimer object to perform automated data collection
class RepeatedTimer(object):
    def __init__(self, interval, function, set_size, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.function   = function
        self.index      = 0
        self.set_size   = set_size
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        # While running, check against desired set_size, stop when set size is reached
        self.index = self.index + 1
        if self.index <= self.set_size:
            self.is_running = False
            self.start()
            self.function(*self.args, **self.kwargs)
        else:
            #self._time.cancel()
            self.is_running = False
            
    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False
###### END OF CUSTOM CLASS ############