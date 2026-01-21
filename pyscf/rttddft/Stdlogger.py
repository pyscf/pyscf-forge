## PLS do not import your own module
## static method only
import numpy as np
import sys
import os
import time
import datetime
class Stdlogger:
    loglevel=1
    # -2:FATAL ERROR  -1:WARNING  0:MILESTONE LOGS  1:DEBUG LOGS
    @staticmethod
    def printout(lvl=1,text="",verbose=0,wt0=None,dtme=False):
        fd=open("Stdlogger.log",'a')
        if( wt0 is not None ):
            wt1=time.time()
            text+="   elapsed: %f    "%(wt1-wt0)
        if( dtme ):
            text+="   \t\t "+str(datetime.datetime.now())
        print(text,file=fd);fd.close()
        if(lvl<( verbose+ Stdlogger.loglevel)):
            print(text)
