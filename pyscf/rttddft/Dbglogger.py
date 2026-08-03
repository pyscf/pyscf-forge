from .futils import futils
class Dbglogger:
  nlog=-1
  filenumber=0
  logfile=None
  @staticmethod
  def set_filenumber(num):
      Dbglogger.filenumber=num
  @staticmethod
  def write(msg,refresh=False,stdout=False):
#    if( world.rank != 0 ):  TODO XXX  MPI RANK 
#        return
    if( Dbglogger.filenumber<0 ):
        print("#log:"+msg)
        return  
    Dbglogger.nlog=Dbglogger.nlog+1
    if( Dbglogger.logfile is None ):
        Dbglogger.logfile="Dbglogger.log"
        if( Dbglogger.filenumber > 0 ):
            Dbglogger.logfile="Dbglogger_%d.log"%(Dbglogger.filenumber)

    fd=futils.fopen(Dbglogger.logfile,("w" if(refresh or Dbglogger.nlog==0) else "a"));
    print(msg,file=fd)
    if( stdout ):
        print(msg)
    futils.fclose(fd)

Dbglogger.write("#logging..")

#def dbglogger(msg,refresh=False,stdout=False):
#    Dbglogger.write(msg,refresh,stdout)
