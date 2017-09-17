import time
from datetime import datetime

import sys

#default mode set to 'append'
DEFAULT_MODE = 'a'
#select encoding automaticly;
#if stream existed then encoding as existed stream
#else encoding='UTF-8'
AUTO_ENCODING = -1


'''
stream_table{
    path1: {stream, mode, encoding, state}
    path2: {stream, mode, encoding, state}
    ...
}

'''

class _FileStream:
    stream_table={}

    def __init__(self):
        stream_table = {}

    def addStream(self, path, mode=DEFAULT_MODE, encoding=AUTO_ENCODING):

        # check if stream already existed
        if path in self.stream_table:
            tb = self.stream_table[path]
            if encoding != AUTO_ENCODING and encoding != tb['encoding']:
                raise IOError('new stream encoding doesn\'t match the existed stream')

        # create new stream

        if encoding == AUTO_ENCODING or sys.version_info.major != 3:
            encoding = 'UTF-8'
        stream = self._open_stream(path, mode, encoding)

        
        # create entry and add to table
        tb = {'stream': stream, 'mode': mode, 'encoding': encoding, 'state': True}
        self.stream_table[path] = tb

    def delStream(self, path):
        if path in self.stream_table:
            tb = self.stream_table[path]
            if tb['state'] == True:
                tb['stream'].close()
            
            del self.stream_table[path]


    def isExist(self, path):
        return path in self.stream_table


    def write(self, path, msg):
        if path in self.stream_table:
            tb = self.stream_table[path]
            if tb['state'] == True:
                tb['stream'].write(msg)
                tb['stream'].flush()
            else:
                raise IOError('stream \'{0}\' is closed'.format(path))
        else:
            raise IOError('stream \'{0}\' doesn\'t exist'.format(path))
    
    def getStreamByPath(self, path):
        if path in self.stream_table:
            tb = self.stream_table[path]
            return tb['stream']
        return None

    def open(self, path):
        if path in self.stream_table:
            tb = self.stream_table[path]
            if tb['state'] == False:
                tb['stream'] = self._open_stream(path, tb['mode'], tb['encoding'])
                tb['state'] = True
        else:
            raise IOError('stream \'{0}\' doesn\'t exist'.format(path))

                
    def _open_stream(self, path, mode, encoding):
        try:
            if sys.version_info.major == 3:
                return open(path, mode, encoding=encoding)
            else:
                return open(path, mode)
        except (OSError, IOError) as e:
            raise e


    def close(self, path):
        if path in self.stream_table:
            tb = self.stream_table[path]
            if tb['state'] == True:
                tb['stream'].close()
                tb['state'] = False

    def closeAll(self):
        for path in self.stream_table:
            tb = self.stream_table[path]
            if tb['state'] == True:
                tb['stream'].close()
                tb['state'] = False

    def clearfile(self, path):
        if path in self.stream_table:
            tb = self.stream_table[path]
            if tb['state'] == True:
                tb['stream'].close()
            tb['stream'] = self._open_stream(path, 'w', tb['encoding'])

    def clear(self):
        self.closeAll()
        self.stream_table = {}

    def __del__(self):
        self.clear()
        


class LogWriter(object):
    stream_table = _FileStream()
    number = 0
    def __init__(self, path='pyLog.log', printout=False, name=None):
        
        # print to terminal
        if path == None:
            self.stream = None
            printout = True
        else:
            # check if there has already exist a stream that has the same path, else create one
            if not self.stream_table.isExist(path):
                self.stream_table.addStream(path, 'a', 'UTF-8') 
            
            self.stream = self.stream_table.getStreamByPath(path)


        # stream info        
        self.filepath = path
        self.isOpened = True
        self.printout = printout
        if name == None:
            name = 'Writer{0}'.format(self.number)
            self.number += 1

        self.name = name
    

    def open(self):
        try:
            self.stream_table.open(self.filepath)
            self.stream = self.stream_table.getStreamByPath(self.filepath)
            self.isOpened = True
        except (OSError, IOError) as e:
            self._print_error('in LogWriter.open', e)

    def path(self):
        return self.filepath
        
    def _write_log(self, lgtype, msg):
        if self.isOpened == False or self.stream.closed:
            self.open()
        try:
            ttmsg = '{0} '.format(datetime.now())
            if lgtype != None:
                ttmsg += '::{0}:: '.format(lgtype)
            ttmsg += '[{0}] {1}'.format(self.name, msg)

            if self.stream != None:
                self.stream.write(ttmsg+'\n')
                self.stream.flush()
            if self.printout or lgtype == 'ERROR':
                print(ttmsg)
        except (OSError, IOError) as e:
            self._print_error('in LogWriter._write_log', e)
        

    def clear(self):
        if self.filepath == None:
            return
        try:
            self.stream_table.clearfile(self.filepath)
            self.stream = self.stream_table.getStreamByPath(self.filepath)
        except (OSError, IOError) as e:
            self._print_error('in LogWriter.clear', e)

    def Log(self, msg):
        self._write_log('INFO', msg)

    def Error(self, msg):
        self._write_log('ERROR', msg)

    def Warning(self, msg):
        self._write_log('WARN', msg)
    
    def Write(self, msg, lgtype=None):
        self._write_log(None, msg)

    def _print_error(self, msg, e):
        print('{0} ::ERROR:: {1}'.format(datetime.now(), msg))
        print('{0} ::ERROR:: [errno {1}] {2}:{3}'.format(datetune.now(), e.errno, type(e).__name__, e.strerror))

    def close(self):
        if self.filepath == None:
            return
        self.stream_table.close(self.filepath)
        self.isOpened = False


    def __del__(self):
        self.close()
            

