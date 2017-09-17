from ..socket.clientSock import clientSock
from ..log.LogWriter import LogWriter

"""
class deviceManager

"""

class deviceManager(object):


    ###
    #   class Device
    #   just a container to store device info
    #
    class Device(object):
        arg_list=['name', 'ip', 'port', 'sock', 'conn', 'state']

        def __init__(self, **kargs):
            [setattr(self, k, None) for k in self.arg_list]
            self.setAttr(**kargs)

        def setAttr(self, **kargs):
            [setattr(self, k, v) for k, v, in kargs.items()]


    ###
    #    __init__
    #
    #   devices_list: the list of the devices want to connect
    #   log: log file path
    #   name: the ID in log file
    #
    #   the format of devices_list is like:
    #   devices_list = [
    #       ['name1', 'ip', port],
    #       ['name2', 'ip', port],
    #       ...
    #   ]
    #
    #   return: None
    #
    def __init__(self, devices_list=[], log=None, name='deviceManager'):
        self.devices={}
        self.log = LogWriter(log, printout=True, name=name)
        self.name = name

        # add new devices to map
        self.addDevices(devices_list)


    ###
    #    addDevices
    #
    #   devices_list: same as __init__
    #
    #   return: None
    #
    def addDevices(self, devices_list=[]):
        for entry in devices_list:
            if entry[0] in self.devices:
                self.log.Error('the device \'{0}\' has already existed'.format(entry[0]))
                continue
            device = deviceManager.Device(name=entry[0], ip=entry[1], port=entry[2],
                                        sock=None, conn=False, state=False)
            
            self.devices[device.name] = device

    ###
    #   waitForConnections
    #
    #   create socket to connect to all the devices which are disconnected
    #   every_time_out: connection timeout of every devices (sec)
    #
    #   return: the number of devices connected successfully
    #
    def waitForConnections(self, every_time_out=5):
        number = 0
        for name, device in self.devices.items():

            # try to connect
            if device.conn == False:
                device.sock = self.__wait_for_connections(device, every_time_out, log = self.log.path())
                if device.sock != None:
                    device.conn = True
                    number += 1
                else:
                    device.conn = False

        return number


    ###
    #   deviceRecovery
    #
    #   reconnect to the device
    #   device: device to reconnect, type can be either str(device name) or deviceManager.Device
    #   time_out: connection timeout
    #
    #   return: True if connected, else False
    #
    def deviceRecovery(self, device, time_out=5):

        if type(device) == str:
            device = self.__get_device_by_name(device)
            if device == None:
                return False

        device.sock = self.__wait_for_connections(self, device, time_out, self.log.path())
        
        if device.sock != None:
            device.conn = True
        else:
            device.conn = False

        return device.conn



    ###
    #   sendMsgToIdleDeviceAndWaitForReply
    #
    #   send message to the device which its state is idle, 
    #       and then wait for reply (block)
    #
    #   msg: message in bytearray
    #
    #   return: device reply in bytearray
    #
    def sendMsgToIdleDeviceAndWaitForReply(self, msg, block=True):
        device = None

        while device == None:
            device = self.__get_idle_device()

            if not block:
                break


        device.state = True
        try:
            device.sock.send(msg)
            reply = device.sock.recv()
            device.state = False
            return reply
        except (OSError, IOError):
            device.sock.close()
            device.conn = False
            device.state = False
            self.log.Error('device {0} lost connection'.format(self.__device_format(device)))
        return None


    ###
    #   sendMsgToDevice
    #
    #   just send message to the device
    #
    #   device: type can be either str(name) or deviceManager.Device
    #   msg: message
    #
    #   return: None, but if failed, return False
    #
    def sendMsgToDevice(self, device, msg):
        if type(device) == str:
            device = self.__get_device_by_name(device)
            if device == None:
                return False
        
        if device.conn == True:
            try:
                device.sock.send(msg)
            except (OSError, IOError):
                device.conn = False
                device.state = False
                self.log.Error('device {0} lost connection'.format(self.__device_format(device)))


    def closeDevice(self, device, block=True):
        if type(device) == str:
            device = self.__get_device_by_name(device)
            if device == None:
                return False

        if device.conn == True:
            if block:
                while device.state:
                    pass
            try:
                device.conn = False
                device.state = False
                device.sock.close()
            except (OSError, IOError):
                self.log.Error('device {0} lost connection'.format(self.__device_format(device)))


    def closeAllDevices(self, block=True):
        for name, device in self.devices.items():
            self.closeDevice(device, block)
        
        self.log.Log('all devices has closed')
        
    ###
    #   __wait_for_connections  (private)
    #
    #   create connection to the device with block
    #   device: deviceManager.Device
    #   time_out: connection timeout
    #   log: socket log
    #
    #   return: clientSock if succeed to connect, else None
    #
    def __wait_for_connections(self, device, time_out, log):
        sock = clientSock(log=log, name=device.name)
        self.log.Log('connecting to deivce {0}'.format(self.__device_format(device)))
        if not sock.connect(ip=device.ip, port=device.port, time_out=time_out):
            self.log.Error('cannot connect to device {0}'.format(self.__device_format(device)))
            return None
        self.log.Log('connection established to device {0}'.format(self.__device_format(device)))
        return sock

    ###
    #   __get_idle_device  (private)
    #
    #   get device which state is idle
    #
    #   return: deviceManager.Device
    #  
    def __get_idle_device(self):
        for name, device in self.devices.items():
            if device.conn == True and device.state == False:
                return device

        return None

    ### 
    #   __get_device_by_name  (private)
    #
    #   get device by device name, if failed then log error msg
    #   name: device name
    #
    #   return: deviceManager.Device
    #
    def __get_device_by_name(self, name):
        try:
            return self.devices[name]
        except KeyError as e:
            self.log.Error('cannot find device \'{0}\' in deviceManager'.format(name))
            return None


    ###
    #   __device_format  (private)
    #
    #   get device info 'name' (ip:port)
    #   device: deviceManager.Device
    #
    #   return: str, device info
    #
    def __device_format(self, device):
        return '\'{0}\' ({1}:{2})'.format(device.name, device.ip, device.port)
