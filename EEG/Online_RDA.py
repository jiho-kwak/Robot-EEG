"""
Simple Python RDA client for the RDA tcpip interface of the BrainVision Recorder
It reads all the information from the recorded EEG,
prints EEG and marker information to the console and calculates and
prints the average power every second


Brain Products GmbH
Gilching/Freiburg, Germany
www.brainproducts.com

"""

# needs socket and struct library
from datetime import datetime
import time
import os
import shutil
import argparse
from socket import *
from struct import *

# Marker class for storing marker information
class Marker:
    def __init__(self):
        self.position = 0
        self.points = 0
        self.channel = -1
        self.type = ""
        self.description = ""

# Helper function for receiving whole message
def RecvData(socket, requestedSize):
    returnStream = ''
    while len(returnStream) < requestedSize:
        databytes = socket.recv(requestedSize - len(returnStream))
        if databytes == '':
            raise RuntimeError
        returnStream += databytes

    return returnStream


# Helper function for splitting a raw array of
# zero terminated strings (C) into an array of python strings
def SplitString(raw):
    stringlist = []
    s = ""
    for i in range(len(raw)):
        if raw[i] != '\x00':
            s = s + raw[i]
        else:
            stringlist.append(s)
            s = ""

    return stringlist


# Helper function for extracting eeg properties from a raw data array
# read from tcpip socket
def GetProperties(rawdata):

    # Extract numerical data
    (channelCount, samplingInterval) = unpack('<Ld', rawdata[:12])

    # Extract resolutions
    resolutions = []
    for c in range(channelCount):
        index = 12 + c * 8
        restuple = unpack('<d', rawdata[index:index+8])
        resolutions.append(restuple[0])

    # Extract channel names
    channelNames = SplitString(rawdata[12 + 8 * channelCount:])

    return (channelCount, samplingInterval, resolutions, channelNames)

##############################################################################################
#
# Main RDA routine
#
##############################################################################################

def RDA(subject, date, mode):
	# Create a tcpip socket
	con = socket(AF_INET, SOCK_STREAM)
	# Connect to recorder host via 32Bit RDA-port
	# adapt to your host, if recorder is not running on local machine
	# change port to 51234 to connect to 16Bit RDA-port
	con.connect(("localhost", 51244))

	# Flag for main loop
	finish = False

	# data buffer for calculation, empty in beginning
	data1s = []

	# block counter to check overflows of tcpip buffer
	lastBlock = -1
	
	currentDir = os.getcwd()

	os.chdir(os.path.join(currentDir, "Subject_Names", subject, "Data"))
	if mode == "Train":
		subjectfile = "EEG_" + subject + '_' + date + "_Train.txt"
	if mode == "Test":
		subjectfile = "EEG_" + subject + '_' + date + "_Test.txt"	
	time.sleep(1)
	file = open(subjectfile, "a")

	#### Main Loop ####
	while not finish:

	    # Get message header as raw array of chars
	    rawhdr = RecvData(con, 24)

	    # Split array into usefull information id1 to id4 are constants
	    (id1, id2, id3, id4, msgsize, msgtype) = unpack('<llllLL', rawhdr)

	    # Get data part of message, which is of variable size
	    rawdata = RecvData(con, msgsize - 24)

	    # Perform action dependend on the message type
	    if msgtype == 1:
	        # Start message, extract eeg properties and display them
	        (channelCount, samplingInterval, resolutions, channelNames) = GetProperties(rawdata)
	        # reset block counter
	        lastBlock = -1

	        print("Start")
	        print("Number of channels: " + str(channelCount))
	        print("Sampling interval: " + str(samplingInterval))
	        print("Resolutions: " + str(resolutions))
	        print("Channel Names: " + str(channelNames))


	    elif msgtype == 4:
	        (block, points, markerCount) = unpack('<LLL', rawdata[:12])

	        # Extract eeg data as array of floats
	        data = []
	        for i in range(points * channelCount):
	            index = 12 + 4 * i
	            value = unpack('<f', rawdata[index:index + 4])
	            data.append(value[0])
	            if(len(data)==channelCount):
	                file.write(str(data)+", ("+str("%.4f" % time.time())+")\n")
	                data=[]


	    elif msgtype == 3:
	        file.close()
	        # Stop message, terminate program
	        print("Stop")
	        finish = True

	# Close tcpip connection
	con.close()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Online_RDA")
	parser.add_argument("-s", "--subject", metavar = '', type = str, required = True, help = "The first name of subject in uppercase")
	parser.add_argument("-d", "--date", metavar = '', type = str, required = True, help = "Today's date (YYMMDD)")
	parser.add_argument("-m", "--mode", metavar = '', type = str, required = True, help = "Select the mode (Train/Test)")

	args = parser.parse_args()

	RDA(args.subject, args.date, args.mode)