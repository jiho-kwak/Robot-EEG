import os
import time
import decimal
import shutil
import copy
import argparse
import numpy as np
import pandas as pd
from IPython.display import display

### Find the number closest to the 'target' in 'array' and return its index and value.
def find_nearest(array, target):
	idx = np.searchsorted(array, target)
	idx1 = max(0, idx-1)
	value = array[np.abs(np.array(array[idx1:idx+1]) - target).argmin() + idx1]
	return array.index(value), value

def str_to_list(str):
	empty = []
	empty.append(str)
	return empty

# Number of EEG data received per second : 500

def Parse(subject, date):
	EEGfile = "EEG_IAPS_" + subject + '_' + date + ".txt"
	logfile = "log_IAPS_" + subject + '_' + date + ".txt"
	currentDir = os.getcwd()
	sortedIAPS = []
	seconds_dig_eeg = []
	seconds_dig_log = []
	seconds_str_eeg = []
	seconds_str_log = []
	list_index = []
	list_EEG = []


	fd = open(EEGfile, 'r')
	lines = fd.readlines()

	for line in lines:
		eeg = line.split('(')[0].rstrip().rstrip(',').lstrip('[').rstrip(']').split(", ")
		sec_dig = decimal.Decimal(line.split('(')[1].rstrip().rstrip(')'))
		sec_str = line.split('(')[1].rstrip().rstrip(')')
		seconds_dig_eeg.append(sec_dig)
		seconds_str_eeg.append(sec_str)
		list_EEG.append(eeg)
	seconds_dig_eeg.sort()
	fd.close()


	fd = open(logfile, 'r')
	lines = fd.readlines()

	for line in lines:
		IAPS_num = line.split(',')[0].rstrip().lstrip()
		sec_dig = decimal.Decimal(line.split('(')[1].rstrip().rstrip(')'))
		sec_str = line.split('(')[1].rstrip().rstrip(')')
		seconds_dig_log.append(sec_dig)
		seconds_str_log.append(sec_str)
		sortedIAPS.append(IAPS_num)
	fd.close()


	for i in range(len(sortedIAPS)):
		np_trial = []

		target1 = seconds_dig_log[i]
		target2 = decimal.Decimal(seconds_str_log[i]) + 3

		idx1, sec1 = find_nearest(seconds_dig_eeg, target1)
		idx2, sec2 = find_nearest(seconds_dig_eeg, target2)

		for j in range(idx1, idx2):
			np_trial.append(str_to_list(sortedIAPS[i]) + list_EEG[j])

		Trial = np.array(np_trial)
		np.save(os.path.join("Subject_Names", subject, "Train", date, sortedIAPS[i] + '_' + subject + '_' + date + ".npy"), Trial)

	shutil.move(os.path.join(currentDir, EEGfile), os.path.join(currentDir, "Subject_Names", subject, "Data", EEGfile))
	shutil.move(os.path.join(currentDir, logfile), os.path.join(currentDir, "Subject_Names", subject, "Data", logfile))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Parsing_IAP")
    parser.add_argument("-s", "--subject", metavar = '', type = str, required = True, help = "The first name of subject in uppercase")
    parser.add_argument("-d", "--date", metavar = '', type = str, required = True, help = "Today's date (YYMMDD)")
    args = parser.parse_args()

    Parse(args.subject, args.date)