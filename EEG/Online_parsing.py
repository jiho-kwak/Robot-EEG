# generic imports
import os
import time
import shutil
import socket
import decimal
import argparse
import numpy as np
import pandas as pd

### FTP import 
from ftplib import FTP


### Return number of lines in file 'fname'.
def file_len(fname):
	length = sum(1 for line in open(fname))
	return length


### Find the number closest to the 'target' in 'array' and return its index and value.
def find_nearest(array, target):
	idx = np.searchsorted(array, target)
	idx1 = max(0, idx-1)
	value = array[np.abs(np.array(array[idx1:idx+1]) - target).argmin() + idx1]
	return array.index(value), value


### Make string 'str' to list ['str']
def str_to_list(str):
	empty = []
	empty.append(str)
	return empty


### Make directory 'path_dir' if it does not exist.
def makedir(path_dir, dirname):
	if not os.path.isdir(path_dir):
		os.mkdir(path_dir)
		print("made {}".format(dirname))


### Main parsing
def Parsing(subject, date, mode):
	### Default Settings
	currentDir = os.getcwd()
	resultDir_train = os.path.join(currentDir, "Subject_Names", subject, "Train", date)
	resultDir_test = os.path.join(currentDir, "Subject_Names", subject, "Test", date)
	logfile = "log_" + subject + '_' + date + ".txt"
	list_col = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "01", "02", "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz", "FC1", "FC2", "CP1", "CP2", "FC5", "FC6", "CP5", "CP6", "TP9", "TP10", "Eog", "Ekg1", "Ekg2", "x_dir", "y_dir", "z_dir"]

	list_index_train = []
	list_index_empty = []
	list_index_full = []
	list_index_cookie = []

	list_train = []
	list_empty = []
	list_full = []
	list_cookie = []

	np_train = []
	np_empty = []
	np_full = []
	np_cookie = []

	idx_train = 1001
	idx_empty = 1
	idx_full = 1
	idx_cookie = 1


	### Select mode
	item = mode
	if item == "Train":
		read_position = 0
		subjectfile = os.path.join(currentDir, "Subject_Names", subject, "Data", "EEG_" + subject + '_' + date + "_Train.txt")
	elif item == "Test":
		read_position = 30
		subjectfile = os.path.join(currentDir, "Subject_Names", subject, "Data", "EEG_" + subject + '_' + date + "_Test.txt")
		item = "Empty"
	print(logfile)


	### Ip address for FTP connection
	ip = "143.248.236.225"
	

	### Main-loop
	while True:
		time.sleep(1)
		try:
			ftp = FTP(ip)
			ftp.login('urp', 'kaist2020')
			ftp.cwd("URP")
			f = open(logfile, "wb")
			ftp.retrbinary("RETR " + logfile, f.write)
			f.close()
		except socket.error:
			time.sleep(1)
			continue

		### Wait for online tasks
		if not os.path.isfile(subjectfile):
			continue
		elif os.path.isfile(logfile):
			fd_log = open(logfile, "r")
			lines_log = fd_log.readlines()
			if len(lines_log) <= 1:
				continue
			elif len(lines_log) <= read_position + 1:
				continue
		else:
			continue

		seconds_dig_log = []
		seconds_str_log = []
		seconds_dig_eeg = []
		seconds_str_eeg = []
		list_EEG = []
		states = []


		### Read and make array of the subject's log file
		for line in lines_log:
			state = line.split('(')[0].rstrip().rstrip(',').lstrip('[').rstrip(']').split(", ")
			sec_dig = decimal.Decimal(line.split('(')[1].rstrip().rstrip(')'))
			sec_str = line.split('(')[1].rstrip().rstrip(')')
			seconds_dig_log.append(sec_dig)
			seconds_str_log.append(sec_str)
			states.append(state)
		seconds_dig_log.sort()
		fd_log.close()


		### Read and make array of the subject's eeg file
		fd_eeg = open(subjectfile, 'r')
		lines = fd_eeg.readlines()

		for line in lines:
			if line == lines[-1]:
				break
			eeg = line.split('(')[0].rstrip().rstrip(',').lstrip('[').rstrip(']').split(", ")
			sec_dig = decimal.Decimal(line.split('(')[1].rstrip().rstrip(')'))
			sec_str = line.split('(')[1].rstrip().rstrip(')')
			seconds_dig_eeg.append(sec_dig)
			seconds_str_eeg.append(sec_str)
			list_EEG.append(eeg)
		seconds_dig_eeg.sort()
		fd_eeg.close()


		### Parsing
		print(read_position)
		target1 = seconds_dig_log[read_position]
		target2 = seconds_dig_log[read_position+1]
		print(states[read_position][0])

		idx1, sec1 = find_nearest(seconds_dig_eeg, target1)
		idx2, sec2 = find_nearest(seconds_dig_eeg, target2)
		print(idx1)
		print(idx2)

		for j in range(idx1, idx2):
			if item == "Train":
				list_train.append(list_EEG[j])
				list_index_train.append(", ".join(states[read_position]))
				np_train.append(str_to_list(", ".join(states[read_position])) + list_EEG[j])

			elif item == "Empty":
				list_empty.append(list_EEG[j])
				list_index_empty.append(", ".join(states[read_position]))
				np_empty.append(str_to_list(", ".join(states[read_position])) + list_EEG[j])

			elif item == "Full":
				list_full.append(list_EEG[j])
				list_index_full.append(", ".join(states[read_position]))
				np_full.append(str_to_list(", ".join(states[read_position])) + list_EEG[j])

			elif item == "Cookie":
				list_cookie.append(list_EEG[j])
				list_index_cookie.append(", ".join(states[read_position]))
				np_cookie.append(str_to_list(", ".join(states[read_position])) + list_EEG[j])


		### Make numpy files (Train)
		if item == "Train":
			if states[read_position+1][0] == "resting":
				TRAIN = np.array(np_train)
				npyfilename_train = str(idx_train) + '_' + subject + '_' + date + ".npy"
				npyfile_train = os.path.join(resultDir_train, npyfilename_train)
				np.save(npyfile_train, TRAIN)
				print("Complete_save_train"+str(idx_train))

				if read_position == 27:
					shutil.move(os.path.join(currentDir, logfile), os.path.join(currentDir, "Subject_Names", subject, "Data", "log_" + subject + '_' + date + "_Train.txt"))
					break

				list_index_train = []
				list_train = []
				np_train = []
				idx_train += 1
				read_position += 2


		### Make numpy files (Test)
		else: 
			if states[read_position+1][0] == "end":
				if states[read_position][2] == "empty":
					EMPTY = np.array(np_empty)
					npyfilename = item + str(idx_empty) + '_' + subject + '_' + date + ".npy"
					makedir(os.path.join(resultDir_test, "EMPTY"), "empts-dir")
					makedir(os.path.join(resultDir_test, "EMPTY", item+str(idx_empty)), "empt"+str(idx_empty)+"-dir")
					npyfile = os.path.join(resultDir_test, "EMPTY", item+str(idx_empty), npyfilename)
					np.save(npyfile, EMPTY)
					print("Complete_save_empty"+str(idx_empty))

					list_index_empty = []
					list_empty = []
					np_empty = []
					idx_empty += 1
					item = "Full"

				elif states[read_position][2] == "full":
					FULL = np.array(np_full)
					npyfilename = item + str(idx_full) + '_' + subject + '_' + date + ".npy"
					makedir(os.path.join(resultDir_test, "FULL"), "full-dir")
					makedir(os.path.join(resultDir_test, "FULL", item+str(idx_full)), "full"+str(idx_full)+"-dir")
					npyfile = os.path.join(resultDir_test, "FULL", item+str(idx_full), npyfilename)
					np.save(npyfile, FULL)
					print("Complete_save_full"+str(idx_full))

					list_index_full = []
					list_full = []
					np_full = []
					idx_full += 1
					item = "Cookie"

				elif states[read_position][2] == "cookie":
					COOKIE = np.array(np_cookie)
					npyfilename = item + str(idx_cookie) + '_' + subject + '_' + date + ".npy"
					makedir(os.path.join(resultDir_test, "COOKIE"), "cookies-dir")
					makedir(os.path.join(resultDir_test, "COOKIE", item+str(idx_cookie)), "ck"+str(idx_cookie)+"-dir")
					npyfile = os.path.join(resultDir_test, "COOKIE", item+str(idx_cookie), npyfilename)
					np.save(npyfile, COOKIE)
					print("Complete_save_cookie"+str(idx_cookie))

					if read_position == 118:
						shutil.copy(os.path.join(currentDir, logfile), os.path.join(currentDir, "Plot"))
						shutil.move(os.path.join(currentDir, logfile), os.path.join(currentDir, "Subject_Names", subject, "Data", logfile))
						break

					list_index_cookie = []
					list_cookie = []
					np_cookie = []
					idx_cookie += 1
					item = "Empty"

				read_position += 1
		read_position += 1


### Main function
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Online_parsing")
	parser.add_argument("-s", "--subject", metavar = '', type = str, required = True, help = "The first name of subject in uppercase")
	parser.add_argument("-d", "--date", metavar = '', type = str, required = True, help = "Today's date (YYMMDD)")
	parser.add_argument("-m", "--mode", metavar = '', type = str, required = True, help = "Select the mode (Train/Test)")
	args = parser.parse_args()

	Parsing(args.subject, args.date, args.mode)