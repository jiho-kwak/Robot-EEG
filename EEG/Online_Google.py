import os
import time
import shutil
import gspread
import argparse
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "Online_Google")
	parser.add_argument("-s", "--subject", metavar = '', type = str, required = True, help = "The first name of subject in uppercase")
	parser.add_argument("-d", "--date", metavar = '', type = str, required = True, help = "Today's date (YYMMDD)")
	args = parser.parse_args()

	Val_Ctrl = []

	shutil.copy("C:\\Users\\URP\\Desktop\\labels_VA.csv", os.path.join(os.getcwd(), "labels_VA.csv"))

	scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

	credentials = ServiceAccountCredentials.from_json_keyfile_name("robot-human-interaction-b8b536ad207e.json", scope)

	gc = gspread.authorize(credentials)

	while True:
		time.sleep(10)
		gc1 = gc.open("Responses").worksheet("Sheet1")

		gc2 = gc1.get_all_values()

		if gc2[-1][1] != args.subject:
			continue
		else:
			row_subject = gc2[-1]

		num_task = (len(row_subject)-4)

		for i in range(num_task):
			Val_Ctrl.append([str(1001+i), row_subject[4+i], 0])
		break


	dataframe = pd.DataFrame(Val_Ctrl)
	dataframe.to_csv("labels_VA.csv", mode = 'a', index = False, header = False)
	shutil.move(os.path.join(os.getcwd(), "labels_VA.csv"), os.path.join(os.getcwd(), "Subject_Names", args.subject, "Train", args.date, "labels_VA.csv"))