# [IROS2022] Robot-EEG
Codes used for Affect-driven Robot Behavior Learning System using EEG Signals for Less Negative Feelings and More Positive Outcomes, IROS 2022.

## Training Session
Run codes as following order:
Let participant's name as ABC, experiment's date as YYMMDD, desired starting time of the code as YYYY-MM-DD/hh:mm:ss, and number of IAPS photos as nn. 
1. cd EEG
2. python Exp_IAP.py -s ABC -d YYMMDD -t YYYY-MM-DD/hh:mm:ss -n nn
3. python Parsing_IAP.py -s ABC -d YYMMDD
4. python Online_RDA.py -s ABC -d YYMMDD -m Train
5. python Online_parsing.py -s ABC -d YYMMDD -m Train
6. conda activate Google
7. python Online_Google.py -s ABC -d 210223
8. cd ../Robot
9. python robot_action.py

## Testing Session
1. cd EEG
2. python Online_RDA.py -s ABC -d 210223 -m Test
3. python Online_parsing.py -s ABC -d 210223 -m Test
4. conda activate AffectRobot
5. python Online_main.py -s ABC -d 210223
6. cd ../Robot
7. python main.py

## Data
The collected data are Server\Data\곽지호\연구\2021 IROS\Data

## Environments
Conda environments are in envs directory.
