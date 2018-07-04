#!/bin/bash
AUDIO_TRAIN=../input/audio_train
AUDIO_TEST=../input/audio_test
DATA=../input/train.csv
MFCC=../input/mfcc/train_mfcc.npy

#check if data exist
if [ ! -e $DATA ]; then
	echo Please put train.csv to ../input/train.csv
fi

if [ ! -e $AUDIO_TRAIN ]; then
	echo Please put audio file to ../input/audio_train
fi

if [ ! -e $AUDIO_TEST ]; then
	echo Please put audio file to ../input/audio_test
fi

MODE=$1
if [ "${MODE}" = "1d_conv" ]
then
	echo "1d_conv"
	python3 train_1d.py

elif [ "${MODE}" = "2d_mfcc" ]
then
	
	if [ ! -e $MFCC ]; then
		echo Please follow Readme to run data_genertor first!
	fi

	echo "2d_mfcc"
	python3 train_mfcc.py
	

else
	echo "No such mode."
fi
