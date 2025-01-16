#!/bin/bash

CORPUS_PATH=$(echo $1 | sed 's:\(.\)/*$:\1:')
EXTENSION=$2

slashes=${CORPUS_PATH//[!\/]}
depth=${#slashes}

echo $CORPUS_PATH
find $CORPUS_PATH -name \*$EXTENSION  -name \*.flac -exec sh -c 'printf "%s\t%s\n" $(echo $1 | cut -d / -f $2-) $(soxi -s $1)' sh {} $(( depth + 2 )) \;

