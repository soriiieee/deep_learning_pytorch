#!/bin/bash

HOME=/home/ysorimachi/work/sori_py2/deepl
BIN=$HOME/bin
SRC=$HOME/src
PY=$HOME/py

DIR=`dirname $0`
. ${DIR}/com.conf
. ${DIR}/prod.conf

SETCHOKI

LOG=${TMP}/ftp1.log


STORE=/work2/ysorimachi/sat/convLSTM/store
INI_J="201901010900"
END_J="201904010900"
TERM=`$TOOL/dtcomp_com $END_J $INI_J 3`

echo "`date +%Y%m%d%H%M` [START] INI_J $INI_J"> $LOG
echo "`date +%Y%m%d%H%M` [START] STORE $STORE">> $LOG
echo "`date +%Y%m%d%H%M` [START] TERM $TERM DAYS !">> $LOG
# exit
# -----
# 
for DD in `seq 0 $TERM`;do
INI_J2=`$TOOL/dtinc_com $INI_J 3 $DD`
for HH in `seq 0 6`;do
INI_J3=`$TOOL/dtinc_com $INI_J2 4 $HH`
for MI in `seq 0 5 55`;do
INI_J4=`$TOOL/dtinc_com $INI_J3 5 $MI`
INI_U4=`$TOOL/dtinc_com $INI_J4 4 -9`

#---------------------
# for BAND in "02" "03" "05" "13" "15";do
for BAND in "03";do
[ ! -e $STORE/$BAND ] && mkdir -p $STORE/$BAND
cd $STORE/$BAND
GET_HIMAWARI8 $BAND $INI_U4 
done
#---------------------

echo "`date` [END]"  $INI_J4 >> $LOG
done #DD
done
done