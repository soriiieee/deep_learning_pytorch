#!/bin/bash


HERE=`pwd`
#----------------2021.09.27---------
# 四国電力エリアの電力予測プロダクトについて
LOCAL=$HERE/data/power
# echo $HERE
cd $LOCAL
for YY in `seq 2016 2021`;do
# URL=https://www.yonden.co.jp/csv/juyo_shikoku_${YY}.csv
URL=https://www.yonden.co.jp/nw/denkiyoho/csv/juyo_shikoku_${YY}.csv
wget $URL
FILE=`basename $URL`
nkf -w --overwrite $FILE
done
#----------------2021.09.27---------