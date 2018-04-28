# nucleus



把nucleus文件夹替换Mask_RCNN/sample/下的nucleus

设定好dataset的位置（dataset文件夹为kaggle数据下载后完全解压的文件夹目录，e.g. dataset/stage1_train中的dataset位置），
然后./run.sh

bash内容，不需要写stage1_train：
#!/bin/bash
for i in `seq 1 7`;
do
    echo "Running cross validation fold $i"
    KERAS_BACKEND=tensorflow python3 nucleus.py train --dataset=../../../ --subset=train --weights=imagenet --cross=$i --logs=../../logs_cross_$i
done
