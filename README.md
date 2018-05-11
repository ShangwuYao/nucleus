# nucleus

## Final Report 框架
1. Introduction(intuition about the idea/purpose of the project)
2. Background/Related Work(intro to competition from kaggle, model Mask R-CNN from He, comparision between Mask R-CNN & U-Net, etc.)
3. Model(details of Mask R-CNN, structure, implementation, advantage and dis.., etc.)
4. Experiments(tuned content from poster)
5. Results
6. Discussion(ideas about the results)
7. References

## How to Run
1. 

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


2.

move cross_valid.npy to the same directory of nucleus.py
