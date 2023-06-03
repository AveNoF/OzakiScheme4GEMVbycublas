#!/bin/bash

./cublas_GemmEx_example 8 1 > result/profile_OO8.txt
./cublas_GemmEx_example 9 1 > result/profile_OO9.txt
./cublas_GemmEx_example 10 1 > result/profile_OO10.txt
./cublas_GemmEx_example 11 1 > result/profile_OO11.txt
./cublas_GemmEx_example 12 1 > result/profile_OO12.txt
./cublas_GemmEx_example 13 1 > result/profile_OO13.txt
./cublas_GemmEx_example 14 1 > result/profile_OO14.txt
./cublas_GemmEx_example 15 1 > result/profile_OO15.txt

./cublas_GemmEx_example 8 4 > result/profile_4OO8.txt
./cublas_GemmEx_example 9 4 > result/profile_4OO9.txt
./cublas_GemmEx_example 10 4 > result/profile_4OO10.txt
./cublas_GemmEx_example 11 4 > result/profile_4OO11.txt
./cublas_GemmEx_example 12 4 > result/profile_4OO12.txt
./cublas_GemmEx_example 13 4 > result/profile_4OO13.txt
./cublas_GemmEx_example 14 4 > result/profile_4OO14.txt
./cublas_GemmEx_example 15 4 > result/profile_4OO15.txt

nv-nsight-cu-cli --set full ./cublas_GemmEx_example 8 1 > profile/profile_OO8.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 9 1 > profile/profile_OO9.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 10 1 > profile/profile_OO10.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 11 1 > profile/profile_OO11.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 12 1 > profile/profile_OO12.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 13 1 > profile/profile_OO13.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 14 1 > profile/profile_OO14.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 15 1 > profile/profile_OO15.txt

nv-nsight-cu-cli --set full ./cublas_GemmEx_example 8 4 > profile/profile_4OO8.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 9 4 > profile/profile_4OO9.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 10 4 > profile/profile_4OO10.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 11 4 > profile/profile_4OO11.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 12 4 > profile/profile_4OO12.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 13 4 > profile/profile_4OO13.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 14 4 > profile/profile_4OO14.txt
nv-nsight-cu-cli --set full ./cublas_GemmEx_example 15 4 > profile/profile_4OO15.txt


