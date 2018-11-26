% Machine Learning (20CS6037-001)
% Assignment 3
% Group Name: LI_LI_SONG_ZENG
% Group Members: Haipeng Li, Xin Li, Ximing Song, Jianfeng Zeng

close all; clear all; clc;

[training_data_set, testing_data_set, data] = makeData();

% [w, b, a] = SMO(data, 0.1, 0.1, 0.5, training_data_set);
[alpha, b, w] = SMO(data, 0.1, 0.1, 0.5, training_data_set);
