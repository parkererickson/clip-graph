#!/bin/bash
python main.py -bs 64 -gm gcn & wait;
python main.py -bs 64 -gm gat & wait;
python main.py -bs 32 -gm gat & wait;
python main.py -bs 64 -pim False & wait;
python main.py -bs 64 -pim False -gm gcn & wait;
python main.py -bs 64 -pim False -gm gat & wait;
python main.py -bs 32 -pim False -gm gat