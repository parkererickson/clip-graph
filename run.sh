#!/bin/bash
python main.py -bs 32 -gm gat -gp mean & wait;
python main.py -bs 32 -gm gat -gp max & wait;
python main.py -bs 32 -pim False -gm gat -gp max & wait;