#!/bin/bash

PART=$1 # csv file: str
RUN=$2 # run index for the folder to keep track of hyperparam optimization: int
GAMMA=$3 # discount factor: float
BETA_TYPE=$4 # all, one, one-all, all-all, one-all-plus-minus: str
BETA_VAL=$5 # float
BATCH_SIZE=$6 # int

# shellcheck disable=SC2207
graph_names=( $(tail -n +2 $PART | cut -d ',' -f2) )


for graph in "${graph_names[@]}"; do
#	for idx in {1..2}; do
	echo "Solving for $graph"
	python rqaoa_agent.py 1 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log1 &
	python rqaoa_agent.py 2 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log2 &
	python rqaoa_agent.py 3 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log3 &
	python rqaoa_agent.py 4 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log4 &
	python rqaoa_agent.py 5 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log5 &
	python rqaoa_agent.py 6 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log6 &
	python rqaoa_agent.py 7 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log7 &
	python rqaoa_agent.py 8 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log8 &
	python rqaoa_agent.py 9 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log9 &
	python rqaoa_agent.py 10 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log10 &
	python rqaoa_agent.py 11 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log11 &
	python rqaoa_agent.py 12 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log12 &
	python rqaoa_agent.py 13 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log13 &
	python rqaoa_agent.py 14 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log14 &
	python rqaoa_agent.py 15 140 $graph 0.001 0.5 $GAMMA $BETA_TYPE $BETA_VAL rqaoa 1 None $BATCH_SIZE True nrg $RUN True &> log15 &
	wait

done