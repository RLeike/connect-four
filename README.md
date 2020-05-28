# Deep Q-Learning for Connect Four

This repository provides code for training a neural network to play Connect Four.

## Prerequisites

python3 3.8

jax

scipy

## Usage

Run the code with

`python3 main.py`.

## Background

Deep Q-Learning was first introduced by [1], who trained a realtively small neural network to play atari games.

## Implementation

`main.py` holds the training loop, in which many rounds of self-play are carried out. The generated trajectories are stored in a Memory object, of which the implementation is inside `memory.py`. This object then provides preprocessed batches for training the agent.
The agent itself is defined in `agent.py`. It consists of a simple feed forward convolutional neural network. 
The agent can be tested by playing against it after training.
Hereby the power is enhanced by recursively looking ahead using a min-max algorithm.


[1] Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Alex Graves, Ioannis Antonoglou, Daan Wierstra and Martin Riedmiller. "Playing Atari with Deep Reinforcement Learning", https://arxiv.org/abs/1312.5602v1
