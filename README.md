# Snake Game AI with Deep Q-Learning

This project is an implementation of a Snake Game AI powered by Deep Q-Learning, a reinforcement learning algorithm. The agent is designed to play the game autonomously, learning through trial and error using neural networks in PyTorch. The project is structured in Python and utilizes the Pygame library for game rendering.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How It Works](#how-it-works)
- [Training the Agent](#training-the-agent)
- [Dependencies](#dependencies)
- [File Descriptions](#file-descriptions)
- [References](#references)

## Overview

The Snake Game AI project employs a simple deep neural network to train an agent capable of learning strategies to maximize its score. The agent observes the game state, chooses an action based on its policy, and updates its knowledge using the Q-Learning algorithm. With sufficient training, the agent learns to survive longer and achieve higher scores by avoiding collisions and seeking out food.

## Project Structure

The project is organized into the following files:

```plaintext
├── model.py            # Defines the neural network and trainer for the agent
├── agent.py            # Contains the reinforcement learning agent class and training loop
├── game.py             # Implementation of the Snake game environment using Pygame
├── helper.py           # Utility functions for plotting the scores
├── README.md           # Instructions and explanations for the project
```

## Setup and Installation

Clone the repository:

```
git clone https://github.com/yourusername/snake_game_ai.git
cd snake_game_ai
```

Create a virtual environment (optional but recommended):

```
python3 -m venv venv
source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
```

Install required dependencies:

```
pip install -r requirements.txt
```

## How It Works

The project uses a Deep Q-Network (DQN) algorithm:

- Agent: The main class responsible for making decisions, storing experiences, and learning from past actions.
- Neural Network: Defined in model.py as a simple two-layer linear network with ReLU activations. It learns to approximate the Q-function.
- Environment: The game.py file implements the Snake game, providing state information and handling the game dynamics.

## Game State Representation

The game state is represented as a vector with details about obstacles and food location relative to the snake's head, as well as the snake's direction.
Action Space

The action space includes three possible moves:

- Turn Left
- Turn Right
- Move Straight

## Reward System

The reward structure is simple:

    +10 for eating food
    -10 for colliding with walls or itself
    -0.1 for each step without eating, discouraging idle movements

## Training the Agent

To start training the AI to play Snake, run the following command:

```
python agent.py
```
