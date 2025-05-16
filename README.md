# STA303-24F-Project: Reinforce Learning on FrozenLake

> Project Code Guidance of Artificial Intelligence B in 2024 Fall, Department of Statistics and Data Science, SUSTech

In this project, the course has provided basic DQN and Q-learning algorithms and visualization code.

On this basis, we have used DQN to perform hyperparameter analysis, formulated a new strategy for fine-tuning evaluation metric, benefit visualization, and implemented Double DQN and Dueling DQN algorithms to achieve a better performance.

## Code Structure

```
Project_6
├── __pycache__
├── agent
│   ├── __pycache__
│   └── base_agent.py
├── output
│   ├── double_dqn
│   ├── dqn
│   └── dueling_dqn
├── frozenlake_double_dqn.ipynb
├── frozenlake_dqn.ipynb
├── frozenlake_dueling_dqn_FinalPerformance.ipynb
├── README.md
├── requirements.txt
└── utils.py

```



## The Guidance of Running the Code

Below, we'll elaborate on how the code executes for all of the algorithms within the scope of our project's exploration. 

First, before running all the code for the algorithm part, make sure you've configured the environment according to the `requirement.txt`:

```txt
tqdm==4.66.5
matplotlib==3.7.5
pandas == 1.5.3
gymnasium==0.29.1
numpy==1.22.0
pygame==2.5.2
```

You can install this environment by running the first line in any of the given notebook files:

```python
!pip install -r requirements.txt
```

When a similar output appears in the terminal, the environment configuration is considered complete:

```python
Requirement already satisfied: contourpy>=1.0.1 in /Users/shellyli/anaconda3/lib/python3.11/site-packages (from matplotlib==3.7.5->-r requirements.txt (line 2)) (1.2.0)
Requirement already satisfied: cycler>=0.10 in /Users/shellyli/anaconda3/lib/python3.11/site-packages (from matplotlib==3.7.5->-r requirements.txt (line 2)) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in /Users/shellyli/anaconda3/lib/python3.11/site-packages (from matplotlib==3.7.5->-r requirements.txt (line 2)) (4.51.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /Users/shellyli/anaconda3/lib/python3.11/site-packages (from matplotlib==3.7.5->-r requirements.txt (line 2)) (1.4.4)
Requirement already satisfied: packaging>=20.0 in /Users/shellyli/anaconda3/lib/python3.11/site-packages (from matplotlib==3.7.5->-r requirements.txt (line 2)) (24.1)
Requirement already satisfied: pillow>=6.2.0 in /Users/shellyli/anaconda3/lib/python3.11/site-packages (from matplotlib==3.7.5->-r requirements.txt (line 2)) (10.4.0)
Requirement already satisfied: pyparsing>=2.3.1 in /Users/shellyli/anaconda3/lib/python3.11/site-packages (from matplotlib==3.7.5->-r requirements.txt (line 2)) (3.1.2)
Requirement already satisfied: python-dateutil>=2.7 in /Users/shellyli/anaconda3/lib/python3.11/site-packages (from matplotlib==3.7.5->-r requirements.txt (line 2)) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /Users/shellyli/anaconda3/lib/python3.11/site-packages (from pandas==1.5.3->-r requirements.txt (line 3)) (2024.1)
INFO: pip is looking at multiple versions of pandas to determine which version is compatible with other requirements. This could take a while.
```



### DQN

We will perform the main hyperparameter analysis for the DQN algorithm, and its file is `forzenlake_dqn.ipynb`.

This notebook contains two parts:

- Run the model with defailt parameters

- Hyperparameters tuning

After making sure that the environment is configured, the default DQN training can be carried out, in the first code block. This code block defines the most basic DQN method, with the default training parameters. After running, you can see the output like this:

```python
Episode: 100, Total Steps: 703, Ep Step: 5, Raw Reward: 0.00, Epsilon: 0.90
Episode: 200, Total Steps: 1458, Ep Step: 4, Raw Reward: 0.00, Epsilon: 0.82
Episode: 300, Total Steps: 2394, Ep Step: 8, Raw Reward: 0.00, Epsilon: 0.74
Episode: 400, Total Steps: 3360, Ep Step: 22, Raw Reward: 0.00, Epsilon: 0.67

~~~~~~Interval Save: Model saved.

Episode: 500, Total Steps: 4420, Ep Step: 17, Raw Reward: 0.00, Epsilon: 0.61
Episode: 600, Total Steps: 5290, Ep Step: 33, Raw Reward: 0.00, Epsilon: 0.55
...
~~~~~~Interval Save: Model saved.

Episode: 10000, Total Steps: 194590, Ep Step: 8, Raw Reward: 0.00, Epsilon: 0.01
Average reward of the last 100 episodes: 0.66
```

And it will output the relevant pictures of visualization.

Next, in the second part, the hyperparameters are fine-tuned, i.e., the next block of code is executed. Before executing the next block of code, find the `param_names` and `param_values` parameters and define them as needed. Here's an example of fine-tuning the `epsilon_max` parameter in increments of 0.01 in the interval [0.78, 0.82]:

``` python
# Parameter name and search space
param_names = ["epsilon_max"]
param_values = [np.arange(0.78, 0.82, 0.01)]  # Define parameter a, b and c: search from a to b, with a change of c each time
param_combinations = list(itertools.product(*param_values))
```

After that, you can run the code block, then the average reward of each last 100 episodes under the condition of this parameter will be calculated, and the relevant reward, loss and other images will be drawn, and finally the policy with the largest return in the interval will be taken for output, such as:

```python
Best hyperparams: {'epsilon_max': 0.8} with average reward: 0.77 and reward variance: 0.18
```

You can get the best policy of the `epsilon_max` parameter in the interval, with the relevant reward.



### Double DQN

The implementation of Double DQN is in notebook  `frozenlake_double_dqn.ipynb` . This notebook contains four parts:

- Environment Construction
- Run the model with defailt parameters
- Hyperparameters tuning
- Repetition of the best performance of Double DQN
- Random parameter adjustment

You can run them one by one to complete the training of the algorithm.



### Dueling DQN

Dueling DQN is the algorithm to achieve the best performance during our model research.

The implementation of Dueling DQN is in notebook  `frozenlake_dueling_dqn_FinalPerformance.ipynb` . This notebook contains two parts:

- Implement Dueling DQN algorithm with default parameters

- Repetition of the best performance of Deuling DQN with the best parameters combination

You can just simply run the first part to complete the construction of environment. And then just run the second part to repeat our best performance of Deuling DQN with the best parameters combinition.

After running the second part, and see the result:

```
Best Performance with average reward of last 100 episodes: 0.98
```

You successfully complete the repetition of the best performance of Dueling DQN, as well as our project!

