# SCII_Bots
<img src="assets/BC.jpg" style="zoom:100%"/>

Several agents able to play *[StarCraft II]((https://starcraft2.com/))* will be built in this repository!



### Initializing

##### Build basic develop environment

First of all, you need to download and install the game. Then follow the instructions below to build awesome battle bots!

The install packages from requirements file are all tested on windows 10, conda 4.9.2 and python 3.7.3.

Create a new conda environment with

```powershell
conda create -n SCII_Bots python=3.7.3
```

Install [git for python](https://anaconda.org/anaconda/git) with 

```powershell
conda install git
```

Install requirements with (*note that some of the packages may not need to be used*)

```powershell
pip install -r requirements.txt
```

If everything goes well, the followed result should be shown in the powershell/prompt.

<img src="assets/requirements_of_py37_clone.png" style="zoom:80%"/>

PS: the newest version of [PySC2](https://github.com/deepmind/pysc2) package is 3.0.0, which most of the codes published on websites are based on 2.x.x, so most of them are no longer available to run. 

PPS: PySC2 is very different with and more complex than [python-sc2](https://github.com/Dentosal/python-sc2). In this repository, we mainly focus on PySC2.



### Learning

##### introduction of game environment 

**State**: obtained from env.observation, including the feature screen, feature minimap and player info.

**Action**: try to determine what to do and where to go to win the game. There are two types of the actions include several basic action (currently 11) and a coordinate position with 64*64 points.

**Reward**: 

<u>*version 1:*</u>
$$
(score + total\_value\_units + total\_value\_structures + 10*killed\_value\_units + 10*killed\_value\_structures + collected\_minerals + collected\_rate\_minerals + 5*spent\_minerals - 8*idle\_work\_time) * 10e-6
$$
need further adjust.

<u>*version 2:*</u>

use `*spent_minerals*` to reward the action; use `*killed_value_units + killed_value_structures*` to reward attack point.

In summary,
$$
reward = [reward_a, reward_p]
$$
where
$$
reward_a = spent\_minerals * (10e^{-2})
$$

$$
reward_p = killed\_value\_units + killed\_value\_structures
$$

In addition, the reward will adjusted further to simulate the returns from environment more precisely. 

------

Run the environment test script as follows with

```powershell
python runner_basic_test.py
```

------

Train an DQN agent to play the game with 

```powershell
python runner_dqn.py
```

I also use the supervised value network to validate if the gradient update is worked, just run with

```python
python runner_nn_test.py
```



Train an A2C agent to play the game with （Coming Soon）

```powershell
python runner_a2c.py
```



##### details of neural agent

The structure of **DQN** neural agent is constructed as follows with pytorch 1.2.0.

The neural network model takes three different types of input tensors include 27 channels screen features, 11channels mini-map features and 11 channels player information features. 

Also, two different functional models, named as operation model and warfare model, shared several layers of the whole network and the inputs. And the two models output action value and value of attack position respectively.

<img src="assets\dqnagent-1621745942426.png" style="zoom:15%"/>



The structure of **A2C** neural agent is constructed as follows with pytorch 1.2.0. (coming soon)

<img src="assets/a2cagent-1621745942426.png" style="zoom:10%"/>



### evaluating

It costs much time but the a2c-agent does learn something like:

- Less Idle workers are better (maximum 6 at first, no more than 3 after take screenshot)

![img](file:///C:/Users/zz-guo/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)Episode 190

![img](file:///C:/Users/zz-guo/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)Episode 191



-  Too many engine bays are not useful (maximum 7 at first, only build one mostly at now)

![img](file:///C:/Users/zz-guo/AppData/Local/Temp/msohtmlclip1/01/clip_image008.jpg) episode 178

![img](file:///C:/Users/zz-guo/AppData/Local/Temp/msohtmlclip1/01/clip_image010.jpg)episode 204

 

- Action sequence about how to train marines and attack (before marines can be trained, minerals must enough, supply deport and barrack must be built)

![img](file:///C:/Users/zz-guo/AppData/Local/Temp/msohtmlclip1/01/clip_image012.jpg)episode 185

![img](file:///C:/Users/zz-guo/AppData/Local/Temp/msohtmlclip1/01/clip_image014.jpg)episode 187

There is no final results yet but COMING SOON ...



And finally,

<u>***En Taro Zeratul !!!***</u>



TODO List:

1. Standardize the game rules in environment;
2. Redesign the reward function.

### references

> https://github.com/deepmind/pysc2
>
> https://github.com/skjb/pysc2-tutorial
>
> https://github.com/Dentosal/python-sc2
>
> https://github.com/ClausewitzCPU0/SC2AI
>
> [Home · Dentosal/python-sc2 Wiki · GitHub](https://github.com/Dentosal/python-sc2/wiki)