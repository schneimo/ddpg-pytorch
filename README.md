# ddpg-pytorch
PyTorch implementation of DDPG for continous control tasks.

This is a PyTorch implementation of Deep Deterministic Policy Gradients proposed in [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/abs/1509.02971).

This implementation is inspired by the OpenAI baselines for [DDPG](https://github.com/openai/baselines/tree/master/baselines/ddpg) and the newer [TD3](https://github.com/sfujim/TD3) implementation. But instead of parameter space noise this iplementation uses the original Ornstein-Uhlenbeck noise process of the original DDPG implementation.

## Tested environments (via [OpenAI Gym](https://gym.openai.com))

* [OpenAI Roboschool](https://github.com/openai/roboschool)

Since 'Roboschool' is deprecated, I highly recommend using [PyBullet](http://pybullet.org) instead (also recommended by OpenAI).

## Requirements

* Python 3
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)
* [OpenAI gym](https://github.com/openai/baselines)
* [OpenAI Roboschool](https://github.com/openai/baselines)

## Training


```bash
python train.py --env-name "RoboschoolInvertedPendulumSwingup-v1"
```

## Pretrained models

Pretrained models can be found in ![the folder 'saved_models'](saved_models/) for the *'RoboschoolInvertedPendulumSwingup-v1'* and the *'RoboschoolInvertedPendulum-v1'* environments.

## Contributions

Contributions are welcome. If you find any bugs, know how to make the code better or want to implement used other used methods regarding DDPG, please open an issue or a pull request.

## Disclaimer

It's extremely difficult to reproduce results for Reinforcement Learning methods. See ["Deep Reinforcement Learning that Matters"](https://arxiv.org/abs/1709.06560) for more information. I tried to reproduce the original paper and the OpenAI implementation as close as possible, but I wanted to use Roboschool also. This made the task more difficult, since there are no benchmarks for DDPG with Roboschool.
