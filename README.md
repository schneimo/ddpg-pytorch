# ddpg-pytorch
PyTorch implementation of DDPG for continuous control tasks.

This is a PyTorch implementation of Deep Deterministic Policy Gradients developed in [CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/abs/1509.02971).

<p align="center"> 
<img src="_assets/roboschool_swingup.gif">
</p>

This implementation is inspired by the OpenAI baseline of [DDPG](https://github.com/openai/baselines/tree/master/baselines/ddpg), the newer [TD3](https://github.com/sfujim/TD3) implementation and also various other resources about DDPG. But instead of parameter space noise this implementation uses the original Ornstein-Uhlenbeck noise process of the original DDPG implementation.

## Tested environments (via [OpenAI Gym](https://gym.openai.com))

* [OpenAI Roboschool](https://github.com/openai/roboschool)

Since 'Roboschool' is deprecated, I highly recommend using [PyBullet](http://pybullet.org) instead (also recommended by OpenAI).

## Requirements

* Python 3
* TensorBoard
* TQDM
* [PyTorch](http://pytorch.org/)
* [OpenAI gym](https://github.com/openai/baselines)
* [OpenAI Roboschool](https://github.com/openai/baselines)

## Training

```bash
python train.py --env "RoboschoolInvertedPendulumSwingup-v1"
```

## Testing

```bash
python test.py --env "RoboschoolInvertedPendulumSwingup-v1"
```

## Pretrained models

Pretrained models can be found in [the folder 'saved_models'](saved_models/) for the *'RoboschoolInvertedPendulumSwingup-v1'* and the *'RoboschoolInvertedPendulum-v1'* environments.

## Contributions

Contributions are welcome. If you find any bugs, know how to make the code better or want to implement other used methods regarding DDPG, please open an issue or a pull request.

## Disclaimer

This repo is an attempt to reproduce results of Reinforcement Learning methods to gain a deeper understanding of the developed concepts. But even with quite numerus other reproductions, an own reproduction is a quite difficult task even today. In ["Deep Reinforcement Learning that Matters"](https://arxiv.org/abs/1709.06560) you can read more about reproducibility of Reinforcement Learning methods. I tried to reproduce the original paper and the OpenAI implementation as close as possible, but I wanted to use Roboschool also. This made the task more difficult, since there are no benchmarks for DDPG with Roboschool and thus the choice of hyperparameters was much more difficult.
