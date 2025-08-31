
# mujoco_erastic_rl

<p align="center">
  <img src="./assets/video/hard_cheetah_13m.gif" width="240" alt="demo gif">
  <img src="./assets/video/erastic_dump_14m.gif" width="240" alt="demo gif">
</p>

This is a reinforcement learning environment that uses the simplest PPO and Mujoco Half-Cheater as elastic bodies in a slippery setting.  
(最も単純なPPOとMujoco Half-Cheaterを滑りやすい設定の弾性体として使用する強化学習環境です。)   

In mujico's typical rigid body physics environment, reinforcement learning often generates unrealistic movements that exploit the repulsive forces at the contact surfaces of rigid bodies.
If this is the optimal movement that can be easily acquired in that environment, we experimented with what would happen if the environment were simply made slippery and bouncy.   
As expected, this alone allowed us to reproduce more natural cheetah movements.   

(mujico の通常の剛体物理環境では、強化学習によって剛体の接触面の反発力を利用した非現実的な動きが生成されることがよくあります。
もしこれがその環境で容易に獲得できる最適な動きだとしたら、環境を単純に滑りやすく弾力性のあるものにしたらどうなるかを実験しました。
予想通り、これだけでより自然なチーターの動きを再現することができました。)   

## Erastic and Slippery Half Cheetah config
assets/half_cheetah_soft.xml

## env (環境)

If you are installing it yourself in a proper environment from scratch, please refer to the following.  
You may encounter some dependency issues.  

(自分で最初から正式な環境でインストールする場合は以下を参考にしてください。
いくつか依存関係の問題が出るかもしれません。)  

 - Gymnasium  
 https://github.com/Farama-Foundation/Gymnasium

 - baselines3  
https://github.com/DLR-RM/stable-baselines3

## Installation

```bash
pip install -r requirement.txt
cd .\Gymnasium-1.1.1\
pip install -e ".[mujoco]"
pip install -e ".[other]"
cd ..
cd stable-baselines3-2.6.0
pip install -e ".[extra]"
cd ..
```

## How to Run
```bash
python main.py
```

## Memo 
   https://github.com/google-deepmind/mujoco/releases
   mujoco-3.3.2-windows-x86_64.zip

