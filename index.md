
---
title: PixyzRL: Bayesian RL Framework with Probabilistic Generative Models
description: Reinforcement Learning with Bayesian Inference and Probabilistic Generative Models
layout: default
---

# PixyzRL: Bayesian RL Framework with Probabilistic Generative Models

![PixyzRL](https://github.com/user-attachments/assets/577b9d4b-30d0-493d-95fc-b83a2f292c28)

## 概要
PixyzRLは、**確率的生成モデル**と**ベイズ理論**に基づいた強化学習フレームワークです。Pixyzライブラリの上に構築され、サンプル効率の向上や不確実性を考慮した意思決定が可能になります。

### 特徴
- **確率分布による方策最適化**
- **PPO（Proximal Policy Optimization）実装**
- **Gymnasium環境対応**
- **ロールアウトバッファによるメモリ管理**
- **強化学習用の柔軟なロガー機能**

## インストール方法

Python 3.10 以上が必要です。

```bash
pip install torch torchaudio torchvision pixyz gymnasium[box2d] torchrl
```

リポジトリをクローンしてインストールする場合:

```bash
git clone https://github.com/ItoMasaki/PixyzRL.git
cd PixyzRL
pip install -e .
```

## クイックスタート

### 1. 環境のセットアップ
```python
from pixyzrl.environments import Env

env = Env("CartPole-v1")
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
```

### 2. Actor-Criticネットワークの定義
```python
import torch
from pixyz.distributions import Categorical, Deterministic
from torch import nn

class Actor(Categorical):
    def __init__(self):
        super().__init__(var=["a"], cond_var=["o"], name="p")
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, o: torch.Tensor):
        return {"probs": self.net(o)}

class Critic(Deterministic):
    def __init__(self):
        super().__init__(var=["v"], cond_var=["o"], name="f")
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, o: torch.Tensor):
        return {"v": self.net(o)}

actor = Actor()
critic = Critic()
```

### 3. PPOエージェントのセットアップ
```python
from pixyzrl.models import PPO

agent = PPO(actor, critic, eps_clip=0.2, lr_actor=3e-4, lr_critic=1e-3, device="cpu", entropy_coef=0.0, mse_coef=1.0)
```

### 4. ロールアウトバッファの作成
```python
from pixyzrl.memory import RolloutBuffer

buffer = RolloutBuffer(
    2048,
    {
        "obs": {"shape": (4,), "map": "o"},
        "value": {"shape": (1,), "map": "v"},
        "action": {"shape": (2,), "map": "a"},
        "reward": {"shape": (1,)},
        "done": {"shape": (1,)},
        "returns": {"shape": (1,), "map": "r"},
        "advantages": {"shape": (1,), "map": "A"},
    },
    "cpu",
    1,
)
```

### 5. モデルのトレーニング
```python
from pixyzrl.trainer import OnPolicyTrainer

trainer = OnPolicyTrainer(env, buffer, agent, "cpu")
trainer.train(1000)
```

## ディレクトリ構成
```
PixyzRL
├── docs
│   └── pixyz
│       └── README.pixyz.md
├── examples  # サンプルコード
├── pixyzrl
│   ├── environments  # 環境ラッパー
│   ├── models
│   │   ├ on_policy  # オンポリシーモデル
│   │   └ off_policy  # オフポリシーモデル
│   ├── memory  # ロールアウトバッファ
│   ├── trainer  # トレーニングマネジメント
│   ├── losses  # 損失関数
│   ├── logger  # ロギング
│   └── utils.py
└── pyproject.toml
```

## ライセンス
PixyzRLはMITライセンスの下で公開されています。

## 作者
- **Masaki Ito** (l1sum [at] icloud.com)
- **Daisuke Nakahara**

## リポジトリ
[GitHub - ItoMasaki/PixyzRL](https://github.com/ItoMasaki/PixyzRL)

## 今後の予定
- [ ] `Trainer` の最適化
- [ ] `Logger` の拡張
- [ ] DQN、DDPG、SACなどのアルゴリズム実装
- [ ] Dreamerなどのモデルベース手法

## コミュニティ & サポート
詳細情報は以下をご覧ください。
[PixyzRL ChatGPT Page](https://chatgpt.com/g/g-67b7c36695fc8191aca4cb7420dad17c-pixyzrl)

