"""
src/agents/ppo_agent.py

stable-baselines3 PPO 래퍼.

BTCGridTradingEnv 위에서 PPO를 학습/평가/저장/로드하는 단일 인터페이스.
하이퍼파라미터는 experiment_config.yaml에서 읽는다.

exp003 이후 지원:
  - VecNormalize: config["vec_normalize"]["enabled"] = True 시 자동 활성화
  - LR 스케줄: config["agent"]["lr_schedule"] = "cosine" | "linear" | "constant"

사용 예:
    agent = PPOAgent(config, df_train, df_val)
    agent.train()
    metrics = agent.evaluate(df_val)
    agent.save("experiments/exp003/model")
"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env.trading_env import BTCGridTradingEnv
from src.evaluation.metrics import compute_all


# ─────────────────────────────────────────────────────────────────────────────
# LR 스케줄 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _build_lr_schedule(agent_cfg: dict) -> float | Callable[[float], float]:
    """
    config["agent"]["lr_schedule"] 에 따라 LR 스케줄 함수 또는 고정값을 반환한다.

    SB3 PPO는 learning_rate에 float 또는 Callable[[progress_remaining], float]를 받는다.
    progress_remaining: 1.0 (학습 시작) → 0.0 (학습 종료)

    지원 스케줄:
        "constant" : 고정 LR (기본값)
        "linear"   : lr_start → lr_end 선형 감소
        "cosine"   : lr_start → lr_end cosine annealing
    """
    lr_start = float(agent_cfg["learning_rate"])
    schedule  = agent_cfg.get("lr_schedule", "constant")

    if schedule == "constant":
        return lr_start

    lr_end = float(agent_cfg.get("lr_end", lr_start * 0.1))

    if schedule == "linear":
        def linear_schedule(progress_remaining: float) -> float:
            return lr_end + (lr_start - lr_end) * progress_remaining
        return linear_schedule

    if schedule == "cosine":
        def cosine_schedule(progress_remaining: float) -> float:
            progress = 1.0 - progress_remaining           # 0 → 1
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return lr_end + (lr_start - lr_end) * cosine_factor
        return cosine_schedule

    raise ValueError(f"알 수 없는 lr_schedule: {schedule!r}. 'constant'|'linear'|'cosine' 중 선택")


# ─────────────────────────────────────────────────────────────────────────────
# VecNormalize 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _vecnorm_path(model_path: str) -> str:
    """모델 경로에서 VecNormalize 저장 경로를 파생한다."""
    # "experiments/exp003/best_model" → "experiments/exp003/best_model_vecnorm.pkl"
    return str(model_path).rstrip(".zip") + "_vecnorm.pkl"


def _get_base_env(vec_env) -> BTCGridTradingEnv:
    """VecNormalize / DummyVecEnv 래퍼에서 실제 BTCGridTradingEnv를 꺼낸다."""
    if isinstance(vec_env, VecNormalize):
        return vec_env.venv.envs[0]
    return vec_env.envs[0]  # DummyVecEnv


# ─────────────────────────────────────────────────────────────────────────────
# Val 평가 콜백
# ─────────────────────────────────────────────────────────────────────────────

class ValMetricsCallback(BaseCallback):
    """
    eval_freq 스텝마다 Val 환경에서 1 에피소드를 실행하고
    Sharpe Ratio / 수익률 / MDD를 계산한다.

    최고 Sharpe 모델을 best_model_path에 자동 저장한다.
    VecNormalize 사용 시 통계도 함께 저장한다.
    MLflow가 활성화된 경우 지표를 함께 기록한다.
    """

    def __init__(
        self,
        df_val: pd.DataFrame,
        config: dict,
        eval_freq: int,
        best_model_path: str,
        train_env,          # DummyVecEnv 또는 VecNormalize — eval 시 obs_rms 복사용
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.df_val          = df_val
        self.config          = config
        self.eval_freq       = eval_freq
        self.best_model_path = best_model_path
        self.train_env       = train_env
        self.best_sharpe     = -np.inf
        self.initial_cash    = config["environment"]["initial_cash"]

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        metrics = _evaluate_once(
            model=self.model,
            df=self.df_val,
            config=self.config,
            train_env=self.train_env,
        )
        sharpe = metrics["sharpe_ratio"]
        ret    = metrics["total_return_pct"]
        mdd    = metrics["max_drawdown_pct"]

        if self.verbose >= 1:
            print(
                f"  [eval] step={self.num_timesteps:>8,} | "
                f"Sharpe={sharpe:+.3f} | "
                f"Return={ret:+.2f}% | "
                f"MDD={mdd:.2f}%"
            )

        # MLflow 로깅
        try:
            import mlflow
            if mlflow.active_run():
                mlflow.log_metrics(
                    {
                        "val_sharpe":     sharpe,
                        "val_return_pct": ret,
                        "val_mdd_pct":    mdd,
                        "val_n_trades":   metrics["n_trades"],
                    },
                    step=self.num_timesteps,
                )
        except Exception:
            pass

        # 최고 Sharpe 모델 저장
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            Path(self.best_model_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.best_model_path)

            # VecNormalize 통계 저장 (있을 경우)
            if isinstance(self.train_env, VecNormalize):
                self.train_env.save(_vecnorm_path(self.best_model_path))

            if self.verbose >= 1:
                print(f"  [best] Sharpe {sharpe:.3f} → {self.best_model_path}")

        return True


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼: 단일 에피소드 평가
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_once(
    model: PPO,
    df: pd.DataFrame,
    config: dict,
    train_env=None,
) -> dict:
    """
    학습된 모델로 df를 1 에피소드 실행하고 metrics dict를 반환한다.
    deterministic=True (탐색 없이 최적 행동).

    Args:
        model      : 학습된 PPO 모델
        df         : 평가 데이터 (val 또는 test)
        config     : 실험 설정 dict
        train_env  : VecNormalize 인스턴스 (있으면 obs 정규화 통계 복사).
                     None이면 정규화 없이 실행.
    """
    initial_cash = config["environment"]["initial_cash"]

    # 평가 환경 생성
    base_env = DummyVecEnv([lambda: BTCGridTradingEnv(df, config)])

    if isinstance(train_env, VecNormalize):
        # 학습 통계를 복사해서 inference-only VecNormalize로 감쌈
        eval_env = VecNormalize(base_env, training=False, norm_reward=False)
        eval_env.obs_rms  = deepcopy(train_env.obs_rms)
        eval_env.clip_obs = train_env.clip_obs
        underlying = eval_env.venv.envs[0]
    else:
        eval_env  = base_env
        underlying = base_env.envs[0]

    obs = eval_env.reset()
    equity_list = [initial_cash]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, dones, _infos = eval_env.step(action)
        done = bool(dones[0])
        price = float(underlying.df.loc[underlying.current_step - 1, "close"])
        equity_list.append(underlying.cash + underlying.holdings * price)

    eval_env.close()

    equity_curve = pd.Series(equity_list, dtype=float)
    return compute_all(
        equity_curve=equity_curve,
        initial_cash=initial_cash,
        n_trades=underlying.n_trades,
        completed_cycles=underlying.completed_cycles,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PPOAgent
# ─────────────────────────────────────────────────────────────────────────────

class PPOAgent:
    """
    BTCGridTradingEnv + stable-baselines3 PPO 래퍼.

    Args:
        config   : experiment_config.yaml 로드 결과
        df_train : Train 파티션 DataFrame
        df_val   : Val 파티션 DataFrame (학습 중 평가용, None이면 콜백 비활성)
    """

    def __init__(
        self,
        config: dict,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame | None = None,
    ):
        self.config   = config
        self.df_train = df_train
        self.df_val   = df_val

        agent_cfg  = config["agent"]
        train_cfg  = config["training"]
        vec_cfg    = config.get("vec_normalize", {})

        self.seed            = train_cfg["seed"]
        self.eval_freq       = train_cfg["eval_freq"]
        self.total_timesteps = agent_cfg["total_timesteps"]
        self.log_dir         = train_cfg["log_dir"]
        self.use_vec_norm    = vec_cfg.get("enabled", False)

        # ── Train 환경 ──────────────────────────────────────────
        base_env = DummyVecEnv([lambda: BTCGridTradingEnv(df_train, config)])

        if self.use_vec_norm:
            self.train_env = VecNormalize(
                base_env,
                norm_obs     = vec_cfg.get("norm_obs", True),
                norm_reward  = vec_cfg.get("norm_reward", True),
                clip_obs     = float(vec_cfg.get("clip_obs", 10.0)),
                clip_reward  = float(vec_cfg.get("clip_reward", 10.0)),
                gamma        = float(vec_cfg.get("gamma", agent_cfg["gamma"])),
            )
        else:
            self.train_env = base_env

        # ── LR 스케줄 ───────────────────────────────────────────
        lr = _build_lr_schedule(agent_cfg)

        # ── TensorBoard ─────────────────────────────────────────
        try:
            import tensorboard  # noqa: F401
            tb_log = os.path.join(self.log_dir, "tb_logs")
        except ImportError:
            tb_log = None

        # ── PPO 모델 ─────────────────────────────────────────────
        self.model = PPO(
            policy         = agent_cfg["policy"],
            env            = self.train_env,
            learning_rate  = lr,
            n_steps        = agent_cfg["n_steps"],
            batch_size     = agent_cfg["batch_size"],
            n_epochs       = agent_cfg["n_epochs"],
            gamma          = agent_cfg["gamma"],
            gae_lambda     = agent_cfg["gae_lambda"],
            clip_range     = agent_cfg["clip_range"],
            ent_coef       = agent_cfg["ent_coef"],
            vf_coef        = agent_cfg["vf_coef"],
            max_grad_norm  = agent_cfg["max_grad_norm"],
            verbose        = 1,
            seed           = self.seed,
            tensorboard_log = tb_log,
        )

    # ── 학습 ──────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: int | None = None,
        best_model_path: str | None = None,
    ) -> None:
        """
        PPO 학습 실행.

        학습 완료 후 VecNormalize 통계를 모델 경로 옆에 자동 저장한다.

        Args:
            total_timesteps : 학습 스텝 수 (None이면 config 값 사용)
            best_model_path : 최고 Sharpe 모델 저장 경로
        """
        total_timesteps = total_timesteps or self.total_timesteps
        best_model_path = best_model_path or os.path.join(self.log_dir, "best_model")

        callbacks = []
        if self.df_val is not None:
            callbacks.append(
                ValMetricsCallback(
                    df_val=self.df_val,
                    config=self.config,
                    eval_freq=self.eval_freq,
                    best_model_path=best_model_path,
                    train_env=self.train_env,
                    verbose=1,
                )
            )

        print(f"\n학습 시작: {total_timesteps:,} 스텝")
        if self.use_vec_norm:
            print("  VecNormalize: 활성 (norm_obs=True, norm_reward=True)")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            reset_num_timesteps=True,
        )
        print("학습 완료.")

    # ── 평가 ──────────────────────────────────────────────────

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        학습된 모델로 df를 1 에피소드 실행하고 metrics dict를 반환한다.

        VecNormalize 활성 시 학습 통계를 복사해 inference-only로 실행한다.

        Returns:
            compute_all() 결과 dict + "equity_curve" + "completed_cycles"
        """
        metrics = _evaluate_once(
            model=self.model,
            df=df,
            config=self.config,
            train_env=self.train_env if self.use_vec_norm else None,
        )

        # equity_curve, completed_cycles는 별도로 수집 (compute_all에 없음)
        # _evaluate_once를 직접 확장하지 않고 한 번 더 실행하는 대신
        # 내부 결과에 equity_curve/completed_cycles를 덧붙임
        initial_cash = self.config["environment"]["initial_cash"]
        base_env = DummyVecEnv([lambda: BTCGridTradingEnv(df, self.config)])

        if self.use_vec_norm:
            eval_env = VecNormalize(base_env, training=False, norm_reward=False)
            eval_env.obs_rms  = deepcopy(self.train_env.obs_rms)
            eval_env.clip_obs = self.train_env.clip_obs
            underlying = eval_env.venv.envs[0]
        else:
            eval_env   = base_env
            underlying = base_env.envs[0]

        obs = eval_env.reset()
        equity_list = [initial_cash]
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _reward, dones, _infos = eval_env.step(action)
            done = bool(dones[0])
            price = float(underlying.df.loc[underlying.current_step - 1, "close"])
            equity_list.append(underlying.cash + underlying.holdings * price)

        eval_env.close()

        equity_curve = pd.Series(equity_list, dtype=float)
        metrics = compute_all(
            equity_curve=equity_curve,
            initial_cash=initial_cash,
            n_trades=underlying.n_trades,
            completed_cycles=underlying.completed_cycles,
        )
        metrics["equity_curve"]      = equity_curve
        metrics["completed_cycles"]  = underlying.completed_cycles
        return metrics

    # ── 저장 / 로드 ────────────────────────────────────────────

    def save(self, path: str) -> None:
        """
        모델을 path에 저장한다.
        VecNormalize 사용 시 {path}_vecnorm.pkl도 함께 저장한다.
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        if self.use_vec_norm:
            vn_path = _vecnorm_path(path)
            self.train_env.save(vn_path)
            if self.model.verbose >= 1:
                print(f"모델 저장: {path}.zip  |  VecNormalize: {vn_path}")
        else:
            if self.model.verbose >= 1:
                print(f"모델 저장: {path}.zip")

    @classmethod
    def load(
        cls,
        path: str,
        config: dict,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame | None = None,
    ) -> "PPOAgent":
        """
        저장된 모델을 로드하고 PPOAgent 인스턴스를 반환한다.
        VecNormalize 통계 파일({path}_vecnorm.pkl)이 있으면 함께 로드한다.
        """
        agent = cls(config, df_train, df_val)
        agent.model = PPO.load(path, env=agent.train_env)

        vn_path = _vecnorm_path(path)
        if agent.use_vec_norm and Path(vn_path).exists():
            agent.train_env = VecNormalize.load(vn_path, agent.train_env.venv)
            agent.train_env.training = False  # inference only
            print(f"VecNormalize 로드: {vn_path}")

        return agent
