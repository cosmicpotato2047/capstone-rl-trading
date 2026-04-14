"""
src/agents/ppo_agent.py

stable-baselines3 PPO 래퍼.

BTCGridTradingEnv 위에서 PPO를 학습/평가/저장/로드하는 단일 인터페이스.
하이퍼파라미터는 experiment_config.yaml에서 읽는다.

사용 예:
    agent = PPOAgent(config, df_train, df_val)
    agent.train()
    metrics = agent.evaluate(df_val)
    agent.save("experiments/exp001/model")
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.env.trading_env import BTCGridTradingEnv
from src.evaluation.metrics import compute_all


# ─────────────────────────────────────────────────────────────────────────────
# Val 평가 콜백 (학습 중 주기적 Sharpe 측정)
# ─────────────────────────────────────────────────────────────────────────────

class ValMetricsCallback(BaseCallback):
    """
    eval_freq 스텝마다 Val 환경에서 1 에피소드를 실행하고
    Sharpe Ratio / 수익률 / MDD를 계산한다.

    최고 Sharpe 모델을 best_model_path에 자동 저장한다.
    MLflow가 활성화된 경우 지표를 함께 기록한다.
    """

    def __init__(
        self,
        df_val: pd.DataFrame,
        config: dict,
        eval_freq: int,
        best_model_path: str,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.df_val         = df_val
        self.config         = config
        self.eval_freq      = eval_freq
        self.best_model_path = best_model_path
        self.best_sharpe    = -np.inf
        self.initial_cash   = config["environment"]["initial_cash"]

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq != 0:
            return True

        metrics = _evaluate_once(self.model, self.df_val, self.config)
        sharpe  = metrics["sharpe_ratio"]
        ret     = metrics["total_return_pct"]
        mdd     = metrics["max_drawdown_pct"]

        if self.verbose >= 1:
            print(
                f"  [eval] step={self.num_timesteps:>8,} | "
                f"Sharpe={sharpe:+.3f} | "
                f"Return={ret:+.2f}% | "
                f"MDD={mdd:.2f}%"
            )

        # MLflow 로깅 (임포트 실패 시 무시)
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
            if self.verbose >= 1:
                print(f"  [best] Sharpe {sharpe:.3f} → {self.best_model_path}")

        return True


# ─────────────────────────────────────────────────────────────────────────────
# 내부 헬퍼: 단일 에피소드 평가
# ─────────────────────────────────────────────────────────────────────────────

def _evaluate_once(model: PPO, df: pd.DataFrame, config: dict) -> dict:
    """
    학습된 모델로 df를 1 에피소드 실행하고 metrics dict를 반환한다.
    deterministic=True (탐색 없이 최적 행동).
    """
    env = BTCGridTradingEnv(df, config)
    obs, _ = env.reset()
    equity_list = [config["environment"]["initial_cash"]]
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        price = float(env.df.loc[env.current_step - 1, "close"])
        equity_list.append(env.cash + env.holdings * price)

    equity_curve = pd.Series(equity_list, dtype=float)
    return compute_all(
        equity_curve=equity_curve,
        initial_cash=config["environment"]["initial_cash"],
        n_trades=env.n_trades,
        completed_cycles=env.completed_cycles,
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
        self.config    = config
        self.df_train  = df_train
        self.df_val    = df_val

        agent_cfg = config["agent"]
        train_cfg = config["training"]

        self.seed          = train_cfg["seed"]
        self.eval_freq     = train_cfg["eval_freq"]
        self.total_timesteps = agent_cfg["total_timesteps"]
        self.log_dir       = train_cfg["log_dir"]

        # Train 환경 (DummyVecEnv — SB3 필수)
        self.train_env = DummyVecEnv([
            lambda: BTCGridTradingEnv(df_train, config)
        ])

        # tensorboard 설치 여부 확인
        try:
            import tensorboard  # noqa: F401
            tb_log = os.path.join(self.log_dir, "tb_logs")
        except ImportError:
            tb_log = None   # tensorboard 미설치 시 비활성

        # PPO 모델 생성
        self.model = PPO(
            policy          = agent_cfg["policy"],
            env             = self.train_env,
            learning_rate   = agent_cfg["learning_rate"],
            n_steps         = agent_cfg["n_steps"],
            batch_size      = agent_cfg["batch_size"],
            n_epochs        = agent_cfg["n_epochs"],
            gamma           = agent_cfg["gamma"],
            gae_lambda      = agent_cfg["gae_lambda"],
            clip_range      = agent_cfg["clip_range"],
            ent_coef        = agent_cfg["ent_coef"],
            vf_coef         = agent_cfg["vf_coef"],
            max_grad_norm   = agent_cfg["max_grad_norm"],
            verbose         = 1,
            seed            = self.seed,
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

        Args:
            total_timesteps : 학습 스텝 수 (None이면 config 값 사용)
            best_model_path : 최고 Sharpe 모델 저장 경로
                              (None이면 {log_dir}/best_model)
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
                    verbose=1,
                )
            )

        print(f"\n학습 시작: {total_timesteps:,} 스텝")
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

        Returns:
            compute_all() 결과 dict +
            "equity_curve": pd.Series
        """
        env = BTCGridTradingEnv(df, self.config)
        obs, _ = env.reset()
        equity_list = [self.config["environment"]["initial_cash"]]
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            price = float(env.df.loc[env.current_step - 1, "close"])
            equity_list.append(env.cash + env.holdings * price)

        equity_curve = pd.Series(equity_list, dtype=float)
        metrics = compute_all(
            equity_curve=equity_curve,
            initial_cash=self.config["environment"]["initial_cash"],
            n_trades=env.n_trades,
            completed_cycles=env.completed_cycles,
        )
        metrics["equity_curve"] = equity_curve
        metrics["completed_cycles"] = env.completed_cycles
        return metrics

    # ── 저장 / 로드 ────────────────────────────────────────────

    def save(self, path: str) -> None:
        """모델을 path에 저장한다 (.zip 자동 추가)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
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
        """저장된 모델을 로드하고 PPOAgent 인스턴스를 반환한다."""
        agent = cls(config, df_train, df_val)
        agent.model = PPO.load(path, env=agent.train_env)
        return agent
