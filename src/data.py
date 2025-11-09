import polars as pl
import numpy as np
import grid2op

from pathlib import Path
from tqdm import tqdm
from typing import Optional

from grid2op.Environment import Environment
from grid2op.Observation import BaseObservation
from grid2op.Action import BaseAction
from grid2op.Backend import Backend
from src.logger import logger


class DatasetLoader:
    def __init__(
        self,
        env_name: str,
        backend: Backend,
        n_busbar: int,
        cache_dir: str,
        episode_count: int,
        n_actions: int,
        overwrite_cache: bool = False,
    ):
        self.env_name = env_name
        self.backend = backend
        self.n_busbar = n_busbar
        self.cache_dir = Path(cache_dir)
        self.episode_count = episode_count
        self.n_actions = n_actions
        self.overwrite_cache = overwrite_cache

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.features_path = self.cache_dir / "df_features.parquet"
        self.targets_path = self.cache_dir / "df_targets.parquet"

        self.env: Optional[Environment] = None
        self.all_actions: list[BaseAction] = []

    @staticmethod
    def _extract_features(obs: BaseObservation, act: BaseAction) -> np.ndarray:
        return np.concatenate(
            [
                obs.gen_p,
                obs.gen_q,
                obs.load_p,
                obs.load_q,
                obs.topo_vect,
                obs.rho,
                act.to_vect(),
            ]
        )

    def _create_realistic_observation(self) -> list[list[BaseObservation]]:
        list_obs = []
        for _ in tqdm(range(self.episode_count), desc="Episodes"):
            obs_per_episode = []
            obs = self.env.reset()
            obs_per_episode.append(obs)
            for _ in range(self.env.chronics_handler.max_timestep()):
                obs, _, done, _ = self.env.step(self.env.action_space())
                if done:
                    break
                obs_per_episode.append(obs)
            list_obs.append(obs_per_episode)
        return list_obs

    def _create_training_data(
        self,
        list_obs: list[list[BaseObservation]],
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        df_features = []
        df_targets = []
        for episode_id, obs_per_episode in enumerate(list_obs):
            for obs in tqdm(
                obs_per_episode,
                desc=f"Episode {episode_id}/{len(list_obs) - 1}",
            ):
                simulator = obs.get_simulator()
                actions = [
                    self.env.action_space.sample() for _ in range(self.n_actions)
                ]
                for act in actions:
                    sim_after_act = simulator.predict(act=act)
                    n_obs = sim_after_act.current_obs
                    df_targets.append(
                        np.array([n_obs.rho.max(), episode_id])
                        if sim_after_act.converged
                        else np.array([np.inf, episode_id])
                    )
                    df_features.append(
                        np.concatenate(
                            [
                                self._extract_features(obs, act),
                                np.array([int(episode_id)]),
                            ]
                        )
                    )

        features_df = pl.DataFrame(
            np.vstack(df_features),
            schema=[f"feature_{i}" for i in range(np.vstack(df_features).shape[1] - 1)]
            + ["episode_id"],
        )
        targets_df = pl.DataFrame(
            np.vstack(df_targets), schema=["target", "episode_id"]
        )
        features_df = features_df.with_columns(pl.col("episode_id").cast(pl.Int64))
        targets_df = targets_df.with_columns(pl.col("episode_id").cast(pl.Int64))
        return features_df, targets_df

    def load(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        if (
            self.features_path.exists()
            and self.targets_path.exists()
            and not self.overwrite_cache
        ):
            logger.info(
                "Dataset found in cache, loading from disk.",
                cache_dir=str(self.cache_dir),
            )
            return pl.read_parquet(self.features_path), pl.read_parquet(
                self.targets_path
            )
        logger.info(
            "Loading data from environment and saving to cache.",
            env_name=self.env_name,
        )
        self.env = grid2op.make(
            self.env_name, backend=self.backend, n_busbar=self.n_busbar
        )

        self.all_actions = [
            self.env.action_space.sample() for _ in range(self.n_actions)
        ]
        self.all_actions.append(self.env.action_space())

        list_obs = self._create_realistic_observation()
        df_features, df_targets = self._create_training_data(list_obs)

        df_features.write_parquet(self.features_path)
        df_targets.write_parquet(self.targets_path)

        return df_features, df_targets
