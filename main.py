import numpy as np
import mujoco, ctypes, os
from pathlib import Path

HERE = Path(__file__).parent

import gymnasium
from gymnasium import spaces
#from gymnasium.wrappers import RecordEpisodeStatistics, RecordVideo
from gymnasium.wrappers import TimeLimit, RecordVideo, RecordEpisodeStatistics
from gymnasium.envs.mujoco import MujocoEnv
from stable_baselines3 import PPO

print("MuJoCo version :", mujoco.__version__)     # → 3.3.0 以上
print("Gymnasium ver. :", gymnasium.__version__)  # 1.1.1 でも問題なし


ASSETS = HERE / "assets"
#ASSETS = HERE / "Gymnasium-1.1.1/gymnasium/envs/mujoco/assets"

XML = ASSETS / "half_cheetah_soft.xml"
#XML = ASSETS / "half_cheetah.xml" # OK

VIDEO_DIR = HERE / "videos"
VIDEO_DIR.mkdir(exist_ok=True)

MAX_EPISODE_STEPS = 2000          # ♠ まとめて変数にしておく

class SoftHalfCheetahEnv(MujocoEnv):
    """Elastic Half-Cheetah that can render rgb_array for video logging."""
    def __init__(self, frame_skip: int = 5, render_mode="rgb_array"):
        # ---------- 1️⃣ 先に MuJoCo モデルを読み込み観測次元を算出 ----------
        model_tmp = mujoco.MjModel.from_xml_path(str(XML))
        obs_dim   = model_tmp.nq - 1 + model_tmp.nv   # qpos[1:] + qvel
        del model_tmp  
 
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32
        )
        # The base class will infer action_space from MuJoCo actuators,
        # so we don’t need to supply it here.
        # --------------------------------------------------------------

        super().__init__(
            model_path=str(XML),
            frame_skip=frame_skip,
            observation_space=obs_space,
            default_camera_config=dict(trackbodyid=2),
            render_mode=render_mode,                   # forwarded to Gymnasium
        )

        # noise range copied from Gymnasium’s stock HalfCheetah
        self.reset_noise_scale = 0.1        

        # ─── 頭部ジオメトリのIDを取得 ───
        # XML で定義されている name="head" ジオメトリの ID
        self.head_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, b"head"
        )  # ‘head’ geom の ID をキャッシュ        

        # ─── 前足位置の取得 ───
        # 前足を前に出せなくて、つんのめりやすい最初から前向きの角度にしておくため、書記位置を取得
        # joint の qpos アドレスを取得しておく
        self.fthigh_qpos_addr = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, b"fthigh"
        )
        self.fshin_qpos_addr = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, b"fshin"
        )

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()        

    # -----------------------------------------------------------
    # NEW: implement reset_model so env.reset() works
    def reset_model(self):
        noise = self.np_random.uniform(
            low=-self.reset_noise_scale,
            high= self.reset_noise_scale,
            size=self.model.nq + self.model.nv,
        )
        qpos = self.init_qpos + noise[: self.model.nq]
        qvel = self.init_qvel + noise[self.model.nq :]

        # 前足を前に出せなくて、つんのめりやすい最初から前向きの角度にしておくため、書記位置を取得
        # fthigh の初期角度を 1.5 rad(推薦0.5) に固定
        qpos[self.fthigh_qpos_addr] = -1.0
        qpos[self.fshin_qpos_addr] = 1.0

        self.set_state(qpos, qvel)
        return self._get_obs()

    # --- stock observation / reward (unchanged) ---
    def _get_obs(self):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        return np.concatenate([qpos[1:], qvel]).astype(np.float32)

    def step(self, action):
        xpos_before = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xpos_after = self.data.qpos[0]
        reward_run  = (xpos_after - xpos_before) / self.dt
        reward_ctrl = -0.1 * np.square(action).sum()
        reward = reward_run + reward_ctrl
        # ─── 頭部が床と接触したかチェック ───
        head_contact  = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if (c.geom1 == self.head_geom_id) or (c.geom2 == self.head_geom_id):
                head_contact  = True
                break
        done = False
        # 失敗ペナルティ
        if head_contact :
            reward += -1.0
            done = True
        obs, info = self._get_obs(), {}        
        return obs, reward, done, False, info



# ---------- helpers ----------
def every_50k_steps(step: int) -> bool:
    """Trigger video every 50 000 env steps (≈ 12.5 k PPO updates)."""
    return step % 50_000 == 0


def make_env():
    # ensure_solid_plugin()
    env = SoftHalfCheetahEnv()
    # MAX_EPISODE_STEPS . Force end of episode on step (ステップでエピソードを強制終了)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)    
    env = RecordEpisodeStatistics(env)          # logs return/length
    env = RecordVideo(
        env,
        video_folder=str(VIDEO_DIR),
        episode_trigger=lambda ep_no: ep_no % 50 == 0,  # 50 by episode
        #video_length=1000,                              # 1エピ全体なら None でもOK
        video_length=MAX_EPISODE_STEPS,          # None ならエピソード全体を録画
        name_prefix="soft-cheetah"
    )    

    return env

def is_plugin_registered(name: str) -> bool:
    """すでにプラグイン name が登録済みなら True"""
    count = mujoco.mj_pluginCount()
    for i in range(count):
        p = mujoco.mj_pluginAt(i)
        if ctypes.string_at(p.contents.name).decode() == name:
            return True
    return False

def ensure_solid_plugin():
    """mujoco.elasticity.solid が無ければ elasticity.dll をロード"""
    if is_plugin_registered("mujoco.elasticity.solid"):
        return  # already availabled.

    # elasticity.dll full path
    dll = Path(__file__).parent / "mujoco_plugin" / "elasticity.dll"
    try:
        mujoco.mj_loadPluginLibrary(str(dll))
    except mujoco.FatalError as e:
        # Only double registration of cable is suppressed. (cable の二重登録だけ握りつぶす)
        if "mujoco.elasticity.cable" in str(e):
            pass
        else:
            raise  # If there is another cause, re-throw. (別原因なら再スロー)


if __name__ == "__main__":
    env = make_env()

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4096,
        batch_size=4096,
        learning_rate=3e-4,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.03,
        verbose=1,
    )

    # —— Train ——
    #model.learn(total_timesteps=1_000_000)
    model.learn(total_timesteps=100_000_000)

    env.close()   # flush videos

    print(f"\nVideos saved to: {VIDEO_DIR.resolve()}")
