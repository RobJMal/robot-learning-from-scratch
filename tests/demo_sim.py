# import os
# os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# import gymnasium as gym
# import gym_pusht  # registers the env
# import imageio.v2 as imageio
# import numpy as np

# env = gym.make("gym_pusht/PushT-v0",
#                obs_type="environment_state_agent_pos",
#                render_mode="rgb_array")
# obs, info = env.reset(seed=0)
# print({k: (v.shape, v.min(), v.max()) for k, v in obs.items()})
# print("action space:", env.action_space)

# frames = [env.render()]
# for step in range(300):
#     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     frames.append(env.render())
#     if step % 50 == 0:
#         print(step, reward, terminated, truncated)
#     if terminated or truncated:
#         print("episode ended at step", step)
#         break

# imageio.mimsave("results/random_rollout.mp4", frames, fps=10)

import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gymnasium as gym
import gym_pusht  # registers the env
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

FPS = 10
waypoints = [(100, 100), (400, 100), (400, 400), (100, 400)]
steps_per_target = 30

env = gym.make("gym_pusht/PushT-v0",
               obs_type="environment_state_agent_pos",
               render_mode="rgb_array")
obs, _ = env.reset(seed=0)

frames, agent, targets = [], [], []
for wp in waypoints:
    target = np.array(wp, dtype=np.float32)
    for _ in range(steps_per_target):
        obs, *_ = env.step(target)
        frames.append(env.render())
        agent.append(obs["agent_pos"].copy())
        targets.append(target.copy())
agent, targets = np.array(agent), np.array(targets)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(frames[0], extent=[0, 512, 512, 0])  # world coords, y pointing down
im = ax.images[0]
ax.set_xlim(0, 512)
ax.set_ylim(512, 0)
state_ring, = ax.plot([], [], "o", mfc="none", mec="lime", ms=16, mew=2,
                      label="state (pusher)")
target_x, = ax.plot([], [], "x", color="red", ms=12, mew=3, label="action (target)")
gap, = ax.plot([], [], "-", color="red", lw=1.5)
ax.legend(loc="upper right", fontsize=8)


def update(i):
    im.set_data(frames[i])
    state_ring.set_data([agent[i, 0]], [agent[i, 1]])
    target_x.set_data([targets[i, 0]], [targets[i, 1]])
    gap.set_data([agent[i, 0], targets[i, 0]], [agent[i, 1], targets[i, 1]])
    dist = np.linalg.norm(agent[i] - targets[i])
    ax.set_title(f"step {i}   gap to target: {dist:.0f} px")
    return im, state_ring, target_x, gap


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS)

if animation.writers.is_available("ffmpeg"):
    out = "results/step_response.mp4"
    writer = animation.FFMpegWriter(fps=FPS)
else:
    print("ffmpeg not found, writing a GIF instead")
    out = "results/step_response.gif"
    writer = animation.PillowWriter(fps=FPS)

anim.save(out, writer=writer, dpi=100)
print("saved", out)