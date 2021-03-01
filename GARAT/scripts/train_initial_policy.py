"""Script good initial policy on some environement"""
import gym
from stable_baselines.common.policies import MlpPolicy as mlp_standard
from stable_baselines.common.policies import FeedForwardPolicy as ffp_standard
from stable_baselines.sac.policies import MlpPolicy as mlp_sac
from stable_baselines.sac.policies import FeedForwardPolicy as ffp_sac
from stable_baselines.td3.policies import MlpPolicy as mlp_td3
from stable_baselines.td3.policies import FeedForwardPolicy as ffp_td3
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines import SAC, TD3, TRPO, PPO2, ACKTR
from stable_baselines.ddpg.noise import NormalActionNoise
from rl_gat.gat import NoisyRealEnv
from stable_baselines.common.callbacks import EvalCallback
import numpy as np
import yaml, shutil, os
from scripts.utils import MujocoNormalized

ALGO = TRPO
# set the environment here :
ENV_NAME = 'HopperArmatureModified-v2'
# set this to the parent environment
PARAMS_ENV = 'Hopper-v2'
TIME_STEPS = 2000000
NOISE_VALUE = 0.0
SAVE_BEST_FOR_20 = False
NORMALIZE = False
MUJOCO_NORMALIZE = False

if NORMALIZE is True and MUJOCO_NORMALIZE is True:
    raise ValueError('That is not possible !')

if 'Bullet' in ENV_NAME:
    from pybullet_envs import *



if NOISE_VALUE == 0.0 or NOISE_VALUE == 0:
    if SAVE_BEST_FOR_20:
        model_name = "data/models/" + ALGO.__name__ + "_initial_policy_steps_" + ENV_NAME + "_" + str(
            TIME_STEPS) + "_best_.pkl"
    else:
        model_name = "data/models/" + ALGO.__name__ + "_initial_policy_steps_" + ENV_NAME + "_" + str(
            TIME_STEPS) + "_.pkl"
else:
    if SAVE_BEST_FOR_20:
        model_name = "data/models/" + ALGO.__name__ + "_initial_policy_steps_" + ENV_NAME + "_" + str(
            TIME_STEPS) + "_NOISY_" + str(NOISE_VALUE) + "_best_.pkl"
    else:
        model_name = "data/models/" + ALGO.__name__ + "_initial_policy_steps_" + ENV_NAME + "_" + str(
            TIME_STEPS) + "_NOISY_" + str(NOISE_VALUE) + "_.pkl"

if 'Dart' in ENV_NAME:
    import rl_gat
    import pybullet


# Separate evaluation env
if SAVE_BEST_FOR_20:
    eval_env = DummyVecEnv([lambda : gym.make(ENV_NAME)])
    if NORMALIZE:
        eval_env = VecNormalize(eval_env,
                                training=True,
                                norm_obs=True,
                                norm_reward=False,
                                clip_reward=1e6,
                                )


    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=model_name[:-4],
                                 n_eval_episodes=30,
                                 eval_freq=5000,
                                 deterministic=True,
                                 render=False,
                                 verbose=1)

    def save_the_model():
        shutil.move(model_name[:-4]+'/best_model.zip', model_name)
        try:
            os.rmdir(model_name[:-4])
            print('Successfully saved the model.')
        except Exception as e:
            print(e)


def evaluate_policy_on_env(env,
                           model,
                           render=True,
                           iters=1,
                           deterministic=True
                           ):
    # model.set_env(env)
    return_list = []
    for i in range(iters):
        return_val = 0
        done = False
        obs = env.reset()
        while not done:
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, rewards, done, info = env.step(action)
            return_val+=rewards
            if render:
                env.render()
                # time.sleep(0.01)

        if not i%15: print('Iteration ', i, ' done.')
        return_list.append(return_val)
    print('***** STATS FOR THIS RUN *****')
    print('MEAN : ', np.mean(return_list))
    print('STD : ', np.std(return_list))
    print('******************************')
    return np.mean(return_list), np.std(return_list)/np.sqrt(len(return_list))


def train_initial_policy(
        model_name,
        algo=ALGO,
        env_name=ENV_NAME,
        time_steps=TIME_STEPS):
    """Uses the specified algorithm on the target environment"""
    print("Using algorithm : ", algo.__name__)
    print("Model saved as : ", "data/models/" +algo.__name__+"_initial_policy_"+env_name+"_.pkl")

    # define the environment here
    env = gym.make(env_name)
    if NOISE_VALUE>0 : env = NoisyRealEnv(env, noise_value=NOISE_VALUE)

    if MUJOCO_NORMALIZE:
        env = MujocoNormalized(env)

    print('~~ ENV Obs RANGE : ', env.observation_space.low, env.observation_space.high)
    print('~~~ ENV Action RANGE : ', env.action_space.low, env.action_space.high)

    if algo.__name__  == "ACKTR":
        print('Using SubprovVecEnv')
        env = SubprocVecEnv([lambda: env for i in range(8)])
    elif algo.__name__ == "SAC":
        print('Using standard gym environment')
        env = env
    else:
        print('Using Dummy Vec Env')
        env = DummyVecEnv([lambda : env])

    if NORMALIZE :
        env = VecNormalize(env,
                           training=True,
                           norm_obs=True,
                           norm_reward=False,
                           clip_reward=1e6,
                           )


    with open('data/target_policy_params.yaml') as file:
        args = yaml.load(file, Loader=yaml.FullLoader)
    args = args[algo.__name__][PARAMS_ENV]
    print('~~ Loaded args file ~~')

    if algo.__name__ == "SAC":
        print('Initializing SAC with RLBaselinesZoo hyperparameters .. ')
        print('using 256 node architecture as in the paper')

        class CustomPolicy(ffp_sac):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy, self).__init__(*args, **kwargs,
                                                   feature_extraction="mlp", layers=[256, 256])

        model = SAC(CustomPolicy, env,
                    verbose=1,
                    tensorboard_log='data/TBlogs/initial_policy_training',
                    batch_size=args['batch_size'],
                    buffer_size=args['buffer_size'],
                    ent_coef=args['ent_coef'],
                    learning_starts=args['learning_starts'],
                    learning_rate=args['learning_rate'],
                    train_freq=args['train_freq'],
                    )
    elif algo.__name__ == "TD3":
        print('Initializing TD3 with RLBaselinesZoo hyperparameters .. ')
        # hyperparameters suggestions from :
        # https://github.com/araffin/rl-baselines-zoo/blob/master/trained_agents/td3/HopperBulletEnv-v0/config.yml
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                         sigma=float(args['noise_std']) * np.ones(n_actions))
        class CustomPolicy2(ffp_td3):
            def __init__(self, *args, **kwargs):
                super(CustomPolicy2, self).__init__(*args, **kwargs,
                                                   feature_extraction="mlp", layers=[400, 300])
        model = TD3(CustomPolicy2, env,
                    verbose = 1,
                    tensorboard_log = 'data/TBlogs/initial_policy_training',
                    batch_size = args['batch_size'],
                    buffer_size = args['buffer_size'],
                    gamma = args['gamma'],
                    gradient_steps = args['gradient_steps'],
                    learning_rate = args['learning_rate'],
                    learning_starts = args['learning_starts'],
                    action_noise = action_noise,
                    train_freq=args['train_freq'],
                    )

    elif algo.__name__ == "TRPO":
        print('Initializing TRPO with RLBaselinesZoo hyperparameters .. ')
        # hyperparameters suggestions from :
        # https://github.com/araffin/rl-baselines-zoo/blob/master/trained_agents/sac/HopperBulletEnv-v0/config.yml
        model = TRPO(mlp_standard, env,
                    verbose=1,
                    tensorboard_log='data/TBlogs/initial_policy_training',
                    timesteps_per_batch=args['timesteps_per_batch'],
                    lam=args['lam'],
                    max_kl=args['max_kl'],
                    gamma=args['gamma'],
                    vf_iters=args['vf_iters'],
                    vf_stepsize=args['vf_stepsize'],
                    entcoeff=args['entcoeff'],
                    cg_damping=args['cg_damping'],
                    cg_iters=args['cg_iters']
                    )

    elif algo.__name__ == "ACKTR":
        print('Initializing ACKTR')
        model = ACKTR(mlp_standard,
                      env,
                      verbose=1,
                      n_steps=128,
                      ent_coef=0.01,
                      lr_schedule='constant',
                      learning_rate=0.0217,
                      max_grad_norm=0.5,
                      gamma=0.99,
                      vf_coef=0.946)

    elif algo.__name__ == "PPO2":
        print('Initializing PPO2')
        print('Num envs : ', env.num_envs)
        model = PPO2(mlp_standard,
                     env,
                     n_steps=int(args['n_steps']/env.num_envs),
                     nminibatches=args['nminibatches'],
                     lam=args['lam'],
                     gamma=args['gamma'],
                     ent_coef=args['ent_coef'],
                     noptepochs=args['noptepochs'],
                     learning_rate=args['learning_rate'],
                     cliprange=args['cliprange'],
                     verbose=1,
                     tensorboard_log='data/TBlogs/initial_policy_training',
                     )

    else:
        print('No algorithm matched. Using SAC .. ')
        model = SAC(CustomPolicy, env,
                    verbose=1,
                    batch_size=args['batch_size'],
                    buffer_size=args['buffer_size'],
                    ent_coef=args['ent_coef'],
                    learning_starts=args['learning_starts'],
                    learning_rate=args['learning_rate'],
                    train_freq=args['train_freq'],
                    )

    # change model name if using normalization
    if NORMALIZE:
        model_name = model_name.replace('.pkl', 'normalized_.pkl')

    elif MUJOCO_NORMALIZE:
        model_name = model_name.replace('.pkl', 'mujoco_norm_.pkl')

    if SAVE_BEST_FOR_20:
        model.learn(total_timesteps=time_steps,
                    tb_log_name=model_name,
                    log_interval=10,
                    callback=eval_callback)
        save_the_model()
        model_name = model_name.replace('best_', '')
        model.save(model_name)

    else:
        model.learn(total_timesteps=time_steps,
                    tb_log_name=model_name.split('/')[-1],
                    log_interval=10,)
        model.save(model_name)
        evaluate_policy_on_env(env, model, render=False, iters=10)

    # save the environment params
    if NORMALIZE:
        # env.save(model_name.replace('.pkl', 'stats_.pkl'))
        env.save('data/models/env_stats/'+env_name+'.pkl')

    print('done :: ', model_name)
    exit()

if __name__ == '__main__':
    train_initial_policy(model_name=model_name)
