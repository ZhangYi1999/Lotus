#### train policy bug
```
Process Process-24:
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/yi/Program/Lotus/lotus/libero/envs/venv.py", line 231, in _worker
    env = env_fn_wrapper.data()
  File "/home/yi/Program/Lotus/lotus/lifelong/metric.py", line 93, in <lambda>
    [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
  File "/home/yi/Program/Lotus/lotus/libero/envs/env_wrapper.py", line 161, in __init__
    super().__init__(**kwargs)
  File "/home/yi/Program/Lotus/lotus/libero/envs/env_wrapper.py", line 56, in __init__
    self.env = TASK_MAPPING[self.problem_name](
  File "/home/yi/Program/Lotus/lotus/libero/envs/problems/libero_floor_manipulation.py", line 37, in __init__
    super().__init__(bddl_file_name, *args, **kwargs)
  File "/home/yi/Program/Lotus/lotus/libero/envs/bddl_base_domain.py", line 135, in __init__
    super().__init__(
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/manipulation/manipulation_env.py", line 162, in __init__
    super().__init__(
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/robot_env.py", line 214, in __init__
    super().__init__(
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/base.py", line 143, in __init__
    self._reset_internal()
  File "/home/yi/Program/Lotus/lotus/libero/envs/bddl_base_domain.py", line 735, in _reset_internal
    super()._reset_internal()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/robot_env.py", line 510, in _reset_internal
    super()._reset_internal()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/base.py", line 299, in _reset_internal
    render_context = MjRenderContextOffscreen(self.sim, device_id=self.render_gpu_device_id)
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 210, in __init__
    super().__init__(sim, offscreen=True, device_id=device_id, max_width=max_width, max_height=max_height)
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 78, in __init__
    self.gl_ctx = GLContext(max_width=max_width, max_height=max_height, device_id=self.device_id)
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/renderers/context/egl_context.py", line 136, in __init__
    self._context = EGL.eglCreateContext(EGL_DISPLAY, config, EGL.EGL_NO_CONTEXT, None)
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/OpenGL/platform/baseplatform.py", line 415, in __call__
    return self( *args, **named )
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/OpenGL/error.py", line 230, in glCheckError
    raise self._errorClass(
OpenGL.raw.EGL._errors.EGLError: EGLError(
        err = EGL_BAD_ALLOC,
        baseOperation = eglCreateContext,
        cArguments = (
                <OpenGL._opaque.EGLDisplay_pointer object at 0x78d5875043c0>,
                <OpenGL._opaque.EGLConfig_pointer object at 0x78d5875042c0>,
                <OpenGL._opaque.EGLContext_pointer object at 0x78d5b3ac28c0>,
                None,
        ),
        result = <OpenGL._opaque.EGLContext_pointer object at 0x78d5875049c0>
)
Exception ignored in: <function EGLGLContext.__del__ at 0x78d5b39740d0>
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/renderers/context/egl_context.py", line 155, in __del__
    self.free()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/renderers/context/egl_context.py", line 146, in free
    if self._context:
AttributeError: 'EGLGLContext' object has no attribute '_context'
Exception ignored in: <function MjRenderContext.__del__ at 0x78d5b3974280>
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 198, in __del__
    self.con.free()
AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
```

Possible Reason: Robosuite not compatible with multiprocessing


```
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
{ 'bddl_folder': None,
  'benchmark_name': 'libero_object',
  'data': { 'action_scale': 1.0,
            'affine_translate': 4,
            'data_modality': ['image', 'proprio'],
            'frame_stack': 1,
            'img_h': 128,
            'img_w': 128,
            'max_word_len': 25,
            'num_kp': 64,
            'obs': { 'modality': { 'depth': [],
                                   'low_dim': [ 'gripper_states',
                                                'joint_states'],
                                   'rgb': [ 'agentview_rgb',
                                            'eye_in_hand_rgb']}},
            'obs_key_mapping': { 'agentview_rgb': 'agentview_image',
                                 'eye_in_hand_rgb': 'robot0_eye_in_hand_image',
                                 'gripper_states': 'robot0_gripper_qpos',
                                 'joint_states': 'robot0_joint_pos'},
            'seq_len': 10,
            'shuffle_task': False,
            'state_dim': None,
            'task_group_size': 1,
            'task_order_index': 0,
            'train_dataset_ratio': 0.8,
            'use_ee': False,
            'use_eye_in_hand': True,
            'use_gripper': True,
            'use_joint': True},
  'device': 'cuda',
  'eval': { 'batch_size': 64,
            'eval': True,
            'eval_every': 5,
            'load_path': '',
            'max_steps': 600,
            'n_eval': 20,
            'num_procs': 20,
            'num_workers': 4,
            'save_sim_states': False,
            'use_mp': True},
  'exp': 'transformer_libero_object',
  'folder': None,
  'goal_modality': 'BUDS',
  'init_states_folder': None,
  'lifelong': {'algo': 'Multitask_Skill', 'eval_in_train': True},
  'load_previous_model': False,
  'meta': { 'color_aug': { 'network': 'BatchWiseImgColorJitterAug',
                           'network_kwargs': { 'brightness': 0.3,
                                               'contrast': 0.3,
                                               'epsilon': 0.1,
                                               'hue': 0.3,
                                               'input_shape': None,
                                               'saturation': 0.3}},
            'embed_size': 64,
            'extra_hidden_size': 128,
            'extra_num_layers': 0,
            'image_encoder': { 'network': 'ResnetEncoder',
                               'network_kwargs': { 'freeze': False,
                                                   'language_fusion': 'film',
                                                   'no_stride': False,
                                                   'pretrained': False,
                                                   'remove_layer_num': 4}},
            'language_encoder': { 'network': 'MLPEncoder',
                                  'network_kwargs': { 'hidden_size': 128,
                                                      'input_size': 768,
                                                      'num_layers': 1,
                                                      'output_size': 128}},
            'policy_head': { 'loss_kwargs': {'loss_coef': 1.0},
                             'network': 'GMMHead',
                             'network_kwargs': { 'activation': 'softplus',
                                                 'hidden_size': 1024,
                                                 'low_eval_noise': False,
                                                 'min_std': 0.0001,
                                                 'num_layers': 2,
                                                 'num_modes': 5}},
            'policy_type': 'BCTransformerPolicy',
            'temporal_position_encoding': { 'network': 'SinusoidalPositionEncoding',
                                            'network_kwargs': { 'factor_ratio': None,
                                                                'input_size': None,
                                                                'inv_freq_factor': 10}},
            'transformer_dropout': 0.1,
            'transformer_head_output_size': 64,
            'transformer_input_size': None,
            'transformer_max_seq_len': 10,
            'transformer_mlp_hidden_size': 256,
            'transformer_num_heads': 6,
            'transformer_num_layers': 4,
            'translation_aug': { 'network': 'TranslationAug',
                                 'network_kwargs': { 'input_shape': None,
                                                     'translation': 8}}},
  'policy': { 'color_aug': { 'network': 'BatchWiseImgColorJitterAug',
                             'network_kwargs': { 'brightness': 0.3,
                                                 'contrast': 0.3,
                                                 'epsilon': 0.1,
                                                 'hue': 0.3,
                                                 'input_shape': None,
                                                 'saturation': 0.3}},
              'embed_size': 64,
              'extra_hidden_size': 128,
              'extra_num_layers': 0,
              'image_encoder': { 'network': 'ResnetEncoder',
                                 'network_kwargs': { 'freeze': False,
                                                     'language_fusion': 'film',
                                                     'no_stride': False,
                                                     'pretrained': False,
                                                     'remove_layer_num': 4}},
              'language_encoder': { 'network': 'MLPEncoder',
                                    'network_kwargs': { 'hidden_size': 128,
                                                        'input_size': 768,
                                                        'num_layers': 1,
                                                        'output_size': 128}},
              'policy_head': { 'loss_kwargs': {'loss_coef': 1.0},
                               'network': 'GMMHead',
                               'network_kwargs': { 'activation': 'softplus',
                                                   'hidden_size': 1024,
                                                   'low_eval_noise': False,
                                                   'min_std': 0.0001,
                                                   'num_layers': 2,
                                                   'num_modes': 5}},
              'policy_type': 'BCTransformerPolicy',
              'temporal_position_encoding': { 'network': 'SinusoidalPositionEncoding',
                                              'network_kwargs': { 'factor_ratio': None,
                                                                  'input_size': None,
                                                                  'inv_freq_factor': 10}},
              'transformer_dropout': 0.1,
              'transformer_head_output_size': 64,
              'transformer_input_size': None,
              'transformer_max_seq_len': 10,
              'transformer_mlp_hidden_size': 256,
              'transformer_num_heads': 6,
              'transformer_num_layers': 4,
              'translation_aug': { 'network': 'TranslationAug',
                                   'network_kwargs': { 'input_shape': None,
                                                       'translation': 8}}},
  'pretrain': False,
  'pretrain_model_path': '',
  'seed': 42,
  'skill_learning': { 'agglomoration': { 'K': 5,
                                         'affinity': 'rbf',
                                         'agglomoration_step': 10,
                                         'dist': 'l2',
                                         'footprint': 'mean',
                                         'min_len_thresh': 20,
                                         'scale': 0.05,
                                         'segment_footprint': 'concat_1',
                                         'segment_scale': 2,
                                         'visualization': False},
                      'eval': {'meta_freq': 5},
                      'exp_name': 'dinov2_libero_object_image_only',
                      'folder': './',
                      'hydra': {'run': {'dir': '.'}},
                      'meta': { 'activation': 'leaky-relu',
                                'affine_translate': 4,
                                'batch_size': 100,
                                'embedding_layer_dims': [300, 400],
                                'id_layer_dims': [300, 400],
                                'img_h': 128,
                                'img_w': 128,
                                'lr': 0.0001,
                                'num_epochs': 1001,
                                'num_kp': 64,
                                'num_workers': 0,
                                'random_affine': False,
                                'rnn_hidden_dim': 100,
                                'rnn_num_layers': 2,
                                'separate_id_prediction': False,
                                'use_cvae': True,
                                'use_eye_in_hand': False,
                                'use_rnn': False,
                                'use_spatial_softmax': False,
                                'visual_feature_dimension': 64},
                      'meta_cvae_cfg': { 'enable': True,
                                         'kl_coeff': 0.01,
                                         'latent_dim': 64},
                      'modality_str': 'dinov2_agentview_eye_in_hand',
                      'multitask': { 'skip_task_id': [5, 6, 8],
                                     'task_id': 0,
                                     'testing_percentage': 1.0,
                                     'training_task_id': -1},
                      'record_states': True,
                      'repr': { 'alpha_kl': 0.05,
                                'modalities': [ 'agentview',
                                                'eye_in_hand',
                                                'proprio'],
                                'no_skip': True,
                                'z_dim': 32},
                      'skill_subgoal_cfg': { 'horizon': 30,
                                             'subgoal_type': 'linear',
                                             'use_eye_in_hand': False,
                                             'use_final_goal': False,
                                             'use_spatial_softmax': True,
                                             'visual_feature_dimension': 32},
                      'skill_training': { 'action_squash': True,
                                          'activation': 'leaky-relu',
                                          'affine_translate': 4,
                                          'agglomoration': {'K': 5},
                                          'batch_size': 128,
                                          'data_modality': ['image', 'proprio'],
                                          'gripper_smoothing': False,
                                          'img_h': 128,
                                          'img_w': 128,
                                          'lr': 0.001,
                                          'min_lr': 0.0001,
                                          'no_skip': True,
                                          'num_epochs': 1001,
                                          'num_kp': 64,
                                          'num_workers': 0,
                                          'policy_layer_dims': [300, 400],
                                          'policy_type': 'normal_subgoal',
                                          'random_affine': True,
                                          'rnn_encoder_mlp_dims': [128, 128],
                                          'rnn_hidden_dim': 100,
                                          'rnn_loss_reduction': 'mean',
                                          'rnn_num_layers': 2,
                                          'run_idx': 0,
                                          'state_dim': 37,
                                          'subtask_id': [],
                                          'use_changepoint': False,
                                          'use_eye_in_hand': True,
                                          'use_gripper': True,
                                          'use_joints': True,
                                          'use_rnn': False,
                                          'visual_feature_dimension': 64},
                      'use_checkpoint': False,
                      'verbose': True,
                      'video': { 'demo_output_dir': 'paper_vis/demo_videos/defaults',
                                 'dir': '',
                                 'fps': 60,
                                 'height': 1024,
                                 'output_dir': 'paper_vis/videos/defaults',
                                 'width': 1024}},
  'task_embedding_format': 'bert',
  'task_embedding_one_hot_offset': 1,
  'train': { 'batch_size': 32,
             'debug': False,
             'grad_clip': 100.0,
             'loss_scale': 1.0,
             'n_epochs': 50,
             'num_workers': 4,
             'optimizer': { 'kwargs': { 'betas': [0.9, 0.999],
                                        'lr': 0.0001,
                                        'weight_decay': 0.0001},
                            'name': 'torch.optim.AdamW'},
             'resume': False,
             'resume_path': '',
             'scheduler': { 'kwargs': {'eta_min': 1e-05, 'last_epoch': -1},
                            'name': 'torch.optim.lr_scheduler.CosineAnnealingLR'},
             'use_augmentation': True},
  'use_wandb': True,
  'wandb_project': 'lifelong learning'}
'Available algorithms:'
{ 'agem': <class 'lotus.lifelong.algos.agem.AGEM'>,
  'er': <class 'lotus.lifelong.algos.er.ER'>,
  'ewc': <class 'lotus.lifelong.algos.ewc.EWC'>,
  'metacontroller': <class 'lotus.lifelong.algos.skill.MetaController'>,
  'multitask': <class 'lotus.lifelong.algos.multitask.Multitask'>,
  'packnet': <class 'lotus.lifelong.algos.packnet.PackNet'>,
  'sequential': <class 'lotus.lifelong.algos.base.Sequential'>,
  'singletask': <class 'lotus.lifelong.algos.single_task.SingleTask'>,
  'subskill': <class 'lotus.lifelong.algos.skill.SubSkill'>}
'Available policies:'
{ 'bcrnnpolicy': <class 'lotus.lifelong.models.bc_rnn_policy.BCRNNPolicy'>,
  'bctransformerpolicy': <class 'lotus.lifelong.models.bc_transformer_policy.BCTransformerPolicy'>,
  'bctransformerskillpolicy': <class 'lotus.lifelong.models.bc_transformer_policy.BCTransformerSkillPolicy'>,
  'bcviltpolicy': <class 'lotus.lifelong.models.bc_vilt_policy.BCViLTPolicy'>}
[info] using task orders [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

============= Initialized Observation Utils with Obs Spec =============

using obs modality: rgb with keys: ['eye_in_hand_rgb', 'agentview_rgb']
using obs modality: depth with keys: []
using obs modality: low_dim with keys: ['joint_states', 'gripper_states']
SequenceDataset: loading dataset into memory...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 2736.69it/s]
/home/yi/Program/Lotus/lotus/libero/../datasets/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
SequenceDataset: loading dataset into memory...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 3057.03it/s]
/home/yi/Program/Lotus/lotus/libero/../datasets/libero_object/pick_up_the_cream_cheese_and_place_it_in_the_basket_demo.hdf5
SequenceDataset: loading dataset into memory...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 3280.23it/s]
/home/yi/Program/Lotus/lotus/libero/../datasets/libero_object/pick_up_the_salad_dressing_and_place_it_in_the_basket_demo.hdf5
SequenceDataset: loading dataset into memory...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 3359.15it/s]
/home/yi/Program/Lotus/lotus/libero/../datasets/libero_object/pick_up_the_bbq_sauce_and_place_it_in_the_basket_demo.hdf5
SequenceDataset: loading dataset into memory...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 3315.24it/s]
/home/yi/Program/Lotus/lotus/libero/../datasets/libero_object/pick_up_the_ketchup_and_place_it_in_the_basket_demo.hdf5
SequenceDataset: loading dataset into memory...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 3295.65it/s]
/home/yi/Program/Lotus/lotus/libero/../datasets/libero_object/pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5
SequenceDataset: loading dataset into memory...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 3542.49it/s]
/home/yi/Program/Lotus/lotus/libero/../datasets/libero_object/pick_up_the_butter_and_place_it_in_the_basket_demo.hdf5
SequenceDataset: loading dataset into memory...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 3420.68it/s]
/home/yi/Program/Lotus/lotus/libero/../datasets/libero_object/pick_up_the_milk_and_place_it_in_the_basket_demo.hdf5
SequenceDataset: loading dataset into memory...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 3382.72it/s]
/home/yi/Program/Lotus/lotus/libero/../datasets/libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5
SequenceDataset: loading dataset into memory...
100%|██████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 3407.12it/s]
/home/yi/Program/Lotus/lotus/libero/../datasets/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5
subtasks distance score: 0.044081632653061226
subtasks distance score: 0.0
subtasks distance score: 0.0
subtasks distance score: 0.04
subtasks distance score: 0.0
subtasks distance score: 0.04
train_dataset_id: [0, 1]

=================== Lifelong Benchmark Information  ===================
 Name: libero_object
 # Tasks: 10
    - Task 1:
        pick up the alphabet soup and place it in the basket
    - Task 2:
        pick up the cream cheese and place it in the basket
    - Task 3:
        pick up the salad dressing and place it in the basket
    - Task 4:
        pick up the bbq sauce and place it in the basket
    - Task 5:
        pick up the ketchup and place it in the basket
    - Task 6:
        pick up the tomato sauce and place it in the basket
    - Task 7:
        pick up the butter and place it in the basket
    - Task 8:
        pick up the milk and place it in the basket
    - Task 9:
        pick up the chocolate pudding and place it in the basket
    - Task 10:
        pick up the orange juice and place it in the basket
=======================================================================

/home/yi/.vscode/extensions/ms-python.debugpy-2024.12.0-linux-x64/bundled/libs/debugpy/adapter/../../debugpy/launcher/../../debugpy/../debugpy/_vendored/force_pydevd.py:18: UserWarning: incompatible copy of pydevd already imported:
 /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/pydevd_plugins/extensions/pydevd_plugin_omegaconf.py
  warnings.warn(msg + ':\n {}'.format('\n  '.join(_unvendored)))
wandb: Currently logged in as: 470620104 (470620104-technical-university-of-munich). Use `wandb login --relogin` to force relogin
wandb: wandb version 0.18.5 is available!  To upgrade, please run:
wandb:  $ pip install wandb --upgrade
wandb: Tracking run with wandb version 0.13.1
wandb: Run data is saved locally in /home/yi/Program/Lotus/lotus/wandb/run-20241031_104637-zwubswqn
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run sleepless-darkness-7
wandb: ⭐️ View project at https://wandb.ai/470620104-technical-university-of-munich/lifelong%20learning
wandb: 🚀 View run at https://wandb.ai/470620104-technical-university-of-munich/lifelong%20learning/runs/zwubswqn
Finish loading subtask_0:  436
Subtask id: 0
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/torch/functional.py:568: UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument. (Triggered internally at  ../aten/src/ATen/native/TensorShape.cpp:2228.)
  return _VF.meshgrid(tensors, **kwargs)  # type: ignore[attr-defined]
[info] start training skill 0
[info] Epoch:   0 | train loss:  5.35 | time: 0.04
[info] Epoch:   1 | train loss:  2.36 | time: 0.08
[info] Epoch:   2 | train loss:  0.55 | time: 0.08
[info] Epoch:   3 | train loss: -1.84 | time: 0.08
[info] Epoch:   4 | train loss: -3.68 | time: 0.08
[info] Epoch:   5 | train loss: -4.20 | time: 0.07
[info] Epoch:   6 | train loss: -5.39 | time: 0.08
[info] Epoch:   7 | train loss: -6.88 | time: 0.07
[info] Epoch:   8 | train loss: -8.33 | time: 0.08
[info] Epoch:   9 | train loss: -9.32 | time: 0.07
[info] Epoch:  10 | train loss: -10.42 | time: 0.07
[info] Epoch:  11 | train loss: -11.95 | time: 0.07
[info] Epoch:  12 | train loss: -9.70 | time: 0.07
[info] Epoch:  13 | train loss: -8.78 | time: 0.07
[info] Epoch:  14 | train loss: -10.08 | time: 0.07
[info] Epoch:  15 | train loss: -10.13 | time: 0.07
[info] Epoch:  16 | train loss: -9.94 | time: 0.08
[info] Epoch:  17 | train loss: -12.03 | time: 0.07
[info] Epoch:  18 | train loss: -7.31 | time: 0.08
[info] Epoch:  19 | train loss: -11.74 | time: 0.08
[info] Epoch:  20 | train loss: -12.92 | time: 0.08
[info] Epoch:  21 | train loss: -12.07 | time: 0.08
[info] Epoch:  22 | train loss: -12.46 | time: 0.08
[info] Epoch:  23 | train loss: -11.67 | time: 0.08
[info] Epoch:  24 | train loss: -13.18 | time: 0.07
[info] Epoch:  25 | train loss: -13.79 | time: 0.08
[info] Epoch:  26 | train loss: -14.02 | time: 0.08
[info] Epoch:  27 | train loss: -14.47 | time: 0.07
[info] Epoch:  28 | train loss: -14.60 | time: 0.08
[info] Epoch:  29 | train loss: -14.76 | time: 0.07
[info] Epoch:  30 | train loss: -14.95 | time: 0.08
[info] Epoch:  31 | train loss: -14.72 | time: 0.07
[info] Epoch:  32 | train loss: -14.74 | time: 0.07
[info] Epoch:  33 | train loss: -15.21 | time: 0.07
[info] Epoch:  34 | train loss: -15.29 | time: 0.08
[info] Epoch:  35 | train loss: -15.00 | time: 0.08
[info] Epoch:  36 | train loss: -14.70 | time: 0.08
[info] Epoch:  37 | train loss: -15.23 | time: 0.07
[info] Epoch:  38 | train loss: -15.34 | time: 0.07
[info] Epoch:  39 | train loss: -15.22 | time: 0.08
[info] Epoch:  40 | train loss: -15.39 | time: 0.07
[info] Epoch:  41 | train loss: -15.19 | time: 0.07
[info] Epoch:  42 | train loss: -15.70 | time: 0.08
[info] Epoch:  43 | train loss: -15.90 | time: 0.08
[info] Epoch:  44 | train loss: -16.05 | time: 0.07
[info] Epoch:  45 | train loss: -16.12 | time: 0.07
[info] Epoch:  46 | train loss: -16.19 | time: 0.08
[info] Epoch:  47 | train loss: -16.12 | time: 0.08
[info] Epoch:  48 | train loss: -15.81 | time: 0.08
[info] Epoch:  49 | train loss: -15.86 | time: 0.08
[info] Epoch:  50 | train loss: -16.24 | time: 0.07
Finish loading subtask_1:  508
Subtask id: 1
[info] start training skill 1
[info] Epoch:   0 | train loss:  5.44 | time: 0.05
[info] Epoch:   1 | train loss:  2.25 | time: 0.09
[info] Epoch:   2 | train loss: -0.52 | time: 0.09
[info] Epoch:   3 | train loss: -2.01 | time: 0.09
[info] Epoch:   4 | train loss: -4.10 | time: 0.08
[info] Epoch:   5 | train loss: -5.38 | time: 0.09
[info] Epoch:   6 | train loss: -5.87 | time: 0.08
[info] Epoch:   7 | train loss: -7.62 | time: 0.08
[info] Epoch:   8 | train loss: -8.15 | time: 0.09
[info] Epoch:   9 | train loss: -9.04 | time: 0.09
[info] Epoch:  10 | train loss: -10.40 | time: 0.09
[info] Epoch:  11 | train loss: -11.33 | time: 0.08
[info] Epoch:  12 | train loss: -12.00 | time: 0.09
[info] Epoch:  13 | train loss: -12.05 | time: 0.08
[info] Epoch:  14 | train loss: -10.88 | time: 0.09
[info] Epoch:  15 | train loss: -11.37 | time: 0.09
[info] Epoch:  16 | train loss: -12.21 | time: 0.09
[info] Epoch:  17 | train loss: -12.72 | time: 0.08
[info] Epoch:  18 | train loss: -11.22 | time: 0.08
[info] Epoch:  19 | train loss: -10.94 | time: 0.09
[info] Epoch:  20 | train loss: -11.78 | time: 0.08
[info] Epoch:  21 | train loss: -12.80 | time: 0.09
[info] Epoch:  22 | train loss: -13.37 | time: 0.08
[info] Epoch:  23 | train loss: -13.04 | time: 0.08
[info] Epoch:  24 | train loss: -13.29 | time: 0.08
[info] Epoch:  25 | train loss: -13.90 | time: 0.08
[info] Epoch:  26 | train loss: -14.18 | time: 0.08
[info] Epoch:  27 | train loss: -14.29 | time: 0.08
[info] Epoch:  28 | train loss: -14.70 | time: 0.09
[info] Epoch:  29 | train loss: -15.04 | time: 0.09
[info] Epoch:  30 | train loss: -15.05 | time: 0.09
[info] Epoch:  31 | train loss: -15.19 | time: 0.09
[info] Epoch:  32 | train loss: -15.55 | time: 0.09
[info] Epoch:  33 | train loss: -15.85 | time: 0.09
[info] Epoch:  34 | train loss: -15.75 | time: 0.09
[info] Epoch:  35 | train loss: -15.98 | time: 0.09
[info] Epoch:  36 | train loss: -16.11 | time: 0.09
[info] Epoch:  37 | train loss: -15.75 | time: 0.09
[info] Epoch:  38 | train loss: -16.24 | time: 0.09
[info] Epoch:  39 | train loss: -16.53 | time: 0.09
[info] Epoch:  40 | train loss: -16.69 | time: 0.08
[info] Epoch:  41 | train loss: -16.93 | time: 0.09
[info] Epoch:  42 | train loss: -17.02 | time: 0.09
[info] Epoch:  43 | train loss: -17.04 | time: 0.09
[info] Epoch:  44 | train loss: -17.25 | time: 0.09
[info] Epoch:  45 | train loss: -17.23 | time: 0.09
[info] Epoch:  46 | train loss: -17.44 | time: 0.08
[info] Epoch:  47 | train loss: -17.41 | time: 0.08
[info] Epoch:  48 | train loss: -17.44 | time: 0.09
[info] Epoch:  49 | train loss: -17.63 | time: 0.09
[info] Epoch:  50 | train loss: -17.70 | time: 0.08
MetaPolicyDataset:  torch.Size([938])
[info] start training meta controller
[info] Epoch:   0 | Train loss: 51.17 | 
Training ce loss: 20.72 | Training embedding loss:  1.61 | Training kl loss: 28.84 | Time: 0.05
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
[robosuite WARNING] No private macro file found! (macros.py:53)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:54)
[robosuite WARNING] To setup, run: python /home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/scripts/setup_macros.py (macros.py:55)
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
Exception ignored in: <function MjRenderContext.__del__ at 0x7405f063db80>
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 198, in __del__
    self.con.free()
AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
Process Process-11:
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/yi/Program/Lotus/lotus/libero/envs/venv.py", line 231, in _worker
    env = env_fn_wrapper.data()
  File "/home/yi/Program/Lotus/lotus/lifelong/metric.py", line 93, in <lambda>
    [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
  File "/home/yi/Program/Lotus/lotus/libero/envs/env_wrapper.py", line 161, in __init__
    super().__init__(**kwargs)
  File "/home/yi/Program/Lotus/lotus/libero/envs/env_wrapper.py", line 56, in __init__
    self.env = TASK_MAPPING[self.problem_name](
  File "/home/yi/Program/Lotus/lotus/libero/envs/problems/libero_floor_manipulation.py", line 37, in __init__
    super().__init__(bddl_file_name, *args, **kwargs)
  File "/home/yi/Program/Lotus/lotus/libero/envs/bddl_base_domain.py", line 135, in __init__
    super().__init__(
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/manipulation/manipulation_env.py", line 162, in __init__
    super().__init__(
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/robot_env.py", line 214, in __init__
    super().__init__(
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/base.py", line 143, in __init__
    self._reset_internal()
  File "/home/yi/Program/Lotus/lotus/libero/envs/bddl_base_domain.py", line 735, in _reset_internal
    super()._reset_internal()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/robot_env.py", line 510, in _reset_internal
    super()._reset_internal()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/base.py", line 299, in _reset_internal
    render_context = MjRenderContextOffscreen(self.sim, device_id=self.render_gpu_device_id)
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 210, in __init__
    super().__init__(sim, offscreen=True, device_id=device_id, max_width=max_width, max_height=max_height)
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 109, in __init__
    self._set_mujoco_context_and_buffers()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 112, in _set_mujoco_context_and_buffers
    self.con = mujoco.MjrContext(self.model._model, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.FatalError: Offscreen framebuffer is not complete, error 0x8cdd
/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/thop/profile.py:12: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  if LooseVersion(torch.__version__) < LooseVersion("1.0.0"):
Process Process-13:
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/yi/Program/Lotus/lotus/libero/envs/venv.py", line 231, in _worker
    env = env_fn_wrapper.data()
  File "/home/yi/Program/Lotus/lotus/lifelong/metric.py", line 93, in <lambda>
    [lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)]
  File "/home/yi/Program/Lotus/lotus/libero/envs/env_wrapper.py", line 161, in __init__
    super().__init__(**kwargs)
  File "/home/yi/Program/Lotus/lotus/libero/envs/env_wrapper.py", line 56, in __init__
    self.env = TASK_MAPPING[self.problem_name](
  File "/home/yi/Program/Lotus/lotus/libero/envs/problems/libero_floor_manipulation.py", line 37, in __init__
    super().__init__(bddl_file_name, *args, **kwargs)
  File "/home/yi/Program/Lotus/lotus/libero/envs/bddl_base_domain.py", line 135, in __init__
    super().__init__(
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/manipulation/manipulation_env.py", line 162, in __init__
    super().__init__(
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/robot_env.py", line 214, in __init__
    super().__init__(
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/base.py", line 143, in __init__
    self._reset_internal()
  File "/home/yi/Program/Lotus/lotus/libero/envs/bddl_base_domain.py", line 735, in _reset_internal
    super()._reset_internal()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/robot_env.py", line 510, in _reset_internal
    super()._reset_internal()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/environments/base.py", line 299, in _reset_internal
    render_context = MjRenderContextOffscreen(self.sim, device_id=self.render_gpu_device_id)
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 210, in __init__
    super().__init__(sim, offscreen=True, device_id=device_id, max_width=max_width, max_height=max_height)
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 109, in __init__
    self._set_mujoco_context_and_buffers()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 112, in _set_mujoco_context_and_buffers
    self.con = mujoco.MjrContext(self.model._model, mujoco.mjtFontScale.mjFONTSCALE_150)
mujoco.FatalError: Offscreen framebuffer is not complete, error 0x8cdd
Exception ignored in: <function MjRenderContext.__del__ at 0x750bdc4b6b80>
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 198, in __del__
    self.con.free()
AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
Exception ignored in: <function EGLGLContext.__del__ at 0x750bdc4b69d0>
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/renderers/context/egl_context.py", line 155, in __del__
    self.free()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/renderers/context/egl_context.py", line 149, in free
    EGL.eglMakeCurrent(EGL_DISPLAY, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/OpenGL/error.py", line 230, in glCheckError
    raise self._errorClass(
OpenGL.raw.EGL._errors.EGLError: EGLError(
        err = EGL_NOT_INITIALIZED,
        baseOperation = eglMakeCurrent,
        cArguments = (
                <OpenGL._opaque.EGLDisplay_pointer object at 0x750bc7b23c40>,
                <OpenGL._opaque.EGLSurface_pointer object at 0x750bdc596240>,
                <OpenGL._opaque.EGLSurface_pointer object at 0x750bdc596240>,
                <OpenGL._opaque.EGLContext_pointer object at 0x750bdc5e1d40>,
        ),
        result = 0
)
Exception ignored in: <function MjRenderContext.__del__ at 0x7bfa24043b80>
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 198, in __del__
    self.con.free()
AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
Exception ignored in: <function EGLGLContext.__del__ at 0x7bfa240439d0>
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/renderers/context/egl_context.py", line 155, in __del__
    self.free()
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/renderers/context/egl_context.py", line 149, in free
    EGL.eglMakeCurrent(EGL_DISPLAY, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/OpenGL/error.py", line 230, in glCheckError
    raise self._errorClass(
OpenGL.raw.EGL._errors.EGLError: Exception ignored in: <function MjRenderContext.__del__ at 0x781a0cb34b80>
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 198, in __del__
    self.con.free()
AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
EGLError(
        err = EGL_NOT_INITIALIZED,
        baseOperation = eglMakeCurrent,
        cArguments = (
                <OpenGL._opaque.EGLDisplay_pointer object at 0x7bf9fdf20c40>,
                <OpenGL._opaque.EGLSurface_pointer object at 0x7bfa24121240>,
                <OpenGL._opaque.EGLSurface_pointer object at 0x7bfa24121240>,
                <OpenGL._opaque.EGLContext_pointer object at 0x7bfa2444fd40>,
        ),
        result = 0
)
Exception ignored in: <function MjRenderContext.__del__ at 0x7efcc6a38b80>
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 198, in __del__
    self.con.free()
AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
Error executing job with overrides: ['seed=42', 'benchmark_name=libero_object', 'policy=bc_transformer_policy', 'lifelong=multitask_skill', 'skill_learning.exp_name=dinov2_libero_object_image_only', 'exp=transformer_libero_object', 'goal_modality=BUDS']
Backend TkAgg is interactive backend. Turning interactive mode on.
Exception ignored in: <function MjRenderContext.__del__ at 0x7d4f0585eb80>
Traceback (most recent call last):
  File "/home/yi/Program/Lotus/.conda/lib/python3.9/site-packages/robosuite/utils/binding_utils.py", line 198, in __del__
    self.con.free()
AttributeError: 'MjRenderContextOffscreen' object has no attribute 'con'
```