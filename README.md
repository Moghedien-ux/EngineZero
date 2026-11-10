This project stems from my passion for low level optimization and understanding how things happen behind the scenes. After following and understanding a PyTorch based course provided by DeepLearning.AI, I realized I was able to train AI models on whatever I wanted, however, whilst I gained the knowledge of how AI models work, I started to wonder: how does PyTorch work and how optimal is it?
Given my background as a Computer Software Engineering student withing the domain of Systems Engineering and Applied Informatics I started to wonder: If I wanted to train a model meant to control a robot, how efficient would PyTorch be?
After conducting some research and by the simple fact that PyTorch is Python based at heart (even though there are C/C++ versions) I realized that for robotic systems that run on not so permissive components, maybe there is a way to make things more efficient,
as such I embarked on an interesting journey: Making a tool similar to PyTorch from scratch in C++ using modern C++ 26 standards (even though C++ 26 wasn't passed as the new standard C++ because of contracts features that turned the vote below 66%) and low level
memory optimization to make a tool that is able to create and train a model on insanely low spec levels (unfortunately it is purely CPU based atm, so PyTorch wins performance if we're talking about a system that has a GPU).

The EngineZero tool can develop different types of models, for example:
 - Transformer / Micro-LLM (focused on self-attention, large matrices, tokenization, and generation).
 - Reinforcement Learning / Robotics in a simulator (focused on real-time interaction with a physical environment/game, DQN/PPO).
 - Computer Vision / CNN Engine (focused on convolutions, im2col, pooling, and classification).

## Racing simulation (DQN)

The project includes an initial, configurable racing environment and a Deep Q-Network training loop:

```text
AI
AI train [episodes] [model-file] [experience-file] [--fresh]
AI race [model-file]
AI live [model-file] [track.csv] [gear.cfg]
```

Running `AI` without arguments now starts training for 200 episodes. The simulator generates the RL training data itself: each action produces a state/action/reward/next-state transition, and DQN learns from those transitions through its replay buffer. Episodes vary the oval geometry, surface friction, vehicle mass, CG height, tire material, steering, and drive/brake parameters. No external labeled dataset is needed.

Generated transitions are written to `racing_experience.csv` and loaded into replay memory on later runs. Model weights are checkpointed every 10 episodes to `racing_model.ez`; training resumes those weights when present. `AI train 500 racing_model.ez` sets the episode count explicitly; add `--fresh` to start new weights while retaining the experience file. An optional experience-file argument selects another CSV. `AI race racing_model.ez` loads the saved model and displays the car's position, speed, tire temperature/wear/grip, and lap progress as it runs.

For portable live inference, create a track CSV containing one `x,y` centerline point per
line (an optional `x,y` header is accepted), then run:

```text
AI live racing_model.ez track.csv gear.cfg
```

The process reads one telemetry sample per line from standard input using
`timestamp,x,y,heading,speed_mps,rpm,gear` and emits one JSON recommendation per line.
Recommendations include the DQN steering/pedal action, route progress, an estimated arrival
time, and a gear/shift suggestion. This interface is advisory-only: it does not write to
CAN, actuators, or any vehicle control bus. The arrival time is a current-speed estimate,
not a validated optimal lap-time prediction. Vehicle-specific gearbox ratios and RPM limits
must be configured before deployment.

The optional gearbox file uses:

```text
ratios=3.50,2.10,1.40,1.00,0.80
final_drive=3.42
redline_rpm=7000
upshift_rpm=6200
downshift_rpm=1800
```

The simulator API is in `include/RacingSimulation.hpp`. Pass a `Track` and `CarConfig` to `RacingSimulator` to set the track centerline/width/surface friction and vehicle dimensions, mass, center-of-gravity height, steering/drive/brake characteristics, tire material, initial tread/wear/temperature, and tire thermal/wear rates. `VehicleState` contains the evolving 3D center-of-gravity position and dynamic vehicle/tire values. The track is modeled in the XY plane; CG height supplies the vertical coordinate. `include/RacingTrainer.hpp` connects its normalized observations and nine discrete steering/pedal commands to the DQN.

This is a research prototype, not a validated vehicle-dynamics package. Its bicycle-style motion, tire grip/heat/wear equations, and oval default track are simplified approximations; they do not model suspension, full tire curves, weather, pit stops, or a full 3D road. The model learns only the behavior represented by this simulator and should not be used to control a real vehicle.

Build and run the focused tests with:

```text
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```
