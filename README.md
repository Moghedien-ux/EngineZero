This project stems from my passion for low level optimization and understanding how things happen behind the scenes. After following and understanding a PyTorch based course provided by DeepLearning.AI, I realized I was able to train AI models on whatever I wanted, however, whilst I gained the knowledge of how AI models work, I started to wonder: how does PyTorch work and how optimal is it?
Given my background as a Computer Software Engineering student withing the domain of Systems Engineering and Applied Informatics I started to wonder: If I wanted to train a model meant to control a robot, how efficient would PyTorch be?
After conducting some research and by the simple fact that PyTorch is Python based at heart (even though there are C/C++ versions) I realized that for robotic systems that run on not so permissive components, maybe there is a way to make things more efficient,
as such I embarked on an interesting journey: Making a tool similar to PyTorch from scratch in C++ using modern C++ 26 standards (even though C++ 26 wasn't passed as the new standard C++ because of contracts features that turned the vote below 66%) and low level
memory optimization to make a tool that is able to create and train a model on insanely low spec levels (unfortunately it is purely CPU based atm, so PyTorch wins performance if we're talking about a system that has a GPU).

The EngineZero tool can develop different types of models, for example:
 - Transformer / Micro-LLM (focused on self-attention, large matrices, tokenization, and generation).
 - Reinforcement Learning / Robotics in a simulator (focused on real-time interaction with a physical environment/game, DQN/PPO).
 - Computer Vision / CNN Engine (focused on convolutions, im2col, pooling, and classification).
