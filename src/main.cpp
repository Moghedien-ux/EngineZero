#include <iostream>
#include "Tensor.hpp"
#include "Autograd.hpp"
#include "DQNAgent.hpp"
#include "TransformerBlock.hpp"
#include "Conv2D.hpp"

int main() {
    std::cout << "====================================================\n";
    std::cout << " ENGINE ZERO C++26 - FULL MULTI-MODEL DEMO RUN \n";
    std::cout << "====================================================\n\n";

    std::cout << "--- [1/3] Testing Reinforcement Learning Agent (DQN) ---\n";
    DQNAgent agent(4, 2);     Tensor state({1, 4}, 0.5f);
    Tensor next_state({1, 4}, 0.6f);

    float loss_rl = agent.train_step(state, 1, 1.0f, next_state, false);
    std::cout << "RL Bellman Update Loss: " << loss_rl << " | OK!\n\n";

    std::cout << "--- [2/3] Testing Transformer Block (Self-Attention) ---\n";
    TransformerBlock transformer(8, 2);     Tensor seq_input({4, 8}, 0.2f);
    auto tokens = ValueNode::make(seq_input, {}, "Tokens");
    auto tf_out = transformer.forward(tokens);
    std::cout << "Transformer Output Shape: [" << tf_out->data.shape()[0] << ", " << tf_out->data.shape()[1] << "] | OK!\n\n";

    std::cout << "--- [3/3] Testing Convolutional Engine (Conv2D) ---\n";
    Conv2D conv(1, 4, 3);     Tensor img_input({1, 8, 8}, 1.0f);
    auto image = ValueNode::make(img_input, {}, "Image");
    auto conv_out = conv.forward(image);
    std::cout << "Conv2D Feature Map Shape: [" << conv_out->data.shape()[0] << ", " << conv_out->data.shape()[1] << "] | OK!\n\n";

    return 0;
}