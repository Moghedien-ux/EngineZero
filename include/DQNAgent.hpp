#pragma once

#include "Module.hpp"
#include "Autograd.hpp"
#include "Linear.hpp"
#include "Ops.hpp"
#include "Optimizer.hpp"
#include <vector>
#include <random>

class DQNAgent : public Module {
public:
    Linear fc1;
    Linear fc2;
    SGD optimizer;
    float gamma;

    DQNAgent(size_t state_dim, size_t action_dim, float lr = 0.01f, float gamma = 0.99f)
        : fc1(state_dim, 32),
          fc2(32, action_dim),
          optimizer(parameters(), lr),
          gamma(gamma) {}

    ValuePtr forward(const ValuePtr& state) {
        auto h = EngineOps::relu(fc1.forward(state));
        return fc2.forward(h);     }

    std::vector<ValuePtr> parameters() const override {
        std::vector<ValuePtr> params;
        for (auto p : fc1.parameters()) params.push_back(p);
        for (auto p : fc2.parameters()) params.push_back(p);
        return params;
    }

        float train_step(const Tensor& state_data, int action, float reward, const Tensor& next_state_data, bool done) {
        optimizer.zero_grad();

        auto s = ValueNode::make(state_data, {}, "state");
        auto ns = ValueNode::make(next_state_data, {}, "next_state");

                auto q_values = forward(s);
        auto next_q_values = forward(ns);

                float max_next_q = next_q_values->data[0, 0];
        for (size_t i = 1; i < next_q_values->data.shape()[1]; ++i) {
            if (next_q_values->data[0, i] > max_next_q) {
                max_next_q = next_q_values->data[0, i];
            }
        }

        float target_q_val = done ? reward : reward + gamma * max_next_q;

        Tensor target_tensor(q_values->data.shape());
        for (size_t i = 0; i < q_values->data.shape()[1]; ++i) {
            target_tensor[0, i] = q_values->data[0, i];
        }
        target_tensor[0, action] = target_q_val;
        auto target = ValueNode::make(target_tensor, {}, "target_q");
        auto loss = EngineOps::mse_loss(q_values, target);

        loss->backward();
        optimizer.step();

        return loss->data[0, 0];
    }
};