#pragma once

#include "Module.hpp"
#include "Autograd.hpp"
#include "Linear.hpp"
#include "Ops.hpp"
#include "Optimizer.hpp"
#include <vector>
#include <random>
#include <cmath>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <string>

class DQNAgent : public Module {
private:
    static size_t checked_dimension(size_t dimension) {
        if (dimension == 0) {
            throw std::invalid_argument("DQN state and action dimensions must be positive.");
        }
        return dimension;
    }

public:
    Linear fc1;
    Linear fc2;
    SGD optimizer;
    float gamma;
    size_t state_dim;
    size_t action_dim;
    std::mt19937 random_engine{std::random_device{}()};

    DQNAgent(size_t state_dim, size_t action_dim, float lr = 0.01f, float gamma = 0.99f)
        : fc1(checked_dimension(state_dim), 32),
          fc2(32, checked_dimension(action_dim)),
          optimizer(parameters(), lr),
          gamma(gamma),
          state_dim(state_dim),
          action_dim(action_dim) {
        if (!std::isfinite(lr) || lr <= 0.0f || !std::isfinite(gamma) ||
            gamma < 0.0f || gamma > 1.0f) {
            throw std::invalid_argument("DQN learning rate must be positive and gamma must be in [0, 1].");
        }
    }

    ValuePtr forward(const ValuePtr& state) {
        auto h = EngineOps::relu(fc1.forward(state));
        return fc2.forward(h);     }

    std::vector<ValuePtr> parameters() const override {
        std::vector<ValuePtr> params;
        for (auto p : fc1.parameters()) params.push_back(p);
        for (auto p : fc2.parameters()) params.push_back(p);
        return params;
    }

    Tensor predict(const Tensor& state_data) const {
        Tensor hidden = EngineOps::matmul(state_data, fc1.W->data);
        hidden = EngineOps::add(hidden, fc1.b->data);
        hidden = EngineOps::relu(hidden);
        Tensor output = EngineOps::matmul(hidden, fc2.W->data);
        return EngineOps::add(output, fc2.b->data);
    }

    int select_action(const Tensor& state_data, float epsilon) {
        if (epsilon < 0.0f || epsilon > 1.0f) {
            throw std::invalid_argument("Epsilon must be between 0 and 1.");
        }
        std::uniform_real_distribution<float> chance(0.0f, 1.0f);
        if (chance(random_engine) < epsilon) {
            std::uniform_int_distribution<int> random_action(0, static_cast<int>(action_dim) - 1);
            return random_action(random_engine);
        }

        Tensor q_values = predict(state_data);
        int best_action = 0;
        for (size_t i = 1; i < action_dim; ++i) {
            if (q_values[0, i] > q_values[0, static_cast<size_t>(best_action)]) {
                best_action = static_cast<int>(i);
            }
        }
        return best_action;
    }

    float train_step(const Tensor& state_data, int action, float reward, const Tensor& next_state_data,
                     bool done, const DQNAgent* target_agent = nullptr) {
        if (state_data.ndim() != 2 || state_data.shape()[0] != 1 || state_data.shape()[1] != state_dim ||
            next_state_data.shape() != state_data.shape()) {
            throw std::invalid_argument("DQN states must have shape [1, state_dim].");
        }
        if (action < 0 || static_cast<size_t>(action) >= action_dim) {
            throw std::out_of_range("DQN action is outside the configured action space.");
        }

        optimizer.zero_grad();

        auto s = ValueNode::make(state_data, {}, "state");
        auto q_values = forward(s);
        Tensor next_q_values_data = target_agent ? target_agent->predict(next_state_data) : predict(next_state_data);

        float max_next_q = next_q_values_data[0, 0];
        for (size_t i = 1; i < action_dim; ++i) {
            if (next_q_values_data[0, i] > max_next_q) {
                max_next_q = next_q_values_data[0, i];
            }
        }

        float target_q_val = done ? reward : reward + gamma * max_next_q;

        Tensor target_tensor = q_values->data;
        target_tensor[0, static_cast<size_t>(action)] = target_q_val;
        auto target = ValueNode::make(target_tensor, {}, "target_q");
        auto loss = EngineOps::mse_loss(q_values, target);

        loss->backward();
        optimizer.step();

        return loss->data[0, 0];
    }

    void copy_parameters_from(const DQNAgent& other) {
        if (state_dim != other.state_dim || action_dim != other.action_dim) {
            throw std::invalid_argument("Cannot copy parameters between differently sized DQN agents.");
        }
        auto destination = parameters();
        auto source = other.parameters();
        for (size_t i = 0; i < destination.size(); ++i) {
            destination[i]->data = source[i]->data;
        }
    }

    void save(std::ostream& output) const {
        output << "ENGINEZERO_DQN_V1\n" << state_dim << ' ' << action_dim << '\n';
        const auto params = parameters();
        output << params.size() << '\n';
        output.precision(std::numeric_limits<float>::max_digits10);
        for (const auto& parameter : params) {
            output << parameter->data.size() << '\n';
            for (size_t i = 0; i < parameter->data.size(); ++i) {
                output << parameter->data.at_flat(i) << (i + 1 == parameter->data.size() ? '\n' : ' ');
            }
        }
        if (!output) {
            throw std::runtime_error("Failed to write DQN model.");
        }
    }

    void load(std::istream& input) {
        std::string signature;
        size_t stored_state_dim = 0;
        size_t stored_action_dim = 0;
        size_t parameter_count = 0;
        if (!(input >> signature >> stored_state_dim >> stored_action_dim >> parameter_count) ||
            signature != "ENGINEZERO_DQN_V1" || stored_state_dim != state_dim ||
            stored_action_dim != action_dim) {
            throw std::runtime_error("Invalid DQN model header or incompatible dimensions.");
        }

        auto params = parameters();
        if (parameter_count != params.size()) {
            throw std::runtime_error("DQN model parameter count does not match this network.");
        }

        std::vector<std::vector<float>> values(params.size());
        for (size_t i = 0; i < params.size(); ++i) {
            size_t count = 0;
            if (!(input >> count) || count != params[i]->data.size()) {
                throw std::runtime_error("DQN model parameter shape does not match this network.");
            }
            values[i].resize(count);
            for (float& value : values[i]) {
                if (!(input >> value) || !std::isfinite(value)) {
                    throw std::runtime_error("DQN model contains invalid parameter data.");
                }
            }
        }

        for (size_t i = 0; i < params.size(); ++i) {
            for (size_t j = 0; j < values[i].size(); ++j) {
                params[i]->data.at_flat(j) = values[i][j];
            }
        }
    }
};