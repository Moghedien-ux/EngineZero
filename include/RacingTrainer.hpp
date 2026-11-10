#pragma once

#include "DQNAgent.hpp"
#include "RacingSimulation.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Racing {

inline Tensor make_state_tensor(const RacingSimulator& simulator) {
    const std::vector<float> values = simulator.observation();
    Tensor state({1, values.size()});
    for (size_t i = 0; i < values.size(); ++i) {
        state[0, i] = values[i];
    }
    return state;
}

class RacingTrainer {
public:
    explicit RacingTrainer(RacingSimulator simulator = RacingSimulator(Track::oval(60.0f, 40.0f)))
        : simulator_(std::move(simulator)),
          base_track_(simulator_.track()),
          base_car_(simulator_.car_config()),
          agent_(RacingSimulator::observation_size, RacingSimulator::action_count, 0.001f, 0.99f),
          target_agent_(RacingSimulator::observation_size, RacingSimulator::action_count, 0.001f, 0.99f),
          random_engine_(std::random_device{}()) {
        target_agent_.copy_parameters_from(agent_);
    }

    void train(size_t episodes, const std::string& model_path, bool resume_existing = true,
               const std::string& experience_path = "racing_experience.csv") {
        if (episodes == 0) {
            throw std::invalid_argument("Training requires at least one episode.");
        }
        if (resume_existing && std::filesystem::exists(model_path)) {
            load_model(model_path);
            std::cout << "Resuming model from " << model_path << '\n';
        }

        constexpr size_t replay_capacity = 20000;
        constexpr size_t batch_warmup = 32;
        constexpr size_t target_update_interval = 250;
        constexpr size_t checkpoint_interval = 10;
        std::vector<Transition> replay;
        replay.reserve(replay_capacity);
        const size_t loaded_experiences = load_experiences(experience_path, replay, replay_capacity);
        const bool write_header = !std::filesystem::exists(experience_path) ||
                                  std::filesystem::file_size(experience_path) == 0;
        const std::filesystem::path experience_file(experience_path);
        if (!experience_file.parent_path().empty()) {
            std::filesystem::create_directories(experience_file.parent_path());
        }
        std::ofstream experience_output(experience_file, std::ios::app);
        if (!experience_output) {
            throw std::runtime_error("Unable to create training experience file: " + experience_path);
        }
        experience_output.precision(9);
        if (write_header) {
            experience_output << "ENGINEZERO_EXPERIENCE_V1," << RacingSimulator::observation_size << '\n';
        }

        size_t replay_cursor = loaded_experiences % replay_capacity;
        size_t updates_since_target_sync = 0;
        float epsilon = 1.0f;
        std::cout << "Training episodes: " << episodes << " | replay loaded: " << loaded_experiences
                  << " | new experience data: " << experience_path << '\n';
        for (size_t episode = 1; episode <= episodes; ++episode) {
            simulator_ = make_training_scenario();
            simulator_.reset();
            Tensor state = make_state_tensor(simulator_);
            float episode_reward = 0.0f;
            bool done = false;

            while (!done) {
                const int action = agent_.select_action(state, epsilon);
                const StepResult result = simulator_.step(action);
                Tensor next_state = make_state_tensor(simulator_);
                episode_reward += result.reward;

                Transition transition{state, action, result.reward, next_state, result.done};
                write_experience(experience_output, transition);
                if (replay.size() < replay_capacity) {
                    replay.push_back(std::move(transition));
                } else {
                    replay[replay_cursor] = std::move(transition);
                    replay_cursor = (replay_cursor + 1) % replay_capacity;
                }

                if (replay.size() >= batch_warmup) {
                    std::uniform_int_distribution<size_t> sample(0, replay.size() - 1);
                    const Transition& item = replay[sample(random_engine_)];
                    agent_.train_step(item.state, item.action, item.reward, item.next_state,
                                      item.done, &target_agent_);
                    ++updates_since_target_sync;
                    if (updates_since_target_sync >= target_update_interval) {
                        target_agent_.copy_parameters_from(agent_);
                        updates_since_target_sync = 0;
                    }
                }

                state = std::move(next_state);
                done = result.done;
            }

            epsilon = std::max(0.05f, epsilon * 0.99f);
            std::cout << "Episode " << episode << '/' << episodes
                      << " | reward " << episode_reward
                      << " | time " << simulator_.state().elapsed_time_s << " s"
                      << " | distance " << simulator_.state().lap_progress_m << '/'
                      << simulator_.track().length() << " m"
                      << " | surface mu " << simulator_.track().surface_friction()
                      << " | mass " << simulator_.car_config().mass_kg << " kg"
                      << " | tire " << simulator_.state().tire_temperature_c << " C, "
                      << simulator_.state().tire_wear * 100.0f << "% worn"
                      << " | epsilon " << epsilon << '\n';
            experience_output.flush();
            if (!experience_output) {
                throw std::runtime_error("Failed while writing training experience data.");
            }
            if (episode % checkpoint_interval == 0) {
                save_model(model_path);
                std::cout << "Checkpoint saved to " << model_path << '\n';
            }
        }

        save_model(model_path);
        std::cout << "Saved trained model to " << model_path << '\n';
    }

    void load_model(const std::string& model_path) {
        std::ifstream input(model_path);
        if (!input) {
            throw std::runtime_error("Unable to open model file: " + model_path);
        }
        agent_.load(input);
        target_agent_.copy_parameters_from(agent_);
    }

    void save_model(const std::string& model_path) const {
        const std::filesystem::path model_file(model_path);
        if (!model_file.parent_path().empty()) {
            std::filesystem::create_directories(model_file.parent_path());
        }
        std::ofstream output(model_file, std::ios::trunc);
        if (!output) {
            throw std::runtime_error("Unable to create model file: " + model_path);
        }
        agent_.save(output);
    }

    void race() {
        simulator_.reset();
        bool done = false;
        while (!done) {
            const Tensor state = make_state_tensor(simulator_);
            const int action = agent_.select_action(state, 0.0f);
            const StepResult result = simulator_.step(action);
            done = result.done;

            if (simulator_.state().steps % 10 == 0 || done) {
                const VehicleState& car = simulator_.state();
                std::cout << "\rTime " << car.elapsed_time_s << " s"
                          << " | pos (" << car.x << ", " << car.y << ", " << car.cg_z << ") m"
                          << " | heading " << car.heading << " rad"
                          << " | yaw rate " << car.yaw_rate_rad_s << " rad/s"
                          << " | speed " << car.speed_mps * 3.6f << " km/h"
                          << " | tire " << car.tire_temperature_c << " C"
                          << " | wear " << car.tire_wear * 100.0f << "%"
                          << " | grip " << car.tire_grip_coefficient
                          << " | lateral " << result.lateral_offset << " m"
                          << " | progress " << car.lap_progress_m << '/'
                          << simulator_.track().length() << " m" << std::flush;
            }
        }

        std::cout << "\n" << (simulator_.state().lap_progress_m >= simulator_.track().length()
            ? "Finished the lap." : "Episode ended before the finish.")
            << " Elapsed time: " << simulator_.state().elapsed_time_s << " s\n";
    }

    DQNAgent& agent() noexcept { return agent_; }

private:
    struct Transition {
        Tensor state;
        int action;
        float reward;
        Tensor next_state;
        bool done;
    };

    static constexpr const char* experience_signature = "ENGINEZERO_EXPERIENCE_V1";

    RacingSimulator make_training_scenario() {
        std::uniform_real_distribution<float> scale(0.85f, 1.15f);
        std::uniform_real_distribution<float> surface_scale(0.8f, 1.1f);
        std::vector<Vec2> centerline = base_track_.centerline();
        const float scale_x = scale(random_engine_);
        const float scale_y = scale(random_engine_);
        for (Vec2& point : centerline) {
            point.x *= scale_x;
            point.y *= scale_y;
        }

        CarConfig car = base_car_;
        car.mass_kg *= scale(random_engine_);
        car.cg_height_m *= scale(random_engine_);
        car.tire_material_friction *= scale(random_engine_);
        car.maximum_steering_angle_rad *= scale(random_engine_);
        car.drive_force_n *= scale(random_engine_);
        car.brake_force_n *= scale(random_engine_);

        const float half_width = std::max(base_track_.half_width() * scale(random_engine_),
                                          car.width_m * 0.5f + 1.0f);
        const float surface_friction = std::max(0.3f,
            base_track_.surface_friction() * surface_scale(random_engine_));
        Track track(std::move(centerline), half_width, surface_friction);
        return RacingSimulator(std::move(track), car, simulator_.time_step_s(),
                               simulator_.maximum_steps());
    }

    static void write_experience(std::ostream& output, const Transition& transition) {
        const size_t state_size = transition.state.size();
        for (size_t i = 0; i < state_size; ++i) {
            output << transition.state.at_flat(i) << ',';
        }
        output << transition.action << ',' << transition.reward << ',';
        for (size_t i = 0; i < transition.next_state.size(); ++i) {
            output << transition.next_state.at_flat(i) << ',';
        }
        output << (transition.done ? 1 : 0) << '\n';
    }

    static std::vector<float> parse_experience_row(const std::string& row) {
        std::vector<float> values;
        values.reserve(RacingSimulator::observation_size * 2 + 3);
        std::stringstream line(row);
        std::string field;
        while (std::getline(line, field, ',')) {
            size_t parsed = 0;
            float value = 0.0f;
            try {
                value = std::stof(field, &parsed);
            } catch (const std::exception&) {
                throw std::runtime_error("Training experience file contains a non-numeric field.");
            }
            if (parsed != field.size() || !std::isfinite(value)) {
                throw std::runtime_error("Training experience file contains invalid numeric data.");
            }
            values.push_back(value);
        }
        if (values.size() != RacingSimulator::observation_size * 2 + 3) {
            throw std::runtime_error("Training experience row has an unexpected number of fields.");
        }
        return values;
    }

    static size_t load_experiences(const std::string& path, std::vector<Transition>& replay,
                                   size_t capacity) {
        if (!std::filesystem::exists(path)) return 0;
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("Unable to read training experience file: " + path);
        }

        std::string header;
        if (!std::getline(input, header) ||
            header != std::string(experience_signature) + ',' +
                          std::to_string(RacingSimulator::observation_size)) {
            throw std::runtime_error("Training experience file has an invalid or incompatible header.");
        }

        size_t loaded = 0;
        std::string row;
        while (std::getline(input, row)) {
            if (row.empty()) continue;
            const std::vector<float> values = parse_experience_row(row);
            const float action_value = values[RacingSimulator::observation_size];
            const float done_value = values.back();
            if (action_value < 0.0f ||
                action_value >= static_cast<float>(RacingSimulator::action_count) ||
                std::floor(action_value) != action_value ||
                (done_value != 0.0f && done_value != 1.0f)) {
                throw std::runtime_error("Training experience file contains an invalid action or terminal flag.");
            }

            Tensor state({1, RacingSimulator::observation_size});
            Tensor next_state({1, RacingSimulator::observation_size});
            for (size_t i = 0; i < RacingSimulator::observation_size; ++i) {
                state[0, i] = values[i];
                next_state[0, i] = values[RacingSimulator::observation_size + 2 + i];
            }
            Transition transition{std::move(state), static_cast<int>(action_value),
                                  values[RacingSimulator::observation_size + 1],
                                  std::move(next_state), done_value == 1.0f};
            if (replay.size() < capacity) {
                replay.push_back(std::move(transition));
            } else {
                replay[loaded % capacity] = std::move(transition);
            }
            ++loaded;
        }
        if (!input.eof()) {
            throw std::runtime_error("Failed while reading training experience file.");
        }
        return loaded;
    }

    RacingSimulator simulator_;
    Track base_track_;
    CarConfig base_car_;
    DQNAgent agent_;
    DQNAgent target_agent_;
    std::mt19937 random_engine_;
};

} // namespace Racing
