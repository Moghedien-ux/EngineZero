#include <iostream>
#include "Tensor.hpp"
#include "Autograd.hpp"
#include "DQNAgent.hpp"
#include "TransformerBlock.hpp"
#include "Conv2d.hpp"
#include "RacingTrainer.hpp"
#include "LiveRacing.hpp"
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        const std::string command = argc > 1 ? argv[1] : "train";
        if (command == "train") {
            const size_t episodes = argc > 2 ? std::stoull(argv[2]) : 200;
            const std::string model_path = argc > 3 ? argv[3] : "racing_model.ez";
            if (argc > 6) {
                throw std::invalid_argument("Too many train arguments.");
            }
            bool fresh = false;
            bool has_experience_path = false;
            std::string experience_path = "racing_experience.csv";
            for (int i = 4; i < argc; ++i) {
                const std::string argument = argv[i];
                if (argument == "--fresh") {
                    if (fresh) throw std::invalid_argument("--fresh may only be specified once.");
                    fresh = true;
                } else if (!has_experience_path) {
                    experience_path = argument;
                    has_experience_path = true;
                } else {
                    throw std::invalid_argument("Only one experience file may be specified.");
                }
            }
            Racing::RacingTrainer trainer;
            trainer.train(episodes, model_path, !fresh, experience_path);
            return 0;
        }
        if (command == "race") {
            if (argc > 3) {
                throw std::invalid_argument("Usage: AI race [model-file]");
            }
            const std::string model_path = argc > 2 ? argv[2] : "racing_model.ez";
            Racing::RacingTrainer trainer;
            trainer.load_model(model_path);
            trainer.race();
            return 0;
        }
        if (command == "live") {
            if (argc > 5) {
                throw std::invalid_argument("Usage: AI live [model-file] [track.csv] [gear.cfg]");
            }
            const std::string model_path = argc > 2 ? argv[2] : "racing_model.ez";
            const std::string track_path = argc > 3 ? argv[3] : "track.csv";
            const Racing::GearConfig gears = argc > 4
                ? Racing::load_gear_config(argv[4]) : Racing::GearConfig{};
            Racing::LiveRacingController controller(
                Racing::load_track_csv(track_path, 6.0f, 1.0f), {}, gears);
            controller.load_model(model_path);
            std::string row;
            bool first_data_row = true;
            while (std::getline(std::cin, row)) {
                if (row.empty()) continue;
                if (first_data_row && row == "timestamp,x,y,heading,speed_mps,rpm,gear") {
                    first_data_row = false;
                    continue;
                }
                first_data_row = false;
                Racing::LiveTelemetry telemetry;
                if (!Racing::parse_live_telemetry(row, telemetry)) {
                    throw std::invalid_argument(
                        "Live telemetry must be CSV: timestamp,x,y,heading,speed_mps,rpm,gear");
                }
                Racing::write_live_recommendation(std::cout, telemetry, controller.update(telemetry));
                std::cout.flush();
            }
            return 0;
        }
        if (command == "--help" || command == "help") {
            std::cout << "Usage:\n"
                      << "  AI                         Start training (200 episodes by default)\n"
                      << "  AI train [episodes] [model-file] [experience-file] [--fresh]\n"
                      << "  AI race [model-file]\n"
                      << "  AI live [model-file] [track.csv] [gear.cfg]\n"
                      << "  AI demo                    Run the tensor/transformer/CNN demo\n"
                      << "Training generates simulated experience and resumes saved model weights.\n";
            return 0;
        }
        if (command != "demo") {
            throw std::invalid_argument("Unknown command. Run AI --help for usage.");
        }

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
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}