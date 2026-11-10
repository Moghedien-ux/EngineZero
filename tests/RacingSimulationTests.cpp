#include "RacingSimulation.hpp"
#include "DQNAgent.hpp"
#include "LiveRacing.hpp"
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void test_simulator_updates_dynamic_state() {
    Racing::CarConfig car;
    car.initial_tire_wear = 0.1f;
    car.initial_tire_temperature_c = 70.0f;
    car.tire_wear_per_second = 0.1f;
    car.tire_heat_generation_c_per_s = 5.0f;
    Racing::RacingSimulator simulator(Racing::Track::oval(35.0f, 25.0f, 8.0f, 0.9f), car);

    const auto initial_grip = simulator.state().tire_grip_coefficient;
    for (size_t i = 0; i < 30; ++i) {
        simulator.step(5);
    }

    const Racing::VehicleState& state = simulator.state();
    require(state.steps == 30, "Simulation step counter did not advance.");
    require(state.cg_z == car.cg_height_m, "The car center of gravity must remain above the track surface.");
    require(state.tire_wear > car.initial_tire_wear, "Tire wear did not increase.");
    require(state.tire_temperature_c != car.initial_tire_temperature_c, "Tire temperature did not change.");
    require(state.tire_grip_coefficient != initial_grip, "Dynamic tire state did not update grip.");
    require(simulator.observation().size() == Racing::RacingSimulator::observation_size,
            "Observation size does not match the DQN input size.");
}

void test_dqn_model_round_trip() {
    DQNAgent original(4, 3);
    Tensor state({1, 4}, 0.25f);
    const Tensor expected = original.predict(state);

    std::stringstream model;
    original.save(model);
    DQNAgent restored(4, 3);
    restored.load(model);
    const Tensor actual = restored.predict(state);

    require(expected.shape() == actual.shape(), "Restored DQN output shape changed.");
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::abs(expected.at_flat(i) - actual.at_flat(i)) < 1e-6f,
                "Restored DQN predictions differ from the saved model.");
    }
}

void test_stalled_vehicle_terminates() {
    Racing::RacingSimulator simulator(Racing::Track::oval(35.0f, 25.0f, 8.0f, 0.9f));
    Racing::StepResult result{};
    for (size_t i = 0; i < 101 && !result.done; ++i) {
        result = simulator.step(1);
    }
    require(result.done, "A stationary car should not run an episode forever.");
    require(simulator.state().elapsed_time_s <= 5.1f, "Stall detection did not respect its time limit.");
}

void test_live_controller_emits_recommendation() {
    Racing::LiveRacingController controller(
        Racing::Track::oval(35.0f, 25.0f, 8.0f, 0.9f));
    Racing::LiveTelemetry telemetry{0.0, 35.0f, 0.0f, 1.5707963f, 10.0f, 4000.0f, 2};
    const Racing::LiveRecommendation recommendation = controller.update(telemetry);
    require(recommendation.recommended_gear == 2, "Live controller changed a stable gear recommendation.");
    require(recommendation.distance_to_finish_m >= 0.0f, "Live controller returned a negative route distance.");
}

} // namespace

int main() {
    try {
        test_simulator_updates_dynamic_state();
        test_dqn_model_round_trip();
        test_stalled_vehicle_terminates();
        test_live_controller_emits_recommendation();
        std::cout << "All racing simulation tests passed.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
}
