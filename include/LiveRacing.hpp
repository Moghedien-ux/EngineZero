#pragma once

#include "DQNAgent.hpp"
#include "RacingSimulation.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Racing {

struct LiveTelemetry {
    double timestamp_s{};
    float x{};
    float y{};
    float heading{};
    float speed_mps{};
    float rpm{};
    int gear{};
};

struct GearConfig {
    std::vector<float> ratios{3.50f, 2.10f, 1.40f, 1.00f, 0.80f};
    float final_drive{3.42f};
    float redline_rpm{7000.0f};
    float upshift_rpm{6200.0f};
    float downshift_rpm{1800.0f};

    void validate() const {
        if (ratios.empty() || !std::all_of(ratios.begin(), ratios.end(),
                                           [](float value) { return std::isfinite(value) && value > 0.0f; }) ||
            !std::isfinite(final_drive) || final_drive <= 0.0f ||
            !std::isfinite(redline_rpm) || redline_rpm <= 0.0f ||
            !std::isfinite(upshift_rpm) || upshift_rpm <= 0.0f || upshift_rpm >= redline_rpm ||
            !std::isfinite(downshift_rpm) || downshift_rpm < 0.0f ||
            downshift_rpm >= upshift_rpm) {
            throw std::invalid_argument("Gear configuration contains invalid ratios or RPM limits.");
        }
    }
};

struct LiveRecommendation {
    int action{};
    DriverAction driving_action{};
    int recommended_gear{};
    std::string gear_change{"hold"};
    float shift_rpm{};
    float route_progress_m{};
    float distance_to_finish_m{};
    float estimated_arrival_time_s{};
    float lateral_offset_m{};
    float heading_error_rad{};
};

inline Track load_track_csv(const std::string& path, float half_width, float surface_friction) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Unable to open track CSV: " + path);
    }

    std::vector<Vec2> points;
    std::string row;
    size_t line_number = 0;
    while (std::getline(input, row)) {
        ++line_number;
        if (row.empty()) continue;
        std::stringstream line(row);
        std::string x_field;
        std::string y_field;
        if (!std::getline(line, x_field, ',') || !std::getline(line, y_field, ',')) {
            if (points.empty()) continue;
            throw std::runtime_error("Track CSV line " + std::to_string(line_number) +
                                     " must contain x,y.");
        }
        try {
            size_t x_parsed = 0;
            size_t y_parsed = 0;
            const float x = std::stof(x_field, &x_parsed);
            const float y = std::stof(y_field, &y_parsed);
            if (x_parsed != x_field.size() || y_parsed != y_field.size() ||
                !std::isfinite(x) || !std::isfinite(y)) {
                throw std::invalid_argument("non-numeric");
            }
            points.push_back({x, y});
        } catch (const std::exception&) {
            if (!points.empty()) {
                throw std::runtime_error("Track CSV line " + std::to_string(line_number) +
                                         " contains invalid coordinates.");
            }
        }
    }
    return Track(std::move(points), half_width, surface_friction);
}

inline GearConfig load_gear_config(const std::string& path) {
    GearConfig config;
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Unable to open gear configuration: " + path);
    std::string row;
    size_t line_number = 0;
    while (std::getline(input, row)) {
        ++line_number;
        if (row.empty() || row[0] == '#') continue;
        const size_t separator = row.find('=');
        if (separator == std::string::npos) {
            throw std::runtime_error("Gear configuration line " + std::to_string(line_number) +
                                     " must use key=value.");
        }
        const std::string key = row.substr(0, separator);
        const std::string value = row.substr(separator + 1);
        try {
            if (key == "ratios") {
                config.ratios.clear();
                std::stringstream values(value);
                std::string ratio;
                while (std::getline(values, ratio, ',')) config.ratios.push_back(std::stof(ratio));
            } else if (key == "final_drive") {
                config.final_drive = std::stof(value);
            } else if (key == "redline_rpm") {
                config.redline_rpm = std::stof(value);
            } else if (key == "upshift_rpm") {
                config.upshift_rpm = std::stof(value);
            } else if (key == "downshift_rpm") {
                config.downshift_rpm = std::stof(value);
            } else {
                throw std::invalid_argument("unknown key");
            }
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid gear configuration line " +
                                     std::to_string(line_number) + ".");
        }
    }
    config.validate();
    return config;
}

class LiveRacingController {
public:
    LiveRacingController(Track track, CarConfig car = {}, GearConfig gears = {})
        : track_(std::move(track)), car_(car), gears_(std::move(gears)),
          agent_(RacingSimulator::observation_size, RacingSimulator::action_count, 0.001f, 0.99f) {
        gears_.validate();
    }

    void load_model(const std::string& path) {
        std::ifstream input(path);
        if (!input) throw std::runtime_error("Unable to open model file: " + path);
        agent_.load(input);
    }

    LiveRecommendation update(const LiveTelemetry& telemetry) const {
        validate(telemetry);
        const TrackProjection projection = track_.project({telemetry.x, telemetry.y});
        const float heading_error = wrap_angle(telemetry.heading - projection.heading);
        Tensor state = make_observation(telemetry, projection, heading_error);
        const int action = agent_.select_action(state, 0.0f);
        const DriverAction driving_action = RacingSimulator::decode_action(action);

        int recommended_gear = telemetry.gear;
        std::string gear_change = "hold";
        float shift_rpm = gears_.upshift_rpm;
        if (telemetry.rpm >= gears_.upshift_rpm && telemetry.gear < static_cast<int>(gears_.ratios.size())) {
            recommended_gear = telemetry.gear + 1;
            gear_change = "upshift";
        } else if (telemetry.rpm <= gears_.downshift_rpm && telemetry.gear > 1) {
            recommended_gear = telemetry.gear - 1;
            gear_change = "downshift";
            shift_rpm = gears_.downshift_rpm;
        }

        const float distance_to_finish = std::max(0.0f, track_.length() - projection.distance);
        const float estimated_time = distance_to_finish / std::max(telemetry.speed_mps, 0.1f);
        return {action, driving_action, recommended_gear, gear_change, shift_rpm,
                projection.distance, distance_to_finish, estimated_time,
                projection.lateral_offset, heading_error};
    }

private:
    static void validate(const LiveTelemetry& telemetry) {
        if (!std::isfinite(telemetry.timestamp_s) || !std::isfinite(telemetry.x) ||
            !std::isfinite(telemetry.y) || !std::isfinite(telemetry.heading) ||
            !std::isfinite(telemetry.speed_mps) || !std::isfinite(telemetry.rpm) ||
            telemetry.speed_mps < 0.0f || telemetry.rpm < 0.0f || telemetry.gear < 1) {
            throw std::invalid_argument("Telemetry contains invalid or negative values.");
        }
    }

    Tensor make_observation(const LiveTelemetry& telemetry, const TrackProjection& projection,
                            float heading_error) const {
        const float tire_temperature = car_.initial_tire_temperature_c;
        const float tire_wear = car_.initial_tire_wear;
        const float tire_grip = track_.surface_friction() * car_.tire_material_friction *
                                car_.initial_tread_coefficient;
        const float half_lane = track_.half_width();
        const float normalized_curvature = std::clamp(projection.curvature * 20.0f, -1.0f, 1.0f);
        const std::vector<float> values = {
            std::clamp(telemetry.speed_mps / 50.0f, 0.0f, 1.5f),
            std::clamp(projection.lateral_offset / half_lane, -2.0f, 2.0f),
            std::sin(heading_error), std::cos(heading_error), normalized_curvature,
            0.0f, std::clamp(tire_temperature / std::max(car_.maximum_tire_temperature_c, 1.0f), 0.0f, 2.0f),
            tire_wear, std::clamp(projection.distance / track_.length(), -1.0f, 1.0f),
            std::clamp(tire_grip / 2.0f, 0.0f, 1.0f), 0.0f,
            std::clamp((half_lane - std::abs(projection.lateral_offset)) / half_lane, -1.0f, 1.0f),
            std::clamp(track_.surface_friction() / 1.5f, 0.0f, 2.0f),
            std::clamp(car_.tire_material_friction / 1.5f, 0.0f, 2.0f),
            std::clamp(car_.mass_kg / 1500.0f, 0.0f, 2.0f),
            std::clamp(car_.cg_height_m / 1.0f, 0.0f, 2.0f),
            std::clamp(half_lane / 10.0f, 0.0f, 2.0f),
            std::clamp(car_.maximum_steering_angle_rad / 0.7f, 0.0f, 2.0f),
            std::clamp(car_.drive_force_n / car_.mass_kg / 8.0f, 0.0f, 2.0f),
            std::clamp(car_.brake_force_n / car_.mass_kg / 12.0f, 0.0f, 2.0f)
        };
        Tensor state({1, values.size()});
        for (size_t i = 0; i < values.size(); ++i) state[0, i] = values[i];
        return state;
    }

    Track track_;
    CarConfig car_;
    GearConfig gears_;
    mutable DQNAgent agent_;
};

inline bool parse_live_telemetry(const std::string& row, LiveTelemetry& telemetry) {
    std::stringstream line(row);
    std::string field;
    std::vector<std::string> fields;
    while (std::getline(line, field, ',')) fields.push_back(field);
    if (fields.size() != 7) return false;
    try {
        size_t parsed = 0;
        telemetry.timestamp_s = std::stod(fields[0], &parsed);
        if (parsed != fields[0].size()) return false;
        telemetry.x = std::stof(fields[1], &parsed);
        if (parsed != fields[1].size()) return false;
        telemetry.y = std::stof(fields[2], &parsed);
        if (parsed != fields[2].size()) return false;
        telemetry.heading = std::stof(fields[3], &parsed);
        if (parsed != fields[3].size()) return false;
        telemetry.speed_mps = std::stof(fields[4], &parsed);
        if (parsed != fields[4].size()) return false;
        telemetry.rpm = std::stof(fields[5], &parsed);
        if (parsed != fields[5].size()) return false;
        const float gear = std::stof(fields[6], &parsed);
        if (parsed != fields[6].size() || std::floor(gear) != gear) return false;
        telemetry.gear = static_cast<int>(gear);
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

inline void write_live_recommendation(std::ostream& output, const LiveTelemetry& telemetry,
                                      const LiveRecommendation& recommendation) {
    output << std::setprecision(9)
           << "{\"timestamp_s\":" << telemetry.timestamp_s
           << ",\"action\":" << recommendation.action
           << ",\"steering\":" << recommendation.driving_action.steering
           << ",\"pedal\":" << recommendation.driving_action.pedal
           << ",\"recommended_gear\":" << recommendation.recommended_gear
           << ",\"gear_change\":\"" << recommendation.gear_change
           << "\",\"shift_rpm\":" << recommendation.shift_rpm
           << ",\"route_progress_m\":" << recommendation.route_progress_m
           << ",\"distance_to_finish_m\":" << recommendation.distance_to_finish_m
           << ",\"estimated_arrival_time_s\":" << recommendation.estimated_arrival_time_s
           << ",\"lateral_offset_m\":" << recommendation.lateral_offset_m
           << ",\"heading_error_rad\":" << recommendation.heading_error_rad << "}\n";
}

} // namespace Racing
