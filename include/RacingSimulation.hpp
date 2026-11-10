#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

namespace Racing {

constexpr float gravity = 9.80665f;
constexpr float pi = 3.14159265358979323846f;

struct Vec2 {
    float x{};
    float y{};
};

inline float wrap_angle(float angle) {
    while (angle > pi) angle -= 2.0f * pi;
    while (angle < -pi) angle += 2.0f * pi;
    return angle;
}

struct TrackProjection {
    float distance{};
    float lateral_offset{};
    float heading{};
    float curvature{};
};

class Track {
public:
    Track(std::vector<Vec2> centerline, float half_width, float surface_friction)
        : centerline_(std::move(centerline)), half_width_(half_width), surface_friction_(surface_friction) {
        if (centerline_.size() < 3 || !std::isfinite(half_width_) ||
            !std::isfinite(surface_friction_) || half_width_ <= 0.0f || surface_friction_ <= 0.0f) {
            throw std::invalid_argument("A track needs at least three points, positive width, and positive friction.");
        }

        cumulative_distance_.resize(centerline_.size(), 0.0f);
        for (size_t i = 0; i < centerline_.size(); ++i) {
            const Vec2& current = centerline_[i];
            const Vec2& next = centerline_[(i + 1) % centerline_.size()];
            if (!std::isfinite(current.x) || !std::isfinite(current.y)) {
                throw std::invalid_argument("Track centerline coordinates must be finite.");
            }
            const float dx = next.x - current.x;
            const float dy = next.y - current.y;
            const float segment_length = std::sqrt(dx * dx + dy * dy);
            if (segment_length <= 1e-4f) {
                throw std::invalid_argument("Track centerline contains duplicate adjacent points.");
            }
            if (i + 1 < centerline_.size()) {
                cumulative_distance_[i + 1] = cumulative_distance_[i] + segment_length;
            }
            length_ += segment_length;
        }
    }

    static Track oval(float radius_x, float radius_y, float half_width = 6.0f,
                      float surface_friction = 1.0f, size_t samples = 256) {
        if (!std::isfinite(radius_x) || !std::isfinite(radius_y) ||
            radius_x <= 0.0f || radius_y <= 0.0f || samples < 8) {
            throw std::invalid_argument("Oval radii must be positive and the track needs at least eight samples.");
        }
        std::vector<Vec2> points;
        points.reserve(samples);
        for (size_t i = 0; i < samples; ++i) {
            const float angle = 2.0f * pi * static_cast<float>(i) / static_cast<float>(samples);
            points.push_back({radius_x * std::cos(angle), radius_y * std::sin(angle)});
        }
        return Track(std::move(points), half_width, surface_friction);
    }

    TrackProjection project(Vec2 point) const {
        float best_distance_squared = INFINITY;
        TrackProjection best{};

        for (size_t i = 0; i < centerline_.size(); ++i) {
            const Vec2& start = centerline_[i];
            const Vec2& end = centerline_[(i + 1) % centerline_.size()];
            const float dx = end.x - start.x;
            const float dy = end.y - start.y;
            const float segment_length_squared = dx * dx + dy * dy;
            const float t = std::clamp(((point.x - start.x) * dx + (point.y - start.y) * dy) /
                                       segment_length_squared, 0.0f, 1.0f);
            const float nearest_x = start.x + t * dx;
            const float nearest_y = start.y + t * dy;
            const float offset_x = point.x - nearest_x;
            const float offset_y = point.y - nearest_y;
            const float distance_squared = offset_x * offset_x + offset_y * offset_y;
            if (distance_squared >= best_distance_squared) continue;

            const float segment_length = std::sqrt(segment_length_squared);
            const float heading = std::atan2(dy, dx);
            const size_t previous = (i + centerline_.size() - 1) % centerline_.size();
            const float previous_heading = std::atan2(
                start.y - centerline_[previous].y, start.x - centerline_[previous].x);
            const float next_heading = std::atan2(
                centerline_[(i + 2) % centerline_.size()].y - end.y,
                centerline_[(i + 2) % centerline_.size()].x - end.x);
            const float curvature = wrap_angle(next_heading - previous_heading) /
                                    std::max(segment_length, 1e-4f);

            best_distance_squared = distance_squared;
            best.distance = cumulative_distance_[i] + t * segment_length;
            best.lateral_offset = (dx * offset_y - dy * offset_x) / segment_length;
            best.heading = heading;
            best.curvature = curvature;
        }
        return best;
    }

    const std::vector<Vec2>& centerline() const noexcept { return centerline_; }
    float half_width() const noexcept { return half_width_; }
    float surface_friction() const noexcept { return surface_friction_; }
    float length() const noexcept { return length_; }

private:
    std::vector<Vec2> centerline_;
    std::vector<float> cumulative_distance_;
    float half_width_{};
    float surface_friction_{};
    float length_{};
};

struct CarConfig {
    float mass_kg{1200.0f};
    float length_m{4.3f};
    float width_m{1.8f};
    float wheelbase_m{2.6f};
    float cg_height_m{0.55f};
    float maximum_steering_angle_rad{0.48f};
    float drive_force_n{6500.0f};
    float brake_force_n{12000.0f};
    float aerodynamic_drag_coefficient{0.42f};
    float frontal_area_m2{2.2f};
    float rolling_resistance{0.015f};
    float tire_material_friction{1.05f};
    float initial_tread_coefficient{1.0f};
    float initial_tire_wear{0.0f};
    float initial_tire_temperature_c{80.0f};
    float optimal_tire_temperature_c{90.0f};
    float maximum_tire_temperature_c{150.0f};
    float tire_heat_capacity_j_per_c{45000.0f};
    float tire_heat_generation_c_per_s{2.5f};
    float tire_cooling_c_per_s{0.04f};
    float tire_wear_per_second{0.00004f};
};

struct VehicleState {
    float x{};
    float y{};
    float cg_z{};
    float heading{};
    float speed_mps{};
    float yaw_rate_rad_s{};
    float tire_temperature_c{};
    float tire_wear{};
    float tire_tread_coefficient{};
    float tire_grip_coefficient{};
    float lateral_acceleration_mps2{};
    float lap_progress_m{};
    float elapsed_time_s{};
    size_t steps{};
};

struct DriverAction {
    float steering{};
    float pedal{};
};

struct StepResult {
    float reward{};
    bool done{};
    bool finished{};
    float lateral_offset{};
    float heading_error{};
};

class RacingSimulator {
public:
    static constexpr size_t observation_size = 20;
    static constexpr size_t action_count = 9;

    RacingSimulator(Track track, CarConfig car = {}, float time_step_s = 0.05f,
                    size_t maximum_steps = 3000)
        : track_(std::move(track)), car_(car), time_step_s_(time_step_s), maximum_steps_(maximum_steps) {
        const float car_parameters[] = {
            car_.mass_kg, car_.length_m, car_.width_m, car_.wheelbase_m, car_.cg_height_m,
            car_.maximum_steering_angle_rad, car_.drive_force_n, car_.brake_force_n,
            car_.aerodynamic_drag_coefficient, car_.frontal_area_m2, car_.rolling_resistance,
            car_.tire_material_friction, car_.initial_tread_coefficient, car_.initial_tire_wear,
            car_.initial_tire_temperature_c, car_.optimal_tire_temperature_c,
            car_.maximum_tire_temperature_c, car_.tire_heat_capacity_j_per_c,
            car_.tire_heat_generation_c_per_s, car_.tire_cooling_c_per_s, car_.tire_wear_per_second,
            time_step_s_
        };
        if (!std::all_of(std::begin(car_parameters), std::end(car_parameters),
                         [](float value) { return std::isfinite(value); }) ||
            car_.mass_kg <= 0.0f || car_.length_m <= 0.0f || car_.width_m <= 0.0f ||
            car_.wheelbase_m <= 0.0f || car_.cg_height_m <= 0.0f ||
            car_.maximum_steering_angle_rad <= 0.0f || car_.drive_force_n <= 0.0f ||
            car_.brake_force_n <= 0.0f || car_.tire_material_friction <= 0.0f ||
            car_.initial_tread_coefficient <= 0.0f || car_.initial_tire_wear < 0.0f ||
            car_.initial_tire_wear > 1.0f || car_.initial_tire_temperature_c < 0.0f ||
            car_.optimal_tire_temperature_c < 0.0f ||
            car_.maximum_tire_temperature_c < car_.optimal_tire_temperature_c ||
            car_.maximum_tire_temperature_c <= 0.0f || car_.tire_heat_capacity_j_per_c <= 0.0f ||
            car_.tire_heat_generation_c_per_s < 0.0f || car_.tire_cooling_c_per_s < 0.0f ||
            car_.tire_wear_per_second < 0.0f || car_.aerodynamic_drag_coefficient < 0.0f ||
            car_.frontal_area_m2 <= 0.0f || car_.rolling_resistance < 0.0f ||
            time_step_s_ <= 0.0f || maximum_steps_ == 0) {
            throw std::invalid_argument("Car, tire, and simulation parameters must be physically valid.");
        }
        if (track_.half_width() <= car_.width_m * 0.5f) {
            throw std::invalid_argument("Track half-width must exceed half the car width.");
        }
        reset();
    }

    void reset() {
        const Vec2 start = track_.centerline().front();
        const TrackProjection projection = track_.project(start);
        state_ = {};
        state_.x = start.x;
        state_.y = start.y;
        state_.cg_z = car_.cg_height_m;
        state_.heading = projection.heading;
        state_.tire_temperature_c = car_.initial_tire_temperature_c;
        state_.tire_wear = car_.initial_tire_wear;
        stalled_steps_ = 0;
        update_tire_grip();
        previous_track_distance_ = projection.distance;
    }

    static DriverAction decode_action(int action_index) {
        if (action_index < 0 || static_cast<size_t>(action_index) >= action_count) {
            throw std::out_of_range("Action index is outside the racing action space.");
        }
        constexpr float steering_levels[] = {-1.0f, 0.0f, 1.0f};
        constexpr float pedal_levels[] = {-1.0f, 0.0f, 1.0f};
        return {steering_levels[action_index / 3], pedal_levels[action_index % 3]};
    }

    std::vector<float> observation() const {
        const TrackProjection projection = track_.project({state_.x, state_.y});
        const float heading_error = wrap_angle(state_.heading - projection.heading);
        const float half_lane = track_.half_width();
        const float temperature_range = std::max(car_.maximum_tire_temperature_c, 1.0f);
        const float normalized_curvature = std::clamp(projection.curvature * 20.0f, -1.0f, 1.0f);
        const float lateral_demand = std::abs(state_.lateral_acceleration_mps2) /
                                     std::max(state_.tire_grip_coefficient * gravity, 0.1f);
        return {
            std::clamp(state_.speed_mps / 50.0f, 0.0f, 1.5f),
            std::clamp(projection.lateral_offset / half_lane, -2.0f, 2.0f),
            std::sin(heading_error),
            std::cos(heading_error),
            normalized_curvature,
            std::clamp(state_.yaw_rate_rad_s / 2.0f, -2.0f, 2.0f),
            std::clamp(state_.tire_temperature_c / temperature_range, 0.0f, 2.0f),
            state_.tire_wear,
            std::clamp(state_.lap_progress_m / track_.length(), -1.0f, 1.0f),
            std::clamp(state_.tire_grip_coefficient / 2.0f, 0.0f, 1.0f),
            std::clamp(lateral_demand, 0.0f, 2.0f),
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
    }

    StepResult step(int action_index) {
        if (state_.steps >= maximum_steps_) {
            return {0.0f, true, false, 0.0f, 0.0f};
        }
        const DriverAction action = decode_action(action_index);
        const TrackProjection before = track_.project({state_.x, state_.y});
        const float heading_error = wrap_angle(state_.heading - before.heading);
        const float grip = std::max(state_.tire_grip_coefficient, 0.1f);
        const float lateral_limit = std::min(grip * gravity,
            gravity * car_.width_m * 0.5f / car_.cg_height_m);
        const float desired_yaw_rate = state_.speed_mps / car_.wheelbase_m *
            std::tan(action.steering * car_.maximum_steering_angle_rad);
        const float requested_lateral_acceleration = state_.speed_mps * desired_yaw_rate;
        const float utilization = std::abs(requested_lateral_acceleration) /
                                  std::max(lateral_limit, 0.1f);
        const float grip_scale = utilization > 1.0f ? 1.0f / utilization : 1.0f;
        state_.yaw_rate_rad_s = desired_yaw_rate * grip_scale;
        state_.lateral_acceleration_mps2 = state_.speed_mps * state_.yaw_rate_rad_s;

        const float air_density = 1.225f;
        const float drag_force = 0.5f * air_density * car_.aerodynamic_drag_coefficient *
                                 car_.frontal_area_m2 * state_.speed_mps * state_.speed_mps;
        const float rolling_force = car_.rolling_resistance * car_.mass_kg * gravity;
        float longitudinal_acceleration = action.pedal >= 0.0f
            ? action.pedal * car_.drive_force_n / car_.mass_kg
            : action.pedal * car_.brake_force_n / car_.mass_kg;
        longitudinal_acceleration -= (drag_force + rolling_force) / car_.mass_kg;
        state_.speed_mps = std::max(0.0f, state_.speed_mps + longitudinal_acceleration * time_step_s_);
        stalled_steps_ = state_.speed_mps < 0.1f ? stalled_steps_ + 1 : 0;
        state_.heading = wrap_angle(state_.heading + state_.yaw_rate_rad_s * time_step_s_);
        state_.x += state_.speed_mps * std::cos(state_.heading) * time_step_s_;
        state_.y += state_.speed_mps * std::sin(state_.heading) * time_step_s_;
        state_.cg_z = car_.cg_height_m;

        const TrackProjection after = track_.project({state_.x, state_.y});
        float progress_delta = after.distance - previous_track_distance_;
        if (progress_delta > track_.length() * 0.5f) progress_delta -= track_.length();
        if (progress_delta < -track_.length() * 0.5f) progress_delta += track_.length();
        state_.lap_progress_m += progress_delta;
        previous_track_distance_ = after.distance;

        const float slip_energy = std::max(0.0f, std::abs(requested_lateral_acceleration) -
            lateral_limit) * car_.mass_kg * state_.speed_mps * time_step_s_;
        const float generated_heat = car_.tire_heat_generation_c_per_s *
            (0.2f + std::min(utilization, 3.0f) * std::min(utilization, 3.0f)) * time_step_s_;
        const float energy_heat = slip_energy / std::max(car_.tire_heat_capacity_j_per_c, 1.0f);
        const float cooling = car_.tire_cooling_c_per_s *
            (state_.tire_temperature_c - car_.initial_tire_temperature_c) * time_step_s_;
        state_.tire_temperature_c = std::max(0.0f,
            state_.tire_temperature_c + generated_heat + energy_heat - cooling);
        state_.tire_wear = std::clamp(state_.tire_wear + car_.tire_wear_per_second *
            (0.2f + std::min(utilization, 2.0f)) * time_step_s_, 0.0f, 1.0f);
        update_tire_grip();
        state_.elapsed_time_s += time_step_s_;
        ++state_.steps;

        const float heading_error_after = wrap_angle(state_.heading - after.heading);
        const bool finished = state_.lap_progress_m >= track_.length();
        const bool off_track = std::abs(after.lateral_offset) >
                               track_.half_width() - car_.width_m * 0.5f;
        const bool rollover = std::abs(state_.lateral_acceleration_mps2) >
                              gravity * car_.width_m * 0.5f / car_.cg_height_m;
        const bool timeout = state_.steps >= maximum_steps_;
        const size_t stall_limit = std::max<size_t>(1, static_cast<size_t>(5.0f / time_step_s_));
        const bool stalled = stalled_steps_ >= stall_limit;
        const bool done = finished || off_track || rollover || timeout || stalled;

        float reward = progress_delta / std::max(time_step_s_ * 30.0f, 1.0f) - 0.01f;
        if (finished) reward += 100.0f;
        if (off_track || rollover) reward -= 100.0f;
        if (stalled) reward -= 10.0f;
        return {reward, done, finished, after.lateral_offset, heading_error_after};
    }

    const VehicleState& state() const noexcept { return state_; }
    const CarConfig& car_config() const noexcept { return car_; }
    const Track& track() const noexcept { return track_; }
    float time_step_s() const noexcept { return time_step_s_; }
    size_t maximum_steps() const noexcept { return maximum_steps_; }

private:
    void update_tire_grip() {
        const float tread_retention = 1.0f - 0.75f * state_.tire_wear;
        const float temperature_delta = std::max(0.0f,
            std::abs(state_.tire_temperature_c - car_.optimal_tire_temperature_c) - 20.0f);
        const float thermal_factor = std::clamp(1.0f - temperature_delta /
            std::max(car_.maximum_tire_temperature_c, 1.0f), 0.45f, 1.0f);
        state_.tire_tread_coefficient = car_.initial_tread_coefficient * tread_retention;
        state_.tire_grip_coefficient = track_.surface_friction() * car_.tire_material_friction *
            state_.tire_tread_coefficient * thermal_factor;
    }

    Track track_;
    CarConfig car_;
    float time_step_s_;
    size_t maximum_steps_;
    VehicleState state_{};
    float previous_track_distance_{};
    size_t stalled_steps_{};
};

} // namespace Racing
