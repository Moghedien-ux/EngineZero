#pragma once

#include "Tensor.hpp"
#include <memory>
#include <functional>
#include <vector>
#include <string>

class ValueNode;
using ValuePtr = std::shared_ptr<ValueNode>;

class ValueNode : public std::enable_shared_from_this<ValueNode> {
public:
    Tensor data;
    Tensor grad;

    std::vector<ValuePtr> parents;
    std::function<void()> backward_fn;
    std::string op_name;

    explicit ValueNode(Tensor data_tensor, std::vector<ValuePtr> parents = {}, std::string op = "");

    static ValuePtr make(Tensor data, std::vector<ValuePtr> parents = {}, std::string op = "");

    void backward();

    void zero_grad();
};