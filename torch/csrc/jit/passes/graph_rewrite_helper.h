#pragma once

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>

namespace torch {
namespace jit {
namespace graph_rewrite_helper {

std::string getFuncName(Value* func_value);
Value* getValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap);
c10::optional<IValue> getIValue(
    const std::string& name,
    const std::unordered_map<const Value*, Value*>& match_vmap,
    const std::unordered_map<std::string, Value*>& vmap);
void replaceConvolutionWithAtenConv(std::shared_ptr<Graph>& graph);

using MatchFilter = std::function<
  bool(const Match&, const std::unordered_map<std::string, Value*>&)>;

// This struct contains a compiled IR patterns slated for use in the
// findPatternMatches function. The struct encapsulates the common
// information from parseIR that is used in conjunction with the
// pattern matching facility. A const instance of this struct can
// also be stored away to cache the compiled IR pattern and reduce
// runtime cost
struct PatternInfo {
  std::string pattern_string;
  std::unique_ptr<Graph> pattern_graph;
  std::unordered_map<std::string, Value*> vmap;
  std::vector<MatchFilter> filters;

  static PatternInfo parse_from_str(std::string pattern_string, const std::vector<MatchFilter>& filters = {}) {
    PatternInfo rv{
      std::move(pattern_string), std::make_unique<Graph>(), decltype(vmap){}, filters};
    parseIR(rv.pattern_string, rv.pattern_graph.get(), rv.vmap);
    return rv;
  }
};

auto aten_add_alpha_is_one = [](const Match& match,
                                const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto alpha = toIValue(match_vmap.at(vmap.at("alpha")));
    return alpha && alpha->isInt() && alpha->toInt() == 1;
};

auto is_functional_relu = [](const Match& match,
                             const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    Value* relu = match_vmap.at(vmap.at("relu"));
    return relu->type()->cast<FunctionType>() && getFuncName(relu) == "relu";
};

} // namespace graph_rewrite_helper
} // namespace jit
} // namespace torch
