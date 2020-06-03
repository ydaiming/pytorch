#include <torch/csrc/jit/passes/quantization/insert_quant_dequant.h>
#include <c10/core/QScheme.h>
#include <torch/csrc/jit/ir/subgraph_matcher.h>
#include <torch/csrc/jit/jit_log.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/graph_rewrite_helper.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/passes/quantization/helper.h>
#include <torch/csrc/jit/passes/subgraph_rewrite.h>

#include <stack>

namespace torch {
namespace jit {

namespace {
using graph_rewrite_helper::getFuncName;
using graph_rewrite_helper::getValue;
using graph_rewrite_helper::PatternInfo;

// dynamic quantization ops for activation: choose_qparams, quant, dequant
using DynamicQuantOps = std::tuple<Node*, Node*, Node*>;

c10::QScheme toAffine(c10::QScheme qscheme) {
  switch (qscheme) {
    case c10::kPerTensorAffine:
    case c10::kPerTensorSymmetric:
      return c10::kPerTensorAffine;
    case c10::kPerChannelAffine:
    case c10::kPerChannelSymmetric:
      return c10::kPerChannelAffine;
    default:
      return qscheme;
  }
}

bool isPerChannel(at::QScheme qscheme) {
  return qscheme == c10::kPerChannelAffine ||
      qscheme == c10::kPerChannelSymmetric;
}

// Go through the CallMethod graph to check if the value is Weight.
bool isWeight(Module& module, Value* v) {
  if (isWeight(v)) {
    return true;
  }
  c10::optional<bool> result;
  auto* self = v->owningGraph()->inputs()[0];
  for (const Use& u : v->uses()) {
    Node* n = u.user;
    if (n->kind() == prim::CallMethod) {
      auto m = getInvokedModule(module, n, self);
      auto g = m.get_method(n->s(attr::name)).graph();
      auto call_method_result = isWeight(m, g->inputs()[u.offset]);
      if (result.has_value()) {
        // Check to make sure all the CallMethods in the graph produce the same
        // output.
        TORCH_CHECK(
            call_method_result == result.value(),
            "Expected all CallMethods to use either weight "
            "or non-weight value.",
            v->debugName());
      } else {
        result = call_method_result;
      }
    }
  }
  return result.has_value() ? result.value() : false;
}

Node* insertChooseQParams(Graph* graph, Value* original_val) {
  std::string choose_qparams_func = "_choose_qparams_per_tensor";
  auto reduce_range = graph->insertConstant(false);
  // choose_qparams_per_tensor has 2 outputs, (scale, zero_point).
  Node* choose_qparams = graph->create(
      at::Symbol::aten(choose_qparams_func),
      {original_val, reduce_range},
      /* num_outputs = */ 2);
  choose_qparams->output(0)->setDebugName(original_val->debugName() + ".scale");
  choose_qparams->output(0)->setType(FloatType::get());
  choose_qparams->output(1)->setDebugName(
      original_val->debugName() + ".zero_point");
  choose_qparams->output(1)->setType(IntType::get());
  graph->insertNode(choose_qparams);
  return choose_qparams;
}

Node* insertQuant(
    Graph* graph,
    const std::vector<Value*>& inputs,
    NodeKind quant_kind,
    const std::string& debugName) {
  Node* quant = graph->create(quant_kind, inputs);
  quant->output()->setDebugName(debugName);
  graph->insertNode(quant);
  return quant;
}

Node* insertDeQuant(
    Graph* graph,
    Value* quantized_val,
    Value* original_val,
    size_t id = 0) {
  Node* dequant = graph->create(Symbol::aten("dequantize"), {quantized_val});
  dequant->output()
      ->setDebugName(
          original_val->debugName() + ".dequant." + c10::guts::to_string(id))
      ->setType(original_val->type());
  graph->insertNode(dequant);
  return dequant;
}

void insertDeQuantForAllUse(
    Graph* graph,
    Value* quantized_val,
    Value* original_val) {
  // copy uses to vector since value->uses() is a reference
  // and changing the graph will also change the uses() list
  const std::vector<Use> uses = original_val->uses();
  for (size_t i = 0; i < uses.size(); ++i) {
    auto* user = uses[i].user;
    // Insert dequantize node right before use node, because
    // we want to make sure use node and dequantize node reside
    // in the same block so that quant fusion can happen
    WithInsertPoint ins(user);
    Node* dequant = insertDeQuant(graph, quantized_val, original_val, i);
    user->replaceInput(uses[i].offset, dequant->output());
  }
}

Node* insertQParam(
    Graph* graph,
    Value* quantized_input,
    NodeKind node_kind,
    const TypePtr& output_type,
    const std::string& param_name) {
  Node* qparam = graph->create(node_kind, {quantized_input});
  qparam->output()
      ->setDebugName(quantized_input->debugName() + "." + param_name)
      ->setType(output_type);
  graph->insertNode(qparam);
  return qparam;
}

DynamicQuantOps insertChooseQParamQuantDequant(
    Graph* graph,
    Value* original_val,
    Value* dtype,
    NodeKind quant_kind) {
  Node* choose_qparams = insertChooseQParams(graph, original_val);
  std::vector<Value*> quant_inputs = {original_val};
  for (auto& out : choose_qparams->outputs()) {
    quant_inputs.push_back(out);
  }
  quant_inputs.push_back(dtype);
  Node* quant = insertQuant(
      graph, quant_inputs, quant_kind, original_val->debugName() + ".quant");
  Node* dequant = insertDeQuant(graph, quant->output(), original_val);
  return std::make_tuple(choose_qparams, quant, dequant);
}

void insertQuantizationOps(
    Module& module,
    Value* self,
    Node* observer,
    bool is_per_channel,
    const std::vector<std::string>& qparam_names,
    bool is_dynamic = false) {
  Graph* g = observer->owningGraph();
  // Observer output
  Value* observer_out = observer->output();
  // Inserting before insert point
  WithInsertPoint ins(observer_out->node()->next());

  std::string quantize_func;
  if (is_per_channel) {
    quantize_func = "quantize_per_channel";
  } else {
    quantize_func = "quantize_per_tensor";
  }
  Value* original_val = observer->input(1);
  Node *quant, *choose_qparams, *dequant;
  if (is_dynamic && !isWeight(module, observer_out)) {
    Value* dtype = g->insertGetAttr(self, qparam_names.back());
    std::tie(choose_qparams, quant, dequant) = insertChooseQParamQuantDequant(
        g, observer_out, dtype, at::Symbol::aten(quantize_func));
  } else {
    std::vector<Value*> inputs = {observer_out};
    // Insert GetAttr nodes for quantization parameters
    for (const auto& qparam_name : qparam_names) {
      inputs.push_back(g->insertGetAttr(self, qparam_name));
    }
    quant = insertQuant(
        g,
        inputs,
        at::Symbol::aten(quantize_func),
        original_val->debugName() + ".quant");
    dequant = insertDeQuant(g, quant->output(), original_val);
  }
  observer_out->replaceAllUsesWith(original_val);
  std::vector<Use> uses = original_val->uses();
  // TODO: use replaceAllUsesAfterNodeWith?
  for (const auto& use : uses) {
    auto* user = use.user;
    if (user != quant && user != observer && user != choose_qparams) {
      user->replaceInputWith(original_val, dequant->output());
    }
  }
}

// find the observer for Value `v` and return the name of the observer
c10::optional<std::string> findObserverName(Value* v) {
  // Note that here we just check for the name of observer, but the ideally
  // we should be comparing the type of observer, this is a temporary
  // work around until data only clone of module.clone is supported.
  Node* n = v->node();
  if (n->kind() == prim::CallMethod && n->s(attr::name) == "forward") {
    auto module_instance = n->inputs().at(0);
    if (module_instance->node()->kind() == prim::GetAttr &&
        module_instance->node()->s(attr::name).find("_observer_") !=
            std::string::npos) {
      return module_instance->node()->s(attr::name);
    }
  }
  return c10::nullopt;
}

void ReplicateChooseQParamsQuantDequant(std::shared_ptr<Graph>& graph) {
  const PatternInfo& dynamic_quant_pattern = PatternInfo::parse_from_str(R"(
    graph(%a, %reduce_range, %a_dtype):
        %a_scale : float, %a_zero_point : int = aten::_choose_qparams_per_tensor(%a, %reduce_range)
        %a_quant = aten::quantize_per_tensor(%a, %a_scale, %a_zero_point, %a_dtype)
        %a_dequant = aten::dequantize(%a_quant)
        return (%a_dequant) )");
  const Graph& dynamic_quant_graph = *dynamic_quant_pattern.pattern_graph;

  const auto& matches = findPatternMatches(dynamic_quant_graph, *graph);
  if (matches.size() == 0) {
    return;
  }

  const auto& vmap = dynamic_quant_pattern.vmap;
  Value* dequant_val = vmap.at("a_dequant");
  Node* pattern_dequant = dequant_val->node();
  Value* quant_val = vmap.at("a_quant");
  Node* pattern_quant = quant_val->node();
  Value* choose_qparam_val = vmap.at("a_scale");
  Node* pattern_choose_qparam = choose_qparam_val->node();

  std::vector<DynamicQuantOps> nodes_to_rewrite;
  std::vector<Node*> choose_qparam_nodes_to_rewrite;
  for (const Match& match : matches) {
    Node* matched_dequantize = match.nodes_map.at(pattern_dequant);
    Node* matched_quantize = match.nodes_map.at(pattern_quant);
    Node* matched_choose_qparam = match.nodes_map.at(pattern_choose_qparam);
    if (matched_dequantize->output()->uses().size() > 1) {
      nodes_to_rewrite.emplace_back(std::make_tuple(
          matched_choose_qparam, matched_quantize, matched_dequantize));
    }
  }
  for (const auto nodes : nodes_to_rewrite) {
    auto quant_node = std::get<1>(nodes);
    auto dequant_node = std::get<2>(nodes);
    // get input of quantize call.
    Value* original_val = quant_node->inputs()[0];
    Value* dequant_out = dequant_node->output();
    Value* dtype = quant_node->inputs()[3];
    std::vector<Use> uses = dequant_out->uses();
    for (const Use& use : uses) {
      auto* user = use.user;
      WithInsertPoint ins(user);
      auto quant_ops = insertChooseQParamQuantDequant(
          graph.get(), original_val, dtype, quant_node->kind());
      user->replaceInputWith(dequant_out, std::get<2>(quant_ops)->output());
    }
  }
  Node *choose_qparams, *quant, *dequant;
  for (const auto& n : nodes_to_rewrite) {
    std::tie(choose_qparams, quant, dequant) = n;
    dequant->removeAllInputs();
    quant->removeAllInputs();
    choose_qparams->removeAllInputs();
  }
  for (const auto& n : nodes_to_rewrite) {
    std::tie(choose_qparams, quant, dequant) = n;
    dequant->destroy();
    quant->destroy();
    choose_qparams->destroy();
  }
}

void RemoveRedundantDequantize(std::shared_ptr<Graph>& graph) {
  const std::string dequantize = R"(
    graph(%a_quant):
        %a_dequant = aten::dequantize(%a_quant)
        return (%a_dequant) )";
  const std::string dequantize_replacement = R"(
    graph(%a):
        return (%a) )";
  auto filter = [&](const Match& match,
                    const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto dequant_node = match_vmap.at(vmap.at("a_dequant"))->node();
    Value* dequant_out = dequant_node->output();
    TORCH_CHECK(
        dequant_out->uses().size() == 1,
        "Expect dequant output to have single use");
    Node* user = dequant_out->uses()[0].user;
    return isTensorInfoNode(user);
  };
  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(dequantize, dequantize_replacement);
  rewriter.runOnGraph(graph, filter);
}

void RemoveRedundantQuantizationOps(std::shared_ptr<Graph>& graph) {
  const std::string dynamic_quant_ops = R"(
    graph(%a, %reduce_range, %a_dtype):
        %a_scale : float, %a_zero_point : int = aten::_choose_qparams_per_tensor(%a, %reduce_range)
        %a_quant = aten::quantize_per_tensor(%a, %a_scale, %a_zero_point, %a_dtype)
        %a_dequant = aten::dequantize(%a_quant)
        return (%a_dequant) )";
  const std::string dynamic_quant_replacement = R"(
    graph(%a, %reduce_range, %a_dtype):
        return (%a) )";
  auto filter = [&](const Match& match,
                    const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto dequant_node = match_vmap.at(vmap.at("a_dequant"))->node();
    Value* dequant_out = dequant_node->output();
    TORCH_CHECK(
        dequant_out->uses().size() == 1,
        "Expect dequant output to have single use");
    Node* user = dequant_out->uses()[0].user;
    return !nodeQuantizable(user, /* is_dynamic */ true);
  };
  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(dynamic_quant_ops, dynamic_quant_replacement);
  rewriter.runOnGraph(graph, filter);
}

void checkCalculateQParamsResult(const IValue& qparams) {
  TORCH_CHECK(
      qparams.isTuple(),
      "`calculate_qparams` function is expected to return a "
      "Tuple, but got:",
      qparams.tagKind());
  auto tp = qparams.toTuple();
  TORCH_CHECK(
      tp->elements().size() == 2,
      "`calculate_qparams` function is expected to return a "
      "Tuple of size 2, got Tuple of size ",
      tp->elements().size());
  // Expect first two elements of the tuple to be Tensor
  for (size_t i = 0; i < 2; ++i) {
    TORCH_CHECK(
        tp->elements()[i].isTensor(),
        "Element of Tuple is expected to be Tensor, but element ",
        i,
        " has type: ",
        tp->elements()[i].tagKind());
  }
}

class InsertQuantDeQuantHelper {
 public:
  void run(Module& module, const std::string& method_name);

  void setDynamicFlag(bool is_dynamic) {
    is_dynamic_ = is_dynamic;
  }
  // Cleanup observer nodes from graph and observer modules
  // from module object and ClassType
  void cleanup(Module& module);

  // In order to propagate quantization ops through the ops that doesn't
  // require observation, we'll first inline the graph, and call the
  // PropgateQuantizationOps pass
  void propagateQuantizationOps(Module& module);

  // Used for dynamic quantization to selectively run the weight observers.
  // It extracts the subgraph corresponding to the weight and runs it with
  // the module instance.
  void runWeightObserver(Module& module, const std::string& method_name);

 private:
  ModuleMethodVector getInvokedMethods(
      Module& module,
      const std::string& method_name);

  // Get quantization parameter map of the given Value in Graph
  // by searching for observer module of the value and extract the
  // quantization parameters from the observer module
  std::tuple<c10::QScheme, QParamVector> getQSchemeAndQParamVector(
      script::Module& module,
      Node* n);
  void checkQScheme(Graph* g, c10::QScheme qscheme) {
    if (qscheme_for_graph_.count(g)) {
      TORCH_CHECK(
          qscheme_for_graph_.at(g) == qscheme ||

              "Quantizing same graph with different types of "
              "QSchemes is not supported.\n",
          " Expecting:",
          c10::toString(qscheme_for_graph_.at(g)),
          " Got:",
          c10::toString(qscheme));
    } else {
      qscheme_for_graph_[g] = toAffine(qscheme);
    }
  }

  void collectObserverNodesAndValueToQuantize(Module& module, Value*);
  void cleanup(Module& module, Graph* g);
  void quantizeTensors(Module& module, Graph* g, Value* self);

  void extractAndRunWeightObserver(Module& module, Value* self, Node* observer);
  void cloneNodeInGraph(Node* node, std::shared_ptr<Graph>& g);
  Value* updateInputValueInGraph(Value* v, std::shared_ptr<Graph>& g);
  void buildObserverSubgraph(
      const std::vector<Node*>& src,
      std::shared_ptr<Graph> dest);

  std::unordered_map<Graph*, std::vector<std::string>>
      observer_modules_to_remove_;
  // We only remove observer module attributes from type in the
  // first encounter of the graph, after that since the attributes
  // is already removed from the ClassType, we'll use the list of slot index to
  // replay this removal
  std::unordered_map<Graph*, std::vector<int>> removed_observer_slots_;
  std::unordered_map<Graph*, std::vector<Node*>> nodes_to_destroy_;
  // Map from Graph to observer node, we can use observer node to
  // get the information of original value that's been observed and
  // the quantization parameters
  std::unordered_map<Graph*, std::vector<Node*>> observer_nodes_for_graph_;
  // A map from qparam name (e.g. _scale) to the attribute name in
  // the module(e.g. weight_scale_0)
  std::unordered_map<Node*, std::unordered_map<std::string, std::string>>
      qparam_name_map_for_node_;
  // Record qscheme for every graph, this is for checking
  // each graph is only quantized with one type of QScheme
  std::unordered_map<Graph*, c10::QScheme> qscheme_for_graph_;

  bool is_dynamic_ = false;

  // Map from values in original graph to corresponding values in the extracted
  // subgraph.
  std::unordered_map<Value*, Value*> remap_values_in_observer_subgraph_;
  // Map from original graph to list of nodes corresponding to the subgraph.
  std::unordered_map<Graph*, std::vector<Node*>> graph_to_subgraph_nodes_;
};

void InsertQuantDeQuantHelper::collectObserverNodesAndValueToQuantize(
    Module& module,
    Value* v) {
  auto* g = v->owningGraph();
  auto observer_name = findObserverName(v);
  if (!observer_name) {
    return;
  }
  observer_modules_to_remove_[g].push_back(observer_name.value());

  Node* observer = v->node();
  TORCH_INTERNAL_ASSERT(
      observer->kind() == prim::CallMethod &&
      observer->s(attr::name) == "forward" &&
      observer->inputs()[0]->node()->kind() == prim::GetAttr &&
      observer->inputs()[0]->node()->s(attr::name) == observer_name);

  // Observer forward call node
  nodes_to_destroy_[g].push_back(observer);
  // GetAttr node for observer module
  nodes_to_destroy_[g].push_back(observer->inputs()[0]->node());
  observer_nodes_for_graph_[g].push_back(observer);
}

void InsertQuantDeQuantHelper::cleanup(Module& module) {
  for (auto& method : module.get_methods()) {
    cleanup(module, method.graph().get());
  }
  for (Module m : module.children()) {
    cleanup(m);
  }
}

void InsertQuantDeQuantHelper::cleanup(Module& module, Graph* g) {
  GRAPH_DUMP("Before Remove Observers:", g);
  if (nodes_to_destroy_.count(g)) {
    for (auto& n : nodes_to_destroy_.at(g)) {
      n->removeAllInputs();
    }
    for (auto& n : nodes_to_destroy_.at(g)) {
      n->destroy();
    }
    nodes_to_destroy_.at(g).clear();
  }

  // 1. If we have seen this graph before, this means the observer
  // attributes has been removed from the type(see step 2) but the slot
  // index of these attributes are kept in the list, we'll replay the observer
  // slots removal using these slot indexes
  if (removed_observer_slots_.count(g)) {
    for (auto slot : removed_observer_slots_.at(g)) {
      module._ivalue()->unsafeRemoveSlot(slot);
    }
  }

  // 2. Remove observer modules from last one to first one in order to
  // reduce the time complexity, assuming all the observer modules
  // are added after the existing modules, we'll have complexity of
  // O(N) where N is number of observer modules with this optimization
  if (observer_modules_to_remove_.count(g)) {
    auto& observers = observer_modules_to_remove_.at(g);
    for (int64_t i = observers.size() - 1; i >= 0; --i) {
      auto observer_name = observers[i];
      GRAPH_DEBUG("Trying to remove: ", observer_name);
      if (module.type()->hasAttribute(observer_name)) {
        // We record the slot index here in order to replay the
        // slot removal in other objects that's sharing the ClassType
        // since we're going to remove attribute in the ClassType here
        removed_observer_slots_[g].push_back(
            module.type()->getAttributeSlot(observer_name));
        module._ivalue()->unsafeRemoveAttr(observer_name);
        module.type()->unsafeRemoveAttribute(observer_name);
      }
    }
    observers.clear();
  }
  GRAPH_DUMP("After remove observers :", g);
}

Value* InsertQuantDeQuantHelper::updateInputValueInGraph(
    Value* v,
    std::shared_ptr<Graph>& g) {
  if (remap_values_in_observer_subgraph_.count(v) == 0) {
    auto new_value = g->block()->addInput();
    remap_values_in_observer_subgraph_[v] = new_value;
    new_value->copyMetadata(v);
    return new_value;
  } else {
    return remap_values_in_observer_subgraph_[v];
  }
}

void InsertQuantDeQuantHelper::cloneNodeInGraph(
    Node* node,
    std::shared_ptr<Graph>& g) {
  auto* block = g->block();
  auto value_fn = [&](Value* v) { return updateInputValueInGraph(v, g); };

  auto new_node = block->appendNode(g->createClone(node, value_fn));
  for (size_t i = 0; i < node->outputs().size(); ++i) {
    auto oo = node->outputs()[i];
    auto no = new_node->outputs()[i];
    remap_values_in_observer_subgraph_[oo] = no;
  }
}
void InsertQuantDeQuantHelper::buildObserverSubgraph(
    const std::vector<Node*>& weight_subgraph,
    std::shared_ptr<Graph> dest_graph) {
  // Build weight subgraph
  for (auto n : weight_subgraph) {
    cloneNodeInGraph(n, dest_graph);
  }
  LintGraph(dest_graph);
  // Add last node output value as subgraph output.
  dest_graph->registerOutput(
      updateInputValueInGraph(weight_subgraph.back()->output(0), dest_graph));
  GRAPH_DUMP("New weight observer subgraph: ", dest_graph);
}

void updateCurrInputSet(
    const std::vector<Value*>& inputs,
    std::unordered_set<Value*>& curr_graph_inputs) {
  for (auto& v : inputs) {
    curr_graph_inputs.insert(v);
  }
}

void InsertQuantDeQuantHelper::extractAndRunWeightObserver(
    Module& module,
    Value* self,
    Node* observer) {
  Value* observed_weight = observer->output();
  TORCH_CHECK(
      isWeight(module, observed_weight),
      "extractAndRunWeightObserver can only be called on weight tensors");
  Graph* g = observer->owningGraph();
  std::vector<Node*> weight_subgraph;
  remap_values_in_observer_subgraph_.clear();

  // If the graph was already visited, return list of relevant nodes directly.
  // Multiple module instances can share the same graph code, so we don't need
  // to re-run the extraction process.
  if (graph_to_subgraph_nodes_.count(g)) {
    weight_subgraph = graph_to_subgraph_nodes_[g];
  } else {
    // Extract the subgraph nodes.
    weight_subgraph.push_back(observer);
    Node* n = observer;
    // Track values in the subgraph for which producer needs to be found.
    std::unordered_set<Value*> subgraph_node_inputs;
    updateCurrInputSet(observer->inputs().vec(), subgraph_node_inputs);
    while (n && n != self->node()) {
      n = n->prev();
      // Check if prev node's outputs exists in value list.
      for (auto& o : n->outputs()) {
        if (subgraph_node_inputs.find(o) != subgraph_node_inputs.end()) {
          // Don't push top level node in the subgraph, since it is not needed
          // to run the subgraph
          if (n->output(0) != self) {
            weight_subgraph.push_back(n);
          }
          subgraph_node_inputs.erase(o);
          updateCurrInputSet(n->inputs().vec(), subgraph_node_inputs);
        }
      }
    }
    TORCH_CHECK(
        subgraph_node_inputs.empty(),
        "DynamicQuant: Didn't process all input values in weight observer subgraph");

    // Reverse to traverse subgraph in correct direction
    std::reverse(weight_subgraph.begin(), weight_subgraph.end());
    graph_to_subgraph_nodes_[g] = weight_subgraph;
  }

  auto observer_subgraph = std::make_shared<Graph>();
  auto build_observer_graph = [&](Function& func) {
    buildObserverSubgraph(weight_subgraph, func.graph());
  };
  auto func = torch::make_unique<GraphFunction>(
      "observer_subgraph", observer_subgraph, build_observer_graph);

  Stack module_inp = {module._ivalue()};
  func->run(module_inp);
}

void InsertQuantDeQuantHelper::quantizeTensors(
    Module& module,
    Graph* g,
    Value* self) {
  if (!observer_nodes_for_graph_.count(g)) {
    return;
  }
  for (auto* n : observer_nodes_for_graph_.at(g)) {
    auto* original_value = n->input(1);
    auto tp = getQSchemeAndQParamVector(module, n);
    auto qscheme = std::get<0>(tp);
    auto qparam_map = std::get<1>(tp);
    checkQScheme(g, qscheme);
    std::vector<std::string> qparam_names;
    for (auto& pr : qparam_map) {
      const auto& name = pr.first;
      const auto& qparam = pr.second;
      size_t uid = 0;
      auto qparam_name =
          original_value->debugName() + name + "_" + c10::to_string(uid++);
      while (module.hasattr(qparam_name)) {
        qparam_name =
            original_value->debugName() + name + "_" + c10::to_string(uid++);
      }
      qparam_name_map_for_node_[n][name] = qparam_name;
      module.register_attribute(qparam_name, qparam.type(), qparam);
      qparam_names.push_back(qparam_name);
    }
    insertQuantizationOps(
        module, self, n, isPerChannel(qscheme), qparam_names, is_dynamic_);
  }
}

std::tuple<c10::QScheme, QParamVector> InsertQuantDeQuantHelper::
    getQSchemeAndQParamVector(script::Module& module, Node* n) {
  // TODO: refactor findObserverName to take Node* as input
  Value* v = n->output();
  TORCH_INTERNAL_ASSERT(
      v->type()->isSubtypeOf(TensorType::get()),
      "Expected output of observer node to be Tensor");
  auto observer_name = findObserverName(v);
  TORCH_INTERNAL_ASSERT(
      observer_name,
      "getQSchemeAndParamMap expects the corresponding observer for ",
      v->debugName(),
      " exists.");
  auto observer_module = module.attr(observer_name.value()).toModule();
  auto calculate_qparams = observer_module.get_method("calculate_qparams");
  IValue result = calculate_qparams(std::vector<IValue>());
  checkCalculateQParamsResult(result);
  auto scalar_type = observer_module.attr("dtype");
  TORCH_CHECK(
      scalar_type.toScalarType() != at::ScalarType::Undefined,
      "dtype of observer can't be undefined");
  auto tp = result.toTuple();
  at::Tensor scale = tp->elements()[0].toTensor().to(at::kFloat);
  at::Tensor zero_point = tp->elements()[1].toTensor().to(at::kInt);
  // quantization parameters should appear in the same order as
  // the argument for quantize_per_tensor/quantize_per_channel function
  QParamVector qparams;
  auto qscheme = observer_module.attr("qscheme").toQScheme();
  if (isPerChannel(qscheme)) {
    auto axis = observer_module.attr("ch_axis");
    qparams.push_back(std::make_pair("_scale", scale));
    qparams.push_back(std::make_pair("_zero_point", zero_point));
    qparams.push_back(std::make_pair("_axis", axis.toInt()));
  } else {
    qparams.push_back(std::make_pair("_scale", scale.item<double>()));
    qparams.push_back(
        std::make_pair("_zero_point", zero_point.item<int64_t>()));
  }
  qparams.push_back(std::make_pair("_scalar_type", scalar_type));
  return std::make_tuple(qscheme, qparams);
}

ModuleMethodVector InsertQuantDeQuantHelper::getInvokedMethods(
    Module& module,
    const std::string& method_name) {
  auto graph = module.get_method(method_name).graph();

  ModuleMethodVector invoked_methods;
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      if (n->kind() == prim::CallMethod) {
        auto module_instance = n->inputs()[0];
        auto module_method_name = n->s(attr::name);
        c10::optional<Module> m;
        // calling method on self
        if (module_instance == graph->inputs()[0]) {
          m = module;
        } else if (
            module_instance->node()->kind() == prim::GetAttr &&
            module_instance->node()->s(attr::name).find("_observer_") ==
                std::string::npos) {
          m = getInvokedModule(module, n, graph->inputs()[0]);
        }
        if (m) {
          invoked_methods.push_back({*m, module_method_name});
        }
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  return invoked_methods;
}

void propagateQParams(
    Value* original_output,
    const std::vector<Value*>& inputs,
    const c10::optional<std::tuple<c10::QScheme, QParamVector>>& qparams_opt =
        c10::nullopt) {
  Node* n = original_output->node();
  Graph* graph = n->owningGraph();
  // for ops like average pool, we'll insert quant dequant after the op
  // We'll assume the tensor is a PerTensorAffine quantized Tensor for
  // now, and may generalize later if this becomes an issue
  TORCH_INTERNAL_ASSERT(
      inputs.size() == 1, "Expecting single input for the aten function");
  // input of the dequantize node
  Value* quantized_input = inputs[0]->node()->input(0);
  // insert ops after the general op
  WithInsertPoint ins(n->next());
  std::vector<Value*> quant_inputs;
  auto quant_kind = Symbol::aten("quantize_per_tensor");
  if (qparams_opt.has_value()) {
    quant_inputs = {original_output};
    auto qscheme = std::get<0>(*qparams_opt);
    auto qparams = std::get<1>(*qparams_opt);
    if (isPerChannel(qscheme)) {
      quant_kind = Symbol::aten("quantize_per_channel");
    }
    for (const auto& qparam : qparams) {
      Value* qparam_val = graph->insertConstant(qparam.second);
      qparam_val->setDebugName(quantized_input->debugName() + qparam.first);
      quant_inputs.push_back(qparam_val);
    }
  } else {
    // Only per tensor affine quantized tensor is supported in this case
    // get quantization parameters from previous quantized op
    Node* scale = insertQParam(
        graph,
        quantized_input,
        at::Symbol::aten("q_scale"),
        FloatType::get(),
        "q_scale");
    Node* zero_point = insertQParam(
        graph,
        quantized_input,
        at::Symbol::aten("q_zero_point"),
        IntType::get(),
        "q_zero_point");
    Node* dtype = insertQParam(
        graph, quantized_input, prim::dtype, IntType::get(), "dtype");
    quant_inputs = {original_output,
                    scale->output(),
                    zero_point->output(),
                    dtype->output()};
  }
  Node* quant = insertQuant(
      graph, quant_inputs, quant_kind, original_output->debugName() + ".quant");
  Value* quantized_output = quant->output();
  // replace uses of original output of the general op with quantized
  // output
  original_output->replaceAllUsesAfterNodeWith(quant, quantized_output);
  insertDeQuantForAllUse(graph, quantized_output, quantized_output);
}

void propagateDequantize(Value* output, const std::vector<Value*> inputs) {
  Node* n = output->node();
  Graph* graph = n->owningGraph();
  // Delete dequantize node, we have one dequantize
  // for each use of the value
  for (auto* dequantized_val : inputs) {
    auto* dequantize_node = dequantized_val->node();
    TORCH_INTERNAL_ASSERT(
        dequantized_val->uses().size() == 1,
        "Expect to have one dequantize node for each use");
    // Replace useses of dequantized_val with the input of
    // dequantize node
    dequantized_val->replaceAllUsesWith(dequantize_node->inputs()[0]);
    dequantize_node->removeAllInputs();
    dequantize_node->destroy();
  }
  insertDeQuantForAllUse(graph, output, output);
}

void propagateQuantizationOps(Block* block) {
  for (Node* n : block->nodes()) {
    if (n->kind() == prim::If) {
      for (Block* subblock : n->blocks()) {
        propagateQuantizationOps(subblock);
      }
      if (n->outputs().size() == 0) {
        continue;
      }
      if (n->outputs().size() > 1) {
        // Factoring out dequantize for if blocks with multiple outputs
        // is not supported right now
        continue;
      }
    }
    for (auto* output : n->outputs()) {
      auto inputs = getPassThroughInputs(output);
      if (inputs.size() > 0) {
        // note that we don't need to recursively check for prim::If
        // here because if all inputs of a prim::If is dequantized
        // the dequantize will be factored out before we get to this
        // point
        bool is_dequantized = true;
        for (auto* input : inputs) {
          is_dequantized &= input->node()->kind() == Symbol::aten("dequantize");
        }
        if (!is_dequantized) {
          continue;
        }
        if (isSingleInputGeneralValueAtenFunction(n)) {
          propagateQParams(output, inputs);
        } else if (auto qparams_opt = getFixedQParams(n)) {
          propagateQParams(output, inputs, qparams_opt);
        } else {
          propagateDequantize(output, inputs);
        }
      }
    }
  }
}

void InsertQuantDeQuantHelper::runWeightObserver(
    Module& module,
    const std::string& method_name) {
  for (auto& invoked_methods : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_methods);
    const auto& invoked_method_name = std::get<1>(invoked_methods);
    runWeightObserver(invoked_module, invoked_method_name);
  }
  Method method = module.get_method(method_name);
  auto graph = method.graph();
  Value* self = graph->inputs()[0];

  std::vector<Value*> weight_values;
  for (size_t idx = 1; idx < method.num_inputs(); ++idx) {
    auto& v = graph->inputs()[idx];
    if (v->type()->isSubtypeOf(TensorType::get())) {
      auto observer_name = findObserverName(v);
      // Ensure that we only track values that are the output of observer node.
      // isWeight returns true if the observed value is a weight of quantized
      // op.
      if (observer_name && isWeight(module, v)) {
        weight_values.push_back(v);
      }
    }
  }
  // Visit all blocks in the current graph to find weight values.
  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end;) {
      Node* n = *it++;
      for (Value* v : n->outputs()) {
        if (!v->type()->isSubtypeOf(TensorType::get())) {
          continue;
        }
        auto observer_name = findObserverName(v);
        if (observer_name && isWeight(module, v)) {
          weight_values.push_back(v);
        }
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  // For all the observed weight values, find the corresponding subgraph that
  // contributes to the weight tensor, and run that subgraph to observe the
  // weight.
  for (const auto& v : weight_values) {
    extractAndRunWeightObserver(module, self, v->node());
  }
}

void InsertQuantDeQuantHelper::run(
    Module& module,
    const std::string& method_name) {
  for (auto& invoked_methods : getInvokedMethods(module, method_name)) {
    auto& invoked_module = std::get<0>(invoked_methods);
    const auto& invoked_method_name = std::get<1>(invoked_methods);
    run(invoked_module, invoked_method_name);
  }

  Method method = module.get_method(method_name);
  auto graph = method.graph();

  // We only need to register new parameters if the graph has
  // been quantized before
  // TODO: dedup this part with code in quantizeTensors
  if (observer_nodes_for_graph_.count(graph.get())) {
    for (auto* n : observer_nodes_for_graph_.at(graph.get())) {
      auto tp = getQSchemeAndQParamVector(module, n);
      checkQScheme(graph.get(), std::get<0>(tp));
      auto qparam_map = std::get<1>(tp);
      TORCH_INTERNAL_ASSERT(
          qparam_name_map_for_node_.count(n),
          "Expected to have a qparam_name_map for node:",
          *n);
      auto qparam_name_map = qparam_name_map_for_node_.at(n);
      for (auto& pr : qparam_map) {
        const auto& name = pr.first;
        const auto& qparam = pr.second;
        module._ivalue()->setAttr(qparam_name_map.at(name), qparam);
      }
    }
    return;
  }

  // prim::Param nodes do not belong to the graph. Hence the Insert
  // point is the beginning of graph node. This also safe guards against
  // observing a potentially mutated value due to some in-place operation
  std::vector<Value*> input_values;
  for (size_t idx = 1; idx < method.num_inputs(); ++idx) {
    auto& v = graph->inputs()[idx];
    if (v->type()->isSubtypeOf(TensorType::get())) {
      input_values.push_back(v);
    }
  }

  std::stack<Block*> blocks_to_visit;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end;) {
      Node* n = *it++;
      for (Value* v : n->outputs()) {
        if (!v->type()->isSubtypeOf(TensorType::get())) {
          continue;
        }
        collectObserverNodesAndValueToQuantize(module, v);
      }

      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }

  for (Value* v : input_values) {
    collectObserverNodesAndValueToQuantize(module, v);
  }
  GRAPH_DUMP("Before Quantize Tensors:", graph);
  Value* self = graph->inputs()[0];
  quantizeTensors(module, graph.get(), self);
  GRAPH_DUMP("After Quantize Tensors:", graph);
}

void InsertQuantDeQuantHelper::propagateQuantizationOps(Module& module) {
  SwapFunctionalLinear(module);
  auto graph = module.get_method("forward").graph();
  Inline(*graph);
  ConstantPropagation(graph);
  ReplicateChooseQParamsQuantDequant(graph);
  RemoveRedundantQuantizationOps(graph);
  ReplicateQuant(graph);
  ReplicateDeQuant(graph);
  RemoveRedundantDequantize(graph);
  PropagateQuantizationOps(graph);
}

} // namespace

void SwapFunctionalLinear(Module& module) {
  for (auto& method : module.get_methods()) {
    std::shared_ptr<Graph> g = method.graph();
    SwapFunctionalLinear(g);
  }
  for (Module m : module.children()) {
    SwapFunctionalLinear(m);
  }
}

void SwapFunctionalLinear(std::shared_ptr<Graph>& graph) {
  std::string functional_linear = R"(
graph(%linear, %input, %weight, %bias):
  %r = prim::CallFunction(%linear, %input, %weight, %bias)
  return (%r) )";
  std::string aten_linear = R"(
graph(%linear, %input, %weight, %bias):
  %r = aten::linear(%input, %weight, %bias)
  return (%r) )";
  auto filter = [](const Match& match,
                   const std::unordered_map<std::string, Value*>& vmap) {
    const auto& match_vmap = match.values_map;
    auto linear = getValue("linear", match_vmap, vmap);
    auto func_name = getFuncName(linear);
    return func_name == "linear";
  };
  SubgraphRewriter rewriter;
  rewriter.RegisterRewritePattern(functional_linear, aten_linear);
  // TODO: runOnGraph takes const ref?
  rewriter.runOnGraph(graph, filter);
}

void ReplicateQuant(std::shared_ptr<Graph>& graph) {
  std::stack<Block*> blocks_to_visit;
  std::vector<Node*> quant_nodes_to_rewrite;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      // find quantize node that quantizes the output of if
      if ((n->kind() == Symbol::aten("quantize_per_tensor") ||
           n->kind() == Symbol::aten("quantize_per_channel")) &&
          n->input(0)->node()->kind() == prim::If) {
        quant_nodes_to_rewrite.push_back(n);
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  for (Node* n : quant_nodes_to_rewrite) {
    Node* if_node = n->input(0)->node();
    // move the nodes that produces the quantization parameters before
    // prim::If
    for (auto i = 1; i < n->inputs().size(); ++i) {
      n->input(i)->node()->moveBefore(if_node);
    }
    // replace all uses of the quantized node with the output of if node
    n->output()->replaceAllUsesWith(if_node->output());
    // add quantize nodes to the end of all blocks
    for (Block* if_block : if_node->blocks()) {
      TORCH_CHECK(
          if_block->outputs().size() == 1,
          "replicate quantize only works for `if` node with one output right now");
      // the original return value of the block
      Value* ret_val = if_block->outputs()[0];
      std::vector<Value*> quantize_inputs = n->inputs().vec();
      quantize_inputs[0] = ret_val;
      WithInsertPoint ins(if_block->return_node());
      Node* quant = graph->create(n->kind(), quantize_inputs);
      if_block->replaceOutput(0, quant->output());
      quant->output()->copyMetadata(ret_val);
      graph->insertNode(quant);
    }
  }

  for (Node* n : quant_nodes_to_rewrite) {
    n->removeAllInputs();
  }
  for (Node* n : quant_nodes_to_rewrite) {
    n->destroy();
  }
}

void ReplicateDeQuant(std::shared_ptr<Graph>& graph) {
  std::stack<Block*> blocks_to_visit;
  std::vector<Node*> dequant_nodes_to_rewrite;
  blocks_to_visit.push(graph->block());
  while (!blocks_to_visit.empty()) {
    Block* b = blocks_to_visit.top();
    blocks_to_visit.pop();
    for (Node* n : b->nodes()) {
      if (n->kind() == Symbol::aten("dequantize") &&
          n->output()->uses().size() > 1) {
        dequant_nodes_to_rewrite.push_back(n);
      }
      for (Block* subblock : n->blocks()) {
        blocks_to_visit.push(subblock);
      }
    }
  }
  for (Node* n : dequant_nodes_to_rewrite) {
    auto* quantized_val = n->input(0);
    auto* dequantized_val = n->output();
    insertDeQuantForAllUse(graph.get(), quantized_val, dequantized_val);
  }

  for (Node* n : dequant_nodes_to_rewrite) {
    n->removeAllInputs();
  }

  for (Node* n : dequant_nodes_to_rewrite) {
    n->destroy();
  }
}

// This is the pass to handle ops that does not require observation
// for example: flatten, average_pool, upsample
// This is called after inline and before graph execution
void PropagateQuantizationOps(std::shared_ptr<Graph>& graph) {
  propagateQuantizationOps(graph->block());
}

Module InsertQuantDeQuant(
    Module& input_module,
    const std::string& method_name,
    bool inplace,
    bool is_dynamic) {
  Module module = input_module.clone(inplace);
  InsertQuantDeQuantHelper h;
  h.setDynamicFlag(is_dynamic);
  if (is_dynamic) {
    h.runWeightObserver(module, method_name);
  }
  h.run(module, method_name);
  h.cleanup(module);
  h.propagateQuantizationOps(module);
  return module;
}

} // namespace jit
} // namespace torch
