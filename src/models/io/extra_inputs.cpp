#include "generator/generators.h"
#include "models/model.h"
#include "models/io/extra_inputs.h"

namespace Generators {

PresetExtraInputs::PresetExtraInputs(State& state)
    : state_(state),
      registry_{
          {"num_logits_to_keep", [&state = state_]() -> std::unique_ptr<OrtValue> {
             std::vector<int64_t> shape{1};
             auto num_logits_to_keep = OrtValue::CreateTensor<int64_t>(state.model_.allocator_cpu_, shape);
             *num_logits_to_keep->GetTensorMutableData<int64_t>() = 0;
             return num_logits_to_keep;
           }}} {}

void PresetExtraInputs::Add() {
  const auto input_names_vector = state_.model_.session_info_.GetInputNames();
  const std::unordered_set<std::string> input_names(state_.input_names_.begin(), state_.input_names_.end());
  std::vector<std::string> unclaimed_input_names;
  // Add any model input for which we don't have a corresponding input in the state to the unclaimed_input_names
  for (const auto& input_name : input_names_vector) {
    if (input_names.find(input_name) == input_names.end()) {
      unclaimed_input_names.push_back(input_name);
    }
  }

  // Try to claim the unclaimed inputs from the registry
  for (const auto& input_name : unclaimed_input_names) {
    auto it = registry_.find(input_name);
    if (it != registry_.end()) {
      extra_input_names_.push_back(input_name);
      extra_inputs_.push_back(it->second());
      state_.input_names_.push_back(extra_input_names_.back().c_str());
      state_.inputs_.push_back(extra_inputs_.back().get());
    } else if (input_name.rfind("onnx::Neg_", 0) == 0) {
      // The unclaimed input has a prefix of onnx::Neg_, which is a special case
      // We treat this as an alias to num_logits_to_keep
      extra_input_names_.push_back(input_name);
      extra_inputs_.push_back(registry_.at("num_logits_to_keep")());
      state_.input_names_.push_back(extra_input_names_.back().c_str());
      state_.inputs_.push_back(extra_inputs_.back().get());
    }
  }
}

ExtraInputs::ExtraInputs(State& state)
    : state_{state} {}

void ExtraInputs::Add(const std::vector<ExtraInput>& extra_inputs, const std::vector<std::string>& required_input_names) {
  std::unordered_set<std::string> required_input_names_set(required_input_names.begin(), required_input_names.end());
  // Add extra user inputs. Iterate in order so that a repeated name (e.g. two OgaGenerator_SetModelInput
  // calls for the same input) binds the most recently supplied tensor.
  for (size_t i = 0; i < extra_inputs.size(); i++) {
    if (!required_input_names_set.empty() && !required_input_names_set.count(extra_inputs[i].name)) {
      continue;
    }

    auto it = name_to_index_.find(extra_inputs[i].name);
    if (it != name_to_index_.end()) {
      // Already bound: rebind the existing slot in place instead of appending a new one, so a second
      // SetExtraInputs call (e.g. a later conversation turn) cannot duplicate or misindex the state's
      // input arrays.
      const size_t idx = it->second;
      owned_names_[idx] = extra_inputs[i].name;
      owned_tensors_[idx] = extra_inputs[i].tensor;
      state_.input_names_[state_slots_[idx]] = owned_names_[idx].c_str();
      state_.inputs_[state_slots_[idx]] = owned_tensors_[idx]->ort_tensor_.get();
    } else {
      owned_names_.push_back(extra_inputs[i].name);
      owned_tensors_.push_back(extra_inputs[i].tensor);
      const size_t idx = owned_names_.size() - 1;
      state_slots_.push_back(state_.input_names_.size());
      state_.input_names_.push_back(owned_names_[idx].c_str());
      state_.inputs_.push_back(owned_tensors_[idx]->ort_tensor_.get());
      name_to_index_.emplace(owned_names_[idx], idx);
    }
  }

  registrar_.Add();
}

}  // namespace Generators
