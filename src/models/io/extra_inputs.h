#pragma once

#include <deque>

namespace Generators {

struct PresetExtraInputs {
  PresetExtraInputs(State& state);
  void Add();

 private:
  using FuncType = std::function<std::unique_ptr<OrtValue>()>;
  State& state_;
  std::unordered_map<std::string, FuncType> registry_;
  std::vector<std::unique_ptr<OrtValue>> extra_inputs_;
  std::vector<std::string> extra_input_names_;
};

struct ExtraInputs {
  ExtraInputs(State& state);
  // Binds each extra input by name into state_.input_names_/inputs_. Safe to call more than once per
  // turn: a name seen before is rebound in place (last call wins) rather than appended again, so the
  // caller can be re-entrant across turns without duplicating or misindexing the state's input arrays.
  void Add(const std::vector<ExtraInput>& extra_inputs, const std::vector<std::string>& required_input_names = {});

 private:
  State& state_;
  const Model& model_{state_.model_};
  PresetExtraInputs registrar_{state_};

  // Owned copies of every distinct extra input ever bound through this instance. The caller's
  // std::vector<ExtraInput> (Generator::extra_inputs_) is cleared at the start of every turn, so nothing
  // bound into state_ may borrow from it; retaining a shared_ptr to the Tensor and a private copy of the
  // name keeps state_.inputs_/input_names_ valid regardless of what the caller does afterward.
  //
  // owned_names_ is a deque, not a vector: binding a new distinct name must not invalidate the c_str()
  // pointers already handed to state_.input_names_ for names bound earlier (deque never relocates
  // existing elements on push_back; a vector could reallocate and dangle them).
  std::deque<std::string> owned_names_;
  std::vector<std::shared_ptr<Tensor>> owned_tensors_;
  std::vector<size_t> state_slots_;                    // state_.inputs_/input_names_ index for owned_names_[i]/owned_tensors_[i]
  std::unordered_map<std::string, size_t> name_to_index_;  // name -> index into the three arrays above
};

}  // namespace Generators
