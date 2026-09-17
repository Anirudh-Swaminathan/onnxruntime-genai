// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cstring>
#include "generator/generators.h"
#include "models/model.h"
#include "models/io/multi_modal_features.h"

namespace Generators {

MultiModalFeatures::MultiModalFeatures(State& state, MultiModalFeatures::Mode mode, const std::string& name,
                                       int64_t batch_size, int64_t num_feature_tokens)
    : state_{state},
      type_{mode == MultiModalFeatures::Mode::Input
                ? model_.session_info_.GetInputDataType(name)
                : model_.session_info_.GetOutputDataType(name)},
      mode_{mode},
      name_{name} {
  const auto dims = mode_ == MultiModalFeatures::Mode::Input
                        ? model_.session_info_.GetInputSymbolicShape(name).size()
                        : model_.session_info_.GetOutputSymbolicShape(name).size();

  // If the model expects 3 dimensions, add a batch dimension
  // batch_size <= 0 signals "skip batch dim even if model has 3D output"
  if (dims == 3 && batch_size > 0) {
    shape_.push_back(batch_size);
  }

  shape_.push_back(num_feature_tokens);
  shape_.push_back(model_.config_->model.decoder.hidden_size);

  // There are four cases for MultiModalFeatures:
  // 1) Created as an output for vision or speech model (num_feature_tokens > 0)
  //    The tensor will be pre-allocated to store the output.
  //    It will be transferred to an input for the embedding model.
  // 2) Created as an output for vision or speech model (num_feature_tokens = 0)
  //    The tensor will be pre-allocated to store the empty output.
  //    It will be transferred to an input for the embedding model.
  // 3) Created as an input for embedding model (num_feature_tokens > 0)
  //    The tensor does not need to be pre-allocated because it will be created during (1).
  // 4) Created as an input for embedding model (num_feature_tokens = 0)
  //    The tensor does not need to be pre-allocated because it will be created during (2).
  if (mode == MultiModalFeatures::Mode::Output) {
    features_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), shape_, type_);
  }

  native_shape_ = shape_;
}

void MultiModalFeatures::Add() {
  if (mode_ == MultiModalFeatures::Mode::Input) {
    // In case the features are an input to a model, they are added
    // as a nullptr to reserve a slot in the inputs. The features
    // input will be overwritten when ReuseFeaturesBuffer is invoked.
    index_ = state_.inputs_.size();
    state_.inputs_.push_back(nullptr);
    state_.input_names_.push_back(name_.c_str());
  } else {
    index_ = state_.outputs_.size();
    state_.outputs_.push_back(features_.get());
    state_.output_names_.push_back(name_.c_str());
  }
}

void MultiModalFeatures::Rebind(int64_t batch_size, int64_t num_feature_tokens) {
  // Rederive shape like the constructor: batch_size may change across turns.
  const int64_t hidden_size = shape_.back();
  const auto dims = mode_ == MultiModalFeatures::Mode::Input
                        ? model_.session_info_.GetInputSymbolicShape(name_).size()
                        : model_.session_info_.GetOutputSymbolicShape(name_).size();
  const int64_t old_count = features_ ? static_cast<int64_t>(features_->GetTensorTypeAndShapeInfo()->GetElementCount()) : 0;

  shape_.clear();
  if (dims == 3 && batch_size > 0) {
    shape_.push_back(batch_size);
  }
  shape_.push_back(num_feature_tokens);
  shape_.push_back(hidden_size);

  int64_t new_count = 1;
  for (auto d : shape_) new_count *= d;

  if (mode_ == MultiModalFeatures::Mode::Output) {
    // Same size: keep data (encoder overwrites it next Run() anyway).
    auto old_features = (old_count == new_count && old_count > 0) ? std::move(features_) : nullptr;
    features_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), shape_, type_);
    if (old_features) {
      ByteWrapTensor(*model_.p_device_, *features_).CopyFrom(ByteWrapTensor(*model_.p_device_, *old_features));
    }
    state_.outputs_[index_] = features_.get();
  } else if (num_feature_tokens > 0) {
    // Nulled here; ReuseFeaturesBuffer fills it before Run() reads it this turn.
    features_.reset();
    state_.inputs_[index_] = nullptr;
  } else {
    // Zero tokens: turn may skip the merge path; bind an empty tensor now.
    features_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), shape_, type_);
    state_.inputs_[index_] = features_.get();
  }
}

void MultiModalFeatures::Update(bool is_prompt) {
  // Shrink to zero tokens once per prompt end; Rebind() then no-ops here.
  if (!is_prompt && shape_[shape_.size() - 2] > 0) {  // if num_image_tokens > 0
    const int64_t batch_size = shape_.size() == 3 ? shape_[0] : -1;
    Rebind(batch_size, 0);
  }
}

void MultiModalFeatures::ReuseFeaturesBuffer(MultiModalFeatures& other) {
  if (mode_ == MultiModalFeatures::Mode::Output || other.mode_ == MultiModalFeatures::Mode::Input) {
    throw std::runtime_error("Incorrect usage of the MultiModalFeatures inputs and outputs.");
  }

  // Take ownership of other's computed tensor as this input.
  shape_ = other.shape_;
  features_ = std::move(other.features_);
  state_.inputs_[index_] = features_.get();

  // Give other a real copy so its output binding has valid data before its next Run() overwrites it.
  other.shape_ = other.native_shape_;
  other.features_ = OrtValue::CreateTensor(other.model_.p_device_->GetAllocator(), other.shape_, other.type_);
  ByteWrapTensor(*other.model_.p_device_, *other.features_)
      .CopyFrom(ByteWrapTensor(*other.model_.p_device_, *features_));
  other.state_.outputs_[other.index_] = other.features_.get();
}

void MultiModalFeatures::AllocateEmptyFeatures() {
  // Skip if already allocated (avoids redundant allocation when called from
  // both EmbeddingState::SetExtraInputs and the pipeline prompt path)
  if (features_ && state_.inputs_[index_] == features_.get()) return;
  features_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), shape_, type_);
  state_.inputs_[index_] = features_.get();
}

void MultiModalFeatures::ReshapeFeatures(std::vector<int64_t> new_shape) {
  if (!features_) return;
  auto old_info = features_->GetTensorTypeAndShapeInfo();
  int64_t old_count = static_cast<int64_t>(old_info->GetElementCount());
  int64_t new_count = 1;
  for (auto d : new_shape) new_count *= d;
  if (old_count != new_count || old_count == 0) return;

  auto old_features = std::move(features_);
  features_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), new_shape, type_);
  auto src = ByteWrapTensor(*model_.p_device_, *old_features);
  auto dst = ByteWrapTensor(*model_.p_device_, *features_);
  dst.CopyFrom(src);
  shape_ = std::move(new_shape);
  if (mode_ == Mode::Output && index_ != ~0U) {
    state_.outputs_[index_] = features_.get();
  }
}

size_t MergeImageFeaturesIntoEmbeddings(OrtValue& inputs_embeds,
                                        const OrtValue& image_features,
                                        std::span<const int64_t> target_token_rows) {
  auto embeddings_shape = inputs_embeds.GetTensorTypeAndShapeInfo()->GetShape();
  auto vision_shape = image_features.GetTensorTypeAndShapeInfo()->GetShape();

  // Hidden size is always the last dimension for both [batch, seq, hidden]/[seq, hidden] embeddings
  // and [num_image_tokens, hidden] image features.
  const int64_t embedding_dim = embeddings_shape.back();
  const int64_t vision_dim = vision_shape.back();
  if (vision_dim != embedding_dim) {
    throw std::runtime_error("MergeImageFeaturesIntoEmbeddings: hidden dimension mismatch - vision_dim=" +
                             std::to_string(vision_dim) + ", embedding_dim=" + std::to_string(embedding_dim));
  }

  // Number of image feature rows, robust to [num_image_tokens, hidden] or [1, num_image_tokens, hidden].
  int64_t vision_element_count = 1;
  for (int64_t dim : vision_shape) vision_element_count *= dim;
  const int64_t num_vision_tokens = vision_dim > 0 ? vision_element_count / vision_dim : 0;

  float* embeddings_data = inputs_embeds.GetTensorMutableData<float>();
  const float* vision_data = image_features.GetTensorData<float>();
  size_t consumed = 0;
  for (int64_t row : target_token_rows) {
    if (consumed >= static_cast<size_t>(num_vision_tokens)) break;
    std::memcpy(embeddings_data + (row * embedding_dim),
                vision_data + (consumed * vision_dim),
                vision_dim * sizeof(float));
    ++consumed;
  }
  return consumed;
}

}  // namespace Generators
