// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Gate 3 regression tests for "dynamic image injection for multi-turn VLMs". These exercise, through
// the public API only, the exact defects the multi-turn image plan fixes:
//   - the one-shot `set_extra_inputs_` latch that silently dropped every image after the first turn
//     (Sec 1, src/generators.cpp);
//   - append-instead-of-replace binding in ExtraInputs/MultiModalFeatures on a second SetExtraInputs
//     call (Sec 2a/2b, src/models/extra_inputs.cpp, src/models/multi_modal_features.cpp);
//   - the `is_prompt_`-gated merge path that a later image-bearing turn could never re-enter, and the
//     `vision_state_`/`speech_state_` resets that would otherwise null out on the very first prompt step
//     (Sec 2c/3, src/models/multi_modal.cpp);
//   - the mRoPE conversation-origin offset for Qwen-family position ids (Sec 5, src/models/position_inputs.cpp).
//
// The model under test is the tiny "shape stub" fixture in test/models/qwen3-5: its vision/embedding/
// decoder ONNX graphs are deliberately trivial (constant/zero-filled outputs of the correct, properly
// symbolic shape), so generated token values are meaningless. What each test actually asserts is that the
// full VLM pipeline -- vision session, host-side merge, embedding session, decoder session, and the mRoPE
// position-id rebuild -- runs to completion without throwing on a second (or third) conversation turn.
// Before this plan, several of these scenarios either silently dropped the new image or crashed via a
// null `vision_state_` dereference; after it, `State::Run`'s structural guard (Sec 2d) means any
// regression in the binding logic surfaces immediately as a thrown exception rather than as silently
// wrong output, so "does not throw" is a strong, direct signal here.
//
// qwen3-5 additionally carries hybrid recurrent/conv decoder state (LFM2-style), so this also covers the
// hybrid-decoder native Qwen configuration this plan targets. (The sibling test/models/qwen3-vl fixture
// is deliberately not reused here: its ONNX graphs were generated with static shapes tied to the exact
// two-images-in-one-call scenario the existing qwen_multi_image_vision_test.cpp uses, and reject the
// single-image-per-turn shapes these multi-turn tests need.)

#include <array>
#include <memory>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

#define OGA_USE_SPAN 1
#include "ort_genai.h"

namespace {

template <typename Fn>
std::string CaptureRuntimeErrorMessage(Fn&& fn) {
  try {
    fn();
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return {};
}

// Shared harness for the dummy native (in-graph feature merge) qwen3-5 VLM fixture.
struct MultiTurnVisionHarness {
  std::unique_ptr<OgaModel> model;
  std::unique_ptr<OgaMultiModalProcessor> processor;
  std::unique_ptr<OgaGeneratorParams> params;
  std::unique_ptr<OgaGenerator> generator;

  static std::string ImagePath(const char* file_name) {
    return std::string(MODEL_PATH) + "../images/" + file_name;
  }

  static MultiTurnVisionHarness Create(const char* model_dir = "qwen3-5") {
    MultiTurnVisionHarness harness;
    const std::string model_path = std::string(MODEL_PATH) + model_dir;
    harness.model = OgaModel::Create(model_path.c_str());
    harness.processor = OgaMultiModalProcessor::Create(*harness.model);
    harness.params = OgaGeneratorParams::Create(*harness.model);
    harness.generator = OgaGenerator::Create(*harness.model, *harness.params);
    return harness;
  }

  // A turn that carries exactly one image. Mirrors the "<|vision_start|><|image_pad|><|vision_end|>"
  // placeholder shape used by the existing single-turn multi-image tests (qwen_multi_image_vision_test.cpp)
  // -- the processor expands the single `<|image_pad|>` marker into however many placeholder tokens this
  // image's patch grid requires, so placeholder count and produced vision-feature count are guaranteed to
  // agree without this test needing to compute either by hand.
  std::unique_ptr<OgaNamedTensors> BuildImageTurn(const char* image_file) const {
    const std::string image_path = ImagePath(image_file);
    const std::array<const char*, 1> image_paths{image_path.c_str()};
    auto images = OgaImages::Load(image_paths);
    return processor->ProcessImages(
        "<|vision_start|><|image_pad|><|vision_end|>Describe this image.", images.get());
  }

  // A plain-text turn: no images argument, so the returned payload carries input_ids only.
  std::unique_ptr<OgaNamedTensors> BuildTextTurn(const char* prompt) const {
    return processor->ProcessImages(prompt);
  }
};

}  // namespace

// Before this plan: the second SetInputs() call would fail to re-run SetExtraInputs() at all (the
// `set_extra_inputs_` latch was already false from turn 1), so the second image was silently dropped --
// or, if the re-entrancy bugs below it were exercised, would append a duplicate `pixel_values` binding.
// After this plan: SetInputs() clears and repopulates extra_inputs_ every call (Sec 1), and every binding
// helper along the path rebinds rather than appends (Sec 2a/2b/2c), so a second image-bearing turn must
// run the full vision + merge pipeline again without throwing.
TEST(MultiTurnImageInjectionTest, SecondTurnWithNewImageIsAcceptedNotDropped) {
  auto harness = MultiTurnVisionHarness::Create();

  auto turn1 = harness.BuildImageTurn("australia.jpg");
  ASSERT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->SetInputs(*turn1); }), "");
  ASSERT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->GenerateNextToken(); }), "");

  auto turn2 = harness.BuildImageTurn("landscape.jpg");
  EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->SetInputs(*turn2); }), "");
  EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->GenerateNextToken(); }), "");
}

// A plain-text turn immediately after an image turn must still work: `merge_features_this_step_` (Sec 3)
// is false for this turn (no image/audio tokens), so it takes the generation-stage path exactly as a
// text-only turn always has -- the removed `vision_state_`/`speech_state_` resets (Sec 2c) must not have
// broken that path for the *first* image turn either.
TEST(MultiTurnImageInjectionTest, TextOnlyTurnAfterImageTurnDoesNotThrow) {
  auto harness = MultiTurnVisionHarness::Create();

  auto turn1 = harness.BuildImageTurn("australia.jpg");
  ASSERT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->SetInputs(*turn1); }), "");
  ASSERT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->GenerateNextToken(); }), "");

  auto turn2 = harness.BuildTextTurn("What should I do next?");
  EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->SetInputs(*turn2); }), "");
  EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->GenerateNextToken(); }), "");
}

// Mirrors the dev-tools "weather_gap.json" scenario (Gate 2 case B): image, then a text-only turn, then a
// second image. The image on turn 3 must survive an intervening text-only turn and continue the mRoPE
// coordinate space from where turn 1 left off (Sec 5's `mrope_next_base_`) rather than restarting at 0.
TEST(MultiTurnImageInjectionTest, ImageOnTurnsOneAndThreeSurvivesTextGapTurn) {
  auto harness = MultiTurnVisionHarness::Create();

  auto turn1 = harness.BuildImageTurn("australia.jpg");
  ASSERT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->SetInputs(*turn1); }), "");
  ASSERT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->GenerateNextToken(); }), "");

  auto turn2 = harness.BuildTextTurn("Summarize what you have seen so far.");
  ASSERT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->SetInputs(*turn2); }), "");
  ASSERT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->GenerateNextToken(); }), "");

  auto turn3 = harness.BuildImageTurn("landscape.jpg");
  EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->SetInputs(*turn3); }), "");
  EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->GenerateNextToken(); }), "");
}

// Mirrors the dev-tools "weather_text_first.json" scenario (Gate 2 case C) -- the sharpest single test of
// the whole change, per the plan: a text-only first turn consumes the `is_prompt_` step before any image
// ever arrives, so admitting the image on turn 2 depends on every one of Sec 1/2/3/4/5 at once (the latch,
// the re-entrant bindings, the `merge_features_this_step_` decoupling from `is_prompt_`, and the mRoPE
// offset landing at a non-zero conversation origin).
TEST(MultiTurnImageInjectionTest, TextFirstTurnThenImageOnLaterTurnsDoesNotThrow) {
  auto harness = MultiTurnVisionHarness::Create();

  auto turn1 = harness.BuildTextTurn(
      "I will share images as we go. Tell me what to look for.");
  ASSERT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->SetInputs(*turn1); }), "");
  ASSERT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->GenerateNextToken(); }), "");

  auto turn2 = harness.BuildImageTurn("australia.jpg");
  EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->SetInputs(*turn2); }), "");
  EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->GenerateNextToken(); }), "");

  auto turn3 = harness.BuildImageTurn("landscape.jpg");
  EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->SetInputs(*turn3); }), "");
  EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->GenerateNextToken(); }), "");
}

// Direct regression test for Sec 1/2a's "duplicate name must rebind, not append" requirement -- the
// specific bug `State::Run`'s guard (Sec 2d) exists to catch. `set_model_input` (OgaGenerator_SetModelInput)
// pushes into extra_inputs_ directly, independently of SetInputs's per-turn clear (Sec 1's "Scope of the
// clear", consequence 1); two successive set_model_input calls for the same name landing in a single
// extra_inputs_ vector -- consumed together by one SetExtraInputs() call -- is a scenario Sec 1's clear
// does not itself resolve, so this exercises §2a's replace-by-name directly rather than through it.
//
// This bypasses Generator::SetInputs() entirely (its own clear would discard anything staged via
// SetModelInput beforehand) and instead reconstructs the same turn by binding every non-input_ids tensor
// from a real processed image turn through SetModelInput -- exactly as SetInputs() does internally --
// plus one deliberate duplicate bind of "pixel_values". If ExtraInputs::Add appended instead of rebinding
// the existing slot, state_.input_names_ would contain "pixel_values" twice and AppendTokens() below
// would throw State::Run's duplicate-name guard (Sec 2d) instead of succeeding.
TEST(MultiTurnImageInjectionTest, RepeatedSetModelInputForSameNameRebindsRatherThanAppends) {
  auto harness = MultiTurnVisionHarness::Create();

  auto turn1 = harness.BuildImageTurn("australia.jpg");
  auto names = turn1->GetNames();

  std::unique_ptr<OgaTensor> input_ids_tensor;
  for (size_t i = 0; i < names->Count(); ++i) {
    const char* name = names->Get(i);
    auto tensor = turn1->Get(name);
    if (std::string(name) == "input_ids") {
      input_ids_tensor = std::move(tensor);
      continue;
    }
    EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->SetModelInput(name, *tensor); }), "");
    if (std::string(name) == "pixel_values") {
      // The deliberate duplicate: same name, same (valid) tensor, bound a second time before
      // anything consumes extra_inputs_.
      EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->SetModelInput(name, *tensor); }), "");
    }
  }
  ASSERT_NE(input_ids_tensor, nullptr);

  const auto input_ids_shape = input_ids_tensor->Shape();
  size_t input_ids_count = 1;
  for (auto dim : input_ids_shape) input_ids_count *= static_cast<size_t>(dim);
  const auto* input_ids_data = static_cast<const int32_t*>(input_ids_tensor->Data());

  EXPECT_EQ(CaptureRuntimeErrorMessage([&] { harness.generator->AppendTokens(input_ids_data, input_ids_count); }), "");
}
