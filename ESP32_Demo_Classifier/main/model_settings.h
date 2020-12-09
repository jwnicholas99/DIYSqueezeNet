#ifndef INTEL6_IMAGE_MODEL_SETTINGS_H_
#define INTEL6_IMAGE_MODEL_SETTINGS_H_

// Keeping these as constant expressions allow us to allocate fixed-sized arrays
// on the stack for our working memory.

// All of these values are derived from the values used during model training,
// if you change your model you'll need to update these constants.
constexpr int kNumCols = 96;
constexpr int kNumRows = 96;
constexpr int kNumChannels = 1;

constexpr int kMaxImageSize = kNumCols * kNumRows * kNumChannels;

constexpr int kCategoryCount = 6;
extern const char* kCategoryLabels[kCategoryCount];

#endif  // INTEL6_IMAGE_MODEL_SETTINGS_H_
