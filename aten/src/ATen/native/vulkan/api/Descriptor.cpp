#include <ATen/native/vulkan/api/Descriptor.h>
#include <ATen/native/vulkan/api/Utils.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>

namespace at {
namespace native {
namespace vulkan {
namespace api {

namespace {

constexpr uint32_t kDescriptorFlushResetBit = 1u << 0;
constexpr uint32_t kDescriptorFlushClearPilesBit = 1u << 1;
constexpr uint32_t kDescriptorFlushAllBits =
    kDescriptorFlushResetBit | kDescriptorFlushClearPilesBit;

inline bool descriptor_flush_probe_mask_overridden() {
  static const bool overridden = []() {
    const char* value =
        std::getenv("PYTORCH_VULKAN_UPDATE_ORDERING_PROBE_DESCRIPTOR_FLUSH_MASK");
    return value != nullptr && value[0] != '\0';
  }();
  return overridden;
}

inline uint32_t descriptor_flush_probe_mask() {
  static const uint32_t mask = []() {
    const char* value =
        std::getenv("PYTORCH_VULKAN_UPDATE_ORDERING_PROBE_DESCRIPTOR_FLUSH_MASK");
    if (value == nullptr || value[0] == '\0') {
      return kDescriptorFlushAllBits;
    }

    char* parse_end = nullptr;
    const unsigned long parsed = std::strtoul(value, &parse_end, 10);
    if (parse_end == value || *parse_end != '\0' ||
        parsed > std::numeric_limits<uint32_t>::max()) {
      return kDescriptorFlushAllBits;
    }

    const uint32_t requested_mask = static_cast<uint32_t>(parsed);
    const uint32_t active_mask = requested_mask & kDescriptorFlushAllBits;
    return (active_mask == 0u) ? kDescriptorFlushAllBits : active_mask;
  }();
  return mask;
}

inline bool descriptor_pool_telemetry_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("PYTORCH_VULKAN_DESCRIPTOR_POOL_TELEMETRY");
    if (value == nullptr || value[0] == '\0') {
      return false;
    }
    return !(value[0] == '0' || value[0] == 'f' || value[0] == 'F' ||
             value[0] == 'n' || value[0] == 'N');
  }();
  return enabled;
}

inline uint64_t descriptor_pool_telemetry_every() {
  static const uint64_t every = []() {
    const char* value =
        std::getenv("PYTORCH_VULKAN_DESCRIPTOR_POOL_TELEMETRY_EVERY");
    if (value == nullptr || value[0] == '\0') {
      return static_cast<uint64_t>(128ull);
    }
    char* parse_end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &parse_end, 10);
    if (parse_end == value || *parse_end != '\0' || parsed == 0ull) {
      return static_cast<uint64_t>(128ull);
    }
    return static_cast<uint64_t>(parsed);
  }();
  return every;
}

} // namespace

//
// DescriptorSet
//

DescriptorSet::DescriptorSet(
    VkDevice device,
    VkDescriptorSet handle,
    ShaderLayout::Signature shader_layout_signature)
    : device_(device),
      handle_(handle),
      shader_layout_signature_(std::move(shader_layout_signature)),
      bindings_{} {}

DescriptorSet::DescriptorSet(DescriptorSet&& other) noexcept
    : device_(other.device_),
      handle_(other.handle_),
      shader_layout_signature_(std::move(other.shader_layout_signature_)),
      bindings_(std::move(other.bindings_)) {
  other.handle_ = VK_NULL_HANDLE;
}

DescriptorSet& DescriptorSet::operator=(DescriptorSet&& other) noexcept {
  device_ = other.device_;
  handle_ = other.handle_;
  shader_layout_signature_ = std::move(other.shader_layout_signature_);
  bindings_ = std::move(other.bindings_);

  other.handle_ = VK_NULL_HANDLE;

  return *this;
}

DescriptorSet& DescriptorSet::bind(
    const uint32_t idx,
    const VulkanBuffer& buffer) {
  VK_CHECK_COND(
      buffer.has_memory(),
      "Buffer must be bound to memory for it to be usable");

  DescriptorSet::ResourceBinding binder{};
  binder.binding_idx = idx; // binding_idx
  binder.descriptor_type = shader_layout_signature_[idx]; // descriptor_type
  binder.is_image = false; // is_image
  binder.resource_info.buffer_info.buffer = buffer.handle(); // buffer
  binder.resource_info.buffer_info.offset = buffer.mem_offset(); // offset
  binder.resource_info.buffer_info.range = buffer.mem_range(); // range
  add_binding(binder);

  return *this;
}

DescriptorSet& DescriptorSet::bind(
    const uint32_t idx,
    const VulkanImage& image) {
  VK_CHECK_COND(
      image.has_memory(), "Image must be bound to memory for it to be usable");

  VkImageLayout binding_layout = image.layout();
  if (shader_layout_signature_[idx] == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
    binding_layout = VK_IMAGE_LAYOUT_GENERAL;
  }

  DescriptorSet::ResourceBinding binder{};
  binder.binding_idx = idx; // binding_idx
  binder.descriptor_type = shader_layout_signature_[idx]; // descriptor_type
  binder.is_image = true; // is_image
  binder.resource_info.image_info.sampler = image.sampler(); // buffer
  binder.resource_info.image_info.imageView = image.image_view(); // imageView
  binder.resource_info.image_info.imageLayout = binding_layout; // imageLayout
  add_binding(binder);

  return *this;
}

VkDescriptorSet DescriptorSet::get_bind_handle() const {
  std::vector<VkWriteDescriptorSet> write_descriptor_sets;

  for (const ResourceBinding& binding : bindings_) {
    VkWriteDescriptorSet write{
        VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, // sType
        nullptr, // pNext
        handle_, // dstSet
        binding.binding_idx, // dstBinding
        0u, // dstArrayElement
        1u, // descriptorCount
        binding.descriptor_type, // descriptorType
        nullptr, // pImageInfo
        nullptr, // pBufferInfo
        nullptr, // pTexelBufferView
    };

    if (binding.is_image) {
      write.pImageInfo = &binding.resource_info.image_info;
    } else {
      write.pBufferInfo = &binding.resource_info.buffer_info;
    }

    write_descriptor_sets.emplace_back(write);
  }

  vkUpdateDescriptorSets(
      device_,
      write_descriptor_sets.size(),
      write_descriptor_sets.data(),
      0u,
      nullptr);

  VkDescriptorSet ret = handle_;

  return ret;
}

void DescriptorSet::add_binding(const ResourceBinding& binding) {
  const auto bindings_itr = std::find_if(
      bindings_.begin(),
      bindings_.end(),
      [binding_idx = binding.binding_idx](const ResourceBinding& other) {
        return other.binding_idx == binding_idx;
      });

  if (bindings_.end() == bindings_itr) {
    bindings_.emplace_back(binding);
  } else {
    *bindings_itr = binding;
  }
}

//
// DescriptorSetPile
//

DescriptorSetPile::DescriptorSetPile(
    const uint32_t pile_size,
    VkDescriptorSetLayout descriptor_set_layout,
    VkDevice device,
    VkDescriptorPool descriptor_pool)
    : pile_size_{pile_size},
      set_layout_{descriptor_set_layout},
      device_{device},
      pool_{descriptor_pool},
      descriptors_{},
      in_use_(0u) {
  descriptors_.resize(pile_size_);
  allocate_new_batch();
}

VkDescriptorSet DescriptorSetPile::get_descriptor_set() {
  // No-ops if there are descriptor sets available
  allocate_new_batch();

  VkDescriptorSet handle = descriptors_[in_use_];
  descriptors_[in_use_] = VK_NULL_HANDLE;

  in_use_++;
  return handle;
}

void DescriptorSetPile::allocate_new_batch() {
  // No-ops if there are still descriptor sets available
  if (in_use_ < descriptors_.size() &&
      descriptors_[in_use_] != VK_NULL_HANDLE) {
    return;
  }

  std::vector<VkDescriptorSetLayout> layouts(descriptors_.size());
  fill(layouts.begin(), layouts.end(), set_layout_);

  const VkDescriptorSetAllocateInfo allocate_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, // sType
      nullptr, // pNext
      pool_, // descriptorPool
      utils::safe_downcast<uint32_t>(layouts.size()), // descriptorSetCount
      layouts.data(), // pSetLayouts
  };

  VK_CHECK(
      vkAllocateDescriptorSets(device_, &allocate_info, descriptors_.data()));

  in_use_ = 0u;
}

//
// DescriptorPool
//

DescriptorPool::DescriptorPool(
    VkDevice device,
    const DescriptorPoolConfig& config)
    : device_(device),
      pool_(VK_NULL_HANDLE),
      config_(config),
      mutex_{},
      piles_{},
      telemetry_get_requests_(0u),
      telemetry_pile_hits_(0u),
      telemetry_pile_misses_(0u),
      telemetry_flush_calls_(0u),
      telemetry_reset_calls_(0u),
      telemetry_clear_calls_(0u) {
  if (config.descriptorPoolMaxSets > 0) {
    init(config);
  }
}

DescriptorPool::~DescriptorPool() {
  if (VK_NULL_HANDLE == pool_) {
    return;
  }
  vkDestroyDescriptorPool(device_, pool_, nullptr);
}

void DescriptorPool::init(const DescriptorPoolConfig& config) {
  VK_CHECK_COND(
      pool_ == VK_NULL_HANDLE,
      "Trying to init a DescriptorPool that has already been created!");

  config_ = config;

  std::vector<VkDescriptorPoolSize> type_sizes{
      {
          VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          config_.descriptorUniformBufferCount,
      },
      {
          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          config_.descriptorStorageBufferCount,
      },
      {
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
          config_.descriptorCombinedSamplerCount,
      },
      {
          VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          config_.descriptorStorageBufferCount,
      },
  };

  const VkDescriptorPoolCreateInfo create_info{
      VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      config_.descriptorPoolMaxSets, // maxSets
      static_cast<uint32_t>(type_sizes.size()), // poolSizeCounts
      type_sizes.data(), // pPoolSizes
  };

  VK_CHECK(vkCreateDescriptorPool(device_, &create_info, nullptr, &pool_));
}

DescriptorSet DescriptorPool::get_descriptor_set(
    VkDescriptorSetLayout set_layout,
    const ShaderLayout::Signature& signature) {
  VK_CHECK_COND(
      pool_ != VK_NULL_HANDLE, "DescriptorPool has not yet been initialized!");

  bool pile_hit = true;
  auto it = piles_.find(set_layout);
  if (piles_.cend() == it) {
    pile_hit = false;
    it = piles_
             .insert({
                 set_layout,
                 DescriptorSetPile(
                     config_.descriptorPileSizes, set_layout, device_, pool_),
             })
             .first;
  }

  ++telemetry_get_requests_;
  if (pile_hit) {
    ++telemetry_pile_hits_;
  } else {
    ++telemetry_pile_misses_;
  }
  if (descriptor_pool_telemetry_enabled()) {
    const uint64_t emit_every = descriptor_pool_telemetry_every();
    if (!pile_hit || (telemetry_get_requests_ % emit_every) == 0u) {
      std::cerr << "[vk_descriptor_pool_telemetry] site=get"
                << " seq=" << telemetry_get_requests_
                << " pile_hit=" << static_cast<int>(pile_hit)
                << " piles_size=" << piles_.size()
                << " pile_hits=" << telemetry_pile_hits_
                << " pile_misses=" << telemetry_pile_misses_
                << " flush_calls=" << telemetry_flush_calls_
                << " reset_calls=" << telemetry_reset_calls_
                << " clear_calls=" << telemetry_clear_calls_ << "\n";
    }
  }

  VkDescriptorSet handle = it->second.get_descriptor_set();

  return DescriptorSet(device_, handle, signature);
}

void DescriptorPool::flush() {
  if (pool_ != VK_NULL_HANDLE) {
    ++telemetry_flush_calls_;
    const uint32_t active_mask = descriptor_flush_probe_mask();
    if (descriptor_pool_telemetry_enabled()) {
      std::cerr << "[vk_descriptor_pool_telemetry] site=flush.begin"
                << " flush_call=" << telemetry_flush_calls_
                << " active_mask=" << active_mask
                << " piles_before=" << piles_.size()
                << " gets=" << telemetry_get_requests_
                << " pile_hits=" << telemetry_pile_hits_
                << " pile_misses=" << telemetry_pile_misses_
                << " reset_calls=" << telemetry_reset_calls_
                << " clear_calls=" << telemetry_clear_calls_ << "\n";
    }
    if (descriptor_flush_probe_mask_overridden()) {
      std::cerr << "[vk_update_ordering_probe] site=descriptor_pool.flush"
                << " descriptor_flush_mask=" << active_mask << "\n";
    }
    if (active_mask & kDescriptorFlushResetBit) {
      VK_CHECK(vkResetDescriptorPool(device_, pool_, 0u));
      ++telemetry_reset_calls_;
    }
    if (active_mask & kDescriptorFlushClearPilesBit) {
      piles_.clear();
      ++telemetry_clear_calls_;
    }
    if (descriptor_pool_telemetry_enabled()) {
      std::cerr << "[vk_descriptor_pool_telemetry] site=flush.end"
                << " flush_call=" << telemetry_flush_calls_
                << " active_mask=" << active_mask
                << " piles_after=" << piles_.size()
                << " gets=" << telemetry_get_requests_
                << " pile_hits=" << telemetry_pile_hits_
                << " pile_misses=" << telemetry_pile_misses_
                << " reset_calls=" << telemetry_reset_calls_
                << " clear_calls=" << telemetry_clear_calls_ << "\n";
    }
  }
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
