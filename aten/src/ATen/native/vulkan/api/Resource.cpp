#include <ATen/native/vulkan/api/Adapter.h>
#include <ATen/native/vulkan/api/Context.h>
#include <ATen/native/vulkan/api/Resource.h>

#include <array>
#include <atomic>
#include <cstdlib>
#include <cstdio>
#include <mutex>

namespace {

using ::at::native::vulkan::api::AllocationTag;

constexpr size_t kAllocationTagCount =
    static_cast<size_t>(AllocationTag::CopyTemp) + 1u;

thread_local AllocationTag tls_allocation_tag = AllocationTag::Unknown;

bool alloc_diagnostics_enabled() {
  const char* env = std::getenv("TORCH_VULKAN_ALLOC_DIAGNOSTICS");
  if (!env || env[0] == '\0') {
    return false;
  }
  if (env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' ||
      env[0] == 'Y') {
    return true;
  }
  return false;
}

bool alloc_budget_diagnostics_enabled() {
  const char* env = std::getenv("TORCH_VULKAN_ALLOC_BUDGET_DIAGNOSTICS");
  if (!env || env[0] == '\0') {
    return false;
  }
  if (env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' ||
      env[0] == 'Y') {
    return true;
  }
  return false;
}

bool alloc_attribution_diagnostics_enabled() {
  const char* env = std::getenv("TORCH_VULKAN_ALLOC_ATTRIBUTION_DIAGNOSTICS");
  if (!env || env[0] == '\0') {
    return false;
  }
  if (env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' ||
      env[0] == 'Y') {
    return true;
  }
  return false;
}

enum class AllocationKind : uint8_t {
  Buffer = 0u,
  Image = 1u,
};

struct AllocationTagStats final {
  uint64_t live_bytes = 0ull;
  uint64_t peak_live_bytes = 0ull;
  uint64_t total_alloc_bytes = 0ull;
  uint64_t live_count = 0ull;
  uint64_t peak_live_count = 0ull;
  uint64_t total_alloc_count = 0ull;
};

struct AllocationAttributionState final {
  std::mutex mutex;
  bool registered_atexit = false;
  std::array<AllocationTagStats, kAllocationTagCount> buffer{};
  std::array<AllocationTagStats, kAllocationTagCount> image{};
};

AllocationAttributionState& allocation_attribution_state() {
  static AllocationAttributionState state{};
  return state;
}

AllocationTagStats& allocation_stats_for(
    AllocationAttributionState& state,
    const AllocationKind kind,
    const AllocationTag tag) {
  const size_t index = static_cast<size_t>(tag);
  return AllocationKind::Image == kind ? state.image[index] : state.buffer[index];
}

const char* allocation_kind_name(const AllocationKind kind) {
  return AllocationKind::Image == kind ? "image" : "buffer";
}

void dump_allocation_attribution_summary() {
  if (!alloc_attribution_diagnostics_enabled()) {
    return;
  }

  AllocationAttributionState& state = allocation_attribution_state();
  std::lock_guard<std::mutex> guard(state.mutex);

  for (const AllocationKind kind : {AllocationKind::Buffer, AllocationKind::Image}) {
    for (size_t index = 0; index < kAllocationTagCount; ++index) {
      const AllocationTag tag = static_cast<AllocationTag>(index);
      const AllocationTagStats& stats = allocation_stats_for(state, kind, tag);
      if (0ull == stats.total_alloc_count && 0ull == stats.peak_live_bytes) {
        continue;
      }
      std::fprintf(
          stderr,
          "[vulkan_alloc_summary] kind=%s tag=%s total_alloc_count=%llu total_alloc_bytes=%llu peak_live_count=%llu peak_live_bytes=%llu live_count=%llu live_bytes=%llu\n",
          allocation_kind_name(kind),
          ::at::native::vulkan::api::allocation_tag_name(tag),
          static_cast<unsigned long long>(stats.total_alloc_count),
          static_cast<unsigned long long>(stats.total_alloc_bytes),
          static_cast<unsigned long long>(stats.peak_live_count),
          static_cast<unsigned long long>(stats.peak_live_bytes),
          static_cast<unsigned long long>(stats.live_count),
          static_cast<unsigned long long>(stats.live_bytes));
    }
  }
}

void maybe_register_allocation_attribution_summary() {
  if (!alloc_attribution_diagnostics_enabled()) {
    return;
  }

  AllocationAttributionState& state = allocation_attribution_state();
  std::lock_guard<std::mutex> guard(state.mutex);
  if (!state.registered_atexit) {
    std::atexit(dump_allocation_attribution_summary);
    state.registered_atexit = true;
  }
}

void record_allocation_attribution(
    const AllocationKind kind,
    const AllocationTag tag,
    const uint64_t request_bytes,
    const uint64_t alloc_bytes) {
  if (!alloc_attribution_diagnostics_enabled()) {
    return;
  }

  maybe_register_allocation_attribution_summary();

  AllocationAttributionState& state = allocation_attribution_state();
  std::lock_guard<std::mutex> guard(state.mutex);
  AllocationTagStats& stats = allocation_stats_for(state, kind, tag);
  stats.total_alloc_count += 1u;
  stats.total_alloc_bytes += alloc_bytes;
  stats.live_count += 1u;
  stats.live_bytes += alloc_bytes;
  if (stats.live_count > stats.peak_live_count) {
    stats.peak_live_count = stats.live_count;
  }
  if (stats.live_bytes > stats.peak_live_bytes) {
    stats.peak_live_bytes = stats.live_bytes;
  }
  std::fprintf(
      stderr,
      "[vulkan_alloc_attribution] event=alloc kind=%s tag=%s request=%llu alloc_size=%llu live_count=%llu live_bytes=%llu peak_live_bytes=%llu\n",
      allocation_kind_name(kind),
      ::at::native::vulkan::api::allocation_tag_name(tag),
      static_cast<unsigned long long>(request_bytes),
      static_cast<unsigned long long>(alloc_bytes),
      static_cast<unsigned long long>(stats.live_count),
      static_cast<unsigned long long>(stats.live_bytes),
      static_cast<unsigned long long>(stats.peak_live_bytes));
}

void release_allocation_attribution(
    const AllocationKind kind,
    const AllocationTag tag,
    const uint64_t alloc_bytes) {
  if (!alloc_attribution_diagnostics_enabled() || 0ull == alloc_bytes) {
    return;
  }

  AllocationAttributionState& state = allocation_attribution_state();
  std::lock_guard<std::mutex> guard(state.mutex);
  AllocationTagStats& stats = allocation_stats_for(state, kind, tag);
  stats.live_count = stats.live_count > 0ull ? stats.live_count - 1ull : 0ull;
  stats.live_bytes =
      stats.live_bytes > alloc_bytes ? (stats.live_bytes - alloc_bytes) : 0ull;
  std::fprintf(
      stderr,
      "[vulkan_alloc_attribution] event=free kind=%s tag=%s alloc_size=%llu live_count=%llu live_bytes=%llu\n",
      allocation_kind_name(kind),
      ::at::native::vulkan::api::allocation_tag_name(tag),
      static_cast<unsigned long long>(alloc_bytes),
      static_cast<unsigned long long>(stats.live_count),
      static_cast<unsigned long long>(stats.live_bytes));
}

struct BudgetSummary final {
  uint64_t device_local_budget = 0ull;
  uint64_t device_local_usage = 0ull;
  uint64_t device_local_headroom = 0ull;
  uint64_t dedicated_budget = 0ull;
  uint64_t dedicated_usage = 0ull;
  uint64_t dedicated_headroom = 0ull;
  uint64_t host_budget = 0ull;
  uint64_t host_usage = 0ull;
  uint64_t host_headroom = 0ull;
};

std::atomic<uint64_t> image_budget_sequence{0u};

bool has_device_local_host_visible(VmaAllocator allocator) {
  const VkPhysicalDeviceMemoryProperties* mem_props = nullptr;
  vmaGetMemoryProperties(allocator, &mem_props);
  for (uint32_t i = 0; i < mem_props->memoryTypeCount; ++i) {
    const VkMemoryPropertyFlags flags = mem_props->memoryTypes[i].propertyFlags;
    if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
      return true;
    }
  }
  return false;
}

bool env_true(const char* env_name, const bool default_value = false) {
  const char* env = std::getenv(env_name);
  if (!env || env[0] == '\0') {
    return default_value;
  }
  return (
      env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' ||
      env[0] == 'Y');
}

uint64_t env_u64_mb(const char* env_name, const uint64_t default_mb) {
  const char* env = std::getenv(env_name);
  if (!env || env[0] == '\0') {
    return default_mb;
  }
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(env, &end, 10);
  if (!end || *end != '\0') {
    return default_mb;
  }
  return static_cast<uint64_t>(parsed);
}

bool alloc_pressure_flush_enabled() {
  return env_true("PYTORCH_VULKAN_ALLOC_PRESSURE_FLUSH", true);
}

uint64_t alloc_pressure_margin_bytes() {
  return env_u64_mb("PYTORCH_VULKAN_ALLOC_PRESSURE_MARGIN_MB", 64ull) *
      1024ull * 1024ull;
}

uint64_t device_local_headroom_bytes(VmaAllocator allocator) {
  const VkPhysicalDeviceMemoryProperties* mem_props = nullptr;
  vmaGetMemoryProperties(allocator, &mem_props);

  VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
  vmaGetHeapBudgets(allocator, budgets);

  uint64_t max_headroom = 0ull;
  for (uint32_t heap_idx = 0; heap_idx < mem_props->memoryHeapCount; ++heap_idx) {
    const VkMemoryHeap& heap = mem_props->memoryHeaps[heap_idx];
    if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0u) {
      continue;
    }

    const uint64_t budget = static_cast<uint64_t>(budgets[heap_idx].budget);
    const uint64_t usage = static_cast<uint64_t>(budgets[heap_idx].usage);
    const uint64_t heap_size = static_cast<uint64_t>(heap.size);
    const uint64_t effective_budget = budget > 0ull ? budget : heap_size;
    const uint64_t headroom =
        usage < effective_budget ? (effective_budget - usage) : 0ull;
    if (headroom > max_headroom) {
      max_headroom = headroom;
    }
  }

  return max_headroom;
}

bool heap_has_host_visible_type(
    const VkPhysicalDeviceMemoryProperties* mem_props,
    const uint32_t heap_idx) {
  for (uint32_t type_idx = 0; type_idx < mem_props->memoryTypeCount; ++type_idx) {
    const VkMemoryType& type = mem_props->memoryTypes[type_idx];
    if (type.heapIndex != heap_idx) {
      continue;
    }
    if ((type.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0u) {
      return true;
    }
  }
  return false;
}

BudgetSummary summarize_allocator_budgets(VmaAllocator allocator) {
  BudgetSummary summary{};
  const VkPhysicalDeviceMemoryProperties* mem_props = nullptr;
  vmaGetMemoryProperties(allocator, &mem_props);

  VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
  vmaGetHeapBudgets(allocator, budgets);

  for (uint32_t heap_idx = 0; heap_idx < mem_props->memoryHeapCount; ++heap_idx) {
    const VkMemoryHeap& heap = mem_props->memoryHeaps[heap_idx];
    const uint64_t budget = static_cast<uint64_t>(budgets[heap_idx].budget);
    const uint64_t usage = static_cast<uint64_t>(budgets[heap_idx].usage);
    const uint64_t heap_size = static_cast<uint64_t>(heap.size);
    const uint64_t effective_budget = budget > 0ull ? budget : heap_size;
    const uint64_t headroom =
        usage < effective_budget ? (effective_budget - usage) : 0ull;

    if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0u) {
      summary.device_local_budget += effective_budget;
      summary.device_local_usage += usage;
      summary.device_local_headroom += headroom;
      if (!heap_has_host_visible_type(mem_props, heap_idx)) {
        summary.dedicated_budget += effective_budget;
        summary.dedicated_usage += usage;
        summary.dedicated_headroom += headroom;
      }
    } else {
      summary.host_budget += effective_budget;
      summary.host_usage += usage;
      summary.host_headroom += headroom;
    }
  }

  return summary;
}

void log_budget_snapshot(
    const char* site,
    VmaAllocator allocator,
    const uint64_t sequence,
    const uint64_t request_bytes,
    const VkResult result,
    const VkExtent3D& extents) {
  if (!alloc_budget_diagnostics_enabled()) {
    return;
  }

  const BudgetSummary summary = summarize_allocator_budgets(allocator);
  std::fprintf(
      stderr,
      "[vulkan_budget] seq=%llu site=%s request=%llu result=%d extent=%ux%ux%u "
      "device_local_budget=%llu device_local_usage=%llu device_local_headroom=%llu "
      "dedicated_budget=%llu dedicated_usage=%llu dedicated_headroom=%llu "
      "host_budget=%llu host_usage=%llu host_headroom=%llu\n",
      static_cast<unsigned long long>(sequence),
      site,
      static_cast<unsigned long long>(request_bytes),
      static_cast<int>(result),
      static_cast<unsigned int>(extents.width),
      static_cast<unsigned int>(extents.height),
      static_cast<unsigned int>(extents.depth),
      static_cast<unsigned long long>(summary.device_local_budget),
      static_cast<unsigned long long>(summary.device_local_usage),
      static_cast<unsigned long long>(summary.device_local_headroom),
      static_cast<unsigned long long>(summary.dedicated_budget),
      static_cast<unsigned long long>(summary.dedicated_usage),
      static_cast<unsigned long long>(summary.dedicated_headroom),
      static_cast<unsigned long long>(summary.host_budget),
      static_cast<unsigned long long>(summary.host_usage),
      static_cast<unsigned long long>(summary.host_headroom));
}

uint64_t dedicated_device_local_headroom_bytes(VmaAllocator allocator) {
  const VkPhysicalDeviceMemoryProperties* mem_props = nullptr;
  vmaGetMemoryProperties(allocator, &mem_props);

  VmaBudget budgets[VK_MAX_MEMORY_HEAPS] = {};
  vmaGetHeapBudgets(allocator, budgets);

  uint64_t selected_heap_size = 0ull;
  uint64_t selected_headroom = 0ull;

  for (uint32_t heap_idx = 0; heap_idx < mem_props->memoryHeapCount; ++heap_idx) {
    const VkMemoryHeap& heap = mem_props->memoryHeaps[heap_idx];
    if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0u) {
      continue;
    }
    if (heap_has_host_visible_type(mem_props, heap_idx)) {
      continue;
    }

    const uint64_t budget = static_cast<uint64_t>(budgets[heap_idx].budget);
    const uint64_t usage = static_cast<uint64_t>(budgets[heap_idx].usage);
    const uint64_t heap_size = static_cast<uint64_t>(heap.size);
    const uint64_t effective_budget = budget > 0ull ? budget : heap_size;
    const uint64_t headroom =
        usage < effective_budget ? (effective_budget - usage) : 0ull;

    if (heap_size > selected_heap_size) {
      selected_heap_size = heap_size;
      selected_headroom = headroom;
    }
  }

  return selected_headroom;
}

bool try_flush_pending_cleanup(const char* reason, const uint64_t request_bytes) {
  ::at::native::vulkan::api::Context* context_p =
      ::at::native::vulkan::api::context();
  if (!context_p || !context_p->has_pending_deferred_clear()) {
    return false;
  }

  auto lock = context_p->try_dispatch_lock();
  if (!lock.owns_lock()) {
    return false;
  }

  context_p->submit_cmd_to_gpu(VK_NULL_HANDLE, true);
  context_p->flush();

  if (alloc_diagnostics_enabled()) {
    std::fprintf(
        stdout,
        "[vulkan_alloc] pressure_flush reason=%s request=%llu\n",
        reason,
        static_cast<unsigned long long>(request_bytes));
  }

  return true;
}

void maybe_pressure_flush_before_alloc(
    VmaAllocator allocator,
    const VmaAllocationCreateInfo& alloc_create_info,
    const uint64_t request_bytes) {
  if (!alloc_pressure_flush_enabled()) {
    return;
  }

  const uint64_t required = request_bytes + alloc_pressure_margin_bytes();

  const bool prefers_device_local =
      alloc_create_info.usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE &&
      (alloc_create_info.requiredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ==
          0u;

  if (prefers_device_local) {
    const uint64_t dedicated_headroom =
        dedicated_device_local_headroom_bytes(allocator);
    if (dedicated_headroom > 0ull && dedicated_headroom < required) {
      if (try_flush_pending_cleanup("dedicated_headroom", request_bytes)) {
        return;
      }
    }
  }

  const uint64_t headroom = device_local_headroom_bytes(allocator);
  if (headroom < required) {
    try_flush_pending_cleanup("headroom", request_bytes);
  }
}

} // namespace

namespace at {
namespace native {
namespace vulkan {
namespace api {

const char* allocation_tag_name(const AllocationTag tag) {
  switch (tag) {
    case AllocationTag::Unknown:
      return "unknown";
    case AllocationTag::LinearOutput:
      return "linear_output";
    case AllocationTag::LinearPackInput:
      return "linear_pack_input";
    case AllocationTag::LinearPackWeight:
      return "linear_pack_weight";
    case AllocationTag::BinaryOutput:
      return "binary_output";
    case AllocationTag::LayerNormBinary:
      return "layer_norm_binary";
    case AllocationTag::LinearEpilogueBinary:
      return "linear_epilogue_binary";
    case AllocationTag::FactoryOutput:
      return "factory_output";
    case AllocationTag::ViewOutput:
      return "view_output";
    case AllocationTag::IndexingOutput:
      return "indexing_output";
    case AllocationTag::LayoutOutput:
      return "layout_output";
    case AllocationTag::UnaryOutput:
      return "unary_output";
    case AllocationTag::ReductionOutput:
      return "reduction_output";
    case AllocationTag::SoftmaxOutput:
      return "softmax_output";
    case AllocationTag::BiasTemp:
      return "bias_temp";
    case AllocationTag::Staging:
      return "staging";
    case AllocationTag::MetadataUniform:
      return "metadata_uniform";
    case AllocationTag::CopyTemp:
      return "copy_temp";
  }
  return "unknown";
}

AllocationTag current_allocation_tag() {
  return tls_allocation_tag;
}

AllocationTagScope::AllocationTagScope(const AllocationTag tag)
    : previous_tag_(tls_allocation_tag) {
  tls_allocation_tag = tag;
}

AllocationTagScope::~AllocationTagScope() {
  tls_allocation_tag = previous_tag_;
}

//
// MemoryBarrier
//

MemoryBarrier::MemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags)
    : handle{
          VK_STRUCTURE_TYPE_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
      } {}

//
// MemoryAllocation
//

MemoryAllocation::MemoryAllocation()
    : memory_requirements{},
      create_info{},
      allocator(VK_NULL_HANDLE),
      allocation(VK_NULL_HANDLE) {}

MemoryAllocation::MemoryAllocation(
    VmaAllocator vma_allocator,
    const VkMemoryRequirements& mem_props,
    const VmaAllocationCreateInfo& create_info)
    : memory_requirements(mem_props),
      create_info(create_info),
      allocator(vma_allocator),
      allocation(VK_NULL_HANDLE) {
  VK_CHECK(vmaAllocateMemory(
      allocator, &memory_requirements, &create_info, &allocation, nullptr));
}

MemoryAllocation::MemoryAllocation(MemoryAllocation&& other) noexcept
    : memory_requirements(other.memory_requirements),
      create_info(other.create_info),
      allocator(other.allocator),
      allocation(other.allocation) {
  other.allocation = VK_NULL_HANDLE;
}

MemoryAllocation& MemoryAllocation::operator=(
    MemoryAllocation&& other) noexcept {
  VmaAllocation tmp_allocation = allocation;

  memory_requirements = other.memory_requirements;
  create_info = other.create_info;
  allocator = other.allocator;
  allocation = other.allocation;

  other.allocation = tmp_allocation;

  return *this;
}

MemoryAllocation::~MemoryAllocation() {
  if (VK_NULL_HANDLE != allocation) {
    vmaFreeMemory(allocator, allocation);
  }
}

//
// VulkanBuffer
//

VulkanBuffer::VulkanBuffer()
    : buffer_properties_{},
      allocator_(VK_NULL_HANDLE),
      memory_{},
      owns_memory_(false),
      handle_(VK_NULL_HANDLE),
      tracked_alloc_size_(0ull),
      allocation_tag_(AllocationTag::Unknown) {}

VulkanBuffer::VulkanBuffer(
    VmaAllocator vma_allocator,
    const VkDeviceSize size,
    const VmaAllocationCreateInfo& allocation_create_info,
    const VkBufferUsageFlags usage,
    const bool allocate_memory)
    : buffer_properties_({
          size,
          0u,
          size,
          usage,
      }),
      allocator_(vma_allocator),
      memory_{},
      owns_memory_(allocate_memory),
      handle_(VK_NULL_HANDLE),
      tracked_alloc_size_(0ull),
      allocation_tag_(tls_allocation_tag) {
  // Only allocate memory if the buffer has non-zero size
  if (size == 0) {
    return;
  }

  const VkBufferCreateInfo buffer_create_info{
      VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      size, // size
      buffer_properties_.buffer_usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
  };

  memory_.create_info = allocation_create_info;

  if (allocate_memory) {
    maybe_pressure_flush_before_alloc(
      allocator_, allocation_create_info, static_cast<uint64_t>(size));

    const bool allow_host_retry =
        (buffer_properties_.buffer_usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) &&
        size >= (1024ull * 1024ull * 1024ull) &&
        has_device_local_host_visible(allocator_) &&
        allocation_create_info.usage == VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkResult alloc_result = vmaCreateBuffer(
        allocator_,
        &buffer_create_info,
        &allocation_create_info,
        &handle_,
        &(memory_.allocation),
        nullptr);
    if (alloc_pressure_flush_enabled() &&
        alloc_result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
      if (try_flush_pending_cleanup("oom_retry", static_cast<uint64_t>(size))) {
        alloc_result = vmaCreateBuffer(
            allocator_,
            &buffer_create_info,
            &allocation_create_info,
            &handle_,
            &(memory_.allocation),
            nullptr);
      }
    }
    if (alloc_result == VK_ERROR_OUT_OF_DEVICE_MEMORY && allow_host_retry) {
      VmaAllocationCreateInfo host_alloc_info = allocation_create_info;
      host_alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
      host_alloc_info.requiredFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
      host_alloc_info.preferredFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      alloc_result = vmaCreateBuffer(
          allocator_,
          &buffer_create_info,
          &host_alloc_info,
          &handle_,
          &(memory_.allocation),
          nullptr);
    }
    VK_CHECK(alloc_result);

    if (alloc_diagnostics_enabled()) {
      VmaAllocationInfo alloc_info{};
      vmaGetAllocationInfo(allocator_, memory_.allocation, &alloc_info);
      tracked_alloc_size_ = static_cast<uint64_t>(alloc_info.size);
      const VkPhysicalDeviceMemoryProperties* mem_props = nullptr;
      vmaGetMemoryProperties(allocator_, &mem_props);
      const VkMemoryPropertyFlags flags =
          mem_props->memoryTypes[alloc_info.memoryType].propertyFlags;
      std::fprintf(
          stdout,
          "[vulkan_alloc] buffer size=%llu alloc_size=%llu type=%u flags=0x%x usage=0x%x tag=%s\n",
          static_cast<unsigned long long>(buffer_properties_.size),
          static_cast<unsigned long long>(alloc_info.size),
          alloc_info.memoryType,
          static_cast<unsigned int>(flags),
          static_cast<unsigned int>(buffer_properties_.buffer_usage),
          allocation_tag_name(allocation_tag_));
    } else if (alloc_attribution_diagnostics_enabled()) {
      VmaAllocationInfo alloc_info{};
      vmaGetAllocationInfo(allocator_, memory_.allocation, &alloc_info);
      tracked_alloc_size_ = static_cast<uint64_t>(alloc_info.size);
    }

    record_allocation_attribution(
        AllocationKind::Buffer,
        allocation_tag_,
        static_cast<uint64_t>(buffer_properties_.size),
        tracked_alloc_size_);
  } else {
    VmaAllocatorInfo allocator_info{};
    vmaGetAllocatorInfo(allocator_, &allocator_info);
    VK_CHECK(vkCreateBuffer(
        allocator_info.device, &buffer_create_info, nullptr, &handle_));
  }
}

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept
    : buffer_properties_(other.buffer_properties_),
      allocator_(other.allocator_),
      memory_(std::move(other.memory_)),
      owns_memory_(other.owns_memory_),
      handle_(other.handle_),
      tracked_alloc_size_(other.tracked_alloc_size_),
      allocation_tag_(other.allocation_tag_) {
  other.handle_ = VK_NULL_HANDLE;
  other.tracked_alloc_size_ = 0ull;
  other.allocation_tag_ = AllocationTag::Unknown;
}

VulkanBuffer& VulkanBuffer::operator=(VulkanBuffer&& other) noexcept {
  VkBuffer tmp_buffer = handle_;
  bool tmp_owns_memory = owns_memory_;

  buffer_properties_ = other.buffer_properties_;
  allocator_ = other.allocator_;
  memory_ = std::move(other.memory_);
  owns_memory_ = other.owns_memory_;
  handle_ = other.handle_;
  tracked_alloc_size_ = other.tracked_alloc_size_;
  allocation_tag_ = other.allocation_tag_;

  other.handle_ = tmp_buffer;
  other.owns_memory_ = tmp_owns_memory;
  other.tracked_alloc_size_ = 0ull;
  other.allocation_tag_ = AllocationTag::Unknown;

  return *this;
}

VulkanBuffer::~VulkanBuffer() {
  if (VK_NULL_HANDLE != handle_) {
    release_allocation_attribution(
        AllocationKind::Buffer, allocation_tag_, tracked_alloc_size_);
    if (owns_memory_) {
      vmaDestroyBuffer(allocator_, handle_, memory_.allocation);
    } else {
      vkDestroyBuffer(this->device(), handle_, nullptr);
    }
    // Prevent the underlying memory allocation from being freed; it was either
    // freed by vmaDestroyBuffer, or this resource does not own the underlying
    // memory
    memory_.allocation = VK_NULL_HANDLE;
  }
}

VkMemoryRequirements VulkanBuffer::get_memory_requirements() const {
  VkMemoryRequirements memory_requirements;
  vkGetBufferMemoryRequirements(this->device(), handle_, &memory_requirements);
  return memory_requirements;
}

//
// MemoryMap
//

MemoryMap::MemoryMap(const VulkanBuffer& buffer, const uint8_t access)
    : access_(access),
      allocator_(buffer.vma_allocator()),
      allocation_(buffer.allocation()),
      data_(nullptr),
      data_len_{buffer.mem_size()} {
  if (allocation_) {
    VK_CHECK(vmaMapMemory(allocator_, allocation_, &data_));
  }
}

MemoryMap::MemoryMap(MemoryMap&& other) noexcept
    : access_(other.access_),
      allocator_(other.allocator_),
      allocation_(other.allocation_),
      data_(other.data_),
      data_len_{other.data_len_} {
  other.allocation_ = VK_NULL_HANDLE;
  other.data_ = nullptr;
}

MemoryMap::~MemoryMap() {
  if (!data_) {
    return;
  }

  if (allocation_) {
    if (access_ & MemoryAccessType::WRITE) {
      // Call will be ignored by implementation if the memory type this
      // allocation belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is
      // the behavior we want. Don't check the result here as the destructor
      // cannot throw.
      vmaFlushAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE);
    }

    vmaUnmapMemory(allocator_, allocation_);
  }
}

void MemoryMap::invalidate() {
  if (access_ & MemoryAccessType::READ && allocation_) {
    // Call will be ignored by implementation if the memory type this allocation
    // belongs to is not HOST_VISIBLE or is HOST_COHERENT, which is the behavior
    // we want.
    VK_CHECK(
        vmaInvalidateAllocation(allocator_, allocation_, 0u, VK_WHOLE_SIZE));
  }
}

//
// BufferMemoryBarrier
//

BufferMemoryBarrier::BufferMemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags,
    const VulkanBuffer& buffer)
    : handle{
          VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
          VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
          VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
          buffer.handle_, // buffer
          buffer.buffer_properties_.mem_offset, // offset
          buffer.buffer_properties_.mem_range, // size
      } {}

//
// ImageSampler
//

static bool operator==(
    const ImageSampler::Properties& _1,
    const ImageSampler::Properties& _2) {
  return (
      _1.filter == _2.filter && _1.mipmap_mode == _2.mipmap_mode &&
      _1.address_mode == _2.address_mode && _1.border_color == _2.border_color);
}

ImageSampler::ImageSampler(
    VkDevice device,
    const ImageSampler::Properties& props)
    : device_(device), handle_(VK_NULL_HANDLE) {
  const VkSamplerCreateInfo sampler_create_info{
      VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      props.filter, // magFilter
      props.filter, // minFilter
      props.mipmap_mode, // mipmapMode
      props.address_mode, // addressModeU
      props.address_mode, // addressModeV
      props.address_mode, // addressModeW
      0.0f, // mipLodBias
      VK_FALSE, // anisotropyEnable
      1.0f, // maxAnisotropy,
      VK_FALSE, // compareEnable
      VK_COMPARE_OP_NEVER, // compareOp
      0.0f, // minLod
      VK_LOD_CLAMP_NONE, // maxLod
      props.border_color, // borderColor
      VK_FALSE, // unnormalizedCoordinates
  };

  VK_CHECK(vkCreateSampler(device_, &sampler_create_info, nullptr, &handle_));
}

ImageSampler::ImageSampler(ImageSampler&& other) noexcept
    : device_(other.device_), handle_(other.handle_) {
  other.handle_ = VK_NULL_HANDLE;
}

ImageSampler::~ImageSampler() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroySampler(device_, handle_, nullptr);
}

size_t ImageSampler::Hasher::operator()(
    const ImageSampler::Properties& props) const {
  size_t seed = 0;
  seed = utils::hash_combine(seed, std::hash<VkFilter>()(props.filter));
  seed = utils::hash_combine(
      seed, std::hash<VkSamplerMipmapMode>()(props.mipmap_mode));
  seed = utils::hash_combine(
      seed, std::hash<VkSamplerAddressMode>()(props.address_mode));
  seed =
      utils::hash_combine(seed, std::hash<VkBorderColor>()(props.border_color));
  return seed;
}

void swap(ImageSampler& lhs, ImageSampler& rhs) noexcept {
  VkDevice tmp_device = lhs.device_;
  VkSampler tmp_handle = lhs.handle_;

  lhs.device_ = rhs.device_;
  lhs.handle_ = rhs.handle_;

  rhs.device_ = tmp_device;
  rhs.handle_ = tmp_handle;
}

//
// VulkanImage
//

VulkanImage::VulkanImage()
    : image_properties_{},
      view_properties_{},
      sampler_properties_{},
      allocator_(VK_NULL_HANDLE),
      memory_{},
      owns_memory_(false),
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
      },
      layout_{},
      tracked_alloc_size_(0ull),
      allocation_tag_(AllocationTag::Unknown) {}

VulkanImage::VulkanImage(
    VmaAllocator vma_allocator,
    const VmaAllocationCreateInfo& allocation_create_info,
    const ImageProperties& image_props,
    const ViewProperties& view_props,
    const SamplerProperties& sampler_props,
    const VkImageLayout layout,
    VkSampler sampler,
    const bool allocate_memory)
    : image_properties_(image_props),
      view_properties_(view_props),
      sampler_properties_(sampler_props),
      allocator_(vma_allocator),
      memory_{},
      owns_memory_{allocate_memory},
      handles_{
          VK_NULL_HANDLE,
          VK_NULL_HANDLE,
          sampler,
      },
      layout_(layout),
      tracked_alloc_size_(0ull),
      allocation_tag_(tls_allocation_tag) {
  VmaAllocatorInfo allocator_info{};
  vmaGetAllocatorInfo(allocator_, &allocator_info);

  // If any dims are zero, then no memory will be allocated for the image.
  if (image_props.image_extents.width == 0 ||
      image_props.image_extents.height == 0 ||
      image_props.image_extents.depth == 0) {
    return;
  }

  const VkImageCreateInfo image_create_info{
      VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      image_properties_.image_type, // imageType
      image_properties_.image_format, // format
      image_properties_.image_extents, // extents
      1u, // mipLevels
      1u, // arrayLayers
      VK_SAMPLE_COUNT_1_BIT, // samples
      VK_IMAGE_TILING_OPTIMAL, // tiling
      image_properties_.image_usage, // usage
      VK_SHARING_MODE_EXCLUSIVE, // sharingMode
      0u, // queueFamilyIndexCount
      nullptr, // pQueueFamilyIndices
      layout_, // initialLayout
  };

  memory_.create_info = allocation_create_info;

  if (allocate_memory) {
    const uint64_t request_bytes = static_cast<uint64_t>(
        image_properties_.image_extents.width) *
        static_cast<uint64_t>(image_properties_.image_extents.height) *
        static_cast<uint64_t>(image_properties_.image_extents.depth) * 4ull;
    const uint64_t image_budget_seq = alloc_budget_diagnostics_enabled()
        ? image_budget_sequence.fetch_add(1u, std::memory_order_relaxed) + 1u
        : 0u;
    maybe_pressure_flush_before_alloc(
      allocator_, allocation_create_info, request_bytes);
    if (image_budget_seq > 0u) {
      log_budget_snapshot(
          "image_prealloc",
          allocator_,
          image_budget_seq,
          request_bytes,
          VK_SUCCESS,
          image_properties_.image_extents);
    }

    VkResult alloc_result = vmaCreateImage(
        allocator_,
        &image_create_info,
        &allocation_create_info,
        &(handles_.image),
        &(memory_.allocation),
        nullptr);
    if (alloc_pressure_flush_enabled() &&
        alloc_result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
      if (try_flush_pending_cleanup("oom_retry", request_bytes)) {
        alloc_result = vmaCreateImage(
            allocator_,
            &image_create_info,
            &allocation_create_info,
            &(handles_.image),
            &(memory_.allocation),
            nullptr);
      }
    }
    if (image_budget_seq > 0u && alloc_result != VK_SUCCESS) {
      log_budget_snapshot(
          "image_result",
          allocator_,
          image_budget_seq,
          request_bytes,
          alloc_result,
          image_properties_.image_extents);
    }
    VK_CHECK(alloc_result);
    VmaAllocationInfo alloc_info{};
    vmaGetAllocationInfo(allocator_, memory_.allocation, &alloc_info);
    tracked_alloc_size_ = static_cast<uint64_t>(alloc_info.size);
    if (alloc_diagnostics_enabled() || alloc_attribution_diagnostics_enabled()) {
      const VkPhysicalDeviceMemoryProperties* mem_props = nullptr;
      vmaGetMemoryProperties(allocator_, &mem_props);
      const VkMemoryPropertyFlags flags =
          mem_props->memoryTypes[alloc_info.memoryType].propertyFlags;
      std::fprintf(
          stdout,
          "[vulkan_alloc] image request=%llu alloc_size=%llu type=%u flags=0x%x usage=0x%x extent=%ux%ux%u format=%u tag=%s\n",
          static_cast<unsigned long long>(request_bytes),
          static_cast<unsigned long long>(alloc_info.size),
          alloc_info.memoryType,
          static_cast<unsigned int>(flags),
          static_cast<unsigned int>(image_properties_.image_usage),
          static_cast<unsigned int>(image_properties_.image_extents.width),
          static_cast<unsigned int>(image_properties_.image_extents.height),
          static_cast<unsigned int>(image_properties_.image_extents.depth),
          static_cast<unsigned int>(image_properties_.image_format),
          allocation_tag_name(allocation_tag_));
    }
    record_allocation_attribution(
        AllocationKind::Image,
        allocation_tag_,
        request_bytes,
        tracked_alloc_size_);
    // Only create the image view if the image has been bound to memory
    create_image_view();
  } else {
    VK_CHECK(vkCreateImage(
        allocator_info.device, &image_create_info, nullptr, &(handles_.image)));
  }
}

VulkanImage::VulkanImage(VulkanImage&& other) noexcept
    : image_properties_(other.image_properties_),
      view_properties_(other.view_properties_),
      sampler_properties_(other.sampler_properties_),
      allocator_(other.allocator_),
      memory_(std::move(other.memory_)),
      owns_memory_(other.owns_memory_),
      handles_(other.handles_),
      layout_(other.layout_),
      tracked_alloc_size_(other.tracked_alloc_size_),
      allocation_tag_(other.allocation_tag_) {
  other.handles_.image = VK_NULL_HANDLE;
  other.handles_.image_view = VK_NULL_HANDLE;
  other.handles_.sampler = VK_NULL_HANDLE;
  other.owns_memory_ = false;
  other.tracked_alloc_size_ = 0ull;
  other.allocation_tag_ = AllocationTag::Unknown;
}

VulkanImage& VulkanImage::operator=(VulkanImage&& other) noexcept {
  VkImage tmp_image = handles_.image;
  VkImageView tmp_image_view = handles_.image_view;
  bool tmp_owns_memory = owns_memory_;

  image_properties_ = other.image_properties_;
  view_properties_ = other.view_properties_;
  sampler_properties_ = other.sampler_properties_;
  allocator_ = other.allocator_;
  memory_ = std::move(other.memory_);
  owns_memory_ = other.owns_memory_;
  handles_ = other.handles_;
  layout_ = other.layout_;
  tracked_alloc_size_ = other.tracked_alloc_size_;
  allocation_tag_ = other.allocation_tag_;

  other.handles_.image = tmp_image;
  other.handles_.image_view = tmp_image_view;
  other.owns_memory_ = tmp_owns_memory;
  other.tracked_alloc_size_ = 0ull;
  other.allocation_tag_ = AllocationTag::Unknown;

  return *this;
}

VulkanImage::~VulkanImage() {
  if (VK_NULL_HANDLE != handles_.image_view) {
    vkDestroyImageView(this->device(), handles_.image_view, nullptr);
  }

  if (VK_NULL_HANDLE != handles_.image) {
    release_allocation_attribution(
        AllocationKind::Image, allocation_tag_, tracked_alloc_size_);
    if (owns_memory_) {
      vmaDestroyImage(allocator_, handles_.image, memory_.allocation);
    } else {
      vkDestroyImage(this->device(), handles_.image, nullptr);
    }
    // Prevent the underlying memory allocation from being freed; it was either
    // freed by vmaDestroyImage, or this resource does not own the underlying
    // memory
    memory_.allocation = VK_NULL_HANDLE;
  }
}

void VulkanImage::create_image_view() {
  VmaAllocatorInfo allocator_info{};
  vmaGetAllocatorInfo(allocator_, &allocator_info);

  const VkComponentMapping component_mapping{
      VK_COMPONENT_SWIZZLE_IDENTITY, // r
      VK_COMPONENT_SWIZZLE_IDENTITY, // g
      VK_COMPONENT_SWIZZLE_IDENTITY, // b
      VK_COMPONENT_SWIZZLE_IDENTITY, // a
  };

  const VkImageSubresourceRange subresource_range{
      VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
      0u, // baseMipLevel
      VK_REMAINING_MIP_LEVELS, // levelCount
      0u, // baseArrayLayer
      VK_REMAINING_ARRAY_LAYERS, // layerCount
  };

  const VkImageViewCreateInfo image_view_create_info{
      VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
      handles_.image, // image
      view_properties_.view_type, // viewType
      view_properties_.view_format, // format
      component_mapping, // components
      subresource_range, // subresourceRange
  };

  VK_CHECK(vkCreateImageView(
      allocator_info.device,
      &(image_view_create_info),
      nullptr,
      &(handles_.image_view)));
}

VkMemoryRequirements VulkanImage::get_memory_requirements() const {
  VkMemoryRequirements memory_requirements;
  vkGetImageMemoryRequirements(
      this->device(), handles_.image, &memory_requirements);
  return memory_requirements;
}

//
// ImageMemoryBarrier
//

ImageMemoryBarrier::ImageMemoryBarrier(
    const VkAccessFlags src_access_flags,
    const VkAccessFlags dst_access_flags,
    const VkImageLayout src_layout_flags,
    const VkImageLayout dst_layout_flags,
    const VulkanImage& image)
    : handle{
          VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, // sType
          nullptr, // pNext
          src_access_flags, // srcAccessMask
          dst_access_flags, // dstAccessMask
          src_layout_flags, // oldLayout
          dst_layout_flags, // newLayout
          VK_QUEUE_FAMILY_IGNORED, // srcQueueFamilyIndex
          VK_QUEUE_FAMILY_IGNORED, // dstQueueFamilyIndex
          image.handles_.image, // image
          {
              // subresourceRange
              VK_IMAGE_ASPECT_COLOR_BIT, // aspectMask
              0u, // baseMipLevel
              VK_REMAINING_MIP_LEVELS, // levelCount
              0u, // baseArrayLayer
              VK_REMAINING_ARRAY_LAYERS, // layerCount
          },
      } {}

//
// SamplerCache
//

SamplerCache::SamplerCache(VkDevice device)
    : cache_mutex_{}, device_(device), cache_{} {}

SamplerCache::SamplerCache(SamplerCache&& other) noexcept
    : cache_mutex_{}, device_(other.device_), cache_(std::move(other.cache_)) {
  std::lock_guard<std::mutex> lock(other.cache_mutex_);
}

SamplerCache::~SamplerCache() {
  purge();
}

VkSampler SamplerCache::retrieve(const SamplerCache::Key& key) {
  std::lock_guard<std::mutex> lock(cache_mutex_);

  auto it = cache_.find(key);
  if (cache_.cend() == it) {
    it = cache_.insert({key, SamplerCache::Value(device_, key)}).first;
  }

  return it->second.handle();
}

void SamplerCache::purge() {
  std::lock_guard<std::mutex> lock(cache_mutex_);
  cache_.clear();
}

//
// MemoryAllocator
//

MemoryAllocator::MemoryAllocator(
    VkInstance instance,
    VkPhysicalDevice physical_device,
    VkDevice device)
    : instance_{},
      physical_device_(physical_device),
      device_(device),
      allocator_{VK_NULL_HANDLE} {
  VmaVulkanFunctions vk_functions{};
  vk_functions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  vk_functions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

  const VmaAllocatorCreateInfo allocator_create_info{
      0u, // flags
      physical_device_, // physicalDevice
      device_, // device
      0u, // preferredLargeHeapBlockSize
      nullptr, // pAllocationCallbacks
      nullptr, // pDeviceMemoryCallbacks
      nullptr, // pHeapSizeLimit
      &vk_functions, // pVulkanFunctions
      instance, // instance
      VK_API_VERSION_1_0, // vulkanApiVersion
      nullptr, // pTypeExternalMemoryHandleTypes
  };

  VK_CHECK(vmaCreateAllocator(&allocator_create_info, &allocator_));
}

MemoryAllocator::MemoryAllocator(MemoryAllocator&& other) noexcept
    : instance_(other.instance_),
      physical_device_(other.physical_device_),
      device_(other.device_),
      allocator_(other.allocator_) {
  other.allocator_ = VK_NULL_HANDLE;
  other.device_ = VK_NULL_HANDLE;
  other.physical_device_ = VK_NULL_HANDLE;
  other.instance_ = VK_NULL_HANDLE;
}

MemoryAllocator::~MemoryAllocator() {
  if (VK_NULL_HANDLE == allocator_) {
    return;
  }
  vmaDestroyAllocator(allocator_);
}

MemoryAllocation MemoryAllocator::create_allocation(
    const VkMemoryRequirements& memory_requirements,
    const VmaAllocationCreateInfo& create_info) {
  VmaAllocationCreateInfo alloc_create_info = create_info;
  // Protect against using VMA_MEMORY_USAGE_AUTO_* flags when allocating memory
  // directly, since those usage flags require that VkBufferCreateInfo and/or
  // VkImageCreateInfo also be available.
  switch (create_info.usage) {
    // The logic for the below usage options are too complex, therefore prevent
    // those from being used with direct memory allocation.
    case VMA_MEMORY_USAGE_AUTO:
    case VMA_MEMORY_USAGE_AUTO_PREFER_HOST:
      VK_THROW(
          "Only the VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE usage flag is compatible with create_allocation()");
      break;
    // Most of the time, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE will simply set the
    // DEVICE_LOCAL_BIT as a preferred memory flag. Therefore the below is a
    // decent approximation for VMA behaviour.
    case VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE:
      alloc_create_info.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      alloc_create_info.usage = VMA_MEMORY_USAGE_UNKNOWN;
      break;
    default:
      break;
  }

  return MemoryAllocation(allocator_, memory_requirements, alloc_create_info);
}

VulkanImage MemoryAllocator::create_image(
    const VkExtent3D& extents,
    const VkFormat image_format,
    const VkImageType image_type,
    const VkImageViewType image_view_type,
    const VulkanImage::SamplerProperties& sampler_props,
    VkSampler sampler,
    const bool allow_transfer,
    const bool allocate_memory) {
  VkImageUsageFlags usage =
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  if (allow_transfer) {
    usage |=
        (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
  }

  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  const VulkanImage::ImageProperties image_props{
      image_type,
      image_format,
      extents,
      usage,
  };

  const VulkanImage::ViewProperties view_props{
      image_view_type,
      image_format,
  };

  const VkImageLayout initial_layout = VK_IMAGE_LAYOUT_UNDEFINED;

  return VulkanImage(
      allocator_,
      alloc_create_info,
      image_props,
      view_props,
      sampler_props,
      initial_layout,
      sampler,
      allocate_memory);
}

VulkanBuffer MemoryAllocator::create_storage_buffer(
    const VkDeviceSize size,
    const bool gpu_only,
    const bool allocate_memory) {
  const VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;


  // The create storage buffer will be accessed by both the CPU and GPU, so set
  // the appropriate flags to indicate that the host device will be accessing
  // the data from this buffer.
  if (!gpu_only) {
    // Deferred memory allocation should only be used for GPU only buffers.
    VK_CHECK_COND(
        allocate_memory,
        "Only GPU-only buffers should use deferred memory allocation");

    alloc_create_info.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
    alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
    alloc_create_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    alloc_create_info.preferredFlags = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
        VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  }

  if (!gpu_only && tls_allocation_tag == AllocationTag::Unknown) {
    AllocationTagScope tag_scope(AllocationTag::Staging);
    return VulkanBuffer(
        allocator_, size, alloc_create_info, buffer_usage, allocate_memory);
  }

  return VulkanBuffer(
      allocator_, size, alloc_create_info, buffer_usage, allocate_memory);
}

VulkanBuffer MemoryAllocator::create_staging_buffer(const VkDeviceSize size) {
  AllocationTagScope tag_scope(AllocationTag::Staging);
  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

  VkBufferUsageFlags buffer_usage =
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

  return VulkanBuffer(allocator_, size, alloc_create_info, buffer_usage);
}

VulkanBuffer MemoryAllocator::create_uniform_buffer(const VkDeviceSize size) {
  AllocationTagScope tag_scope(AllocationTag::MetadataUniform);
  VmaAllocationCreateInfo alloc_create_info = {};
  alloc_create_info.flags = DEFAULT_ALLOCATION_STRATEGY |
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  alloc_create_info.usage = VMA_MEMORY_USAGE_AUTO;

  VkBufferUsageFlags buffer_usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;

  VulkanBuffer uniform_buffer(
      allocator_, size, alloc_create_info, buffer_usage);
  return uniform_buffer;
}

//
// VulkanFence
//

VulkanFence::VulkanFence()
    : device_(VK_NULL_HANDLE), handle_(VK_NULL_HANDLE), waiting_(false) {}

VulkanFence::VulkanFence(VkDevice device)
    : device_(device), handle_(VK_NULL_HANDLE), waiting_(VK_NULL_HANDLE) {
  const VkFenceCreateInfo fence_create_info{
      VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, // sType
      nullptr, // pNext
      0u, // flags
  };

  VK_CHECK(vkCreateFence(device_, &fence_create_info, nullptr, &handle_));
}

VulkanFence::VulkanFence(VulkanFence&& other) noexcept
    : device_(other.device_), handle_(other.handle_), waiting_(other.waiting_) {
  other.handle_ = VK_NULL_HANDLE;
  other.waiting_ = false;
}

VulkanFence& VulkanFence::operator=(VulkanFence&& other) noexcept {
  device_ = other.device_;
  handle_ = other.handle_;
  waiting_ = other.waiting_;

  other.device_ = VK_NULL_HANDLE;
  other.handle_ = VK_NULL_HANDLE;
  other.waiting_ = false;

  return *this;
}

VulkanFence::~VulkanFence() {
  if (VK_NULL_HANDLE == handle_) {
    return;
  }
  vkDestroyFence(device_, handle_, nullptr);
}

void VulkanFence::wait() {
  // if get_submit_handle() has not been called, then this will no-op
  if (waiting_) {
    VkResult fence_status = VK_NOT_READY;
    // Run the wait in a loop to keep the CPU hot. A single call to
    // vkWaitForFences with no timeout may cause the calling thread to be
    // scheduled out.
    do {
      // The timeout (last) arg is in units of ns
      fence_status = vkWaitForFences(device_, 1u, &handle_, VK_TRUE, 100000);

      VK_CHECK_COND(
          fence_status != VK_ERROR_DEVICE_LOST,
          "Vulkan Fence: Device lost while waiting for fence!");
    } while (fence_status != VK_SUCCESS);

    VK_CHECK(vkResetFences(device_, 1u, &handle_));

    waiting_ = false;
  }
}

} // namespace api
} // namespace vulkan
} // namespace native
} // namespace at
