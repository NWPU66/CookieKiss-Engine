/**
 * @file streamline_wrapper.hpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-11-28
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

// c

// cpp

// 3rdparty - vulkan
#include "vulkan/vulkan_core.h"

// 3rdparty - streamline
#include "Streamline/include/sl.h"
#include "Streamline/include/sl_dlss.h"
#include "Streamline/include/sl_dlss_g.h"
#include "Streamline/include/sl_helpers_vk.h"
#include "Streamline/include/sl_pcl.h"
#include "Streamline/include/sl_reflex.h"

// users

#if SL_MANUAL_HOOKING

#    include "Streamline/include/sl_security.h"

class StreamlineWrapper {
public:
    struct FunctionTableSL
    {
        PFun_slInit*                   Init;
        PFun_slShutdown*               Shutdown;
        PFun_slIsFeatureSupported*     IsFeatureSupported;
        PFun_slSetTag*                 SetTag;
        PFun_slSetConstants*           SetConstants;
        PFun_slGetFeatureRequirements* GetFeatureRequirements;
        PFun_slEvaluateFeature*        EvaluateFeature;
        PFun_slGetFeatureFunction*     GetFeatureFunction;
        PFun_slGetNewFrameToken*       GetNewFrameToken;
        PFun_slSetVulkanInfo*          SetVulkanInfo;
    } sl = {};

    struct FunctionTableVK
    {
        PFN_vkCreateSwapchainKHR    CreateSwapchainKHR;
        PFN_vkDestroySwapchainKHR   DestroySwapchainKHR;
        PFN_vkGetSwapchainImagesKHR GetSwapchainImagesKHR;
        PFN_vkAcquireNextImageKHR   AcquireNextImageKHR;
        PFN_vkQueuePresentKHR       QueuePresentKHR;
    } vk = {};

    static StreamlineWrapper& get()
    {
        static StreamlineWrapper wrapper;
        return wrapper;
    }

    bool hasLoaded() const { return m_interposerModule != nullptr; }

    void initFunctions()
    {
        sl.Init = reinterpret_cast<PFun_slInit*>(GetProcAddress(m_interposerModule, "slInit"));
        sl.Shutdown =
            reinterpret_cast<PFun_slShutdown*>(GetProcAddress(m_interposerModule, "slShutdown"));
        sl.IsFeatureSupported = reinterpret_cast<PFun_slIsFeatureSupported*>(
            GetProcAddress(m_interposerModule, "slIsFeatureSupported"));
        sl.SetTag =
            reinterpret_cast<PFun_slSetTag*>(GetProcAddress(m_interposerModule, "slSetTag"));
        sl.SetConstants = reinterpret_cast<PFun_slSetConstants*>(
            GetProcAddress(m_interposerModule, "slSetConstants"));
        sl.GetFeatureRequirements = reinterpret_cast<PFun_slGetFeatureRequirements*>(
            GetProcAddress(m_interposerModule, "slGetFeatureRequirements"));
        sl.EvaluateFeature = reinterpret_cast<PFun_slEvaluateFeature*>(
            GetProcAddress(m_interposerModule, "slEvaluateFeature"));
        sl.GetFeatureFunction = reinterpret_cast<PFun_slGetFeatureFunction*>(
            GetProcAddress(m_interposerModule, "slGetFeatureFunction"));
        sl.GetNewFrameToken = reinterpret_cast<PFun_slGetNewFrameToken*>(
            GetProcAddress(m_interposerModule, "slGetNewFrameToken"));
        sl.SetVulkanInfo = reinterpret_cast<PFun_slSetVulkanInfo*>(
            GetProcAddress(m_interposerModule, "slSetVulkanInfo"));
    }

    void initVulkanHooks(VkDevice device)
    {
        const auto getDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
            GetProcAddress(m_interposerModule, "vkGetDeviceProcAddr"));

        vk.CreateSwapchainKHR = reinterpret_cast<PFN_vkCreateSwapchainKHR>(
            getDeviceProcAddr(device, "vkCreateSwapchainKHR"));
        vk.DestroySwapchainKHR = reinterpret_cast<PFN_vkDestroySwapchainKHR>(
            getDeviceProcAddr(device, "vkDestroySwapchainKHR"));
        vk.GetSwapchainImagesKHR = reinterpret_cast<PFN_vkGetSwapchainImagesKHR>(
            getDeviceProcAddr(device, "vkGetSwapchainImagesKHR"));
        vk.AcquireNextImageKHR = reinterpret_cast<PFN_vkAcquireNextImageKHR>(
            getDeviceProcAddr(device, "vkAcquireNextImageKHR"));
        vk.QueuePresentKHR =
            reinterpret_cast<PFN_vkQueuePresentKHR>(getDeviceProcAddr(device, "vkQueuePresentKHR"));
    }

private:
    HMODULE m_interposerModule;

    StreamlineWrapper()
    {
        // Get absolute path to interposer DLL
        std::wstring interposerModulePath(MAX_PATH, L'\0');
        interposerModulePath.resize(GetModuleFileNameW(
            nullptr, interposerModulePath.data(), static_cast<DWORD>(interposerModulePath.size())));
        interposerModulePath.erase(interposerModulePath.rfind('\\'));
        interposerModulePath.append(L"\\sl.interposer.dll");

#    if 0
    // Optionally verify that the interposer DLL is signed by NVIDIA
    if (!sl::security::verifyEmbeddedSignature(interposerModulePath.c_str()))
      return;
#    endif

        m_interposerModule = LoadLibraryW(interposerModulePath.c_str());
        if (hasLoaded()) { initFunctions(); }
    }

    ~StreamlineWrapper()
    {
        if (hasLoaded()) { FreeLibrary(m_interposerModule); }
    }

    StreamlineWrapper(const StreamlineWrapper&)            = delete;
    StreamlineWrapper& operator=(const StreamlineWrapper&) = delete;
};

// Dynamically load Streamline functions on first call
inline sl::Result slInit(const sl::Preferences& pref, uint64_t sdkVersion)
{
    if (!StreamlineWrapper::get().hasLoaded()) { return sl::Result::eErrorNotInitialized; }

    return StreamlineWrapper::get().sl.Init(pref, sdkVersion);
}
inline sl::Result slShutdown()
{
    return StreamlineWrapper::get().sl.Shutdown();
}
inline sl::Result slIsFeatureSupported(sl::Feature feature, const sl::AdapterInfo& adapterInfo)
{
    return StreamlineWrapper::get().sl.IsFeatureSupported(feature, adapterInfo);
}
inline sl::Result slSetTag(const sl::ViewportHandle& viewport,
                           const sl::ResourceTag*    tags,
                           uint32_t                  numTags,
                           sl::CommandBuffer*        cmdBuffer)
{
    return StreamlineWrapper::get().sl.SetTag(viewport, tags, numTags, cmdBuffer);
}
inline sl::Result slSetConstants(const sl::Constants&      values,
                                 const sl::FrameToken&     frame,
                                 const sl::ViewportHandle& viewport)
{
    return StreamlineWrapper::get().sl.SetConstants(values, frame, viewport);
}
inline sl::Result slGetFeatureRequirements(sl::Feature              feature,
                                           sl::FeatureRequirements& requirements)
{
    return StreamlineWrapper::get().sl.GetFeatureRequirements(feature, requirements);
}
inline sl::Result slEvaluateFeature(sl::Feature               feature,
                                    const sl::FrameToken&     frame,
                                    const sl::BaseStructure** inputs,
                                    uint32_t                  numInputs,
                                    sl::CommandBuffer*        cmdBuffer)
{
    return StreamlineWrapper::get().sl.EvaluateFeature(feature, frame, inputs, numInputs,
                                                       cmdBuffer);
}
inline sl::Result
slGetFeatureFunction(sl::Feature feature, const char* functionName, void*& function)
{
    return StreamlineWrapper::get().sl.GetFeatureFunction(feature, functionName, function);
}
inline sl::Result slGetNewFrameToken(sl::FrameToken*& token, const uint32_t* frameIndex)
{
    return StreamlineWrapper::get().sl.GetNewFrameToken(token, const_cast<uint32_t*>(frameIndex));
}
inline sl::Result slSetVulkanInfo(const sl::VulkanInfo& info)
{
    return StreamlineWrapper::get().sl.SetVulkanInfo(info);
}

// 重写链接的 Vulkan 函数，使其重定向到 Streamline
// 在实际应用中，应该直接使用函数指针，但为了简化，本示例链接了 vulkan-1.lib，因此需要重写这些导入
extern VkResult VKAPI_CALL vkCreateSwapchainKHR(VkDevice                        device,
                                                const VkSwapchainCreateInfoKHR* pCreateInfo,
                                                const VkAllocationCallbacks*    pAllocator,
                                                VkSwapchainKHR*                 pSwapchain)
{
    return StreamlineWrapper::get().vk.CreateSwapchainKHR(device, pCreateInfo, pAllocator,
                                                          pSwapchain);
}
extern void VKAPI_CALL vkDestroySwapchainKHR(VkDevice                     device,
                                             VkSwapchainKHR               swapchain,
                                             const VkAllocationCallbacks* pAllocator)
{
    StreamlineWrapper::get().vk.DestroySwapchainKHR(device, swapchain, pAllocator);
}
extern VkResult VKAPI_CALL vkGetSwapchainImagesKHR(VkDevice       device,
                                                   VkSwapchainKHR swapchain,
                                                   uint32_t*      pSwapchainImageCount,
                                                   VkImage*       pSwapchainImages)
{
    return StreamlineWrapper::get().vk.GetSwapchainImagesKHR(
        device, swapchain, pSwapchainImageCount, pSwapchainImages);
}
extern VkResult VKAPI_CALL vkAcquireNextImageKHR(VkDevice       device,
                                                 VkSwapchainKHR swapchain,
                                                 uint64_t       timeout,
                                                 VkSemaphore    semaphore,
                                                 VkFence        fence,
                                                 uint32_t*      pImageIndex)
{
    return StreamlineWrapper::get().vk.AcquireNextImageKHR(device, swapchain, timeout, semaphore,
                                                           fence, pImageIndex);
}
extern VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo)
{
    return StreamlineWrapper::get().vk.QueuePresentKHR(queue, pPresentInfo);
}

// NOTE - 函数拦截与重定向
// 这里的extern关键字用来做函数拦截，链接到vulkan-1.lib动态库时
// 自定义的实现在动态库之前加载，被优先使用，以达到拦截函数的目的

#endif
