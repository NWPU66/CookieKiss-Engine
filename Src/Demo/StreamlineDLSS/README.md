# DLSS Super Resolution and DLSS Frame Generation via Streamline

This sample demonstrates integration of Streamline into a Vulkan-based application and using it to add NVIDIA Reflex, DLSS Super Resolution, and DLSS Frame Generation.

Streamline generally supports two methods of integrating it, either automatically by linking against the sl.interposer.lib library instead of vulkan-1.lib (assuming the application was previously statically linking Vulkan already), or manually by only retrieving the relevant Vulkan entry points from Streamline and having the rest go directly through Vulkan as usual (an application can use the vkGetInstanceProcAddr/vkGetDeviceProcAddr provided by Streamline to fill its dispatch tables). This sample implements both methods, which can be toggled between via a CMake option called STREAMLINE_MANUAL_HOOKING.
When enabled, the sample will link against Vulkan normally, dynamically load Streamline at runtime and only get the required Vulkan functions from it to call. When disabled, the sample will link against sl.interposer.lib instead of vulkan-1.lib.
The manual hooking method can offer better performance because of less overhead (avoids having to redirect all Vulkan calls through Streamline) and is not too difficult to implement when an application already loads all the Vulkan entry points dynamically. It does however also require querying and adding all the necessary Vulkan extensions and features Streamline wants, while the automatic method will add those during device creation behind the scenes without changes to the application.

To debug DLSS, you can replace the DLSS DLLs installed by CMake with their development variants and enable their overlay with a special registry key.
In addition, check out the Streamline ImGui plugin documentation. It can be enabled by adding sl::kFeatureImGUI to the SL_FEATURES array at the top of main.cpp.

本示例演示了将 Streamline 集成到基于 Vulkan 的应用程序中，并使用它添加 NVIDIA Reflex、DLSS 超分辨率和 DLSS 帧生成。

Streamline 通常支持两种集成方法，第一种是自动集成，通过链接 sl.interposer.lib 库而不是 vulkan-1.lib（假设应用程序之前已经静态链接了 Vulkan）；第二种是手动集成，仅从 Streamline 获取相关的 Vulkan 入口点，并让其余部分像往常一样直接通过 Vulkan（应用程序可以使用 Streamline 提供的 vkGetInstanceProcAddr/vkGetDeviceProcAddr 来填充其调度表）。此示例实现了这两种方法，可以通过名为 STREAMLINE_MANUAL_HOOKING 的 CMake 选项进行切换。

启用时，示例将正常链接 Vulkan，动态加载 Streamline 并仅获取所需的 Vulkan 函数进行调用。禁用时，示例将链接 sl.interposer.lib 而不是 vulkan-1.lib。
手动钩挂方法可以提供更好的性能，因为开销较少（避免将所有 Vulkan 调用重定向通过 Streamline），并且在应用程序已经动态加载所有 Vulkan 入口点时实现起来也不太困难。然而，这也需要查询并添加 Streamline 所需的所有 Vulkan 扩展和特性，而自动方法将在设备创建过程中在后台添加这些内容，而无需对应用程序进行更改。

要调试 DLSS，您可以用其开发版本替换 CMake 安装的 DLSS DLL，并通过特殊注册表键启用其覆盖。
此外，请查看 Streamline ImGui 插件文档。可以通过在 main.cpp 顶部的 SL_FEATURES 数组中添加 sl::kFeatureImGUI 来启用它。
