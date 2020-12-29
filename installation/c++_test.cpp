#include <iostream>
#include "onnxruntime_cxx_api.h"

#define USE_CPU // Chnage CPU to USE_CUDA, USE_TENSORRT or USE_OPENVINO

#ifdef USE_CUDA
#include "cuda_provider_factory.h"
#endif  // CUDA GPU Enabled
#ifdef USE_TENSORRT
#include "tensorrt_provider_factory.h"
#endif  // TensorRT GPU Enabled
#ifdef USE_OPENVINO
#include "openvino_provider_factory.h"
#endif  // OpenVINO Enabled

int main()
{
	auto providers = Ort::GetAvailableProviders();
	for (auto provider : providers)
		std::cout << provider << std::endl;

	Ort::Env env = Ort::Env{ ORT_LOGGING_LEVEL_VERBOSE, "Default" };
	Ort::SessionOptions session_options;

#ifdef USE_CUDA
	Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#endif  // CUDA GPU Enabled
#ifdef USE_TENSORRT
	Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0));
#endif  // TensorRT GPU Enabled
#ifdef USE_OPENVINO
	Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_OpenVINO(session_options, ""));
#endif  // OpenVINO Enabled

	Ort::Session session(env, L"D:\\ibrahim\\MobileNetV2.onnx", session_options);
	std::cout << "Model Loaded Successfully!\n";
	system("PAUSE");
}

