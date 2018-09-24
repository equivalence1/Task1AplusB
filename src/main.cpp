#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <vector>
#include <sstream>
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <cassert>


template <typename T>
std::string to_string(T value)
{
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line)
{
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

cl_device_type getDevType(cl_device_id device)
{
    cl_device_type devType;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(devType), &devType, nullptr));
    return devType;
}

bool isGPU(cl_device_id device)
{
    return (getDevType(device) & CL_DEVICE_TYPE_GPU);
}

bool isCPU(cl_device_id device)
{
    return (getDevType(device) & CL_DEVICE_TYPE_CPU);
}

cl_device_id getBestDevice()
{
    cl_device_id cpu = nullptr;

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            cl_device_id device = devices[deviceIndex];
            if (isGPU(device)) {
                return device;
            } else if (isCPU(device)) {
                cpu = device;
            }
        }
    }

    if (cpu == nullptr) {
        throw std::runtime_error("No devices supporting OpenCL were found");
    }

    return cpu;
}

cl_platform_id getDevicePlatformId(cl_device_id device)
{
    cl_platform_id platformId;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(platformId), &platformId, nullptr));
    return platformId;
}

void printDeviceInfo(cl_device_id device)
{
    cl_platform_id platform = getDevicePlatformId(device);
    size_t platformNameSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
    std::vector<unsigned char> platformName(platformNameSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
    std::cout << "    Platform name: " << platformName.data() << std::endl;

    size_t deviceNameSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
    std::vector<unsigned char> deviceName(deviceNameSize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
    std::cout << "    Device name: " << deviceName.data() << std::endl;
}

cl_context createDeviceContext(cl_device_id device)
{
    cl_context ctx;
    cl_int errcode_ret;
    cl_platform_id platformId = getDevicePlatformId(device);
    const cl_context_properties properties[] = {CL_CONTEXT_PLATFORM, (intptr_t)platformId, 0};
    ctx = clCreateContext(properties, 1, &device, nullptr, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    return ctx;
}

int main()
{    
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_device_id device = getBestDevice();

    std::cout << "Using device:" << std::endl;
    printDeviceInfo(device);

    cl_context ctx = createDeviceContext(device);

    cl_int err;
    cl_command_queue queue = clCreateCommandQueue(ctx, device, 0, &err);
    OCL_SAFE_CALL(err);

    unsigned int n = 100*1000*1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    cl_mem aGpu = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, as.data(), &err);
    OCL_SAFE_CALL(err);
    cl_mem bGpu = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, bs.data(), &err);
    OCL_SAFE_CALL(err);
    cl_mem cGpu = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,                       sizeof(float) * n, nullptr, &err);
    OCL_SAFE_CALL(err);

    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        // std::cout << kernel_sources << std::endl;
    }

    const char *sources = kernel_sources.c_str();
    const size_t lengths = kernel_sources.size();
    cl_program program = clCreateProgramWithSource(ctx, 1, &sources, &lengths, &err);
    OCL_SAFE_CALL(err);

    OCL_SAFE_CALL(clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));

    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }

    cl_kernel kernel = clCreateKernel(program, "aplusb", &err);
    OCL_SAFE_CALL(err);

    {
         unsigned int i = 0;
         OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(aGpu), &aGpu));
         OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(bGpu), &bGpu));
         OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cGpu), &cGpu));
         OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(n), &n));
        (void)sizeof(i);
    }

    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        timer t; // Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &event));
            clWaitForEvents(1, &event);
            clReleaseEvent(event);
            t.nextLap(); // При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считаются не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклониение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще) достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GFlops: " << n / (1e9) / t.lapAvg() << std::endl;
        std::cout << "VRAM bandwidth: " << sizeof(float) * n * 3.0 / (1ULL << 30) / t.lapAvg() << " GB/s" << std::endl;
    }

    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, cGpu, CL_TRUE, 0, n * sizeof(float), cs.data(), 0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << sizeof(float) * n * 1.0 / (1ULL << 30) / t.lapAvg() << " GB/s" << std::endl;
    }

    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            std::cout << "i = " << i << ", c[i] = " << cs[i] << ", a[i] = " << as[i] << ", b[i] = " << bs[i] << std::endl;
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseKernel(kernel);
    clReleaseMemObject(aGpu);
    clReleaseMemObject(bGpu);
    clReleaseMemObject(cGpu);
    clReleaseContext(ctx);

    return 0;
}
