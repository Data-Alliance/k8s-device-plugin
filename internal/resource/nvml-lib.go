/**
# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
**/

package resource

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/NVIDIA/go-nvlib/pkg/nvlib/device"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"k8s.io/klog/v2"
)

type nvmlLib struct {
	nvml.Interface
	devicelib device.Interface
}

// NewNVMLManager creates a new manager that uses NVML to query and manage devices
func NewNVMLManager(nvmllib nvml.Interface, devicelib device.Interface) Manager {
	m := nvmlLib{
		Interface: nvmllib,
		devicelib: devicelib,
	}
	return m
}

// GetCudaDriverVersion : Return the cuda v using NVML
func (l nvmlLib) GetCudaDriverVersion() (int, int, error) {
	v, ret := l.SystemGetCudaDriverVersion()
	if ret != nvml.SUCCESS {
		return 0, 0, ret
	}
	major := v / 1000
	minor := v % 1000 / 10

	return major, minor, nil
}

// GetDevices returns the NVML devices for the manager
func (l nvmlLib) GetDevices() ([]Device, error) {
	libdevices, err := l.devicelib.GetDevices()
	if err != nil {
		return nil, err
	}

	// Check if gpu-select file exists and filter devices if needed
	const gpuSelectPath = "/var/lib/kubelet/device-plugins/gpu-select"
	selectedUUID, err := readGPUSelectFile(gpuSelectPath)

	// Get device count
	count, ret := l.DeviceGetCount()
	if ret != nvml.SUCCESS {
		klog.Warningf("Failed to get device count: %v", ret)
	}

	var devices []Device
	for _, d := range libdevices {
		// If gpu-select file exists, device count >= 2, filter devices by UUID only
		if err == nil && count >= 2 && selectedUUID != "" {
			// Get device UUID
			deviceUUID, ret := d.GetUUID()
			if ret != nvml.SUCCESS {
				klog.Warningf("Failed to get device UUID: %v", ret)
				continue
			}

			// Check if this device matches the selected UUID
			if deviceUUID != selectedUUID {
				klog.Infof("Skipping GPU device (UUID: %s) - not matching selected GPU (UUID: %s)",
					deviceUUID, selectedUUID)
				continue
			}

			klog.Infof("Selected GPU device (UUID: %s) matches gpu-select file", deviceUUID)
		}

		device := nvmlDevice{
			Device:    d,
			devicelib: l.devicelib,
		}
		devices = append(devices, device)
	}

	return devices, nil
}

// readGPUSelectFile reads the gpu-select file and returns the UUID
// The file format is just "uuid" (e.g., "GPU-12345678-1234-1234-1234-123456789012")
// Returns empty string and error if file doesn't exist or is invalid
func readGPUSelectFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			klog.Infof("GPU select file %s does not exist, using all GPUs", path)
			return "", err
		}
		klog.Warningf("Failed to open GPU select file %s: %v", path, err)
		return "", err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	if !scanner.Scan() {
		err := fmt.Errorf("GPU select file %s is empty", path)
		klog.Warningf("%v", err)
		return "", err
	}

	line := strings.TrimSpace(scanner.Text())
	if line == "" {
		err := fmt.Errorf("GPU select file %s contains only whitespace", path)
		klog.Warningf("%v", err)
		return "", err
	}

	uuid := line
	klog.Infof("Read GPU select file: uuid=%s", uuid)
	return uuid, nil
}

// GetDriverVersion returns the driver version
func (l nvmlLib) GetDriverVersion() (string, error) {
	v, ret := l.SystemGetDriverVersion()
	if ret != nvml.SUCCESS {
		return "", ret
	}
	return v, nil
}

// Init initialises the library
func (l nvmlLib) Init() error {
	ret := l.Interface.Init()
	if ret != nvml.SUCCESS {
		return ret
	}
	return nil
}

// Shutdown shuts down the library
func (l nvmlLib) Shutdown() error {
	ret := l.Interface.Shutdown()
	if ret != nvml.SUCCESS {
		return ret
	}
	return nil
}
