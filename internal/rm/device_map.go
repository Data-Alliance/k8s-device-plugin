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

package rm

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/NVIDIA/go-nvlib/pkg/nvlib/device"
	"github.com/NVIDIA/go-nvlib/pkg/nvlib/info"
	"github.com/NVIDIA/go-nvml/pkg/nvml"
	"k8s.io/klog/v2"

	spec "github.com/NVIDIA/k8s-device-plugin/api/config/v1"
)

type deviceMapBuilder struct {
	device.Interface
	migStrategy         *string
	resources           *spec.Resources
	replicatedResources *spec.ReplicatedResources

	newGPUDevice func(i int, gpu nvml.Device) (string, deviceInfo)
}

// DeviceMap stores a set of devices per resource name.
type DeviceMap map[spec.ResourceName]Devices

// NewDeviceMap creates a device map for the specified NVML library and config.
func NewDeviceMap(infolib info.Interface, devicelib device.Interface, config *spec.Config) (DeviceMap, error) {
	b := deviceMapBuilder{
		Interface:           devicelib,
		migStrategy:         config.Flags.MigStrategy,
		resources:           &config.Resources,
		replicatedResources: config.Sharing.ReplicatedResources(),
		newGPUDevice:        newNvmlGPUDevice,
	}

	if infolib.ResolvePlatform() == info.PlatformWSL {
		b.newGPUDevice = newWslGPUDevice
	}

	return b.build()
}

// build builds a map of resource names to devices.
func (b *deviceMapBuilder) build() (DeviceMap, error) {
	devices, err := b.buildDeviceMapFromConfigResources()
	if err != nil {
		return nil, fmt.Errorf("error building device map from config.resources: %v", err)
	}
	devices, err = updateDeviceMapWithReplicas(b.replicatedResources, devices)
	if err != nil {
		return nil, fmt.Errorf("error updating device map with replicas from replicatedResources config: %v", err)
	}
	return devices, nil
}

// buildDeviceMapFromConfigResources builds a map of resource names to devices from spec.Config.Resources
func (b *deviceMapBuilder) buildDeviceMapFromConfigResources() (DeviceMap, error) {
	deviceMap, err := b.buildGPUDeviceMap()
	if err != nil {
		return nil, fmt.Errorf("error building GPU device map: %v", err)
	}

	if *b.migStrategy == spec.MigStrategyNone {
		return deviceMap, nil
	}

	migDeviceMap, err := b.buildMigDeviceMap()
	if err != nil {
		return nil, fmt.Errorf("error building MIG device map: %v", err)
	}

	var requireUniformMIGDevices bool
	if *b.migStrategy == spec.MigStrategySingle {
		requireUniformMIGDevices = true
	}

	err = b.assertAllMigDevicesAreValid(requireUniformMIGDevices)
	if err != nil {
		return nil, fmt.Errorf("invalid MIG configuration: %v", err)
	}

	if requireUniformMIGDevices && !deviceMap.isEmpty() && !migDeviceMap.isEmpty() {
		return nil, fmt.Errorf("all devices on the node must be configured with the same migEnabled value")
	}

	deviceMap.merge(migDeviceMap)

	return deviceMap, nil
}

// buildGPUDeviceMap builds a map of resource names to GPU devices
func (b *deviceMapBuilder) buildGPUDeviceMap() (DeviceMap, error) {
	devices := make(DeviceMap)
	var errors []error
	successCount := 0

	// Check if gpu-select file exists and read it
	const gpuSelectPath = "/var/lib/kubelet/device-plugins/gpu-select"
	selectedIndex, selectedUUID, gpuSelectErr := readGPUSelectFile(gpuSelectPath)

	// Get device count by getting all devices
	allDevices, devErr := b.GetDevices()
	if devErr != nil {
		klog.Warningf("Failed to get devices for counting: %v", devErr)
	}
	count := len(allDevices)

	// Determine if we should filter devices
	shouldFilter := gpuSelectErr == nil && count >= 2 && selectedIndex >= 0 && selectedUUID != ""
	if shouldFilter {
		klog.Infof("GPU filtering enabled: will only use GPU %d (UUID: %s)", selectedIndex, selectedUUID)
	}

	err := b.VisitDevices(func(i int, gpu device.Device) error {
		// Apply GPU selection filter if enabled
		if shouldFilter {
			deviceIndex, ret := gpu.GetIndex()
			if ret != nvml.SUCCESS {
				klog.Warningf("Failed to get device index for GPU %d: %v", i, ret)
				return nil // Continue with other GPUs
			}

			deviceUUID, ret := gpu.GetUUID()
			if ret != nvml.SUCCESS {
				klog.Warningf("Failed to get device UUID for GPU %d: %v", i, ret)
				return nil // Continue with other GPUs
			}

			// Skip if this device doesn't match the selected index and UUID
			if deviceIndex != selectedIndex || deviceUUID != selectedUUID {
				klog.Infof("Skipping GPU %d (UUID: %s) - not matching selected GPU (index: %d, UUID: %s)",
					deviceIndex, deviceUUID, selectedIndex, selectedUUID)
				return nil // Continue with other GPUs
			}

			klog.Infof("GPU %d (UUID: %s) matches gpu-select file", deviceIndex, deviceUUID)
		}

		name, ret := gpu.GetName()
		if ret != nvml.SUCCESS {
			err := fmt.Errorf("error getting product name for GPU %d: %v", i, ret)
			errors = append(errors, err)
			klog.Warningf("Skipping GPU %d due to error: %v", i, err)
			return nil // Continue with other GPUs
		}
		migEnabled, err := gpu.IsMigEnabled()
		if err != nil {
			err := fmt.Errorf("error checking if MIG is enabled on GPU %d: %v", i, err)
			errors = append(errors, err)
			klog.Warningf("Skipping GPU %d due to MIG check error: %v", i, err)
			return nil // Continue with other GPUs
		}
		if migEnabled && *b.migStrategy != spec.MigStrategyNone {
			return nil
		}

		matched := false
		for _, resource := range b.resources.GPUs {
			if resource.Pattern.Matches(name) {
				index, info := b.newGPUDevice(i, gpu)
				if err := devices.setEntry(resource.Name, index, info); err != nil {
					err := fmt.Errorf("error setting device entry for GPU %d: %v", i, err)
					errors = append(errors, err)
					klog.Warningf("Skipping GPU %d due to device entry error: %v", i, err)
					return nil // Continue with other GPUs
				}
				matched = true
				successCount++
				klog.Infof("Successfully added GPU %d (%s) to resource %s", i, name, resource.Name)
				break
			}
		}

		if !matched {
			err := fmt.Errorf("GPU %d name '%v' does not match any resource patterns", i, name)
			errors = append(errors, err)
			klog.Warningf("Skipping GPU %d: %v", i, err)
		}

		return nil // Always continue with other GPUs
	})

	// Log summary of GPU processing
	if len(errors) > 0 {
		klog.Warningf("Encountered %d errors while processing GPUs, but %d GPUs were successfully added", len(errors), successCount)
		for _, err := range errors {
			klog.Warningf("GPU processing error: %v", err)
		}
	}

	if successCount > 0 {
		klog.Infof("Successfully processed %d GPUs", successCount)
		return devices, err // Return the original VisitDevices error if any
	}

	// If no GPUs were successfully processed, return an error
	if len(errors) > 0 {
		return devices, fmt.Errorf("failed to process any GPUs: %d errors encountered", len(errors))
	}

	return devices, err
}

// buildMigDeviceMap builds a map of resource names to MIG devices
func (b *deviceMapBuilder) buildMigDeviceMap() (DeviceMap, error) {
	devices := make(DeviceMap)
	var errors []error
	successCount := 0

	err := b.VisitMigDevices(func(i int, d device.Device, j int, mig device.MigDevice) error {
		migProfile, err := mig.GetProfile()
		if err != nil {
			err := fmt.Errorf("error getting MIG profile for MIG device at index '(%v, %v)': %v", i, j, err)
			errors = append(errors, err)
			klog.Warningf("Skipping MIG device (%d, %d) due to profile error: %v", i, j, err)
			return nil // Continue with other MIG devices
		}

		matched := false
		for _, resource := range b.resources.MIGs {
			if resource.Pattern.Matches(migProfile.String()) {
				index, info := newMigDevice(i, j, mig)
				if err := devices.setEntry(resource.Name, index, info); err != nil {
					err := fmt.Errorf("error setting device entry for MIG device (%d, %d): %v", i, j, err)
					errors = append(errors, err)
					klog.Warningf("Skipping MIG device (%d, %d) due to device entry error: %v", i, j, err)
					return nil // Continue with other MIG devices
				}
				matched = true
				successCount++
				klog.Infof("Successfully added MIG device (%d, %d) with profile %s to resource %s", i, j, migProfile.String(), resource.Name)
				break
			}
		}

		if !matched {
			err := fmt.Errorf("MIG device (%d, %d) profile '%v' does not match any resource patterns", i, j, migProfile)
			errors = append(errors, err)
			klog.Warningf("Skipping MIG device (%d, %d): %v", i, j, err)
		}

		return nil // Always continue with other MIG devices
	})

	// Log summary of MIG device processing
	if len(errors) > 0 {
		klog.Warningf("Encountered %d errors while processing MIG devices, but %d MIG devices were successfully added", len(errors), successCount)
		for _, err := range errors {
			klog.Warningf("MIG device processing error: %v", err)
		}
	}

	if successCount > 0 {
		klog.Infof("Successfully processed %d MIG devices", successCount)
	}

	return devices, err
}

// assertAllMigDevicesAreValid ensures that each MIG-enabled device has at least one MIG device
// associated with it.
func (b *deviceMapBuilder) assertAllMigDevicesAreValid(uniform bool) error {
	err := b.VisitDevices(func(i int, d device.Device) error {
		isMigEnabled, err := d.IsMigEnabled()
		if err != nil {
			return err
		}
		if !isMigEnabled {
			return nil
		}
		migDevices, err := d.GetMigDevices()
		if err != nil {
			return err
		}
		if uniform && len(migDevices) == 0 {
			return fmt.Errorf("device %v has no MIG devices configured", i)
		}
		if !uniform && len(migDevices) == 0 {
			klog.Warningf("device %v has no MIG devices configured", i)
		}
		return nil
	})
	if err != nil {
		return fmt.Errorf("at least one device with migEnabled=true was not configured correctly: %v", err)
	}

	if !uniform {
		return nil
	}

	var previousAttributes *nvml.DeviceAttributes
	return b.VisitMigDevices(func(i int, d device.Device, j int, m device.MigDevice) error {
		attrs, ret := m.GetAttributes()
		if ret != nvml.SUCCESS {
			return fmt.Errorf("error getting device attributes: %v", ret)
		}
		if previousAttributes == nil {
			previousAttributes = &attrs
		} else if attrs != *previousAttributes {
			return fmt.Errorf("more than one MIG device type present on node")
		}

		return nil
	})
}

// setEntry sets the DeviceMap entry for the specified resource
func (d DeviceMap) setEntry(name spec.ResourceName, index string, device deviceInfo) error {
	dev, err := BuildDevice(index, device)
	if err != nil {
		return fmt.Errorf("error building Device: %v", err)
	}
	d.insert(name, dev)
	return nil
}

// insert adds the specified device to the device map
func (d DeviceMap) insert(name spec.ResourceName, dev *Device) {
	if d[name] == nil {
		d[name] = make(Devices)
	}
	d[name][dev.ID] = dev
}

// merge merges two devices maps
func (d DeviceMap) merge(o DeviceMap) {
	for name, devices := range o {
		for _, device := range devices {
			d.insert(name, device)
		}
	}
}

// isEmpty checks whether a device map is empty
func (d DeviceMap) isEmpty() bool {
	for _, devices := range d {
		if len(devices) > 0 {
			return false
		}
	}
	return true
}

// getIDsOfDevicesToReplicate returns a list of dervice IDs that we want to replicate.
func (d DeviceMap) getIDsOfDevicesToReplicate(r *spec.ReplicatedResource) ([]string, error) {
	devices, exists := d[r.Name]
	if !exists {
		return nil, nil
	}

	// If all devices for this resource type are to be replicated.
	if r.Devices.All {
		return devices.GetIDs(), nil
	}

	// If a specific number of devices for this resource type are to be replicated.
	if r.Devices.Count > 0 {
		if r.Devices.Count > len(devices) {
			return nil, fmt.Errorf("requested %d devices to be replicated, but only %d devices available", r.Devices.Count, len(devices))
		}
		return devices.GetIDs()[:r.Devices.Count], nil
	}

	// If a specific set of devices for this resource type are to be replicated.
	if len(r.Devices.List) > 0 {
		var ids []string
		for _, ref := range r.Devices.List {
			if ref.IsUUID() {
				d := devices.GetByID(string(ref))
				if d == nil {
					return nil, fmt.Errorf("no matching device with UUID: %v", ref)
				}
				ids = append(ids, d.ID)
			}
			if ref.IsGPUIndex() || ref.IsMigIndex() {
				d := devices.GetByIndex(string(ref))
				if d == nil {
					return nil, fmt.Errorf("no matching device at index: %v", ref)
				}
				ids = append(ids, d.ID)
			}
		}
		return ids, nil
	}

	return nil, fmt.Errorf("unexpected error")
}

// updateDeviceMapWithReplicas returns an updated map of resource names to devices with replica
// information from the active replicated resources config.
func updateDeviceMapWithReplicas(replicatedResources *spec.ReplicatedResources, oDevices DeviceMap) (DeviceMap, error) {
	devices := make(DeviceMap)

	// Begin by walking replicatedResources.Resources and building a map of just the resource names.
	names := make(map[spec.ResourceName]bool)
	for _, r := range replicatedResources.Resources {
		names[r.Name] = true
	}

	// Copy over all devices from oDevices without a resource reference in TimeSlicing.Resources.
	for r, ds := range oDevices {
		if !names[r] {
			devices[r] = ds
		}
	}

	// Walk shared Resources and update devices in the device map as appropriate.
	for _, resource := range replicatedResources.Resources {
		r := resource
		// Get the IDs of the devices we want to replicate from oDevices
		ids, err := oDevices.getIDsOfDevicesToReplicate(&r)
		if err != nil {
			return nil, fmt.Errorf("unable to get IDs of devices to replicate for '%v' resource: %v", r.Name, err)
		}
		// Skip any resources not matched in oDevices
		if len(ids) == 0 {
			continue
		}

		// Add any devices we don't want replicated directly into the device map.
		for _, d := range oDevices[r.Name].Difference(oDevices[r.Name].Subset(ids)) {
			devices.insert(r.Name, d)
		}

		// Create replicated devices add them to the device map.
		// Rename the resource for replicated devices as requested.
		name := r.Name
		if r.Rename != "" {
			name = r.Rename
		}
		for _, id := range ids {
			for i := 0; i < r.Replicas; i++ {
				annotatedID := string(NewAnnotatedID(id, i))
				replicatedDevice := *(oDevices[r.Name][id])
				replicatedDevice.ID = annotatedID
				replicatedDevice.Replicas = r.Replicas
				devices.insert(name, &replicatedDevice)
			}
		}
	}

	return devices, nil
}

// readGPUSelectFile reads the gpu-select file and returns the index and UUID
// The file format is "index uuid" (e.g., "0 GPU-12345678-1234-1234-1234-123456789012")
// Returns -1, empty string and error if file doesn't exist or is invalid
func readGPUSelectFile(path string) (int, string, error) {
	file, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			klog.Infof("GPU select file %s does not exist, using all GPUs", path)
			return -1, "", err
		}
		klog.Warningf("Failed to open GPU select file %s: %v", path, err)
		return -1, "", err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	if !scanner.Scan() {
		err := fmt.Errorf("GPU select file %s is empty", path)
		klog.Warningf("%v", err)
		return -1, "", err
	}

	line := strings.TrimSpace(scanner.Text())
	parts := strings.Fields(line)
	if len(parts) != 2 {
		err := fmt.Errorf("invalid format in GPU select file %s: expected 'index uuid', got '%s'", path, line)
		klog.Warningf("%v", err)
		return -1, "", err
	}

	index, err := strconv.Atoi(parts[0])
	if err != nil {
		err := fmt.Errorf("invalid index in GPU select file %s: %v", path, err)
		klog.Warningf("%v", err)
		return -1, "", err
	}

	uuid := parts[1]
	klog.Infof("Read GPU select file: index=%d, uuid=%s", index, uuid)
	return index, uuid, nil
}
