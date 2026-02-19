using UnityEngine;
using UnityEngine.Rendering;

namespace VoxelEngine
{
    /// <summary>
    /// Handles throttled CPU readback of GPU voxel data, supporting both 
    /// synchronous and async readback with double-buffered slots.
    /// </summary>
    public class VoxelCpuReadback
    {
        private const int SlotCount = 2;
        private uint[][] _slots = new uint[SlotCount][];
        private int _frontSlot;
        private int _inFlightSlot = -1;
        private float _lastReadbackTime = -999f;
        private bool _ready;
        private bool _requestPending;

        public bool IsReady => _ready;

        /// <summary>
        /// Call once per frame to keep the readback cache up-to-date.
        /// </summary>
        public void UpdateCache(GraphicsBuffer readBuffer, int totalVoxels,
            float readbackInterval, bool useAsync)
        {
            if (readBuffer == null) return;

            EnsureSlots(totalVoxels);
            if (Time.time - _lastReadbackTime < readbackInterval) return;

            if (!useAsync)
            {
                readBuffer.GetData(_slots[_frontSlot]);
                _ready = true;
                _lastReadbackTime = Time.time;
                return;
            }

            RequestAsync(readBuffer, readbackInterval, force: false);
        }

        /// <summary>
        /// Get the latest CPU-side voxel data. Returns null when not yet available.
        /// </summary>
        public uint[] GetData(GraphicsBuffer readBuffer, int totalVoxels,
            float readbackInterval, bool useAsync, bool forceRefresh = false)
        {
            if (readBuffer == null) return null;

            EnsureSlots(totalVoxels);

            if (!Application.isPlaying)
            {
                readBuffer.GetData(_slots[_frontSlot]);
                _ready = true;
                return _slots[_frontSlot];
            }

            if (!useAsync)
            {
                if (forceRefresh || Time.time - _lastReadbackTime >= readbackInterval)
                {
                    readBuffer.GetData(_slots[_frontSlot]);
                    _ready = true;
                    _lastReadbackTime = Time.time;
                }
            }
            else if (forceRefresh)
            {
                RequestAsync(readBuffer, readbackInterval, force: true);
            }

            if (!_ready) return null;
            return _slots[_frontSlot];
        }

        public void Reset()
        {
            for (int i = 0; i < SlotCount; i++)
                _slots[i] = null;
            _frontSlot = 0;
            _inFlightSlot = -1;
            _requestPending = false;
            _ready = false;
            _lastReadbackTime = -999f;
        }

        // ----- Internal -----

        private void EnsureSlots(int totalVoxels)
        {
            bool resized = false;
            for (int i = 0; i < SlotCount; i++)
            {
                if (_slots[i] == null || _slots[i].Length != totalVoxels)
                {
                    _slots[i] = new uint[totalVoxels];
                    resized = true;
                }
            }
            if (resized) _ready = false;
        }

        private void RequestAsync(GraphicsBuffer readBuffer, float readbackInterval, bool force)
        {
            if (readBuffer == null) return;
            if (_requestPending) return;
            if (!force && Time.time - _lastReadbackTime < readbackInterval) return;

            int targetSlot = 1 - _frontSlot;
            _requestPending = true;
            _inFlightSlot = targetSlot;
            _lastReadbackTime = Time.time;

            AsyncGPUReadback.Request(readBuffer, (request) =>
            {
                _requestPending = false;
                int slotIndex = _inFlightSlot;
                _inFlightSlot = -1;

                if (request.hasError) return;
                if (slotIndex < 0 || slotIndex >= SlotCount) return;

                var data = request.GetData<uint>();
                var slot = _slots[slotIndex];
                if (slot == null || slot.Length != data.Length)
                {
                    slot = new uint[data.Length];
                    _slots[slotIndex] = slot;
                }

                data.CopyTo(slot);
                _frontSlot = slotIndex;
                _ready = true;
            });
        }
    }
}
