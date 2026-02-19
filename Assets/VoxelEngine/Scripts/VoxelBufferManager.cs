using UnityEngine;

namespace VoxelEngine
{
    /// <summary>
    /// Manages GPU voxel buffers and the ping-pong double-buffering scheme.
    /// </summary>
    public class VoxelBufferManager
    {
        private GraphicsBuffer _voxelBufferA;
        private GraphicsBuffer _voxelBufferB;
        private GraphicsBuffer _brickMapBuffer;
        private bool _pingPong;

        public int WorldSize { get; private set; }
        public int BrickSize { get; private set; }
        public int BrickMapSize => WorldSize / BrickSize;
        public int TotalVoxels => WorldSize * WorldSize * WorldSize;

        public GraphicsBuffer ReadBuffer => _pingPong ? _voxelBufferB : _voxelBufferA;
        public GraphicsBuffer WriteBuffer => _pingPong ? _voxelBufferA : _voxelBufferB;
        public GraphicsBuffer BrickMapBuffer => _brickMapBuffer;

        public GraphicsBuffer VoxelBufferA => _voxelBufferA;
        public GraphicsBuffer VoxelBufferB => _voxelBufferB;

        public void Create(int worldSize, int brickSize)
        {
            WorldSize = worldSize;
            BrickSize = brickSize;

            int total = TotalVoxels;
            int brickTotal = BrickMapSize * BrickMapSize * BrickMapSize;

            _voxelBufferA = new GraphicsBuffer(GraphicsBuffer.Target.Structured, total, sizeof(uint));
            _voxelBufferB = new GraphicsBuffer(GraphicsBuffer.Target.Structured, total, sizeof(uint));
            _brickMapBuffer = new GraphicsBuffer(GraphicsBuffer.Target.Structured, brickTotal, sizeof(uint));

            // Clear buffers
            var clearData = new uint[total];
            _voxelBufferA.SetData(clearData);
            _voxelBufferB.SetData(clearData);

            var clearBrick = new uint[brickTotal];
            _brickMapBuffer.SetData(clearBrick);

            Debug.Log($"[VoxelBufferManager] Created buffers: {worldSize}³ = {total:N0} voxels ({total * 4 / 1024f / 1024f:F1} MB per buffer)");
        }

        public void SwapBuffers()
        {
            _pingPong = !_pingPong;
        }

        public void Release()
        {
            _voxelBufferA?.Release();
            _voxelBufferB?.Release();
            _brickMapBuffer?.Release();
            _voxelBufferA = null;
            _voxelBufferB = null;
            _brickMapBuffer = null;
        }
    }
}
