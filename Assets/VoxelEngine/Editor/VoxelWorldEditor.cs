using UnityEngine;
using UnityEditor;
using VoxelEngine;

namespace VoxelEngine.Editor
{
    [CustomEditor(typeof(VoxelWorld))]
    public class VoxelWorldEditor : UnityEditor.Editor
    {
        private SerializedProperty _worldSize;
        private SerializedProperty _brickSize;
        private SerializedProperty _voxelScale;
        private SerializedProperty _terrainGenShader;
        private SerializedProperty _terrainScale;
        private SerializedProperty _caveScale;
        private SerializedProperty _caveThreshold;
        private SerializedProperty _seed;
        private SerializedProperty _simulationShader;
        private SerializedProperty _brickMapShader;
        private SerializedProperty _enableSimulation;
        private SerializedProperty _simulationStepsPerFrame;
        private SerializedProperty _rayMarchShader;
        private SerializedProperty _maxRaySteps;
        private SerializedProperty _maxShadowSteps;
        private SerializedProperty _enableShadows;
        private SerializedProperty _ambientColor;
        private SerializedProperty _sunColor;
        private SerializedProperty _sunIntensity;
        private SerializedProperty _sunDirection;
        private SerializedProperty _fogDensity;
        private SerializedProperty _fogColor;

        private bool _showWorldInfo = true;
        private bool _showMemoryInfo = true;

        private void OnEnable()
        {
            _worldSize = serializedObject.FindProperty("worldSize");
            _brickSize = serializedObject.FindProperty("brickSize");
            _voxelScale = serializedObject.FindProperty("voxelScale");
            _terrainGenShader = serializedObject.FindProperty("terrainGenShader");
            _terrainScale = serializedObject.FindProperty("terrainScale");
            _caveScale = serializedObject.FindProperty("caveScale");
            _caveThreshold = serializedObject.FindProperty("caveThreshold");
            _seed = serializedObject.FindProperty("seed");
            _simulationShader = serializedObject.FindProperty("simulationShader");
            _brickMapShader = serializedObject.FindProperty("brickMapShader");
            _enableSimulation = serializedObject.FindProperty("enableSimulation");
            _simulationStepsPerFrame = serializedObject.FindProperty("simulationStepsPerFrame");
            _rayMarchShader = serializedObject.FindProperty("rayMarchShader");
            _maxRaySteps = serializedObject.FindProperty("maxRaySteps");
            _maxShadowSteps = serializedObject.FindProperty("maxShadowSteps");
            _enableShadows = serializedObject.FindProperty("enableShadows");
            _ambientColor = serializedObject.FindProperty("ambientColor");
            _sunColor = serializedObject.FindProperty("sunColor");
            _sunIntensity = serializedObject.FindProperty("sunIntensity");
            _sunDirection = serializedObject.FindProperty("sunDirection");
            _fogDensity = serializedObject.FindProperty("fogDensity");
            _fogColor = serializedObject.FindProperty("fogColor");
        }

        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            var world = (VoxelWorld)target;

            // Header
            EditorGUILayout.Space(5);
            EditorGUILayout.LabelField("Voxel Engine", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("GPU-accelerated voxel rendering & simulation", EditorStyles.miniLabel);
            EditorGUILayout.Space(10);

            // World info
            _showWorldInfo = EditorGUILayout.Foldout(_showWorldInfo, "World Information", true);
            if (_showWorldInfo)
            {
                EditorGUI.indentLevel++;
                int ws = _worldSize.intValue;
                long totalVoxels = (long)ws * ws * ws;
                int bms = ws / _brickSize.intValue;

                EditorGUILayout.LabelField("Total Voxels", $"{totalVoxels:N0}");
                EditorGUILayout.LabelField("Brick Map", $"{bms}³ = {bms * bms * bms:N0} bricks");
                EditorGUILayout.LabelField("World Extent", $"{ws * _voxelScale.floatValue:F1}m");

                long memBytes = totalVoxels * 4 * 2; // Two ping-pong buffers
                memBytes += bms * bms * bms * 4; // Brick map
                EditorGUILayout.LabelField("Est. VRAM", $"{memBytes / 1024f / 1024f:F1} MB");

                EditorGUI.indentLevel--;
                EditorGUILayout.Space(5);
            }

            // Draw properties with headers
            DrawDefaultInspector();

            EditorGUILayout.Space(10);

            // Quick actions
            EditorGUILayout.LabelField("Actions", EditorStyles.boldLabel);

            if (GUILayout.Button("Regenerate Terrain (Play Mode)"))
            {
                if (Application.isPlaying)
                {
                    // Force regeneration by disabling and re-enabling
                    world.enabled = false;
                    world.enabled = true;
                }
                else
                {
                    EditorUtility.DisplayDialog("Info", "Terrain generation only works in Play Mode.", "OK");
                }
            }

            if (GUILayout.Button("Randomize Seed"))
            {
                _seed.floatValue = Random.Range(0f, 10000f);
                serializedObject.ApplyModifiedProperties();
            }

            serializedObject.ApplyModifiedProperties();
        }
    }
}
