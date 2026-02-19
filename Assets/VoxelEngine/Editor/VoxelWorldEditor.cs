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
        private SerializedProperty _seed;
        private SerializedProperty _simulationShader;
        private SerializedProperty _brickMapShader;
        private SerializedProperty _rayMarchShader;
        private SerializedProperty _glassDomeShader;

        private bool _showWorldInfo = true;
        private bool _showGlassDomeInfo = true;

        private void OnEnable()
        {
            _worldSize = serializedObject.FindProperty("worldSize");
            _brickSize = serializedObject.FindProperty("brickSize");
            _voxelScale = serializedObject.FindProperty("voxelScale");
            _terrainGenShader = serializedObject.FindProperty("terrainGenShader");
            _seed = serializedObject.FindProperty("seed");
            _simulationShader = serializedObject.FindProperty("simulationShader");
            _brickMapShader = serializedObject.FindProperty("brickMapShader");
            _rayMarchShader = serializedObject.FindProperty("rayMarchShader");
            _glassDomeShader = serializedObject.FindProperty("glassDomeShader");
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

            if (!Application.isPlaying && GUILayout.Button("Run Editor Setup"))
            {
                VoxelSceneSetup.SetupScene();
                EditorGUIUtility.PingObject(target);
            }

            if (_terrainGenShader.objectReferenceValue == null ||
                _simulationShader.objectReferenceValue == null ||
                _brickMapShader.objectReferenceValue == null ||
                _rayMarchShader.objectReferenceValue == null)
            {
                EditorGUILayout.HelpBox(
                    "Missing shader references detected. Use 'Run Editor Setup' to auto-assign required assets.",
                    MessageType.Warning);
            }

            if (_glassDomeShader != null && _glassDomeShader.objectReferenceValue == null)
            {
                EditorGUILayout.HelpBox(
                    "Glass Dome shader not assigned. Use 'Run Editor Setup' to auto-assign, or assign manually.",
                    MessageType.Info);
            }

            EditorGUILayout.Space(5);

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

            // Glass Dome & Camera info (play mode)
            _showGlassDomeInfo = EditorGUILayout.Foldout(_showGlassDomeInfo, "Glass Dome & Camera", true);
            if (_showGlassDomeInfo)
            {
                EditorGUI.indentLevel++;

                // Show glass dome status
                var glassDome = world.GetComponent<VoxelGlassDome>();
                if (glassDome != null)
                {
                    EditorGUILayout.LabelField("Glass Dome", "Attached");
                }
                else
                {
                    EditorGUILayout.LabelField("Glass Dome", "Not attached");
                    if (GUILayout.Button("Add Glass Dome Component"))
                    {
                        Undo.AddComponent<VoxelGlassDome>(world.gameObject);
                    }
                }

                // Show camera mode status
                if (Application.isPlaying)
                {
                    var dualCam = Object.FindFirstObjectByType<VoxelDualCamera>();
                    if (dualCam != null)
                    {
                        EditorGUILayout.LabelField("Camera Mode", dualCam.CameraMode.ToString());
                        EditorGUILayout.HelpBox("Press Tab to toggle between God View and Inside View.", MessageType.Info);

                        EditorGUILayout.BeginHorizontal();
                        if (GUILayout.Button("Switch to God View"))
                            dualCam.SetCameraMode(VoxelCameraMode.GodView);
                        if (GUILayout.Button("Switch to Inside View"))
                            dualCam.SetCameraMode(VoxelCameraMode.InsideView);
                        EditorGUILayout.EndHorizontal();
                    }
                    else
                    {
                        EditorGUILayout.LabelField("Dual Camera", "Not found in scene");
                    }
                }
                else
                {
                    var dualCam = Object.FindFirstObjectByType<VoxelDualCamera>();
                    if (dualCam != null)
                        EditorGUILayout.LabelField("Dual Camera", "Ready (Tab to toggle in Play Mode)");
                    else
                        EditorGUILayout.LabelField("Dual Camera", "Not in scene — run Editor Setup");
                }

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
