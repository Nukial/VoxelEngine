using UnityEngine;
using UnityEditor;
using VoxelEngine;

namespace VoxelEngine.Editor
{
    /// <summary>
    /// Menu item to automatically set up a voxel world scene.
    /// Creates all necessary GameObjects with correct component references.
    /// </summary>
    public static class VoxelSceneSetup
    {
        [MenuItem("VoxelEngine/Setup Voxel Scene", false, 100)]
        public static void SetupScene()
        {
            // Find compute shaders
            var terrainGen = FindAsset<ComputeShader>("TerrainGeneration");
            var simulation = FindAsset<ComputeShader>("VoxelSimulation");
            var brickMap = FindAsset<ComputeShader>("BrickMapUpdate");
            var rayMarchShader = Shader.Find("VoxelEngine/RayMarch");

            if (terrainGen == null || simulation == null || brickMap == null)
            {
                Debug.LogError("[VoxelEngine] Could not find compute shaders! Make sure they are in the project.");
                return;
            }

            if (rayMarchShader == null)
            {
                Debug.LogError("[VoxelEngine] Could not find VoxelEngine/RayMarch shader!");
                return;
            }

            // Create root object
            var existingWorld = Object.FindFirstObjectByType<VoxelWorld>();
            if (existingWorld != null)
            {
                if (!EditorUtility.DisplayDialog("Voxel Scene Setup",
                    "A VoxelWorld already exists in the scene. Replace it?", "Yes", "No"))
                    return;
                Undo.DestroyObjectImmediate(existingWorld.gameObject);
            }

            // --- Voxel World ---
            var worldGO = new GameObject("VoxelWorld");
            Undo.RegisterCreatedObjectUndo(worldGO, "Create Voxel World");

            var world = worldGO.AddComponent<VoxelWorld>();
            var indirectRenderer = worldGO.AddComponent<VoxelIndirectInstanceRenderer>();

            // Set serialized fields via SerializedObject
            var so = new SerializedObject(world);
            so.FindProperty("terrainGenShader").objectReferenceValue = terrainGen;
            so.FindProperty("simulationShader").objectReferenceValue = simulation;
            so.FindProperty("brickMapShader").objectReferenceValue = brickMap;
            so.FindProperty("rayMarchShader").objectReferenceValue = rayMarchShader;
            so.FindProperty("indirectInstanceRenderer").objectReferenceValue = indirectRenderer;
            so.FindProperty("worldSize").intValue = 128;
            so.FindProperty("voxelScale").floatValue = 0.25f;
            so.FindProperty("enableShadows").boolValue = true;
            so.FindProperty("maxRaySteps").intValue = 320;
            so.FindProperty("maxShadowSteps").intValue = 80;
            so.FindProperty("enableAdaptiveQuality").boolValue = true;
            so.FindProperty("movingRaySteps").intValue = 160;
            so.FindProperty("movingShadowSteps").intValue = 40;
            var reduceShadowsProp = so.FindProperty("reduceShadowsWhileMoving");
            if (reduceShadowsProp != null)
                reduceShadowsProp.boolValue = true;
            so.FindProperty("simulationTickRate").floatValue = 30f;
            so.FindProperty("simulationStepsPerFrame").intValue = 1;
            so.FindProperty("sunDirection").vector3Value = new Vector3(0.5f, 0.8f, 0.3f);
            so.ApplyModifiedPropertiesWithoutUndo();

            // --- Camera ---
            var cam = Camera.main;
            if (cam == null)
            {
                var camGO = new GameObject("Main Camera");
                camGO.tag = "MainCamera";
                cam = camGO.AddComponent<Camera>();
                camGO.AddComponent<AudioListener>();
                Undo.RegisterCreatedObjectUndo(camGO, "Create Voxel Camera");
            }

            // Position camera above terrain
            float extent = 128 * 0.25f; // worldSize * voxelScale
            cam.transform.position = new Vector3(extent * 0.5f, extent * 0.7f, -extent * 0.2f);
            cam.transform.LookAt(new Vector3(extent * 0.5f, extent * 0.3f, extent * 0.5f));
            cam.nearClipPlane = 0.1f;
            cam.farClipPlane = 500f;

            // Add camera controller
            if (cam.GetComponent<VoxelCamera>() == null)
                cam.gameObject.AddComponent<VoxelCamera>();

            // Add interaction
            var interaction = cam.GetComponent<VoxelInteraction>();
            if (interaction == null)
                interaction = cam.gameObject.AddComponent<VoxelInteraction>();

            var interactionSO = new SerializedObject(interaction);
            interactionSO.FindProperty("voxelWorld").objectReferenceValue = world;
            interactionSO.FindProperty("mainCamera").objectReferenceValue = cam;
            interactionSO.ApplyModifiedPropertiesWithoutUndo();

            // --- Debug UI ---
            var debugGO = new GameObject("VoxelDebugUI");
            debugGO.transform.SetParent(worldGO.transform);
            var debugUI = debugGO.AddComponent<VoxelDebugUI>();
            var debugSO = new SerializedObject(debugUI);
            debugSO.FindProperty("voxelWorld").objectReferenceValue = world;
            debugSO.ApplyModifiedPropertiesWithoutUndo();

            // --- Collision (optional) ---
            var collisionGO = new GameObject("VoxelCollision");
            collisionGO.transform.SetParent(worldGO.transform);
            collisionGO.AddComponent<MeshCollider>();
            var collision = collisionGO.AddComponent<VoxelCollision>();
            var collisionSO = new SerializedObject(collision);
            collisionSO.FindProperty("voxelWorld").objectReferenceValue = world;
            collisionSO.FindProperty("trackTarget").objectReferenceValue = cam.transform;
            collisionSO.FindProperty("enableCollisionMeshing").boolValue = false;
            collisionSO.FindProperty("updateInterval").floatValue = 0.4f;
            collisionSO.FindProperty("collisionRadius").intValue = 12;
            collisionSO.ApplyModifiedPropertiesWithoutUndo();

            // --- Directional Light ---
            var existingLight = Object.FindFirstObjectByType<Light>();
            if (existingLight == null || existingLight.type != LightType.Directional)
            {
                var lightGO = new GameObject("Directional Light");
                var light = lightGO.AddComponent<Light>();
                light.type = LightType.Directional;
                light.color = new Color(1f, 0.95f, 0.85f);
                light.intensity = 1.2f;
                lightGO.transform.rotation = Quaternion.Euler(50, 30, 0);
                Undo.RegisterCreatedObjectUndo(lightGO, "Create Directional Light");
            }

            // Select the world object
            Selection.activeGameObject = worldGO;

            Debug.Log("[VoxelEngine] Scene setup complete! Press Play to see the voxel world.");
            Debug.Log("[VoxelEngine] Controls: WASD=Move, Mouse=Look, LMB=Dig, RMB=Place, 1-9=Material");
        }

        [MenuItem("VoxelEngine/Quick Test (128³)", false, 200)]
        public static void QuickTest128()
        {
            SetupWithSize(128);
        }

        [MenuItem("VoxelEngine/Quick Test (64³)", false, 201)]
        public static void QuickTest64()
        {
            SetupWithSize(64);
        }

        [MenuItem("VoxelEngine/Quick Test (256³)", false, 202)]
        public static void QuickTest256()
        {
            SetupWithSize(256);
        }

        private static void SetupWithSize(int size)
        {
            SetupScene();
            var world = Object.FindFirstObjectByType<VoxelWorld>();
            if (world != null)
            {
                var so = new SerializedObject(world);
                so.FindProperty("worldSize").intValue = size;
                so.ApplyModifiedProperties();
            }
        }

        private static T FindAsset<T>(string name) where T : Object
        {
            string[] guids = AssetDatabase.FindAssets($"{name} t:{typeof(T).Name}");
            foreach (string guid in guids)
            {
                string path = AssetDatabase.GUIDToAssetPath(guid);
                T asset = AssetDatabase.LoadAssetAtPath<T>(path);
                if (asset != null && asset.name == name)
                    return asset;
            }
            return null;
        }
    }
}
