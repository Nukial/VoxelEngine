using UnityEngine;
using UnityEditor;
using VoxelEngine;

namespace VoxelEngine.Editor
{
    /// <summary>
    /// Scene bootstrap utilities for VoxelEngine.
    /// Designed to be resilient when VoxelWorld serialized fields evolve.
    /// </summary>
    public static class VoxelSceneSetup
    {
        private const int DefaultWorldSize = 128;
        private const float DefaultVoxelScale = 0.25f;

        [MenuItem("VoxelEngine/Editor Tools/Setup Voxel Scene", false, 100)]
        public static void SetupScene()
        {
            SetupWithSize(DefaultWorldSize);
        }

        [MenuItem("VoxelEngine/Editor Tools/Setup Voxel Scene", true)]
        private static bool ValidateSetupScene()
        {
            return !Application.isPlaying;
        }

        [MenuItem("VoxelEngine/Editor Tools/Quick Test (64³)", false, 201)]
        public static void QuickTest64()
        {
            SetupWithSize(64);
        }

        [MenuItem("VoxelEngine/Editor Tools/Quick Test (128³)", false, 200)]
        public static void QuickTest128()
        {
            SetupWithSize(128);
        }

        [MenuItem("VoxelEngine/Editor Tools/Quick Test (256³)", false, 202)]
        public static void QuickTest256()
        {
            SetupWithSize(256);
        }

        [MenuItem("VoxelEngine/Editor Tools/Quick Test (64³)", true)]
        [MenuItem("VoxelEngine/Editor Tools/Quick Test (128³)", true)]
        [MenuItem("VoxelEngine/Editor Tools/Quick Test (256³)", true)]
        private static bool ValidateQuickTests()
        {
            return !Application.isPlaying;
        }

        private static void SetupWithSize(int worldSize)
        {
            if (Application.isPlaying)
            {
                Debug.LogWarning("[VoxelEngine] Setup is disabled while Play Mode is active.");
                return;
            }

            int undoGroup = Undo.GetCurrentGroup();
            Undo.SetCurrentGroupName("VoxelEngine Setup Scene");

            var terrainGen = FindAssetByNames<ComputeShader>("TerrainGeneration", "VoxelTerrainGeneration");
            var simulation = FindAssetByNames<ComputeShader>("VoxelSimulation", "Simulation");
            var brickMap = FindAssetByNames<ComputeShader>("BrickMapUpdate", "VoxelBrickMapUpdate");
            var rayMarchShader = Shader.Find("VoxelEngine/RayMarch") ?? FindAssetByNames<Shader>("RayMarch");

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

            VoxelWorld world = Object.FindFirstObjectByType<VoxelWorld>();
            if (world != null)
            {
                if (!EditorUtility.DisplayDialog("Voxel Scene Setup",
                    "A VoxelWorld already exists in the scene. Update setup on existing object?", "Update", "Cancel"))
                    return;
            }
            else
            {
                var worldGO = new GameObject("VoxelWorld");
                Undo.RegisterCreatedObjectUndo(worldGO, "Create Voxel World");
                world = worldGO.AddComponent<VoxelWorld>();
            }

            var worldObject = world.gameObject;
            var indirectRenderer = EnsureComponent<VoxelIndirectInstanceRenderer>(worldObject);

            ApplyVoxelWorldDefaults(world, indirectRenderer, terrainGen, simulation, brickMap, rayMarchShader, worldSize);

            var cam = Camera.main;
            if (cam == null)
            {
                var camGO = new GameObject("Main Camera");
                camGO.tag = "MainCamera";
                cam = camGO.AddComponent<Camera>();
                camGO.AddComponent<AudioListener>();
                Undo.RegisterCreatedObjectUndo(camGO, "Create Voxel Camera");
            }

            float extent = worldSize * DefaultVoxelScale;
            cam.transform.position = new Vector3(extent * 0.5f, extent * 0.7f, -extent * 0.2f);
            cam.transform.LookAt(new Vector3(extent * 0.5f, extent * 0.3f, extent * 0.5f));
            cam.nearClipPlane = 0.05f;
            cam.farClipPlane = 500f;

            EnsureComponent<VoxelCamera>(cam.gameObject);

            var interaction = EnsureComponent<VoxelInteraction>(cam.gameObject);

            var interactionSO = new SerializedObject(interaction);
            SetObject(interactionSO, "voxelWorld", world);
            SetObject(interactionSO, "mainCamera", cam);
            interactionSO.ApplyModifiedPropertiesWithoutUndo();

            var debugGO = GetOrCreateChild(worldObject.transform, "VoxelDebugUI");
            var debugUI = EnsureComponent<VoxelDebugUI>(debugGO);
            var debugSO = new SerializedObject(debugUI);
            SetObject(debugSO, "voxelWorld", world);
            debugSO.ApplyModifiedPropertiesWithoutUndo();

            var collisionGO = GetOrCreateChild(worldObject.transform, "VoxelCollision");
            EnsureComponent<MeshCollider>(collisionGO);
            var collision = EnsureComponent<VoxelCollision>(collisionGO);
            var collisionSO = new SerializedObject(collision);
            SetObject(collisionSO, "voxelWorld", world);
            SetObject(collisionSO, "trackTarget", cam.transform);
            SetBool(collisionSO, "enableCollisionMeshing", false);
            SetFloat(collisionSO, "updateInterval", 0.4f);
            SetInt(collisionSO, "collisionRadius", 12);
            collisionSO.ApplyModifiedPropertiesWithoutUndo();

            Light mainDirectional = EnsureDirectionalLight();
            BindDirectionalLightOverride(world, mainDirectional);

            Selection.activeGameObject = worldObject;

            Undo.CollapseUndoOperations(undoGroup);

            Debug.Log("[VoxelEngine] Scene setup complete! Press Play to see the voxel world.");
            Debug.Log("[VoxelEngine] Controls: WASD=Move, Mouse=Look, LMB=Dig, RMB=Place, 1-9=Material");
        }

        private static void ApplyVoxelWorldDefaults(
            VoxelWorld world,
            VoxelIndirectInstanceRenderer indirectRenderer,
            ComputeShader terrainGen,
            ComputeShader simulation,
            ComputeShader brickMap,
            Shader rayMarchShader,
            int worldSize)
        {
            var worldSO = new SerializedObject(world);

            SetObject(worldSO, "terrainGenShader", terrainGen);
            SetObject(worldSO, "simulationShader", simulation);
            SetObject(worldSO, "brickMapShader", brickMap);
            SetObject(worldSO, "rayMarchShader", rayMarchShader);
            SetObject(worldSO, "indirectInstanceRenderer", indirectRenderer);

            SetInt(worldSO, "worldSize", worldSize);
            SetInt(worldSO, "brickSize", 8);
            SetFloat(worldSO, "voxelScale", DefaultVoxelScale);

            SetFloat(worldSO, "terrainScale", 0.02f);
            SetFloat(worldSO, "caveScale", 0.06f);
            SetFloat(worldSO, "caveThreshold", 0.72f);

            SetBool(worldSO, "enableSimulation", true);
            SetInt(worldSO, "simulationStepsPerFrame", 1);
            SetFloat(worldSO, "simulationTickRate", 30f);
            SetInt(worldSO, "lightPropagationPasses", 5);

            SetBool(worldSO, "enableShadows", true);
            SetInt(worldSO, "maxRaySteps", 320);
            SetInt(worldSO, "maxShadowSteps", 80);

            SetFloat(worldSO, "maxRenderDistance", 80f);
            SetFloat(worldSO, "distanceQualityFactor", 0.6f);
            SetFloat(worldSO, "renderDistanceFadeRatio", 0.8f);

            SetBool(worldSO, "enableAdaptiveQuality", true);
            SetInt(worldSO, "movingRaySteps", 160);
            SetInt(worldSO, "movingShadowSteps", 40);
            SetBool(worldSO, "enableGpuAdaptiveQuality", true);
            SetBool(worldSO, "reduceShadowsWhileMoving", true);
            SetBool(worldSO, "stabilizeLightingWhileMoving", true);

            SetColor(worldSO, "ambientColor", new Color(0.15f, 0.18f, 0.25f));
            SetColor(worldSO, "sunColor", new Color(1f, 0.95f, 0.85f));
            SetFloat(worldSO, "sunIntensity", 1.2f);
            SetVector3(worldSO, "sunDirection", new Vector3(0.5f, 0.8f, 0.3f));

            SetFloat(worldSO, "fogDensity", 0.003f);
            SetColor(worldSO, "fogColor", new Color(0.6f, 0.75f, 0.9f));

            worldSO.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void BindDirectionalLightOverride(VoxelWorld world, Light directionalLight)
        {
            if (world == null || directionalLight == null) return;

            var worldSO = new SerializedObject(world);
            SetObject(worldSO, "directionalLightOverride", directionalLight);
            SetBool(worldSO, "syncWithUnityDirectionalLight", true);
            worldSO.ApplyModifiedPropertiesWithoutUndo();
        }

        private static Light EnsureDirectionalLight()
        {
            var allLights = Object.FindObjectsByType<Light>(FindObjectsInactive.Exclude, FindObjectsSortMode.None);
            for (int i = 0; i < allLights.Length; i++)
            {
                if (allLights[i] != null && allLights[i].type == LightType.Directional)
                    return allLights[i];
            }

            var lightGO = new GameObject("Directional Light");
            Undo.RegisterCreatedObjectUndo(lightGO, "Create Directional Light");
            var light = lightGO.AddComponent<Light>();
            light.type = LightType.Directional;
            light.color = new Color(1f, 0.95f, 0.85f);
            light.intensity = 1.2f;
            lightGO.transform.rotation = Quaternion.Euler(50f, 30f, 0f);
            return light;
        }

        private static T EnsureComponent<T>(GameObject go) where T : Component
        {
            var component = go.GetComponent<T>();
            if (component == null)
                component = Undo.AddComponent<T>(go);
            return component;
        }

        private static GameObject GetOrCreateChild(Transform parent, string childName)
        {
            var existing = parent.Find(childName);
            if (existing != null) return existing.gameObject;

            var go = new GameObject(childName);
            Undo.RegisterCreatedObjectUndo(go, $"Create {childName}");
            go.transform.SetParent(parent, false);
            return go;
        }

        private static T FindAssetByNames<T>(params string[] candidateNames) where T : Object
        {
            if (candidateNames == null || candidateNames.Length == 0)
                return null;

            string[] guids = AssetDatabase.FindAssets($"t:{typeof(T).Name}");
            T fallbackContains = null;

            for (int i = 0; i < guids.Length; i++)
            {
                string path = AssetDatabase.GUIDToAssetPath(guids[i]);
                T asset = AssetDatabase.LoadAssetAtPath<T>(path);
                if (asset == null)
                    continue;

                string assetName = asset.name;
                for (int j = 0; j < candidateNames.Length; j++)
                {
                    if (assetName == candidateNames[j])
                        return asset;

                    if (fallbackContains == null &&
                        assetName.IndexOf(candidateNames[j], System.StringComparison.OrdinalIgnoreCase) >= 0)
                    {
                        fallbackContains = asset;
                    }
                }
            }

            return fallbackContains;
        }

        private static void SetObject(SerializedObject so, string propertyName, Object value)
        {
            SerializedProperty property = so.FindProperty(propertyName);
            if (property != null)
                property.objectReferenceValue = value;
        }

        private static void SetBool(SerializedObject so, string propertyName, bool value)
        {
            SerializedProperty property = so.FindProperty(propertyName);
            if (property != null)
                property.boolValue = value;
        }

        private static void SetInt(SerializedObject so, string propertyName, int value)
        {
            SerializedProperty property = so.FindProperty(propertyName);
            if (property != null)
                property.intValue = value;
        }

        private static void SetFloat(SerializedObject so, string propertyName, float value)
        {
            SerializedProperty property = so.FindProperty(propertyName);
            if (property != null)
                property.floatValue = value;
        }

        private static void SetColor(SerializedObject so, string propertyName, Color value)
        {
            SerializedProperty property = so.FindProperty(propertyName);
            if (property != null)
                property.colorValue = value;
        }

        private static void SetVector3(SerializedObject so, string propertyName, Vector3 value)
        {
            SerializedProperty property = so.FindProperty(propertyName);
            if (property != null)
                property.vector3Value = value;
        }
    }
}
