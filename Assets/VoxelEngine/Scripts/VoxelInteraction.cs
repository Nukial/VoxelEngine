using UnityEngine;

namespace VoxelEngine
{
    /// <summary>
    /// Player interaction with the voxel world.
    /// Left click: Dig (remove voxel), Right click: Place voxel
    /// Uses CPU-side DDA raycast for picking.
    /// </summary>
    public class VoxelInteraction : MonoBehaviour
    {
        [Header("References")]
        [SerializeField] private VoxelWorld voxelWorld;
        [SerializeField] private Camera mainCamera;

        [Header("Interaction")]
        [SerializeField] private float maxReach = 50f;
        [SerializeField] private float brushRadius = 1f;
        [SerializeField] [Range(0, 14)] private int placeMaterial = 4; // Sand by default
        [SerializeField] [Min(0.01f)] private float targetUpdateInterval = 0.03f;

        [Header("Continuous Interaction")]
        [SerializeField] private bool allowContinuous = true;
        [SerializeField] private float continuousInterval = 0.05f;

        [Header("Visual Feedback")]
        [SerializeField] private bool showCrosshair = true;
        [SerializeField] private bool showTargetHighlight = true;

        private float _lastInteractTime;
        private Vector3Int _lastHitPos;
        private Vector3Int _lastHitNormal;
        private bool _hasTarget;
        private uint _lastHitMaterial;
        private float _lastTargetUpdateTime;

        // Public accessors for UI
        public bool HasTarget => _hasTarget;
        public Vector3Int TargetPos => _lastHitPos;
        public uint TargetMaterial => _lastHitMaterial;
        public int PlaceMaterial { get => placeMaterial; set => placeMaterial = Mathf.Clamp(value, 0, (int)VoxelData.MAT_COUNT - 1); }
        public float BrushRadius { get => brushRadius; set => brushRadius = Mathf.Clamp(value, 0.5f, 10f); }

        private void Start()
        {
            if (voxelWorld == null)
                voxelWorld = FindFirstObjectByType<VoxelWorld>();
            if (mainCamera == null)
                mainCamera = Camera.main;
        }

        private void Update()
        {
            if (voxelWorld == null || mainCamera == null) return;
            if (Cursor.lockState != CursorLockMode.Locked) return;

            // Raycast to find target voxel
            UpdateTarget();

            // Handle material selection with number keys
            HandleMaterialSelection();

            // Handle brush size
            HandleBrushSize();

            // Handle interactions
            HandleInteraction();
        }

        private void UpdateTarget()
        {
            if (Time.time - _lastTargetUpdateTime < targetUpdateInterval)
                return;

            _lastTargetUpdateTime = Time.time;
            Ray ray = mainCamera.ScreenPointToRay(new Vector3(Screen.width / 2f, Screen.height / 2f, 0));
            _hasTarget = voxelWorld.RaycastVoxel(ray, maxReach, out _lastHitPos, out _lastHitNormal, out _lastHitMaterial);
        }

        private void HandleMaterialSelection()
        {
            // Number keys 1-9 to select material
            if (GetDigitPressedThisFrame(1)) SetPlaceMaterial(1);
            if (GetDigitPressedThisFrame(2)) SetPlaceMaterial(2);
            if (GetDigitPressedThisFrame(3)) SetPlaceMaterial(3);
            if (GetDigitPressedThisFrame(4)) SetPlaceMaterial(4);
            if (GetDigitPressedThisFrame(5)) SetPlaceMaterial(5);
            if (GetDigitPressedThisFrame(6)) SetPlaceMaterial(6);
            if (GetDigitPressedThisFrame(7)) SetPlaceMaterial(7);
            if (GetDigitPressedThisFrame(8)) SetPlaceMaterial(8);
            if (GetDigitPressedThisFrame(9)) SetPlaceMaterial(9);

            // 0 for material 10+
            if (GetDigitPressedThisFrame(0))
            {
                placeMaterial = (placeMaterial + 1) % (int)VoxelData.MAT_COUNT;
                if (placeMaterial == 0) placeMaterial = 1;
                Debug.Log($"[Voxel] Selected material: {VoxelData.MaterialNames[placeMaterial]}");
            }
        }

        private void HandleBrushSize()
        {
            // [ and ] to change brush size
            if (GetLeftBracketPressedThisFrame())
            {
                brushRadius = Mathf.Max(0.5f, brushRadius - 0.5f);
                Debug.Log($"[Voxel] Brush radius: {brushRadius}");
            }
            if (GetRightBracketPressedThisFrame())
            {
                brushRadius = Mathf.Min(10f, brushRadius + 0.5f);
                Debug.Log($"[Voxel] Brush radius: {brushRadius}");
            }
        }

        private void HandleInteraction()
        {
            if (!_hasTarget) return;

            bool canInteract = Time.time - _lastInteractTime >= continuousInterval;
            if (!canInteract && allowContinuous) return;

            // Left mouse: Dig
            if (IsLeftMousePressed())
            {
                Dig();
                _lastInteractTime = Time.time;
            }

            // Right mouse: Place
            if (IsRightMousePressed())
            {
                Place();
                _lastInteractTime = Time.time;
            }

            // Middle mouse: Pick material
            if (IsMiddleMousePressedThisFrame())
            {
                PickMaterial();
            }
        }

        private void SetPlaceMaterial(int index)
        {
            placeMaterial = Mathf.Clamp(index, 1, (int)VoxelData.MAT_COUNT - 1);
            Debug.Log($"[Voxel] Selected material: {VoxelData.MaterialNames[placeMaterial]}");
        }

        private static bool GetDigitPressedThisFrame(int digit)
        {
    #if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            if (keyboard == null) return false;
            return digit switch
            {
            0 => keyboard.digit0Key.wasPressedThisFrame,
            1 => keyboard.digit1Key.wasPressedThisFrame,
            2 => keyboard.digit2Key.wasPressedThisFrame,
            3 => keyboard.digit3Key.wasPressedThisFrame,
            4 => keyboard.digit4Key.wasPressedThisFrame,
            5 => keyboard.digit5Key.wasPressedThisFrame,
            6 => keyboard.digit6Key.wasPressedThisFrame,
            7 => keyboard.digit7Key.wasPressedThisFrame,
            8 => keyboard.digit8Key.wasPressedThisFrame,
            9 => keyboard.digit9Key.wasPressedThisFrame,
            _ => false
            };
    #else
            return Input.GetKeyDown(KeyCode.Alpha0 + Mathf.Clamp(digit, 0, 9));
    #endif
        }

        private static bool GetLeftBracketPressedThisFrame()
        {
    #if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            return keyboard != null && keyboard.leftBracketKey.wasPressedThisFrame;
    #else
            return Input.GetKeyDown(KeyCode.LeftBracket);
    #endif
        }

        private static bool GetRightBracketPressedThisFrame()
        {
    #if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            return keyboard != null && keyboard.rightBracketKey.wasPressedThisFrame;
    #else
            return Input.GetKeyDown(KeyCode.RightBracket);
    #endif
        }

        private static bool IsLeftMousePressed()
        {
    #if ENABLE_INPUT_SYSTEM
            var mouse = UnityEngine.InputSystem.Mouse.current;
            return mouse != null && mouse.leftButton.isPressed;
    #else
            return Input.GetMouseButton(0);
    #endif
        }

        private static bool IsRightMousePressed()
        {
    #if ENABLE_INPUT_SYSTEM
            var mouse = UnityEngine.InputSystem.Mouse.current;
            return mouse != null && mouse.rightButton.isPressed;
    #else
            return Input.GetMouseButton(1);
    #endif
        }

        private static bool IsMiddleMousePressedThisFrame()
        {
    #if ENABLE_INPUT_SYSTEM
            var mouse = UnityEngine.InputSystem.Mouse.current;
            return mouse != null && mouse.middleButton.wasPressedThisFrame;
    #else
            return Input.GetMouseButtonDown(2);
    #endif
        }

        private void Dig()
        {
            if (brushRadius <= 0.6f)
            {
                // Single voxel
                voxelWorld.SetVoxel(_lastHitPos, 0);
            }
            else
            {
                // Sphere brush
                Vector3 center = new Vector3(_lastHitPos.x, _lastHitPos.y, _lastHitPos.z);
                voxelWorld.SetVoxelSphere(center, brushRadius, VoxelData.MAT_AIR);
            }
        }

        private void Place()
        {
            // Place adjacent to the hit face
            Vector3Int placePos = _lastHitPos + _lastHitNormal;

            if (!VoxelData.IsInBounds(placePos, voxelWorld.WorldSize))
                return;

            if (brushRadius <= 0.6f)
            {
                uint voxel = VoxelData.PackWithDefaultColor((uint)placeMaterial);
                voxelWorld.SetVoxel(placePos, voxel);
            }
            else
            {
                Vector3 center = new Vector3(placePos.x, placePos.y, placePos.z);
                voxelWorld.SetVoxelSphere(center, brushRadius, (uint)placeMaterial);
            }
        }

        private void PickMaterial()
        {
            placeMaterial = (int)_lastHitMaterial;
            Debug.Log($"[Voxel] Picked material: {VoxelData.MaterialNames[Mathf.Clamp(placeMaterial, 0, (int)VoxelData.MAT_COUNT - 1)]}");
        }

        // =====================================================================
        // UI Drawing
        // =====================================================================

        private void OnGUI()
        {
            if (!showCrosshair || Cursor.lockState != CursorLockMode.Locked) return;

            // Crosshair
            float cx = Screen.width / 2f;
            float cy = Screen.height / 2f;
            float sz = 12f;
            float th = 2f;

            GUI.color = _hasTarget ? Color.white : new Color(1, 1, 1, 0.3f);
            GUI.DrawTexture(new Rect(cx - sz, cy - th / 2, sz * 2, th), Texture2D.whiteTexture);
            GUI.DrawTexture(new Rect(cx - th / 2, cy - sz, th, sz * 2), Texture2D.whiteTexture);

            // Target info
            if (_hasTarget && showTargetHighlight)
            {
                string matName = _lastHitMaterial < VoxelData.MAT_COUNT
                    ? VoxelData.MaterialNames[_lastHitMaterial]
                    : "Unknown";

                GUIStyle style = new GUIStyle(GUI.skin.label);
                style.alignment = TextAnchor.UpperCenter;
                style.normal.textColor = Color.white;
                style.fontSize = 14;

                GUI.Label(new Rect(cx - 100, cy + 20, 200, 30),
                    $"[{matName}] ({_lastHitPos.x}, {_lastHitPos.y}, {_lastHitPos.z})", style);
            }
        }
    }
}
