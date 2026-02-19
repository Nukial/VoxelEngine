using UnityEngine;

namespace VoxelEngine
{
    /// <summary>
    /// Camera mode for the sandbox glass dome gameplay.
    /// </summary>
    public enum VoxelCameraMode
    {
        /// <summary>
        /// God-view: camera stays OUTSIDE the glass dome.
        /// Frustum culling is ON for performance (back-face culling on the voxel volume).
        /// The player observes the entire voxel world from outside like a terrarium.
        /// </summary>
        GodView,

        /// <summary>
        /// Inside-view: camera is fully INSIDE the voxel world.
        /// No culling transition needed - the camera never crosses the boundary.
        /// Optimised for interior rendering without edge-case handling.
        /// </summary>
        InsideView
    }

    /// <summary>
    /// Dual-mode camera controller for sandbox-in-glass gameplay.
    /// 
    /// Mode 1 (GodView): Orbits/flies around the outside of the glass dome.
    ///   - Frustum culling enabled for performance
    ///   - Orbital controls (rotate around center, zoom, pan)
    ///   - Cannot enter the dome boundary
    ///
    /// Mode 2 (InsideView): First-person fly cam fully inside the world.
    ///   - No culling transition overhead (always inside)
    ///   - Clamped to world bounds so camera never reaches the glass boundary
    ///   - Standard WASD + mouse look
    /// </summary>
    public class VoxelDualCamera : MonoBehaviour
    {
        [Header("Camera Mode")]
        [SerializeField] private VoxelCameraMode cameraMode = VoxelCameraMode.GodView;
        [SerializeField] private KeyCode toggleModeKey = KeyCode.Tab;

        [Header("God View - Orbit")]
        [SerializeField] private float orbitDistance = 60f;
        [SerializeField] private float orbitMinDistance = 20f;
        [SerializeField] private float orbitMaxDistance = 200f;
        [SerializeField] private float orbitSensitivity = 2f;
        [SerializeField] private float orbitZoomSpeed = 10f;
        [SerializeField] private float orbitSmoothing = 0.08f;
        [SerializeField] private float orbitMinPitch = -85f;
        [SerializeField] private float orbitMaxPitch = 85f;

        [Header("Inside View - Fly")]
        [SerializeField] private float flySpeed = 10f;
        [SerializeField] private float flySprintMultiplier = 3f;
        [SerializeField] private float flySensitivity = 2f;
        [SerializeField] private float flySmoothing = 0.1f;
        [SerializeField] private float flyScrollSpeedMult = 1.2f;
        [SerializeField] private float flyMaxPitch = 89f;

        [Header("World Reference")]
        [SerializeField] private VoxelWorld voxelWorld;
        [Tooltip("Margin in world units to keep camera inside/outside the dome boundary")]
        [SerializeField] private float boundaryMargin = 2f;

        // --- State ---
        private Camera _camera;
        private float _orbitYaw;
        private float _orbitPitch = 30f;
        private float _currentOrbitDist;
        private Vector3 _orbitCenter;

        private float _flyYaw;
        private float _flyPitch;
        private Vector3 _flyVelocity;

        private bool _cursorLocked;
        private bool _initialized;

        // Cached world bounds
        private Vector3 _worldMin;
        private Vector3 _worldMax;
        private Vector3 _worldCenter;
        private float _worldExtent;

        // =================================================================
        // Public API
        // =================================================================

        public VoxelCameraMode CameraMode
        {
            get => cameraMode;
            set => SetCameraMode(value);
        }

        public void SetCameraMode(VoxelCameraMode mode)
        {
            if (cameraMode == mode) return;
            cameraMode = mode;
            OnCameraModeChanged();
        }

        // =================================================================
        // Lifecycle
        // =================================================================

        private void Start()
        {
            _camera = GetComponent<Camera>();
            if (_camera == null)
                _camera = gameObject.AddComponent<Camera>();

            if (voxelWorld == null)
                voxelWorld = FindFirstObjectByType<VoxelWorld>();

            InitializeWorldBounds();
            InitializeCameraForMode();
            LockCursor(true);
            _initialized = true;
        }

        private void Update()
        {
            if (!_initialized) return;

            HandleCursorLock();
            HandleModeToggle();

            if (!_cursorLocked) return;

            switch (cameraMode)
            {
                case VoxelCameraMode.GodView:
                    UpdateGodView();
                    break;
                case VoxelCameraMode.InsideView:
                    UpdateInsideView();
                    break;
            }
        }

        // =================================================================
        // Initialization
        // =================================================================

        private void InitializeWorldBounds()
        {
            if (voxelWorld == null) return;

            _worldExtent = voxelWorld.WorldExtent;
            _worldMin = voxelWorld.WorldOrigin;
            _worldMax = _worldMin + Vector3.one * _worldExtent;
            _worldCenter = (_worldMin + _worldMax) * 0.5f;
            _orbitCenter = _worldCenter;
            _currentOrbitDist = orbitDistance;
        }

        private void InitializeCameraForMode()
        {
            if (cameraMode == VoxelCameraMode.GodView)
            {
                // Position camera outside looking at center
                _currentOrbitDist = orbitDistance;
                _orbitYaw = 45f;
                _orbitPitch = 30f;
                ApplyOrbitTransform();

                // Enable culling — the camera is always outside the volume, 
                // so Unity's frustum culling works correctly
                _camera.useOcclusionCulling = true;
                _camera.nearClipPlane = 0.3f;
                _camera.farClipPlane = orbitMaxDistance + _worldExtent * 2f;
            }
            else
            {
                // Position camera at world center
                transform.position = _worldCenter;
                _flyYaw = transform.eulerAngles.y;
                _flyPitch = 0f;
                _flyVelocity = Vector3.zero;

                // Inside view: disable occlusion culling to avoid popping
                // Camera never exits the volume so no transition cost
                _camera.useOcclusionCulling = false;
                _camera.nearClipPlane = 0.05f;
                _camera.farClipPlane = _worldExtent * 2f;
            }
        }

        private void OnCameraModeChanged()
        {
            if (!_initialized) return;
            InitializeCameraForMode();
        }

        // =================================================================
        // God View (Outside Orbit)
        // =================================================================

        private void UpdateGodView()
        {
            // Orbit rotation
            Vector2 mouseDelta = GetMouseDelta();
            _orbitYaw += mouseDelta.x * orbitSensitivity;
            _orbitPitch -= mouseDelta.y * orbitSensitivity;
            _orbitPitch = Mathf.Clamp(_orbitPitch, orbitMinPitch, orbitMaxPitch);

            // Zoom
            float scroll = GetScrollDelta();
            if (Mathf.Abs(scroll) > 0.01f)
            {
                _currentOrbitDist -= scroll * orbitZoomSpeed;
                _currentOrbitDist = Mathf.Clamp(_currentOrbitDist, orbitMinDistance, orbitMaxDistance);
            }

            // Pan with middle mouse (shift + drag)
            if (IsSprintPressed())
            {
                Vector3 right = transform.right * mouseDelta.x * _currentOrbitDist * 0.01f;
                Vector3 up = transform.up * mouseDelta.y * _currentOrbitDist * 0.01f;
                _orbitCenter += right + up;
            }

            ApplyOrbitTransform();

            // Enforce outside-boundary constraint
            EnforceOutsideBoundary();
        }

        private void ApplyOrbitTransform()
        {
            Quaternion rotation = Quaternion.Euler(_orbitPitch, _orbitYaw, 0);
            Vector3 offset = rotation * (Vector3.back * _currentOrbitDist);

            Vector3 targetPos = _orbitCenter + offset;
            transform.position = Vector3.Lerp(transform.position, targetPos,
                Time.deltaTime / Mathf.Max(orbitSmoothing, 0.001f));
            transform.LookAt(_orbitCenter);
        }

        private void EnforceOutsideBoundary()
        {
            // Keep camera outside the dome with margin
            Vector3 pos = transform.position;
            float minDist = _worldExtent * 0.5f + boundaryMargin;

            Vector3 toCenter = pos - _worldCenter;
            float dist = toCenter.magnitude;
            if (dist < minDist)
            {
                pos = _worldCenter + toCenter.normalized * minDist;
                transform.position = pos;
            }
        }

        // =================================================================
        // Inside View (First Person Fly)
        // =================================================================

        private void UpdateInsideView()
        {
            // Mouse look
            Vector2 mouseDelta = GetMouseDelta();
            _flyYaw += mouseDelta.x * flySensitivity;
            _flyPitch -= mouseDelta.y * flySensitivity;
            _flyPitch = Mathf.Clamp(_flyPitch, -flyMaxPitch, flyMaxPitch);
            transform.rotation = Quaternion.Euler(_flyPitch, _flyYaw, 0);

            // Movement
            float speed = flySpeed;
            if (IsSprintPressed())
                speed *= flySprintMultiplier;

            Vector3 input = Vector3.zero;
            if (IsMoveForwardPressed()) input += transform.forward;
            if (IsMoveBackwardPressed()) input -= transform.forward;
            if (IsMoveLeftPressed()) input -= transform.right;
            if (IsMoveRightPressed()) input += transform.right;
            if (IsMoveUpPressed()) input += Vector3.up;
            if (IsMoveDownPressed()) input -= Vector3.up;

            Vector3 target = input.normalized * speed;
            _flyVelocity = Vector3.Lerp(_flyVelocity, target,
                Time.deltaTime / Mathf.Max(flySmoothing, 0.001f));
            transform.position += _flyVelocity * Time.deltaTime;

            // Speed scroll
            float scroll = GetScrollDelta();
            if (Mathf.Abs(scroll) > 0.01f)
            {
                flySpeed *= scroll > 0 ? flyScrollSpeedMult : 1f / flyScrollSpeedMult;
                flySpeed = Mathf.Clamp(flySpeed, 1f, 200f);
            }

            // Clamp inside world bounds — camera stays fully inside,
            // no boundary transition needed
            EnforceInsideBoundary();
        }

        private void EnforceInsideBoundary()
        {
            Vector3 pos = transform.position;
            float margin = boundaryMargin;

            pos.x = Mathf.Clamp(pos.x, _worldMin.x + margin, _worldMax.x - margin);
            pos.y = Mathf.Clamp(pos.y, _worldMin.y + margin, _worldMax.y - margin);
            pos.z = Mathf.Clamp(pos.z, _worldMin.z + margin, _worldMax.z - margin);

            transform.position = pos;
        }

        // =================================================================
        // Mode Toggle
        // =================================================================

        private void HandleModeToggle()
        {
            if (GetKeyDownThisFrame(toggleModeKey))
            {
                SetCameraMode(cameraMode == VoxelCameraMode.GodView
                    ? VoxelCameraMode.InsideView
                    : VoxelCameraMode.GodView);
            }
        }

        // =================================================================
        // Cursor Lock
        // =================================================================

        private void HandleCursorLock()
        {
            if (GetKeyDownThisFrame(KeyCode.Escape))
                LockCursor(false);

            if (GetLeftMouseDownThisFrame() && !_cursorLocked)
                LockCursor(true);
        }

        private void LockCursor(bool locked)
        {
            _cursorLocked = locked;
            Cursor.lockState = locked ? CursorLockMode.Locked : CursorLockMode.None;
            Cursor.visible = !locked;
        }

        // =================================================================
        // Input Abstraction (Input System / Legacy)
        // =================================================================

        private static bool GetKeyDownThisFrame(KeyCode key)
        {
#if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            if (keyboard == null) return false;
            return keyboard[KeyToInputSystemKey(key)].wasPressedThisFrame;
#else
            return Input.GetKeyDown(key);
#endif
        }

        private static bool GetLeftMouseDownThisFrame()
        {
#if ENABLE_INPUT_SYSTEM
            var mouse = UnityEngine.InputSystem.Mouse.current;
            return mouse != null && mouse.leftButton.wasPressedThisFrame;
#else
            return Input.GetMouseButtonDown(0);
#endif
        }

        private static Vector2 GetMouseDelta()
        {
#if ENABLE_INPUT_SYSTEM
            var mouse = UnityEngine.InputSystem.Mouse.current;
            if (mouse == null) return Vector2.zero;
            return mouse.delta.ReadValue() * 0.01f;
#else
            return new Vector2(Input.GetAxis("Mouse X"), Input.GetAxis("Mouse Y"));
#endif
        }

        private static float GetScrollDelta()
        {
#if ENABLE_INPUT_SYSTEM
            var mouse = UnityEngine.InputSystem.Mouse.current;
            return mouse == null ? 0f : mouse.scroll.ReadValue().y;
#else
            return Input.GetAxis("Mouse ScrollWheel");
#endif
        }

        private static bool IsSprintPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            return kb != null && kb.leftShiftKey.isPressed;
#else
            return Input.GetKey(KeyCode.LeftShift);
#endif
        }

        private static bool IsMoveForwardPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            return kb != null && kb.wKey.isPressed;
#else
            return Input.GetKey(KeyCode.W);
#endif
        }

        private static bool IsMoveBackwardPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            return kb != null && kb.sKey.isPressed;
#else
            return Input.GetKey(KeyCode.S);
#endif
        }

        private static bool IsMoveLeftPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            return kb != null && kb.aKey.isPressed;
#else
            return Input.GetKey(KeyCode.A);
#endif
        }

        private static bool IsMoveRightPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            return kb != null && kb.dKey.isPressed;
#else
            return Input.GetKey(KeyCode.D);
#endif
        }

        private static bool IsMoveUpPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            return kb != null && (kb.eKey.isPressed || kb.spaceKey.isPressed);
#else
            return Input.GetKey(KeyCode.E) || Input.GetKey(KeyCode.Space);
#endif
        }

        private static bool IsMoveDownPressed()
        {
#if ENABLE_INPUT_SYSTEM
            var kb = UnityEngine.InputSystem.Keyboard.current;
            return kb != null && (kb.qKey.isPressed || kb.leftCtrlKey.isPressed);
#else
            return Input.GetKey(KeyCode.Q) || Input.GetKey(KeyCode.LeftControl);
#endif
        }

#if ENABLE_INPUT_SYSTEM
        private static UnityEngine.InputSystem.Key KeyToInputSystemKey(KeyCode key)
        {
            switch (key)
            {
                case KeyCode.Tab: return UnityEngine.InputSystem.Key.Tab;
                case KeyCode.Escape: return UnityEngine.InputSystem.Key.Escape;
                case KeyCode.F1: return UnityEngine.InputSystem.Key.F1;
                case KeyCode.F2: return UnityEngine.InputSystem.Key.F2;
                case KeyCode.F3: return UnityEngine.InputSystem.Key.F3;
                case KeyCode.Alpha1: return UnityEngine.InputSystem.Key.Digit1;
                case KeyCode.Alpha2: return UnityEngine.InputSystem.Key.Digit2;
                default: return UnityEngine.InputSystem.Key.Tab;
            }
        }
#endif

        // =================================================================
        // Gizmos
        // =================================================================

        private void OnDrawGizmosSelected()
        {
            if (voxelWorld == null) return;

            float extent = voxelWorld.WorldExtent;
            Vector3 center = voxelWorld.WorldOrigin + Vector3.one * extent * 0.5f;

            // Draw dome boundary
            Gizmos.color = new Color(0.4f, 0.7f, 1f, 0.3f);
            Gizmos.DrawWireCube(center, Vector3.one * (extent + boundaryMargin * 2f));

            // Draw camera constraint zone
            if (cameraMode == VoxelCameraMode.InsideView)
            {
                Gizmos.color = new Color(0f, 1f, 0f, 0.15f);
                Gizmos.DrawWireCube(center, Vector3.one * (extent - boundaryMargin * 2f));
            }
        }
    }
}
