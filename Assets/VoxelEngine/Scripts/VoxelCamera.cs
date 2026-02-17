using UnityEngine;

namespace VoxelEngine
{
    /// <summary>
    /// First-person fly camera controller for voxel world navigation.
    /// WASD movement, mouse look, shift to speed up, scroll to change speed.
    /// </summary>
    public class VoxelCamera : MonoBehaviour
    {
        [Header("Movement")]
        [SerializeField] private float moveSpeed = 10f;
        [SerializeField] private float sprintMultiplier = 3f;
        [SerializeField] private float scrollSpeedMultiplier = 1.2f;
        [SerializeField] private float smoothing = 0.1f;

        [Header("Look")]
        [SerializeField] private float mouseSensitivity = 2f;
        [SerializeField] private float maxPitch = 89f;

        private float _pitch;
        private float _yaw;
        private Vector3 _velocity;
        private bool _cursorLocked;

        private void Start()
        {
            _yaw = transform.eulerAngles.y;
            _pitch = transform.eulerAngles.x;
            LockCursor(true);
        }

        private void Update()
        {
            HandleCursorLock();

            if (_cursorLocked)
            {
                HandleLook();
                HandleMovement();
            }

            HandleSpeedScroll();
        }

        private void HandleCursorLock()
        {
            if (GetEscapePressedThisFrame())
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

        private void HandleLook()
        {
            Vector2 delta = GetMouseDelta();
            float mx = delta.x * mouseSensitivity;
            float my = delta.y * mouseSensitivity;

            _yaw += mx;
            _pitch -= my;
            _pitch = Mathf.Clamp(_pitch, -maxPitch, maxPitch);

            transform.rotation = Quaternion.Euler(_pitch, _yaw, 0);
        }

        private void HandleMovement()
        {
            float speed = moveSpeed;
            if (IsSprintPressed())
                speed *= sprintMultiplier;

            Vector3 input = Vector3.zero;
            if (IsMoveForwardPressed()) input += transform.forward;
            if (IsMoveBackwardPressed()) input -= transform.forward;
            if (IsMoveLeftPressed()) input -= transform.right;
            if (IsMoveRightPressed()) input += transform.right;
            if (IsMoveUpPressed()) input += Vector3.up;
            if (IsMoveDownPressed()) input -= Vector3.up;

            Vector3 target = input.normalized * speed;
            _velocity = Vector3.Lerp(_velocity, target, Time.deltaTime / Mathf.Max(smoothing, 0.001f));
            transform.position += _velocity * Time.deltaTime;
        }

        private void HandleSpeedScroll()
        {
            float scroll = GetScrollDelta();
            if (Mathf.Abs(scroll) > 0.01f)
            {
                moveSpeed *= scroll > 0 ? scrollSpeedMultiplier : 1f / scrollSpeedMultiplier;
                moveSpeed = Mathf.Clamp(moveSpeed, 1f, 200f);
            }
        }

        private static bool GetEscapePressedThisFrame()
        {
    #if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            return keyboard != null && keyboard.escapeKey.wasPressedThisFrame;
    #else
            return Input.GetKeyDown(KeyCode.Escape);
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
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            return keyboard != null && keyboard.leftShiftKey.isPressed;
    #else
            return Input.GetKey(KeyCode.LeftShift);
    #endif
        }

        private static bool IsMoveForwardPressed()
        {
    #if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            return keyboard != null && keyboard.wKey.isPressed;
    #else
            return Input.GetKey(KeyCode.W);
    #endif
        }

        private static bool IsMoveBackwardPressed()
        {
    #if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            return keyboard != null && keyboard.sKey.isPressed;
    #else
            return Input.GetKey(KeyCode.S);
    #endif
        }

        private static bool IsMoveLeftPressed()
        {
    #if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            return keyboard != null && keyboard.aKey.isPressed;
    #else
            return Input.GetKey(KeyCode.A);
    #endif
        }

        private static bool IsMoveRightPressed()
        {
    #if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            return keyboard != null && keyboard.dKey.isPressed;
    #else
            return Input.GetKey(KeyCode.D);
    #endif
        }

        private static bool IsMoveUpPressed()
        {
    #if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            return keyboard != null && (keyboard.eKey.isPressed || keyboard.spaceKey.isPressed);
    #else
            return Input.GetKey(KeyCode.E) || Input.GetKey(KeyCode.Space);
    #endif
        }

        private static bool IsMoveDownPressed()
        {
    #if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            return keyboard != null && (keyboard.qKey.isPressed || keyboard.leftCtrlKey.isPressed);
    #else
            return Input.GetKey(KeyCode.Q) || Input.GetKey(KeyCode.LeftControl);
    #endif
        }
    }
}
