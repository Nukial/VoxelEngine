using UnityEngine;

namespace VoxelEngine
{
    /// <summary>
    /// Debug UI overlay showing performance metrics and voxel world info.
    /// </summary>
    public class VoxelDebugUI : MonoBehaviour
    {
        [SerializeField] private VoxelWorld voxelWorld;
        [SerializeField] private bool showDebugUI = true;
        [SerializeField] private KeyCode toggleKey = KeyCode.F1;

        // FPS tracking
        private float _fpsTimer;
        private int _fpsCounter;
        private float _currentFPS;
        private float _avgFrameTime;

        // Stats
        private float _minFPS = float.MaxValue;
        private float _maxFPS;

        private GUIStyle _boxStyle;
        private GUIStyle _labelStyle;
        private GUIStyle _headerStyle;

        private void Start()
        {
            if (voxelWorld == null)
                voxelWorld = FindFirstObjectByType<VoxelWorld>();
        }

        private void Update()
        {
            if (IsTogglePressedThisFrame())
                showDebugUI = !showDebugUI;

            // FPS calculation
            _fpsCounter++;
            _fpsTimer += Time.unscaledDeltaTime;
            if (_fpsTimer >= 0.5f)
            {
                _currentFPS = _fpsCounter / _fpsTimer;
                _avgFrameTime = _fpsTimer / _fpsCounter * 1000f;
                _fpsCounter = 0;
                _fpsTimer = 0;

                if (_currentFPS > 0)
                {
                    _minFPS = Mathf.Min(_minFPS, _currentFPS);
                    _maxFPS = Mathf.Max(_maxFPS, _currentFPS);
                }
            }
        }

        private void OnGUI()
        {
            if (!showDebugUI) return;

            InitStyles();

            float x = 10;
            float y = 10;
            float w = 280;
            float lineH = 20;

            // Background
            float totalH = 220;
            GUI.Box(new Rect(x - 5, y - 5, w + 10, totalH + 10), "", _boxStyle);

            // Title
            GUI.Label(new Rect(x, y, w, lineH), "VOXEL ENGINE DEBUG", _headerStyle);
            y += lineH + 5;

            // FPS
            Color fpsColor = _currentFPS >= 60 ? Color.green :
                             _currentFPS >= 30 ? Color.yellow : Color.red;
            string fpsColorHex = ColorUtility.ToHtmlStringRGB(fpsColor);
            GUI.Label(new Rect(x, y, w, lineH),
                $"FPS: <color=#{fpsColorHex}>{_currentFPS:F1}</color>  ({_avgFrameTime:F1}ms)", _labelStyle);
            y += lineH;

            GUI.Label(new Rect(x, y, w, lineH),
                $"Min/Max: {_minFPS:F1} / {_maxFPS:F1}", _labelStyle);
            y += lineH + 5;

            if (voxelWorld != null)
            {
                // World info
                GUI.Label(new Rect(x, y, w, lineH),
                    $"World: {voxelWorld.WorldSize}³ ({voxelWorld.TotalVoxels:N0} voxels)", _labelStyle);
                y += lineH;

                GUI.Label(new Rect(x, y, w, lineH),
                    $"Brick Map: {voxelWorld.BrickMapSize}³ (brick={voxelWorld.BrickSize}³)", _labelStyle);
                y += lineH;

                float memMB = voxelWorld.TotalVoxels * 4f * 2f / 1024f / 1024f;
                GUI.Label(new Rect(x, y, w, lineH),
                    $"VRAM (buffers): ~{memMB:F1} MB", _labelStyle);
                y += lineH;

                GUI.Label(new Rect(x, y, w, lineH),
                    $"Voxel Scale: {voxelWorld.VoxelScale}m", _labelStyle);
                y += lineH;

                GUI.Label(new Rect(x, y, w, lineH),
                    $"World Extent: {voxelWorld.WorldExtent:F1}m", _labelStyle);
                y += lineH + 5;
            }

            // Controls
            GUI.Label(new Rect(x, y, w, lineH), "CONTROLS", _headerStyle);
            y += lineH;
            GUI.Label(new Rect(x, y, w, lineH), "WASD: Move | Mouse: Look", _labelStyle);
            y += lineH;
            GUI.Label(new Rect(x, y, w, lineH), "LMB: Dig | RMB: Place | MMB: Pick", _labelStyle);
            y += lineH;
            GUI.Label(new Rect(x, y, w, lineH), "1-9: Material | []: Brush size", _labelStyle);
            y += lineH;
            GUI.Label(new Rect(x, y, w, lineH), "Scroll: Speed | Shift: Sprint | F1: Toggle", _labelStyle);
        }

        private void InitStyles()
        {
            if (_boxStyle != null) return;

            _boxStyle = new GUIStyle(GUI.skin.box);
            _boxStyle.normal.background = MakeTexture(1, 1, new Color(0, 0, 0, 0.7f));

            _labelStyle = new GUIStyle(GUI.skin.label);
            _labelStyle.normal.textColor = Color.white;
            _labelStyle.fontSize = 13;
            _labelStyle.richText = true;

            _headerStyle = new GUIStyle(_labelStyle);
            _headerStyle.fontStyle = FontStyle.Bold;
            _headerStyle.fontSize = 14;
            _headerStyle.normal.textColor = new Color(0.5f, 0.9f, 1f);
        }

        private static Texture2D MakeTexture(int w, int h, Color color)
        {
            var tex = new Texture2D(w, h);
            var pixels = new Color[w * h];
            for (int i = 0; i < pixels.Length; i++)
                pixels[i] = color;
            tex.SetPixels(pixels);
            tex.Apply();
            return tex;
        }

        private bool IsTogglePressedThisFrame()
        {
#if ENABLE_INPUT_SYSTEM
            var keyboard = UnityEngine.InputSystem.Keyboard.current;
            if (keyboard == null) return false;
            return toggleKey switch
            {
                KeyCode.F1 => keyboard.f1Key.wasPressedThisFrame,
                KeyCode.F2 => keyboard.f2Key.wasPressedThisFrame,
                KeyCode.F3 => keyboard.f3Key.wasPressedThisFrame,
                KeyCode.F4 => keyboard.f4Key.wasPressedThisFrame,
                KeyCode.F5 => keyboard.f5Key.wasPressedThisFrame,
                KeyCode.F6 => keyboard.f6Key.wasPressedThisFrame,
                KeyCode.F7 => keyboard.f7Key.wasPressedThisFrame,
                KeyCode.F8 => keyboard.f8Key.wasPressedThisFrame,
                KeyCode.F9 => keyboard.f9Key.wasPressedThisFrame,
                KeyCode.F10 => keyboard.f10Key.wasPressedThisFrame,
                KeyCode.F11 => keyboard.f11Key.wasPressedThisFrame,
                KeyCode.F12 => keyboard.f12Key.wasPressedThisFrame,
                _ => keyboard.f1Key.wasPressedThisFrame
            };
#else
            return Input.GetKeyDown(toggleKey);
#endif
        }
    }
}
