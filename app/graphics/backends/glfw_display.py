from __future__ import annotations

import glfw


class GlfwDisplay:
    """Owns the native OpenGL window and its swapchain."""

    def __init__(
        self,
        width: int,
        height: int,
        title: str,
        *,
        vsync: bool = True,
        visible: bool = True,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("display dimensions must be positive")
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")

        self._closed = False
        self._key_events: list[int] = []
        try:
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE if visible else glfw.FALSE)
            self._window = glfw.create_window(width, height, title, None, None)
            if self._window is None:
                raise RuntimeError("GLFW could not create an OpenGL 3.3 window")
            glfw.make_context_current(self._window)
            glfw.swap_interval(1 if vsync else 0)
            glfw.set_key_callback(self._window, self._on_key)
        except Exception:
            glfw.terminate()
            raise

    @property
    def framebuffer_size(self) -> tuple[int, int]:
        return glfw.get_framebuffer_size(self._window)

    @property
    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self._window))

    def make_current(self) -> None:
        glfw.make_context_current(self._window)

    def poll_events(self) -> None:
        glfw.poll_events()

    def swap_buffers(self) -> None:
        glfw.swap_buffers(self._window)

    def consume_key_events(self) -> tuple[int, ...]:
        events = tuple(self._key_events)
        self._key_events.clear()
        return events

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        glfw.destroy_window(self._window)
        glfw.terminate()

    def _on_key(
        self,
        window: object,
        key: int,
        scancode: int,
        action: int,
        mods: int,
    ) -> None:
        del window, scancode, mods
        if action != glfw.PRESS:
            return
        key_map = {
            glfw.KEY_ESCAPE: 27,
            glfw.KEY_Q: ord("q"),
            glfw.KEY_1: ord("1"),
            glfw.KEY_2: ord("2"),
            glfw.KEY_3: ord("3"),
            glfw.KEY_SPACE: ord(" "),
            glfw.KEY_S: ord("s"),
            glfw.KEY_W: ord("w"),
            glfw.KEY_V: ord("v"),
            glfw.KEY_T: ord("t"),
            glfw.KEY_M: ord("m"),
            glfw.KEY_P: ord("p"),
            glfw.KEY_K: ord("k"),
            glfw.KEY_R: ord("r"),
            glfw.KEY_C: ord("c"),
        }
        if key in key_map:
            self._key_events.append(key_map[key])
