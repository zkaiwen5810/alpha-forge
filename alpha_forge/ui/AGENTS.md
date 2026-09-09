# UI principles

- Isolate application integration at the UI boundary. Keep lower components
  independent through small interfaces and explicit callbacks.
- Give each state value one owner. Keep local state with its component and
  coordinate shared behavior in the nearest common parent.
- Prefer composition and established framework interfaces over custom
  abstractions. Avoid reaching through components into their internals.
- Make update and notification flow explicit. Notify after state changes;
  keep redraw notifications separate from application events and actions.
- Mark inherited framework methods with `@override`. Document protocol methods
  and registered callbacks with who calls them; do not mark these as overrides.
  Clearly distinguish application-owned APIs from framework hooks.
- Keep interaction policy with its owning component. Account for focus, modal
  behavior, event propagation, and terminal capabilities when changing input.
- Keep constructor parameters only when callers need them. Preserve behavior
  during refactors; verify affected focus, input, mouse, and layout scenarios.
